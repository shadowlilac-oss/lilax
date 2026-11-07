"""Shared training inputs and native training lifecycle."""

from contextlib import ExitStack, nullcontext

import jax

from lilax._backend.common.goodput import (
    RECORD_JOB_START_TIME, create_goodput_recorder, maybe_monitor_goodput, record_goodput,
)
from lilax._backend.common.profiler import xprof_server
from lilax._backend.configs import pyconfig
from lilax.data_pipeline.trainer import create_data_iterators, dataset_overrides, validate_eval_dataset
from lilax._processing.preferences import processing_overrides
from lilax.callbacks._configuration import callback_overrides
from ._post_base_checkpoint import prepare_training_checkpoint
from ._post_base_precision import check_precision_options, check_precision_overrides
from ..arguments.training_args import TFDS_C4_FIELDS
from .callbacks import TrainerEvents
from .checkpoint_request import checkpoint_request


class _PostTrainer(TrainerEvents):
  def __init__(
      self, model, args=None, train_dataset=None, eval_dataset=None, processing_class=None,
      data_collator=None, optimizers=(None, None), callbacks=None, *, resolved_config=None, defer_materialization=False,
  ):
    validate_eval_dataset(eval_dataset)
    callbacks = [callback() if isinstance(callback, type) else callback for callback in (callbacks or [])]
    overrides = args._overrides() if args is not None else {}
    overrides.update(callback_overrides(callbacks))
    overrides.update(processing_overrides(processing_class))
    overrides.update(dataset_overrides(train_dataset))
    graph_config = model.existing_graph_config
    if graph_config is not None:
      self._check_precision_overrides(graph_config, overrides)
    elif resolved_config is None:
      model.runtime.initialize(
          overrides=overrides, train_dataset_provided=train_dataset is not None,
          eval_dataset_provided=eval_dataset is not None,
      )
      self._check_precision_options(model.runtime.config, overrides)
      if not defer_materialization:
        model.materialize(runtime=model.runtime)
    config = resolved_config
    if config is not None:
      updates = processing_overrides(processing_class) | dataset_overrides(train_dataset) | callback_overrides(callbacks)
      config = pyconfig.HyperParameters(config._pydantic_config.model_copy(
          update={name: value for name, value in updates.items() if getattr(config, name) != value},
      ))
    if config is None:
      config = args._resolve(
          model.runtime, train_dataset_provided=train_dataset is not None, eval_dataset_provided=eval_dataset is not None,
      ) if args is not None else model.runtime.config
    unsupported_data = sorted(set(TFDS_C4_FIELDS) & overrides.keys()) if config.dataset_type != "c4_mlperf" else []
    if unsupported_data:
      raise ValueError(f"Training data pipeline does not use these settings: {unsupported_data}")
    construction = model.existing_graph_config or model.runtime.config
    if config.max_target_length != construction.max_target_length:
      raise ValueError("Training sequence length must match the constructed model's max_target_length.")
    if (
        config.micro_batch_size_to_train_on != construction.micro_batch_size_to_train_on
        and (construction.using_pipeline_parallelism or (construction.quantization and construction.use_qwix_quantization))
    ):
      raise ValueError("Training batch size must match the batch size used to construct this model.")
    if construction.use_multimodal and config.per_device_batch_size != construction.per_device_batch_size:
      raise ValueError("Training batch size must match the batch size used to construct this multimodal model.")
    if (
        (config.gradient_accumulation_steps > 1) != (construction.gradient_accumulation_steps > 1)
        and (construction.num_experts > 1 or construction.hardware in ("gpu", "gpu_multiprocess"))
    ):
      raise ValueError("Gradient accumulation must match the mode used to construct this model.")
    if model.runtime.mesh is not None and any(
        getattr(config, name) != getattr(model.runtime.config, name)
        for name in ("mesh_axes", "ici_parallelism", "dcn_parallelism", "data_sharding", "logical_axis_rules")
    ):
      raise ValueError("Training configuration cannot change an initialized mesh or its sharding.")
    if not model.is_materialized and not defer_materialization:
      model.materialize(runtime=model.runtime)
    if model.is_materialized:
      config = self._with_model_config(config, model)
    self.model = model
    self.config = config
    self.args = args
    self.train_dataset = train_dataset
    self.eval_dataset = eval_dataset
    self.processing_class = processing_class
    self.data_collator = data_collator
    self.optimizers = optimizers
    self.native_trainer = None
    self._init_events(callbacks)

  @staticmethod
  def _check_precision_overrides(graph_config, overrides):
    return check_precision_overrides(graph_config, overrides)

  @staticmethod
  def _check_precision_options(config, overrides):
    return check_precision_options(config, overrides)

  @staticmethod
  def _with_model_config(config, model):
    updates = {}
    if config.scan_layers != model.construction_config.scan_layers:
      updates["scan_layers"] = model.construction_config.scan_layers
    lora = model.construction_config.lora
    if lora.enable_lora:
      updates["lora"] = lora.model_copy(deep=True)
    if not updates:
      return config
    updated = config._pydantic_config.model_copy(update=updates, deep=True)
    return pyconfig.HyperParameters(updated)

  def _prepare_training_checkpoint(self, checkpoint_step, manager, *, resume=True):
    manager, checkpoint_step = prepare_training_checkpoint(
        checkpoint_step, manager, self.args, self.config, self.model, resume=resume,
    )
    self.model.materialize(runtime=self.model.runtime)
    self.config = self._with_model_config(self.config, self.model)
    qat_config = self.model.existing_graph_qat_config
    manager.training_graph = {
        "version": 1, "kind": "qat" if qat_config is not None else "fp",
        "qat_config": qat_config.model_dump(mode="json") if qat_config is not None else None,
    }
    return manager, checkpoint_step

  def _reject_qat_checkpointing(self):
    if self.model.existing_graph_qat_config is not None:
      raise NotImplementedError("Qwix QAT training checkpoint save and restore are not supported.")

  def train(self, resume_from_checkpoint=None, *, checkpoint_step=None):
    request = checkpoint_request(resume_from_checkpoint, checkpoint_step)
    if request.directory is not None:
      self.config = pyconfig.HyperParameters(self.config._pydantic_config.model_copy(
          update={"checkpoint_dir": request.directory},
      ))
    runtime = self.model.runtime
    tunix_config = self._tunix_config(self.config)
    with ExitStack() as resources:
      manager = self._checkpoint_manager(tunix_config)
      resources.callback(manager.close)
      manager, checkpoint_step = self._prepare_training_checkpoint(request.step, manager, resume=request.resume)
      train_iterator, eval_iterator = self._data_iterators()
      for iterator in (train_iterator, eval_iterator):
        if iterator is not None and hasattr(iterator, "close"):
          resources.callback(iterator.close)
      resources.callback(manager.wait_until_finished)
      recorder = create_goodput_recorder(self.config)
      report_to = []
      record_goodput(recorder, RECORD_JOB_START_TIME)
      xprof = runtime._profile.xprof if runtime._profile is not None else None
      server = (xprof_server(self.config.tensorboard_dir, xprof.port, jax.process_index())
                if xprof is not None and xprof.launch_server else nullcontext())
      with self._reporting_context(), maybe_monitor_goodput(self.config), server:
        self.native_trainer, _ = self._native_train(
            self.config, recorder, model=self.model.native_model, mesh=runtime.mesh,
            train_iterator=train_iterator, eval_iterator=eval_iterator, checkpoint_step=checkpoint_step,
            report_to=report_to,
            checkpoint_manager=manager,
            **self._backend_options(),
        )
      return self._train_output()

  def _backend_options(self):
    return {"optimizer": self.optimizers[0], "learning_rate_schedule": self.optimizers[1], "lifecycle": self}

  def _data_iterators(self, *, eval_dataset=None, evaluation_only=False):
    dataset = self.eval_dataset if eval_dataset is None else eval_dataset
    if isinstance(dataset, dict):
      dataset = next(iter(dataset.values()))
    return create_data_iterators(
        self.config, self.model.runtime.mesh, train_dataset=self.train_dataset, eval_dataset=dataset,
        processing_class=self.processing_class, data_collator=self.data_collator,
        formatting_func=getattr(self, "formatting_func", None),
        dataset_text_field=getattr(self.args, "dataset_text_field", "text"), evaluation_only=evaluation_only,
    )
