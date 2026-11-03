"""Shared training inputs and native training lifecycle."""

from functools import partial
from contextlib import nullcontext

import jax

from lilax._backend.common.goodput import (
    RECORD_JOB_START_TIME, create_goodput_recorder, maybe_monitor_goodput, record_goodput,
)
from lilax._backend.common.profiler import xprof_server
from lilax._backend.configs import pyconfig
from lilax._backend.input_pipeline.input_pipeline_interface import create_data_iterator
from ._post_base_checkpoint import prepare_training_checkpoint
from ._post_base_precision import check_precision_options, check_precision_overrides
from .training_args import TFDS_C4_FIELDS
from .reporting import MetricReporter


class _PostTrainer:
  def __init__(
      self, model, args=None, train_dataset=None, eval_dataset=None, processing_class=None,
      data_collator=None, optimizers=(None, None), callbacks=None, *, resolved_config=None, defer_materialization=False,
  ):
    if data_collator is not None:
      raise NotImplementedError("Native post-training does not accept a data collator.")
    if optimizers != (None, None) or callbacks:
      raise NotImplementedError("Native post-training does not accept custom optimizers or callbacks.")
    overrides = args._overrides() if args is not None else {}
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
    if config is None:
      config = args.resolve(
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
    if (train_dataset is not None or eval_dataset is not None) and config.dataset_type != "hf":
      raise ValueError("Supplying datasets requires dataset_type='hf'.")
    if eval_dataset is not None and config.eval_interval <= 0:
      raise ValueError("Supplying an evaluation dataset requires a positive eval_interval.")
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
    self.native_trainer = None

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

  def _prepare_training_checkpoint(self, checkpoint_step, manager):
    manager, checkpoint_step = prepare_training_checkpoint(checkpoint_step, manager, self.args, self.config, self.model)
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

  def train(self, resume_from_checkpoint=None):
    if resume_from_checkpoint is None or resume_from_checkpoint is True:
      checkpoint_step = None
    elif (
        isinstance(resume_from_checkpoint, int) and not isinstance(resume_from_checkpoint, bool)
        and resume_from_checkpoint >= 0
    ):
      checkpoint_step = resume_from_checkpoint
    else:
      raise ValueError("resume_from_checkpoint must be None, True, or a nonnegative checkpoint step.")
    runtime = self.model.runtime
    tunix_config = self._tunix_config(self.config)
    manager = self._checkpoint_manager(tunix_config)
    manager, checkpoint_step = self._prepare_training_checkpoint(checkpoint_step, manager)
    train_iterator, eval_iterator = self._data_iterators()
    recorder = create_goodput_recorder(self.config)
    report_to = self.args.report_to if self.args is not None else None
    reporter_factory = partial(MetricReporter, report_to=report_to) if report_to else None
    record_goodput(recorder, RECORD_JOB_START_TIME)
    server = (
        xprof_server(self.config.tensorboard_dir, self.args.xprof_port, jax.process_index())
        if self.args is not None and self.args.launch_xprof else nullcontext()
    )
    with maybe_monitor_goodput(self.config), server:
      self.native_trainer, _ = self._native_train(
          self.config, recorder, model=self.model.native_model, mesh=runtime.mesh,
          train_iterator=train_iterator, eval_iterator=eval_iterator, checkpoint_step=checkpoint_step,
          reporter_factory=reporter_factory, report_to=report_to,
          checkpoint_manager=manager,
      )
    return self.native_trainer

  def _data_iterators(self):
    train_iterator = eval_iterator = None
    if self.train_dataset is not None or self.eval_dataset is not None or self.processing_class is not None:
      train_iterator, eval_iterator = create_data_iterator(
          self.config, self.model.runtime.mesh, train_dataset=self.train_dataset, eval_dataset=self.eval_dataset,
          processing_class=self.processing_class,
      )
    return train_iterator, eval_iterator
