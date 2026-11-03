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
from lilax._backend.trainers.post_train.checkpoint_selection import select_training_checkpoint
from lilax.configuration import QwixQATConfig
from lilax.peft import PeftModel
from .training_args import PRECISION_FIELDS, TFDS_C4_FIELDS
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
    changed = [
        name for name in PRECISION_FIELDS
        if name in overrides and overrides[name] != getattr(graph_config, name)
    ]
    if changed:
      raise ValueError(f"Cannot change precision settings on a materialized model: {', '.join(changed)}.")

  @staticmethod
  def _check_precision_options(config, overrides):
    quant = config.quantization
    qwix = config.use_qwix_quantization and not config.use_batch_split_schedule
    qwix_modes = ("int4", "int8", "fp4", "fp8", "fp8_e5m2", "fp8_e4m3", "fp8_full", "fp8_gpu", "fp8_nanoo")
    if qwix and quant and quant not in qwix_modes:
      raise ValueError(f"Qwix has no provider for quantization={quant!r}.")
    if overrides.get("use_qwix_quantization") and not quant:
      raise ValueError("use_qwix_quantization requires a quantization recipe.")
    full = quant == "fp8_full" and (qwix or (config.use_batch_split_schedule and not config.use_manual_quantization))
    applicable = {
        "jax_mx_gradient_mode": quant in ("jax_mxfp4", "jax_mxfp8"),
        "quant_cfg_path": quant == "intmp" and not qwix,
        "quantization_local_shard_count": quant == "int8" and not qwix and not config.use_batch_split_schedule,
        "replicate_quant_scale": quant in ("int8", "intmp") and not qwix and not config.use_batch_split_schedule,
        "qwix_bwd_weight_grad_tile_count": qwix and quant in qwix_modes and quant != "fp8_full",
        "weight_quantization_calibration_method": full,
        "act_quantization_calibration_method": full or quant == "serve_fp8_weight",
        "bwd_quantization_calibration_method": full,
        "drhs_grad_quantization_calibration_method": qwix and quant == "fp8_full",
        "quantize_mtp": qwix and quant == "fp8_full",
        "quantize_router_proj": qwix and quant in qwix_modes,
        "quantize_logits_proj": qwix and quant == "fp8_full",
        "logits_proj_quant_calibration_method": qwix and quant == "fp8_full" and config.quantize_logits_proj,
        "logits_proj_bwd_quant_calibration_method": qwix and quant == "fp8_full" and config.quantize_logits_proj,
        "sa_qproj_bwd_bf16": qwix and quant == "fp8_full",
        "use_manual_quantization": config.use_batch_split_schedule,
    }
    unused = [name for name, supported in applicable.items() if name in overrides and not supported]
    if unused:
      raise ValueError(f"Precision settings do not apply to the selected recipe: {', '.join(unused)}.")

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
    manager.save_only_lora_params = getattr(self.args, "save_only_lora_params", None)
    manager.restore_only_lora_params = getattr(self.args, "restore_only_lora_params", None)
    manager, checkpoint_step, marker = select_training_checkpoint(
        self.config.checkpoint_dir, None, step=checkpoint_step, manager=manager,
    )
    if marker is not None:
      if (not isinstance(marker, dict) or set(marker) != {"version", "kind", "qat_config"}
          or type(marker["version"]) is not int or marker["version"] != 1
          or marker["kind"] not in ("qat", "fp")
          or (marker["kind"] == "fp" and marker["qat_config"] is not None)
          or (marker["kind"] == "qat" and not isinstance(marker["qat_config"], dict))):
        raise ValueError("Invalid full-state training graph metadata.")
      if marker["kind"] == "qat":
        if self.config.lora.lora_restore_path:
          raise ValueError("Cannot combine a QAT full-state restore with lora_restore_path.")
        self.model.prepare_for_full_state_restore(
            QwixQATConfig.model_validate(marker["qat_config"]), runtime=self.model.runtime,
        )
      elif self.model.pending_qat_config is not None:
        raise ValueError("The selected full-state checkpoint has an FP graph, not the requested QAT graph.")
    elif checkpoint_step is not None and self.model.pending_qat_config is not None:
      raise ValueError("Cannot restore QAT full state without training graph metadata.")
    if (self.model.pending_qat_config is not None
        and (isinstance(self.model, PeftModel) or self.config.lora.enable_lora or self.config.lora.lora_restore_path)):
      if manager.save_only_lora_params is not False or manager.restore_only_lora_params is not False:
        raise NotImplementedError("QAT with LoRA requires full-model save and restore to preserve QAT statistics.")
      if self.config.lora.lora_restore_path or getattr(self.config, "learn_to_init_mode", False):
        raise NotImplementedError("QAT with LoRA requires full-model checkpointing with optimizer state.")
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
