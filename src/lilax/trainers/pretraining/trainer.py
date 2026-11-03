"""Pretraining through the native MaxText training loop."""

from functools import partial

from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
import numpy as np

from lilax._backend.common.checkpoints import checkpointing
from lilax._backend.common.goodput import (
    RECORD_JOB_START_TIME, create_goodput_recorder, maybe_monitor_goodput, record_goodput,
)
from lilax._backend.configs.runtime import resolve_runtime_config
from lilax._backend.input_pipeline import input_pipeline_interface
from lilax._backend.input_pipeline.hf_data_processing import make_hf_eval_iterator
from lilax._backend.trainers.pre_train import train as native_train
from lilax._backend.utils import sharding, train_utils
from lilax._backend.utils.globals import EPS
from lilax.configuration import QwixQATConfig

from ..lifecycle._base import _PostTrainer
from .run import train_pretraining
from ..telemetry.reporting import MetricReporter


class Trainer(_PostTrainer):
  def __init__(
      self, model, args=None, train_dataset=None, eval_dataset=None, processing_class=None,
      data_collator=None, optimizers=(None, None), callbacks=None, *, model_factory=None,
  ):
    super().__init__(
        model, args, train_dataset, eval_dataset, processing_class, data_collator, optimizers, callbacks,
        defer_materialization=True,
    )
    self.state = None
    self._setup = None
    self._graphdef = None
    self.model_factory = model_factory

  def train(self, resume_from_checkpoint=None):
    if resume_from_checkpoint is None or resume_from_checkpoint is True:
      checkpoint_step = -1
    elif (
        isinstance(resume_from_checkpoint, int) and not isinstance(resume_from_checkpoint, bool)
        and resume_from_checkpoint >= 0
    ):
      checkpoint_step = resume_from_checkpoint
    else:
      raise ValueError("resume_from_checkpoint must be None, True, or a nonnegative checkpoint step.")
    config = self.config
    runtime = self.model.runtime
    train_utils.validate_train_config(config)
    recorder = create_goodput_recorder(config)
    initial_attempt = True
    rebuild_model = False
    training_graph = {}

    def setup_attempt(attempt_config, attempt_recorder, retry_cache):
      nonlocal initial_attempt, rebuild_model
      restore_step = checkpoint_step if initial_attempt else -1
      self.config = attempt_config
      if runtime.mesh is None:
        runtime.initialize(retry_cache)
      mesh = runtime.mesh
      selected_source = None
      checkpoint_manager = None
      supported_manager = not (
          attempt_config.enable_emergency_checkpoint or attempt_config.enable_multi_tier_checkpointing
          or attempt_config.elastic_enabled
      )
      if supported_manager:
        checkpoint_manager = train_utils.create_checkpoint_manager(attempt_config, mesh, None)
        if restore_step >= 0 and checkpoint_manager is None:
          raise ValueError(f"Cannot restore checkpoint step {restore_step} without a checkpoint manager.")
        selected_source = checkpointing.select_restore_source(
            checkpoint_manager, attempt_config.load_parameters_path, attempt_config.load_full_state_path,
            step=restore_step, source_checkpoint_layout=attempt_config.source_checkpoint_layout,
        )
        marker = checkpointing.read_training_graph_metadata(selected_source)
        if marker is not None:
          if (not isinstance(marker, dict) or set(marker) != {"version", "kind", "qat_config"}
              or type(marker["version"]) is not int or marker["version"] != 1
              or marker["kind"] not in ("qat", "fp")
              or (marker["kind"] == "fp" and marker["qat_config"] is not None)
              or (marker["kind"] == "qat" and not isinstance(marker["qat_config"], dict))):
            raise ValueError("Invalid full-state training graph metadata.")
          if marker["kind"] == "qat":
            if selected_source.kind in ("manager", "full_state") and attempt_config.lora.lora_restore_path:
              raise ValueError("Cannot combine a QAT full-state restore with lora_restore_path.")
            selected_qat_config = QwixQATConfig.model_validate(marker["qat_config"])
            self.model.prepare_for_full_state_restore(selected_qat_config, runtime=runtime)
      native_model = self.model_factory(attempt_config, mesh) if rebuild_model else self.model.native_model
      attempt_config = self._with_model_config(attempt_config, self.model)
      self.config = attempt_config
      qat_config = self.model.existing_graph_qat_config
      if qat_config is not None:
        if not supported_manager:
          raise NotImplementedError("QAT full-state checkpointing does not support elastic or emergency state.")
        if attempt_config.lora.enable_lora and not nnx.state(native_model, nnx.LoRAParam):
          raise ValueError("QAT LoRA checkpointing requires a supplied model with LoRA parameters.")
        if selected_source.kind in ("manager", "full_state") and marker is None:
          raise ValueError("Cannot restore QAT full state without training graph metadata.")
        if selected_source.kind == "dynamic":
          raise NotImplementedError("QAT full-state restore does not support dynamic checkpoint sources.")
      if supported_manager and marker is not None and marker["kind"] == "fp" and qat_config is not None:
        raise ValueError("Cannot restore an FP optimizer state into a QAT graph.")
      if supported_manager and marker is not None and marker["kind"] == "qat" and qat_config != selected_qat_config:
        raise ValueError("The model graph does not match the selected full-state QAT config.")
      training_graph.update({
          "version": 1, "kind": "qat" if qat_config is not None else "fp",
          "qat_config": qat_config.model_dump(mode="json") if qat_config is not None else None,
      })
      train_iterator, eval_iterator = self._data_iterators()
      selection = {"checkpoint_manager": checkpoint_manager, "selected_source": selected_source} if supported_manager else {}
      setup = train_utils.setup_train_loop(
          attempt_config, attempt_recorder, elastic_retry_cache=retry_cache,
          model=native_model, mesh=mesh, runtime=runtime,
          data_iterator=train_iterator, eval_data_iterator=eval_iterator, checkpoint_step=restore_step, **selection,
      )
      self.model.native_model = native_model
      self._setup = setup
      self._graphdef = setup[3] if attempt_config.enable_diloco else nnx.graphdef(setup[10])
      initial_attempt = False
      rebuild_model = False
      return setup

    def reinitialize():
      nonlocal rebuild_model
      if self.model_factory is None:
        raise NotImplementedError("Elastic resize for a supplied model requires model_factory.")
      overrides = runtime._overrides | (self.args._overrides() if self.args is not None else {})
      overrides["run_name"] = runtime.config.run_name
      runtime.config = resolve_runtime_config(
          runtime.public_config, overrides=overrides, train_dataset_provided=self.train_dataset is not None,
          eval_dataset_provided=self.eval_dataset is not None,
      )
      runtime.mesh = None
      attempt_config = self.args.resolve(
          runtime, train_dataset_provided=self.train_dataset is not None, eval_dataset_provided=self.eval_dataset is not None,
      ) if self.args is not None else runtime.config
      attempt_config = self._with_model_config(attempt_config, self.model)
      train_utils.validate_train_config(attempt_config)
      rebuild_model = True
      return attempt_config, create_goodput_recorder(attempt_config)

    record_goodput(recorder, RECORD_JOB_START_TIME)
    report_to = self.args.report_to if self.args is not None else None
    reporter_factory = partial(MetricReporter, report_to=report_to) if report_to else None
    with maybe_monitor_goodput(config):
      setup = None
      if not config.elastic_enabled:
        setup = setup_attempt(config, recorder, None)
        config = self.config
      else:
        self.model.materialize(runtime=runtime)
        self._reject_qat_checkpointing()
        config = self._with_model_config(config, self.model)
        self.config = config
      self.state = native_train.run_elastic(
          config, recorder, setup=setup, setup_fn=setup_attempt if setup is None else None,
          reinitialize=reinitialize, reporter_factory=reporter_factory,
          launch_xprof=self.args.launch_xprof if self.args is not None else False,
          xprof_port=self.args.xprof_port if self.args is not None else 8791,
          training_graph=training_graph,
      )
    if not self.config.enable_diloco:
      nnx.update(self.model.native_model, self.state.model)
    return self.state

  def evaluate(self, eval_dataset=None):
    if self._setup is None or self.state is None:
      raise RuntimeError("Training must complete before evaluation.")
    config = self.config
    mesh = self.model.runtime.mesh
    if eval_dataset is not None:
      if config.dataset_type != "hf" or config.eval_interval <= 0:
        raise ValueError("Supplying an evaluation dataset requires dataset_type='hf' and a positive eval_interval.")
      process_indices = input_pipeline_interface.get_process_loading_real_data(
          config.data_sharding, config.global_batch_size_to_load_eval, config.global_batch_size_to_eval_on,
          config.max_target_length, mesh,
      )
      eval_iterator = input_pipeline_interface.create_process_specific_iterator(
          config, mesh, process_indices,
          partial(make_hf_eval_iterator, dataset=eval_dataset, processing_class=self.processing_class), is_training=False,
      )
    else:
      eval_iterator = self._setup[9]
    if eval_iterator is None:
      raise ValueError("No evaluation iterator is configured.")
    if hasattr(eval_iterator, "reset"):
      eval_iterator.reset()
    graphdef = self._graphdef
    state_shardings = self._setup[2]
    if not config.enable_diloco:
      graphdef = native_train.build_eval_graphdef(config, graphdef, self.state) or graphdef
      _, state_shardings = sharding.maybe_update_params_sharding_with_opt(config, state_shardings)
    data_sharding = sharding.get_input_data_sharding(config, mesh, rules=config.logical_axis_rules_for_eval)
    eval_step = train_utils.jit_eval_step(config, graphdef, state_shardings, data_sharding, native_train.eval_step)
    dropless_step = None
    if not config.enable_diloco and config.moe_dropless_fallback == "step":
      dropless_graphdef = native_train.build_dropless_graphdef(config, self._graphdef, self.state)
      dropless_step = train_utils.jit_eval_step(
          config, dropless_graphdef, state_shardings, data_sharding, native_train.eval_step
      )
    totals = {}
    count = 0
    for batch in eval_iterator:
      batch = jax.device_put(batch, data_sharding)
      rng_args = (jax.random.fold_in(self._setup[0], self.state.step),) if config.enable_diloco else ()
      with jax.set_mesh(mesh), logical_axis_rules(config.logical_axis_rules_for_eval):
        metrics = eval_step(self.state, batch, *rng_args)
        if dropless_step is not None and bool(metrics.get("has_moe_overflow")):
          metrics = dropless_step(self.state, batch, *rng_args)
      for name, value in metrics["scalar"].items():
        totals[name] = totals.get(name, 0.0) + float(value)
      count += 1
      if 0 < config.eval_steps <= count:
        break
    if not count:
      raise ValueError("The evaluation iterator is empty.")
    loss = totals["evaluation/total_loss"] / (totals["evaluation/total_weights"] + EPS)
    return {
        "eval_loss": loss, "eval_perplexity": float(np.exp(loss)),
        "eval_total_weights": totals["evaluation/total_weights"],
        "eval_moe_lb_loss": totals["evaluation/moe_lb_loss"] / count,
        "eval_indexer_loss": totals["evaluation/indexer_loss"] / count,
        "eval_mtp_loss": totals["evaluation/mtp_loss"] / count,
        "eval_mtp_acceptance_rate_percent": totals["evaluation/mtp_acceptance_rate_percent"] / count,
        "eval_z_loss": totals["evaluation/z_loss"] / count,
    }

  def save_state(self):
    if self._setup is None or self.state is None:
      raise RuntimeError("Training must complete before saving its state.")
    checkpoint_manager = self._setup[1]
    if checkpoint_manager is None:
      raise ValueError("Checkpointing is disabled in the training configuration.")
    qat_config = self.model.existing_graph_qat_config
    training_graph = {
        "version": 1, "kind": "qat" if qat_config is not None else "fp",
        "qat_config": qat_config.model_dump(mode="json") if qat_config is not None else None,
    }
    checkpointing.maybe_save_checkpoint(
        checkpoint_manager, self.state, self.config, self._setup[6], training_graph=training_graph,
    )
    checkpointing.wait_until_finished(checkpoint_manager)

  def save_model(self, output_dir=None):
    if output_dir is not None and str(output_dir) != self.config.checkpoint_dir:
      raise ValueError("Native checkpoints can only be saved to the configured checkpoint directory.")
    self.save_state()
