"""Pretraining through the native MaxText training loop."""

from contextlib import ExitStack, closing

from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
import numpy as np

from lilax._backend.common.checkpoints import checkpointing
from lilax._backend.common.checkpoints import restore_source
from lilax._backend.common.goodput import (
    RECORD_JOB_START_TIME, create_goodput_recorder, maybe_monitor_goodput, record_goodput,
)
from lilax._backend.configs.runtime import resolve_runtime_config
from lilax._backend.configs.pyconfig import HyperParameters
from lilax._backend.input_pipeline import input_pipeline_interface
from lilax._backend.input_pipeline.hf_data_processing import make_hf_eval_iterator
from lilax._backend.trainers.pre_train import run as native_run
from lilax._backend.trainers.pre_train.moe import graphdefs as native_graphdefs
from lilax._backend.trainers.pre_train.steps import eval_step as native_eval_step
from lilax._backend.trainers.pre_train.start_step import get_first_step
from lilax._backend.trainers.diloco import diloco_sharding
from lilax._backend.common.checkpoints.restore_source_types import RestoreSource
from lilax._backend.distributed import sharding
from lilax._backend.utils import train_utils
from lilax._backend.utils.globals import EPS
from lilax.configuration import QwixQATConfig
from lilax.data_pipeline.trainer import validate_eval_dataset

from ..lifecycle._base import _PostTrainer
from ..lifecycle.checkpoint_request import checkpoint_request


class Trainer(_PostTrainer):
  def __init__(
      self, model=None, *, args, data_collator=None, train_dataset=None, eval_dataset=None, processing_class=None,
      callbacks=None, optimizers=(None, None), model_init=None,
  ):
    if model is None:
      if model_init is None:
        raise ValueError("Trainer requires model or model_init.")
      model = model_init()
    super().__init__(
        model, args, train_dataset, eval_dataset, processing_class, data_collator, optimizers, callbacks,
        defer_materialization=True,
    )
    self._train_state = None
    self._setup = None
    self._graphdef = None
    self.model_init = model_init

  def train(self, resume_from_checkpoint=None, *, checkpoint_step=None):
    policy = self.args.checkpoint_config
    if policy is not None and (policy.save_only_lora_params is not None or policy.restore_only_lora_params is not None):
      raise NotImplementedError("LoRA-only checkpoint selection is supported by the Tunix post-training backends.")
    request = checkpoint_request(resume_from_checkpoint, checkpoint_step)
    checkpoint_step = -1 if request.step is None else request.step
    if request.directory is not None:
      self.config = HyperParameters(self.config._pydantic_config.model_copy(update={"checkpoint_dir": request.directory}))
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
        selected_source = (
            RestoreSource("none") if initial_attempt and not request.resume else restore_source.select_restore_source(
                checkpoint_manager, attempt_config.load_parameters_path, attempt_config.load_full_state_path,
                step=restore_step, source_checkpoint_layout=attempt_config.source_checkpoint_layout,
            )
        )
        if request.resume and selected_source.kind == "none":
          raise ValueError("No training checkpoint is available to resume.")
        marker = restore_source.read_training_graph_metadata(selected_source)
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
      if rebuild_model:
        self.model = self.model_init()
        self.model.runtime.initialize()
      native_model = self.model.native_model
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
      for iterator in (train_iterator, eval_iterator):
        if iterator is not None and hasattr(iterator, "close"):
          resources.callback(iterator.close)
      selection = {"checkpoint_manager": checkpoint_manager, "selected_source": selected_source} if supported_manager else (
          {"selected_source": RestoreSource("none")} if initial_attempt and not request.resume else {}
      )
      setup = train_utils.setup_train_loop(
          attempt_config, attempt_recorder, elastic_retry_cache=retry_cache,
          model=native_model, mesh=mesh, runtime=runtime,
          data_iterator=train_iterator, eval_data_iterator=eval_iterator, checkpoint_step=restore_step, **selection,
          tx=self.optimizers[0], learning_rate_schedule=self.optimizers[1],
      )
      self.model.native_model = native_model
      self._setup = setup
      if setup[1] is not None:
        resources.callback(checkpointing.wait_until_finished, setup[1])
      self._graphdef = setup[3] if attempt_config.enable_diloco else nnx.graphdef(setup[10])
      self.callback_handler.optimizer = setup[10].optimizer if not attempt_config.enable_diloco else self.optimizers[0]
      self.callback_handler.lr_scheduler = setup[5]
      initial_attempt = False
      rebuild_model = False
      return setup

    def reinitialize():
      nonlocal rebuild_model
      if self.model_init is None:
        raise NotImplementedError("Elastic resize for a supplied model requires model_init.")
      overrides = runtime._overrides | (self.args._overrides() if self.args is not None else {})
      overrides["run_name"] = runtime.config.run_name
      runtime.config = resolve_runtime_config(
          runtime.public_config, overrides=overrides, train_dataset_provided=self.train_dataset is not None,
          eval_dataset_provided=self.eval_dataset is not None,
      )
      runtime.mesh = None
      attempt_config = self.args._resolve(
          runtime, train_dataset_provided=self.train_dataset is not None, eval_dataset_provided=self.eval_dataset is not None,
      ) if self.args is not None else runtime.config
      attempt_config = self._with_model_config(attempt_config, self.model)
      train_utils.validate_train_config(attempt_config)
      rebuild_model = True
      return attempt_config, create_goodput_recorder(attempt_config)

    record_goodput(recorder, RECORD_JOB_START_TIME)
    xprof = runtime._profile.xprof if runtime._profile is not None else None
    with ExitStack() as resources, self._reporting_context(), maybe_monitor_goodput(config):
      setup = None
      if not config.elastic_enabled:
        setup = setup_attempt(config, recorder, None)
        config = self.config
      else:
        self.model.materialize(runtime=runtime)
        self._reject_qat_checkpointing()
        config = self._with_model_config(config, self.model)
        self.config = config
      self._train_state = native_run.run_elastic(
          config, recorder, setup=setup, setup_fn=setup_attempt if setup is None else None,
          reinitialize=reinitialize,
          launch_xprof=xprof is not None and xprof.launch_server,
          xprof_port=xprof.port if xprof is not None else 8791,
          training_graph=training_graph, lifecycle=self,
      )
    self._sync_model()
    self.state.global_step = get_first_step(None, self._train_state)
    return self._train_output()

  def _sync_model(self):
    if self._train_state is None:
      return
    if self.config.enable_diloco:
      parameters = nnx.state(self.model.native_model, nnx.Param)
      nnx.replace_by_pure_dict(parameters, self._train_state.params)
      nnx.update(self.model.native_model, parameters)
    else:
      nnx.update(self.model.native_model, self._train_state.model)

  def _end_step(self, step, loss):
    self._sync_model()
    return super()._end_step(step, loss)

  def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
    dataset = self.eval_dataset if eval_dataset is None else eval_dataset
    validate_eval_dataset(dataset)
    if dataset is None:
      raise ValueError("Trainer.evaluate requires an evaluation dataset.")
    if isinstance(dataset, dict):
      metrics = {}
      for name, source in dataset.items():
        metrics.update(self.evaluate(source, ignore_keys, f"{metric_key_prefix}_{name}"))
      return metrics
    _, eval_iterator = self._data_iterators(eval_dataset=dataset, evaluation_only=True)
    if eval_iterator is None:
      raise ValueError("No evaluation iterator is configured.")
    with closing(eval_iterator):
      return self._evaluate_iterator(eval_iterator, metric_key_prefix)

  def _evaluate_iterator(self, eval_iterator, metric_key_prefix):
    if self._setup is None or self._train_state is None:
      self._setup = train_utils.setup_train_loop(
          self.config, create_goodput_recorder(self.config), model=self.model.native_model,
          mesh=self.model.runtime.mesh, runtime=self.model.runtime, data_iterator=iter(()),
          eval_data_iterator=eval_iterator, checkpoint_manager=None, selected_source=RestoreSource("none"),
          tx=self.optimizers[0], learning_rate_schedule=self.optimizers[1],
      )
      self._graphdef = self._setup[3] if self.config.enable_diloco else nnx.graphdef(self._setup[10])
      self._train_state = self._setup[10] if self.config.enable_diloco else nnx.state(self._setup[10])
    config = self.config
    mesh = self.model.runtime.mesh
    if hasattr(eval_iterator, "reset"):
      eval_iterator.reset()
    graphdef = self._graphdef
    state_shardings = self._setup[2]
    if not config.enable_diloco:
      graphdef = native_graphdefs.build_eval_graphdef(config, graphdef, self._train_state) or graphdef
      _, state_shardings = sharding.maybe_update_params_sharding_with_opt(config, state_shardings)
    data_sharding = sharding.get_input_data_sharding(config, mesh, rules=config.logical_axis_rules_for_eval)
    eval_step = train_utils.jit_eval_step(config, graphdef, state_shardings, data_sharding, native_eval_step.eval_step)
    dropless_step = None
    if not config.enable_diloco and config.moe_dropless_fallback == "step":
      dropless_graphdef = native_graphdefs.build_dropless_graphdef(config, self._graphdef, self._train_state)
      dropless_step = train_utils.jit_eval_step(
          config, dropless_graphdef, state_shardings, data_sharding, native_eval_step.eval_step
      )
    totals = {}
    count = 0
    for batch in eval_iterator:
      if config.enable_diloco:
        batch = diloco_sharding.reshape_first_axis_with_diloco(config.num_diloco_replicas, batch)
      batch = jax.device_put(batch, data_sharding)
      rng_args = (jax.random.fold_in(self._setup[0], self._train_state.step),) if config.enable_diloco else ()
      with jax.set_mesh(mesh), logical_axis_rules(config.logical_axis_rules_for_eval):
        metrics = eval_step(self._train_state, batch, *rng_args)
        if dropless_step is not None and bool(metrics.get("has_moe_overflow")):
          metrics = dropless_step(self._train_state, batch, *rng_args)
      for name, value in metrics["scalar"].items():
        totals[name] = totals.get(name, 0.0) + float(value)
      count += 1
    if not count:
      raise ValueError("The evaluation iterator is empty.")
    loss = totals["evaluation/total_loss"] / (totals["evaluation/total_weights"] + EPS)
    return self._evaluation_metrics({
        "eval_loss": loss, "eval_perplexity": float(np.exp(loss)),
        "eval_total_weights": totals["evaluation/total_weights"],
        "eval_moe_lb_loss": totals["evaluation/moe_lb_loss"] / count,
        "eval_indexer_loss": totals["evaluation/indexer_loss"] / count,
        "eval_mtp_loss": totals["evaluation/mtp_loss"] / count,
        "eval_mtp_acceptance_rate_percent": totals["evaluation/mtp_acceptance_rate_percent"] / count,
        "eval_z_loss": totals["evaluation/z_loss"] / count,
    }, metric_key_prefix)

  def save_state(self):
    if self._setup is None or self._train_state is None:
      raise RuntimeError("Training must complete before saving its state.")
    checkpoint_manager = self._setup[1]
    if checkpoint_manager is None:
      raise ValueError("Checkpointing is disabled in the training configuration.")
    qat_config = self.model.existing_graph_qat_config
    training_graph = {
        "version": 1, "kind": "qat" if qat_config is not None else "fp",
        "qat_config": qat_config.model_dump(mode="json") if qat_config is not None else None,
    }
    receipt = checkpointing.maybe_save_checkpoint(
        checkpoint_manager, self._train_state, self.config, self._setup[6], training_graph=training_graph,
        force=True, **self._save_action(self.state.global_step), return_result=True,
    )
    self._resolve_save(receipt, checkpoint_manager)
    checkpointing.wait_until_finished(checkpoint_manager)

  def save_model(self, output_dir=None):
    self._sync_model()
    self.model.save_pretrained(self.args.output_dir if output_dir is None else output_dir)
    if self.processing_class is not None:
      self.processing_class.save_pretrained(self.args.output_dir if output_dir is None else output_dir)
