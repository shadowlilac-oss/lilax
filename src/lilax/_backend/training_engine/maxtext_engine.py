"""Native MaxText training engine."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import contextlib
from typing import Any

from absl import logging
from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
import jax.numpy as jnp
from lilax._backend.common import common_types
from lilax._backend.common import train_state_nnx
from lilax._backend.configs import pyconfig
from lilax._backend.m3 import models as m3_models
from lilax._backend.training_engine import abstract_engine
from lilax._backend.training_engine import batch_sharding
from lilax._backend.training_engine import batch_signature
from lilax._backend.training_engine import checkpointing
from lilax._backend.training_engine import compiled_kernels
from lilax._backend.training_engine import gradient_accumulation
from lilax._backend.training_engine import gradient_sharding
from lilax._backend.training_engine import inflight_throttler
from lilax._backend.training_engine import kernel_compilation
from lilax._backend.training_engine import loss_invocation
from lilax._backend.training_engine import metrics as metrics_module
from lilax._backend.training_engine import micro_step_profiler as profiler_module
from lilax._backend.training_engine import payload_inputs
from lilax._backend.training_engine import pure_state
from lilax._backend.training_engine import router_replay
from lilax._backend.training_engine import state_placement
from lilax._backend.training_engine import step_kernels
from lilax._backend.training_engine import weight_sync
from lilax._backend.training_engine import zero1
from lilax._backend.utils import train_utils
from lilax._backend.utils.model_creation import pretrained

EMPTY_METRICS_BUFFER_ID = -1


class MaxTextTrainingEngine(abstract_engine.AbstractTrainingEngine):
  """Native NNX training engine."""

  def __init__(
      self,
      training_config: pyconfig.HyperParameters,
      mesh: jax.sharding.Mesh | None = None,
      wrap_with_tunix_adapter: bool = False,
      tokenizer_pad_id: int | None = None,
      *,
      model: nnx.Module | None = None,
  ) -> None:
    if not isinstance(training_config, pyconfig.HyperParameters):
      raise TypeError(
          "MaxTextTrainingEngine requires a pyconfig.HyperParameters instance," f" got {type(training_config).__name__}"
      )
    if wrap_with_tunix_adapter:
      if tokenizer_pad_id is None:
        raise ValueError(
            "wrap_with_tunix_adapter=True requires tokenizer_pad_id. Without it the adapter cannot build "
            "segment_ids, so pad positions are attended to and trainer log-probs are silently wrong."
        )
      if mesh is None:
        raise ValueError("wrap_with_tunix_adapter=True requires a mesh; the adapter is built under it.")
    if getattr(getattr(training_config, "lora", None), "enable_lora", False):
      raise NotImplementedError(
          "MaxTextTrainingEngine does not support LoRA, but lora.enable_lora=True was set. "
          "This engine trains all parameters, set lora.enable_lora=False to train with this engine."
      )
    self._config = training_config
    self._mesh = mesh
    self._init_rng = jax.random.PRNGKey(training_config.init_weights_seed)
    self._loss_fn: Callable[..., Any] | None = None
    self._has_aux: bool = True
    self._gen_model_input_fn: Callable[[Any], dict[str, Any]] | None = None
    self._kernels = compiled_kernels.CompiledKernels()
    self._recompile_detector = batch_signature.RecompileDetector()
    if not training_config.model_name:
      raise ValueError("training_config.model_name must be specified")
    self._model = model if model is not None else self._build_model(wrap_with_tunix_adapter, tokenizer_pad_id)
    self._batch_sharder = batch_sharding.BatchSharder(self._config, self._mesh)
    self._state: Any = None
    self._pure_state = pure_state.PureStateCache()
    self._grad_plan = gradient_sharding.GradientShardingPlan()
    self._zero1 = zero1.Zero1Planner(self._config)
    self._accumulation = gradient_accumulation.GradientAccumulation()
    self._total_micro_steps = 0
    self._learning_rate_schedule, tx = train_utils.create_training_optimizer(self._config, self._model, mesh=self._mesh)
    self._optimizer = self._build_optimizer(tx)
    if self._config.use_m3_model:
      m3_models.configure_engine(self)
    self._train_step: int = 0
    self._checkpoint_manager = checkpointing.CheckpointManager(checkpoint_dir=self._checkpoint_dir(), config=self._config)
    self._metrics_recorder = metrics_module.MetricsRecorder()
    self._eval_metrics_recorder = metrics_module.MetricsRecorder(mode=metrics_module.Mode.EVAL)
    self._metrics_logger = metrics_module.MetricsLogger(config=self._config)
    self._throttler = inflight_throttler.InflightThrottler(config=self._config, metrics_logger=self._metrics_logger)
    self._profiler = profiler_module.MicroStepProfiler(self._config)
    self._weight_sync = weight_sync.WeightSyncStager(self._config)

  def _build_model(self, wrap_with_tunix_adapter: bool, tokenizer_pad_id: int | None) -> Any:
    model_or_model_mesh_pair = pretrained.from_pretrained(
        config=self._config,
        mesh=self._mesh,
        model_mode=common_types.MODEL_MODE_TRAIN,
        rng_key=self._init_rng,
        wrap_with_tunix_adapter=wrap_with_tunix_adapter,
        tokenizer_pad_id=tokenizer_pad_id,
    )
    if self._mesh is not None:
      return model_or_model_mesh_pair
    model, self._mesh = model_or_model_mesh_pair
    return model

  def _build_optimizer(self, tx: Any) -> Any:
    return nnx.Optimizer(self._model, tx, wrt=nnx.Param)

  def _checkpoint_dir(self) -> str:
    return self._config.checkpoint_dir

  @property
  def total_micro_steps(self) -> int:
    return self._total_micro_steps

  @property
  def model(self) -> Any:
    return self._model

  @model.setter
  def model(self, new_model: Any) -> None:
    self._model = new_model
    self._kernels.invalidate_train()
    self._kernels.invalidate_eval()
    self._pure_state.model_graphdef = None
    self._pure_state.invalidate()

  @property
  def optimizer(self) -> Any:
    return self._optimizer

  @optimizer.setter
  def optimizer(self, new_optimizer: Any) -> None:
    self._optimizer = new_optimizer
    self._kernels.invalidate_train()
    self._pure_state.state_graphdef = None
    self._pure_state.invalidate()
    self._kernels.invalidate_eval()

  @property
  def train_step(self) -> int:
    return self._train_step

  @train_step.setter
  def train_step(self, step: int) -> None:
    self._train_step = step

  @property
  def state(self) -> Any:
    if self._state is None and self._model is not None and self._optimizer is not None:
      self._state = train_state_nnx.TrainStateNNX(self._model, self._optimizer)
    return self._state

  @state.setter
  def state(self, new_state: Any) -> None:
    self._state = new_state
    self._kernels.invalidate_train()
    self._pure_state.state_graphdef = None
    self._pure_state.invalidate()
    self._kernels.invalidate_eval()

  @property
  def micro_step_count(self) -> int:
    return self._accumulation.micro_step_count

  @property
  def has_accumulated_grads(self) -> bool:
    return self._accumulation.grads is not None

  def with_loss_fn(self, customized_fn: Callable[..., Any], has_aux: bool = False) -> "MaxTextTrainingEngine":
    self._loss_fn = customized_fn
    self._has_aux = has_aux
    self._kernels.compiled = False
    self._kernels.invalidate_eval()
    return self

  def with_gen_model_input_fn(self, gen_model_input_fn: Callable[[Any], dict[str, Any]]) -> "MaxTextTrainingEngine":
    self._gen_model_input_fn = gen_model_input_fn
    self._kernels.compiled = False
    self._kernels.invalidate_eval()
    return self

  @contextlib.contextmanager
  def _sharding_ctx(self):
    if self._mesh is None:
      yield
      return
    with jax.set_mesh(self._mesh), logical_axis_rules(self._config.logical_axis_rules):
      yield

  def _ensure_state(self) -> None:
    if self._state is None:
      self._state = train_state_nnx.TrainStateNNX(self._model, self._optimizer)

  def _refresh_pure_state(self) -> None:
    self._ensure_state()
    self._pure_state.refresh(self._state, getattr(self._state, pure_state.MODEL_STATE_KEY, self._model))

  def _place_leaf(self, leaf: Any, target: jax.sharding.Sharding) -> Any:
    return jax.device_put(leaf, target)

  def _adopt_placed_state(self, placed: Any) -> None:
    with self._sharding_ctx():
      nnx.update(self._state, placed)
    self._pure_state.invalidate()
    self._refresh_pure_state()

  def _place_state_on_mesh(self) -> None:
    if self._mesh is None or self._state is None:
      return
    placed = state_placement.place_state_on_mesh(self._mesh, self._pure_state.read_state(self._state), self._place_leaf)
    if placed is not None:
      self._adopt_placed_state(placed)

  def _shard_optimizer_state_over_data(self) -> None:
    if zero1.zero1_active(self._config, self._mesh) is not None:
      return
    optimizer_pure = zero1.shard_optimizer_state(self._mesh, self._pure_state.read_state(self._state), self._place_leaf)
    if optimizer_pure is not None:
      self._adopt_placed_state(nnx.State({pure_state.OPTIMIZER_STATE_KEY: optimizer_pure.raw_mapping}))

  def _mesh_shardings(self, tree: Any) -> Any:
    return jax.tree.map(lambda leaf: state_placement.mesh_sharding(self._mesh, leaf), tree)

  def _loss_spec(self) -> loss_invocation.LossSpec:
    return loss_invocation.LossSpec(self._loss_fn, self._has_aux, self._gen_model_input_fn is not None)

  def _fwd_bwd_kernel(self, params, rest, batch, acc_grads=None, acc_denom=None):
    return step_kernels.fwd_bwd_kernel(
        self._config, self._loss_spec(), self._pure_state.model_graphdef, self._grad_plan,
        params, rest, batch, acc_grads, acc_denom,
    )

  def _update_kernel(self, state_pure, accumulated_grads, accumulated_denominator, mean_loss):
    return step_kernels.update_kernel(
        self._config, self._pure_state.state_graphdef, self._optimizer, self._grad_plan,
        state_pure, accumulated_grads, accumulated_denominator, mean_loss,
    )

  def _eval_kernel(self, params, rest, batch):
    return step_kernels.eval_kernel(self._config, self._loss_spec(), self._pure_state.model_graphdef, params, rest, batch)

  def _compile_for_batch(self, dynamic_batch: Any, static_batch: dict[str, Any]) -> None:
    self._refresh_pure_state()
    self._place_state_on_mesh()
    self._shard_optimizer_state_over_data()
    state_pure = self._pure_state.read_state(self._state)
    params_pure, rest_pure = self._pure_state.read_model(getattr(self._state, pure_state.MODEL_STATE_KEY, self._model))

    def fwd_bwd(params, rest, dynamic):
      return self._fwd_bwd_kernel(params, rest, batch_signature.merge_static(dynamic, static_batch))

    def fwd_bwd_accum(params, rest, dynamic, acc_grads, acc_denom):
      return self._fwd_bwd_kernel(params, rest, batch_signature.merge_static(dynamic, static_batch), acc_grads, acc_denom)

    def update(state_pure, accumulated_grads, accumulated_denominator, mean_loss):
      return self._update_kernel(state_pure, accumulated_grads, accumulated_denominator, mean_loss)

    if self._mesh is not None:
      state_mesh_shardings = self._mesh_shardings(state_pure)
      params_shardings = self._mesh_shardings(params_pure)
      rest_shardings = self._mesh_shardings(rest_pure)
      batch_shardings = self._batch_sharder.data_shardings(dynamic_batch)
      reduced_params_shardings, unreduced_grad_shardings = gradient_sharding.deferred_all_reduce_shardings(
          self._config, self._mesh, params_shardings
      )
      zero1_params_shardings = self._zero1.params_shardings(self._mesh, params_pure, params_shardings)
      self._grad_plan = gradient_sharding.plan_gradient_shardings(
          params_shardings, reduced_params_shardings, unreduced_grad_shardings, zero1_params_shardings
      )
      shardings = kernel_compilation.train_kernel_shardings(
          self._mesh, state_mesh_shardings, params_shardings, rest_shardings, batch_shardings, self._grad_plan.grads
      )
      if self._accumulation.grads is not None:
        with self._sharding_ctx():
          self._accumulation.grads = jax.tree.map(
              gradient_sharding.conform_accumulator, self._accumulation.grads, self._grad_plan.grads
          )
    else:
      shardings = kernel_compilation.TrainKernelShardings()
      self._zero1.note_declined(zero1.zero1_active(self._config, self._mesh))
      self._grad_plan = gradient_sharding.GradientShardingPlan()

    jitted = kernel_compilation.jit_train_kernels(fwd_bwd, fwd_bwd_accum, update, shardings)
    self._kernels.fwd_bwd = jitted["fwd_bwd"]
    self._kernels.fwd_bwd_accum = jitted["fwd_bwd_accum"]
    self._kernels.update = jitted["update"]
    self._kernels.jitted = jitted
    self._kernels.signature = batch_signature.batch_signature(dynamic_batch, static_batch)
    self._kernels.compiled = True

  def _compile_eval_for_batch(self, dynamic_batch: Any, static_batch: dict[str, Any]) -> None:
    self._pure_state.model_graphdef, params_pure, rest_pure = nnx.split(self._model, nnx.Param, ...)

    def kernel(params, rest, dynamic):
      return self._eval_kernel(params, rest, batch_signature.merge_static(dynamic, static_batch))

    if self._mesh is not None:
      params_shardings = self._mesh_shardings(params_pure)
      rest_shardings = self._mesh_shardings(rest_pure)
      eval_in_shardings = (params_shardings, rest_shardings, self._batch_sharder.data_shardings(dynamic_batch))
      eval_out_shardings = (None, None)
    else:
      eval_in_shardings = None
      eval_out_shardings = None
    self._kernels.eval = jax.jit(kernel, in_shardings=eval_in_shardings, out_shardings=eval_out_shardings)
    self._kernels.eval_signature = batch_signature.batch_signature(dynamic_batch, static_batch)

  def compile(
      self, dummy_data: abstract_engine.TrainerPayload | None = None, compiler_options: dict[str, Any] | None = None
  ) -> None:
    self._kernels.requested = True
    if self._kernels.compiled:
      return
    if dummy_data is None:
      logging.info(
          "MaxTextTrainingEngine.compile() was called without dummy_data; compiling "
          "against the first fwd_bwd payload instead."
      )
      return
    dummy_data = router_replay.expand_dummy_routed_experts(self._config, dummy_data)
    compiled = self.compile_kernels(dummy_data, compiler_options)
    self._kernels.fwd_bwd = compiled["fwd_bwd"]
    self._kernels.fwd_bwd_accum = compiled["fwd_bwd_accum"]
    self._kernels.update = compiled["update"]
    self._compile_eval(dummy_data, compiler_options)

  def compile_kernels(
      self, dummy_data: abstract_engine.TrainerPayload, compiler_options: dict[str, Any] | None = None
  ) -> dict[str, jax.stages.Compiled]:
    options = kernel_compilation.xla_options(self._config, compiler_options)
    lowered = self._lower_kernels(dummy_data)
    with self._sharding_ctx():
      return {name: lowered[name].compile(compiler_options=options) for name in kernel_compilation.KERNEL_NAMES}

  def _compile_eval(self, dummy_data: abstract_engine.TrainerPayload, compiler_options: dict[str, Any] | None) -> None:
    dynamic_batch, static_batch = batch_signature.split_static_and_dynamic(
        payload_inputs.prepare_batch(dummy_data, self._gen_model_input_fn)
    )
    self._compile_eval_for_batch(dynamic_batch, static_batch)
    params_pure, rest_pure = self._pure_state.read_model(getattr(self._state, pure_state.MODEL_STATE_KEY, self._model))
    to_aval = kernel_compilation.to_aval
    with self._sharding_ctx():
      self._kernels.eval = self._kernels.eval.lower(
          jax.tree.map(to_aval, params_pure), jax.tree.map(to_aval, rest_pure), jax.tree.map(to_aval, dynamic_batch)
      ).compile(compiler_options=kernel_compilation.xla_options(self._config, compiler_options))

  def _lower_kernels(self, dummy_data: abstract_engine.TrainerPayload) -> dict[str, jax.stages.Lowered]:
    if dummy_data is None:
      raise ValueError(
          "compile_kernels() needs a dummy payload -- unlike compile(), it cannot defer to the first real batch."
      )
    dynamic_batch, static_batch = batch_signature.split_static_and_dynamic(
        payload_inputs.prepare_batch(dummy_data, self._gen_model_input_fn)
    )
    self._compile_for_batch(dynamic_batch, static_batch)
    to_aval = kernel_compilation.to_aval
    state_aval = jax.tree.map(to_aval, self._pure_state.read_state(self._state))
    params_pure, rest_pure = self._pure_state.read_model(getattr(self._state, pure_state.MODEL_STATE_KEY, self._model))
    params_aval = jax.tree.map(to_aval, params_pure)
    rest_aval = jax.tree.map(to_aval, rest_pure)
    batch_aval = jax.tree.map(to_aval, dynamic_batch)
    mean_loss_aval = jax.ShapeDtypeStruct((), jnp.float32) if self._config.skip_step_on_spikes else None
    with self._sharding_ctx():
      return kernel_compilation.lower_train_kernels(
          self._kernels.jitted, state_aval, params_aval, rest_aval, batch_aval, mean_loss_aval
      )

  @profiler_module.profiled_step("fwd_bwd", control_profile=True)
  def fwd_bwd(self, payload: abstract_engine.TrainerPayload) -> None:
    batch = payload_inputs.prepare_batch(payload, self._gen_model_input_fn)
    model = getattr(self._state, "model", None) if self._state is not None else self._model
    if not isinstance(model, nnx.Module):
      raise TypeError("MaxTextTrainingEngine requires an NNX model (flax.nnx.Module), got" f" {type(model).__name__}")
    self._throttler.wait_for_next()
    self._ensure_state()
    model = getattr(self._state, pure_state.MODEL_STATE_KEY, self._model)
    accumulation = self._accumulation
    if self._kernels.requested:
      dynamic_batch, static_batch = batch_signature.split_static_and_dynamic(batch)
      signature = batch_signature.batch_signature(dynamic_batch, static_batch)
      if not self._kernels.compiled or self._recompile_detector.needs_recompile(signature, self._kernels.signature):
        self._compile_for_batch(dynamic_batch, static_batch)
      params, rest = self._pure_state.read_model(model)
      with self._sharding_ctx():
        if accumulation.grads is None:
          loss, aux, new_rest, acc_grads, acc_denom = self._kernels.fwd_bwd(params, rest, dynamic_batch)
        else:
          loss, aux, new_rest, acc_grads, acc_denom = self._kernels.fwd_bwd_accum(
              params, rest, dynamic_batch, accumulation.grads, accumulation.denominator
          )
    else:
      params, rest = self._pure_state.read_model(model)
      with self._sharding_ctx():
        loss, aux, new_rest, acc_grads, acc_denom = self._fwd_bwd_kernel(
            params, rest, batch, accumulation.grads, accumulation.denominator
        )
    nnx.update(model, new_rest)
    self._pure_state.publish_model_rest(new_rest)
    self._throttler.add_computation(computation=loss, metrics=None)
    if isinstance(loss, abstract_engine.WeightedMetric):
      self.record_metrics("loss", loss)
    if isinstance(aux, dict):
      for key, value in aux.items():
        if value is not None:
          self.record_metrics(key, value)
    accumulation.add(loss, acc_grads, acc_denom)
    self._total_micro_steps += 1

  @profiler_module.profiled_step("update")
  def update(self) -> int:
    accumulation = self._accumulation
    if accumulation.grads is None:
      return self.train_step
    if self._learning_rate_schedule is not None:
      lr = self._learning_rate_schedule(self.train_step)
      self.record_metrics("learning_rate", lr)
    self._throttler.wait_for_next()
    self._ensure_state()
    state_pure = self._pure_state.read_state(self._state)
    mean_loss = accumulation.mean_loss(self._config.skip_step_on_spikes)
    with self._sharding_ctx():
      if self._kernels.compiled and hasattr(self._kernels, "update"):
        new_state_pure, grad_norm, is_skipped = self._kernels.update(
            state_pure, accumulation.grads, accumulation.denominator, mean_loss
        )
      else:
        new_state_pure, grad_norm, is_skipped = self._update_kernel(
            state_pure, accumulation.grads, accumulation.denominator, mean_loss
        )
    nnx.update(self._state, new_state_pure)
    self._pure_state.publish_state(new_state_pure)
    if grad_norm is not None:
      self.record_metrics("gradient_norm", grad_norm)
    if is_skipped is not None:
      self.record_metrics("step_skipped", is_skipped)
    self._throttler.add_computation(
        grad_norm if grad_norm is not None else (self._state if self._state is not None else self._model),
        self._metrics_recorder.get_step_metrics(self.train_step),
    )
    accumulation.reset()
    self._train_step += 1
    return self.train_step

  @contextlib.contextmanager
  def eval_context(self):
    logging.info("Running evaluation on train step %d.", self.train_step)
    self._throttler.wait_for_all()
    try:
      yield
    finally:
      for buffer in self._eval_metrics_recorder.get_metrics_history(clear_cache=True):
        logging.info("Writing buffered eval metrics for train step %d.", buffer.id)
        self._metrics_logger.write_metrics(buffer, mode=metrics_module.Mode.EVAL)
      self._eval_metrics_recorder.cleanup()

  def eval_step(self, payload: abstract_engine.TrainerPayload) -> None:
    batch = payload_inputs.prepare_batch(payload, self._gen_model_input_fn)
    model = getattr(self._state, "model", self._model) if self._state is not None else self._model
    if not isinstance(model, nnx.Module):
      raise TypeError("MaxTextTrainingEngine requires an NNX model (flax.nnx.Module), got" f" {type(model).__name__}")
    self._pure_state.model_graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    self._throttler.wait_for_next()
    if self._kernels.requested:
      dynamic_batch, static_batch = batch_signature.split_static_and_dynamic(batch)
      signature = batch_signature.batch_signature(dynamic_batch, static_batch)
      if self._kernels.eval is None or self._recompile_detector.needs_recompile(signature, self._kernels.eval_signature):
        self._compile_eval_for_batch(dynamic_batch, static_batch)
      loss, aux = self._kernels.eval(params, rest, dynamic_batch)
    else:
      loss, aux = self._eval_kernel(params, rest, batch)
    self._throttler.add_computation(computation=loss, metrics=None)
    if isinstance(loss, abstract_engine.WeightedMetric):
      self.record_metrics("loss", loss, mode=metrics_module.Mode.EVAL)
    else:
      logging.warning("Eval loss is not a WeightedMetric, so it will not be logged. Got %s.", type(loss).__name__)
    if isinstance(aux, dict):
      for key, value in aux.items():
        if value is not None:
          self.record_metrics(key, value, mode=metrics_module.Mode.EVAL)

  def fwd_only(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    self._ensure_state()
    model = getattr(self._state, pure_state.MODEL_STATE_KEY, self._model)
    if not isinstance(model, nnx.Module):
      raise TypeError("MaxTextTrainingEngine requires an NNX model (flax.nnx.Module), got" f" {type(model).__name__}")
    self._throttler.wait_for_next()
    with self._sharding_ctx():
      args, kwargs = self._batch_sharder.place_read_only_inputs((args, kwargs))
      out = fn(model, *args, **kwargs)
    self._throttler.add_computation(
        computation=[leaf for leaf in jax.tree.leaves(out) if isinstance(leaf, jax.Array)], metrics=None
    )
    return out

  def save_checkpoint(self, metadata: Any = None, *, step: int | None = None, force: bool = False) -> None:
    if self._accumulation.micro_step_count > 0:
      raise ValueError(
          "Mid-step checkpointing is not supported; micro_step_count must be 0, got "
          f"{self._accumulation.micro_step_count}."
      )
    if not self._config.enable_checkpointing or not self._checkpoint_dir():
      logging.info("Checkpointing is disabled in config; skipping save_checkpoint.")
      return
    self._throttler.wait_for_all()
    if step is None and isinstance(metadata, Mapping):
      step = metadata.get("step")
    if step is None:
      step = self.train_step
    logging.info("Saving checkpoint at step %d.", step)
    custom_metadata = {}
    if metadata:
      custom_metadata["additional_metadata"] = metadata
    ckpt_saved = self._checkpoint_manager.save_checkpoint(
        step=step,
        checkpoint_state=checkpointing.CheckpointState(model=self.model, optimizer=self.optimizer),
        custom_metadata=custom_metadata,
        force=force,
    )
    if ckpt_saved:
      logging.info("Checkpoint saved at step %d.", step)

  def restore_checkpoint(self, *, step: int | None = None) -> Any:
    restored_step, _, restored_metadata = self._checkpoint_manager.restore_checkpoint(
        checkpoint_state=checkpointing.CheckpointState(model=self.model, optimizer=self.optimizer), step=step
    )
    if restored_step is None:
      return None
    logging.info("Checkpoint restored from step %d.", restored_step)
    self._pure_state.invalidate()
    self.train_step = restored_step
    self._metrics_recorder.cleanup()
    self._accumulation.reset()
    return restored_metadata.get("additional_metadata", None) if restored_metadata else None

  def record_metrics(
      self,
      name: str,
      metric: abstract_engine.WeightedMetric | jax.Array | float | int | dict[str, Any],
      aggregation_fn: Callable[[jax.Array], Any] | None = None,
      mode: metrics_module.Mode = metrics_module.Mode.TRAIN,
  ) -> None:
    if metric is None or name == "intermediate_outputs" or isinstance(metric, (tuple, list)):
      return
    if isinstance(metric, dict):
      for sub_key, sub_value in metric.items():
        if sub_value is not None:
          self.record_metrics(
              f"{name}/{sub_key}" if name else sub_key, sub_value, aggregation_fn=aggregation_fn, mode=mode
          )
    elif mode == metrics_module.Mode.TRAIN:
      self._metrics_recorder.buffer_metrics(
          train_step=self.train_step, name=name, metric=metric, aggregation_fn=aggregation_fn
      )
    else:
      self._eval_metrics_recorder.buffer_metrics(
          train_step=self.train_step, name=name, metric=metric, aggregation_fn=aggregation_fn
      )

  def get_metrics(self, clear_cache: bool = True) -> abstract_engine.MetricsBuffer:
    history = self._metrics_recorder.get_metrics_history(clear_cache=clear_cache)
    if not history:
      return abstract_engine.MetricsBuffer(id=EMPTY_METRICS_BUFFER_ID)
    if len(history) > 1:
      logging.warning(
          "get_metrics() is returning the buffer for step %s and dropping %d older buffer(s); "
          "call get_metrics() once per update, or use MetricsRecorder.get_metrics_history() "
          "to read every step.", history[-1].id, len(history) - 1,
      )
    return history[-1]

  def _get_trainable_params_state(self) -> Any:
    model = getattr(self._state, "model", None) if self._state is not None else self._model
    if isinstance(model, nnx.Module):
      return nnx.state(model, nnx.Param)
    return self.model

  def set_target_state(self, target_state: Any) -> None:
    self._weight_sync.set_target_state(target_state)

  def prepare_weight_sync(self, *, sync_request: Any = None, staging_transport: str = "raiden") -> Any:
    del sync_request
    if staging_transport == "raiden":
      staged = self._weight_sync.reuse_staged(self.train_step)
      if staged is not None:
        return staged
      self._throttler.wait_for_all()
      return self._weight_sync.stage(self._get_trainable_params_state(), self.train_step)
    raise ValueError(f"unknown staging_transport {staging_transport!r}; expected 'raiden'.")

  def release_weight_sync(self, *, sync_request: Any = None) -> Any:
    del sync_request
    self._weight_sync.release()
    return True

  def close(self) -> None:
    if self._profiler is not None:
      self._profiler.close(blocking_object=self._pure_state.read_state(self._state) if self._state is not None else None)
    self._weight_sync.close()
    if self._config.enable_checkpointing and self._checkpoint_dir() and self._checkpoint_manager:
      if self._accumulation.micro_step_count > 0:
        logging.warning(
            "Closing mid-step (micro_step_count=%d); skipping the final checkpoint.", self._accumulation.micro_step_count
        )
      else:
        self.save_checkpoint(metadata=None, force=True)
      self._checkpoint_manager.close()
    self._throttler.cleanup()
    self._metrics_recorder.cleanup()
    self._metrics_logger.cleanup()
