"""Native MaxText training engine."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import contextlib
import dataclasses
import functools
from typing import Any, Optional

from absl import logging
from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
import jax.numpy as jnp
from lilax._backend.common import common_types
from lilax._backend.common import train_state_nnx
from lilax._backend.configs import pyconfig
from lilax._backend.integration.tunix.weight_mapping import raiden_unscan
from lilax._backend.integration.vllm.convert_utils import (
    is_verify_weights_enabled,
    resolve_prefuse_moe_weights,
    resolve_rollout_tp,
)
from lilax._backend.m3 import models as m3_models
from lilax._backend.trainers.pre_train import train as maxtext_train
from lilax._backend.training_engine import abstract_engine
from lilax._backend.training_engine import checkpointing
from lilax._backend.training_engine import inflight_throttler
from lilax._backend.training_engine import metrics as metrics_module
from lilax._backend.training_engine import micro_step_profiler as profiler_module
from lilax._backend.utils import max_utils
from lilax._backend.utils import maxtext_utils
from lilax._backend.utils import model_creation_utils
from lilax._backend.utils import sharding
from lilax._backend.utils import train_utils
import numpy as np
from tunix.dist_rl.weight_sync import raiden_synchronizer

EMPTY_METRICS_BUFFER_ID = -1
_MODEL_STATE_KEY = "model"
_OPTIMIZER_STATE_KEY = "optimizer"
KERNEL_NAMES = ("fwd_bwd", "fwd_bwd_accum", "update")
_PURE_STATE_FALLBACK_WARNING = (
    "Cannot keep the train state as a pure pytree across steps (%s), so every fwd_bwd and "
    "update will re-walk the NNX module graph. That is correct but slow -- the two "
    "`nnx.split` calls cost ~92 ms per step on an unrolled 28-layer qwen3-0.6b, against "
    "~2 ms for the pure-state equivalent. Logged once per engine instance."
)


def _is_jax_dynamic(value: Any) -> bool:
  leaves = jax.tree.leaves(value)
  if not leaves:
    return True
  return any(isinstance(leaf, (jax.Array, jax.ShapeDtypeStruct, np.ndarray, np.generic)) for leaf in leaves)


def _split_static_and_dynamic(batch: Any) -> tuple[Any, dict[str, Any]]:
  if not isinstance(batch, dict):
    return batch, {}
  dynamic: dict[str, Any] = {}
  static: dict[str, Any] = {}
  for key, value in batch.items():
    if isinstance(value, (int, float, bool)) or not _is_jax_dynamic(value):
      static[key] = value
    else:
      dynamic[key] = value
  return dynamic, static


def _batch_signature(dynamic_batch: Any, static_batch: dict[str, Any]) -> Any:
  leaves, treedef = jax.tree.flatten(dynamic_batch)
  shapes = tuple((jnp.shape(leaf), jnp.result_type(leaf)) for leaf in leaves)
  return (treedef, shapes, static_batch)


_REPLICATED_BATCH_DIM_WARNING = (
    "Loss input with batch dim %d does not divide mesh axis %r (size %d), so that "
    "dimension is replicated instead of sharded: every device along the axis holds and "
    "computes the whole micro-batch, %dx the work a sharded one would do there. Results "
    "stay correct. If it was not deliberate -- a sequence-packed micro-batch is always "
    "size 1 and has no alternative -- make the micro-batch a multiple of the axis size."
)
_UNCOMPARABLE_SIGNATURE_WARNING = (
    "Could not compare %s between fwd_bwd calls (%s), so the engine cannot tell whether "
    "the compiled kernel is still valid and will recompile on EVERY fwd_bwd from now on. %s"
)
_UNCOMPARABLE_STATIC_HINT = (
    "This happens when a `gen_model_input_fn` returns a fresh object each call that holds "
    "an array -- a dataclass or SimpleNamespace whose `__eq__` then compares elementwise, "
    "where `bool()` is ambiguous. (Reusing one instance is fine: comparison short-circuits "
    "on identity.) Fix by returning arrays as top-level entries of the batch dict, so they "
    "are traced rather than closed over, or by holding the config object fixed across calls."
)
_UNCOMPARABLE_STRUCTURE_HINT = (
    "This is unexpected: the structural half is built from pytree treedefs, shapes and "
    "dtypes, which compare cleanly. Recompilation stays correct, but the cost is now paid "
    "every step, so this is worth reporting rather than living with."
)
_DATA_AXIS = "data"


def _tag_sharding(named_sharding: jax.sharding.NamedSharding, field: str) -> jax.sharding.NamedSharding:
  if _DATA_AXIS in sharding.get_mesh_axes_used_by_tensor_spec(named_sharding.spec.partitions):
    return named_sharding
  return named_sharding.update(spec=named_sharding.spec.update(**{field: {_DATA_AXIS}}))


def _deferred_all_reduce_shardings(config: Any, mesh: Any, params_shardings: Any) -> tuple[Any, Any]:
  if getattr(config, "shard_mode", None) != common_types.ShardMode.EXPLICIT:
    return None, None
  if mesh is None or mesh.shape.get(_DATA_AXIS, 1) <= 1:
    return None, None
  if any(axis_type != jax.sharding.AxisType.Explicit for axis_type in mesh.axis_types):
    return None, None
  if any(size > 1 for axis, size in mesh.shape.items() if axis != _DATA_AXIS):
    return None, None
  try:
    batch_axes = sharding.batch_mesh_axes(mesh, rules=config.logical_axis_rules)
  except (KeyError, ValueError, IndexError):
    return None, None
  if batch_axes != frozenset({_DATA_AXIS}):
    return None, None
  return (
      jax.tree.map(lambda s: _tag_sharding(s, "reduced"), params_shardings),
      jax.tree.map(lambda s: _tag_sharding(s, "unreduced"), params_shardings),
  )


_ZERO1_DECLINED_WARNING = (
    "`shard_optimizer_over_data` (Zero-1) is set, but this engine cannot honour it (%s), so the "
    "optimizer state stays replicated over the data axis. Logged once per engine instance."
)


def _zero1_active(config: Any, mesh: Any) -> str | None:
  if not getattr(config, "shard_optimizer_over_data", False):
    return "it is not enabled"
  if getattr(config, "shard_mode", None) != common_types.ShardMode.EXPLICIT:
    return "it needs shard_mode=explicit"
  if mesh is None:
    return "the engine has no mesh"
  if mesh.shape.get(_DATA_AXIS, 1) <= 1:
    return f"the mesh has no {_DATA_AXIS!r} axis to shard the optimizer over"
  if any(axis_type != jax.sharding.AxisType.Explicit for axis_type in mesh.axis_types):
    return "the mesh has non-Explicit axes"
  return None


def _zero1_sharding(mesh: Any, aval: Any, base: jax.sharding.NamedSharding | None) -> jax.sharding.NamedSharding | None:
  if base is None or not hasattr(aval, "shape"):
    return None
  try:
    target = sharding.add_data_to_sharding(mesh, (), aval, base)
  except AssertionError:
    return None
  return None if target == base else target


def _conform_accumulator(value: Any, target: jax.sharding.NamedSharding) -> Any:
  current = getattr(value, "sharding", None)
  if current == target:
    return value
  if getattr(current, "spec", None) is not None and current.spec.unreduced:
    return jax.sharding.reshard(value, target)
  return jax.device_put(value, target)


def _normalize_loss_output(out: Any, has_aux: bool) -> abstract_engine.LossOutput:
  if isinstance(out, abstract_engine.LossOutput):
    return out
  if isinstance(out, abstract_engine.WeightedMetric):
    return abstract_engine.LossOutput(primary_loss=out, aux_metrics={})
  if isinstance(out, (tuple, list)) and len(out) == 2:
    loss_val, aux = out
    if isinstance(loss_val, abstract_engine.WeightedMetric):
      primary_loss = loss_val
    elif isinstance(aux, dict) and "xent_sum" in aux and "total_weights" in aux:
      primary_loss = abstract_engine.WeightedMetric(unreduced_sum=aux["xent_sum"], denominator=aux["total_weights"])
    else:
      raise TypeError(
          f"Cannot construct WeightedMetric from 2-tuple loss return with elements "
          f"of type ({type(loss_val).__name__}, {type(aux).__name__}). Expected first element to be a "
          "WeightedMetric, or second element to be a dict containing 'xent_sum' and 'total_weights'."
      )
    return abstract_engine.LossOutput(primary_loss=primary_loss, aux_metrics=aux if (has_aux and isinstance(aux, dict)) else {})
  raise TypeError(
      f"Unsupported return type from loss function: {type(out)}. "
      "Expected abstract_engine.LossOutput, abstract_engine.WeightedMetric, "
      "or a 2-element tuple/list: (loss, aux_metrics)."
  )


def _to_aval(value: Any) -> Any:
  if not hasattr(value, "shape") or not hasattr(value, "dtype"):
    return value
  return jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=getattr(value, "sharding", None))


def _row_and_mask(payload: abstract_engine.RLTrainerPayload) -> tuple[jax.Array, jax.Array]:
  parts = [jnp.asarray(payload.prompt_ids), jnp.asarray(payload.completion_ids)]
  masks = [
      jnp.ones_like(part) if mask is None else jnp.asarray(mask).astype(jnp.int32)
      for part, mask in zip(parts, (payload.prompt_mask, payload.completion_mask))
  ]
  return jnp.concatenate(parts, axis=-1), jnp.concatenate(masks, axis=-1).astype(jnp.int32)


def _aligned(field: Any, name: str, width: int) -> jax.Array | None:
  if field is None:
    return None
  array = jnp.asarray(field)
  if array.shape[-1] != width:
    raise ValueError(
        f"{name} has width {array.shape[-1]} but the model row is {width} wide. Tunix sets "
        f"{name} only on the packed path, where `prompt_ids` is zero-width and the two agree; "
        "a completion-width array cannot be aligned against a non-empty prompt."
    )
  return array


def router_replay_gen_model_input_fn(payload: abstract_engine.RLTrainerPayload) -> dict[str, Any]:
  token_ids, token_mask = _row_and_mask(payload)
  width = token_ids.shape[-1]
  segment_ids = _aligned(payload.segment_ids, "segment_ids", width)
  segment_positions = _aligned(payload.segment_positions, "segment_positions", width)
  if segment_ids is None:
    segment_ids = token_mask
  if segment_positions is not None:
    positions = segment_positions.astype(jnp.int32)
  else:
    real = (token_mask != 0).astype(jnp.int32)
    preceding = jnp.cumsum(real, axis=-1) - real
    starts = jnp.concatenate(
        [jnp.ones((segment_ids.shape[0], 1), dtype=bool), segment_ids[:, 1:] != segment_ids[:, :-1]], axis=-1
    )
    segment_start_count = jax.lax.cummax(jnp.where(starts, preceding, -1), axis=preceding.ndim - 1)
    positions = (preceding - segment_start_count).astype(jnp.int32)
  targets_segmentation = jnp.roll(token_mask, -1, axis=-1)
  targets_segmentation = targets_segmentation.at[:, -1].set(0)
  same_segment = segment_ids[:, :-1] == segment_ids[:, 1:]
  targets_segmentation = targets_segmentation.at[:, :-1].multiply(same_segment.astype(token_mask.dtype))
  kwargs = {
      "inputs": token_ids,
      "inputs_position": positions,
      "inputs_segmentation": segment_ids,
      "targets": jnp.roll(token_ids, -1, axis=-1),
      "targets_segmentation": targets_segmentation,
  }
  routed_experts = getattr(payload, "routed_experts", None)
  if routed_experts is not None:
    kwargs["forced_routed_experts"] = jnp.asarray(routed_experts)
  return kwargs


def make_router_replay_loss_fn(
    config: pyconfig.HyperParameters, dropout_rng: jax.Array | None = None
) -> Callable[..., Any]:
  if config.enable_dropout and dropout_rng is None:
    raise ValueError(
        "make_router_replay_loss_fn requires a dropout_rng when"
        " config.enable_dropout is set; pass one, or set enable_dropout=False"
        " for router replay."
    )

  def router_replay_loss_fn(
      model, inputs, inputs_position, inputs_segmentation, targets, targets_segmentation, forced_routed_experts=None
  ):
    data = {
        "inputs": inputs,
        "inputs_position": inputs_position,
        "inputs_segmentation": inputs_segmentation,
        "targets": targets,
        "targets_segmentation": targets_segmentation,
    }
    if forced_routed_experts is not None:
      data["forced_routed_experts"] = forced_routed_experts
    return maxtext_train.loss_fn(model, config, data, dropout_rng=dropout_rng, params=None, is_train=True)

  return router_replay_loss_fn


def _profiled_step(name: str, control_profile: bool = False) -> Callable[..., Any]:
  def decorator(method: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(method)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
      step_to_profile = self.total_micro_steps
      current_micro_step_count = self.micro_step_count
      if control_profile:
        self._profiler.maybe_activate(step_to_profile, current_micro_step_count, self.train_step, blocking_object=self._state)
      with jax.profiler.StepTraceAnnotation(name, step_num=self.train_step):
        result = method(self, *args, **kwargs)
      if control_profile:
        self._profiler.maybe_deactivate(step_to_profile, current_micro_step_count, self.train_step, blocking_object=self._state)
      return result

    return wrapper

  return decorator


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
    self._compiled = False
    self._compile_requested = False
    self._compiled_signature: Any = None
    self._jitted_kernels: dict[str, Any] = {}
    self._compiled_eval: Any = None
    self._compiled_eval_signature: Any = None
    self._signature_compare_warned: bool = False
    self._replicated_batch_warned: bool = False
    if not training_config.model_name:
      raise ValueError("training_config.model_name must be specified")
    self._model = model if model is not None else self._build_model(wrap_with_tunix_adapter, tokenizer_pad_id)
    self._state: Any = None
    self._params_pure: Any = None
    self._rest_pure: Any = None
    self._state_pure: Any = None
    self._pure_state_warned: bool = False
    self._accumulated_grads: Any = None
    self._accumulated_denominator: Any = None
    self._reduced_params_shardings: Any = None
    self._unreduced_grad_shardings: Any = None
    self._plain_grad_shardings: Any = None
    self._zero1_params_shardings: Any = None
    self._gathered_params_shardings: Any = None
    self._zero1_warned = False
    self._micro_step_count = 0
    self._total_micro_steps = 0
    self._cached_losses: list[abstract_engine.WeightedMetric | jax.Array] = []
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
    self._raiden_sync: Any = None
    self._last_staged_step: Optional[int] = None
    self._staged_metadata: Any = None
    self._target_state: Any = None
    self._use_weight_converter = bool(self._config.use_weight_converter)
    self._rollout_backend = self._config.rollout_backend
    rollout_tp = resolve_rollout_tp(self._config)
    kv_tp = self._config.kv_tp_size or rollout_tp
    moe_tp = self._config.moe_mlp_tp_size or rollout_tp
    prefuse_moe = resolve_prefuse_moe_weights(self._config)
    if self._use_weight_converter:
      from lilax._backend.integration.vllm.weight_converter import WeightConverter

      self._weight_converter = WeightConverter(
          config=self._config,
          tp=rollout_tp,
          kv_tp_size=kv_tp,
          moe_mlp_tp_size=moe_tp,
          prefuse_moe_weights=prefuse_moe,
          rollout_backend=self._rollout_backend,
          debug=self._config.weight_sync_debug,
      )
    else:
      self._weight_converter = None

  def _build_model(self, wrap_with_tunix_adapter: bool, tokenizer_pad_id: int | None) -> Any:
    model_or_model_mesh_pair = model_creation_utils.from_pretrained(
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
    self._compiled = False
    self._compiled_fwd_bwd = None
    self._compiled_fwd_bwd_accum = None
    self._compiled_update = None
    self._compiled_eval = None
    self._compiled_eval_signature = None
    self._model_graphdef = None
    self._invalidate_pure_state()

  @property
  def optimizer(self) -> Any:
    return self._optimizer

  @optimizer.setter
  def optimizer(self, new_optimizer: Any) -> None:
    self._optimizer = new_optimizer
    self._compiled = False
    self._compiled_fwd_bwd = None
    self._compiled_fwd_bwd_accum = None
    self._compiled_update = None
    self._state_graphdef = None
    self._invalidate_pure_state()
    self._compiled_eval = None
    self._compiled_eval_signature = None

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
    self._compiled = False
    self._compiled_fwd_bwd = None
    self._compiled_fwd_bwd_accum = None
    self._compiled_update = None
    self._state_graphdef = None
    self._invalidate_pure_state()
    self._compiled_eval = None
    self._compiled_eval_signature = None

  @property
  def micro_step_count(self) -> int:
    return self._micro_step_count

  @property
  def has_accumulated_grads(self) -> bool:
    return self._accumulated_grads is not None

  def with_loss_fn(self, customized_fn: Callable[..., Any], has_aux: bool = False) -> "MaxTextTrainingEngine":
    self._loss_fn = customized_fn
    self._has_aux = has_aux
    self._compiled = False
    self._compiled_eval = None
    self._compiled_eval_signature = None
    return self

  def with_gen_model_input_fn(self, gen_model_input_fn: Callable[[Any], dict[str, Any]]) -> "MaxTextTrainingEngine":
    self._gen_model_input_fn = gen_model_input_fn
    self._compiled = False
    self._compiled_eval = None
    self._compiled_eval_signature = None
    return self

  @contextlib.contextmanager
  def _sharding_ctx(self):
    if self._mesh is None:
      yield
      return
    with jax.set_mesh(self._mesh), logical_axis_rules(self._config.logical_axis_rules):
      yield

  def _invalidate_pure_state(self) -> None:
    self._params_pure = None
    self._rest_pure = None
    self._state_pure = None

  def _disable_pure_state(self, reason: str) -> None:
    self._invalidate_pure_state()
    if not self._pure_state_warned:
      self._pure_state_warned = True
      logging.warning(_PURE_STATE_FALLBACK_WARNING, reason)

  @staticmethod
  def _with_model_state(state_pure: Any, model_pure: Any) -> Any:
    return nnx.State({**state_pure.raw_mapping, _MODEL_STATE_KEY: model_pure.raw_mapping})

  def _check_pure_state_reusable(self, state_pure: Any, params_pure: Any, rest_pure: Any) -> str | None:
    if not isinstance(state_pure, nnx.State) or _MODEL_STATE_KEY not in state_pure:
      return f"the train state's pure form has no {_MODEL_STATE_KEY!r} entry"
    if not hasattr(state_pure, "raw_mapping"):
      return "this version of flax.nnx.State does not expose raw_mapping"
    rebuilt = self._with_model_state(state_pure, nnx.merge_state(params_pure, rest_pure))
    if jax.tree.structure(rebuilt) != jax.tree.structure(state_pure):
      return f"state[{_MODEL_STATE_KEY!r}] is not the model's own state"
    return None

  def _refresh_pure_state(self) -> None:
    if self._state is None:
      self._state = train_state_nnx.TrainStateNNX(self._model, self._optimizer)
    model = getattr(self._state, _MODEL_STATE_KEY, self._model)
    self._state_graphdef, state_pure = nnx.split(self._state)
    self._model_graphdef, params_pure, rest_pure = nnx.split(model, nnx.Param, ...)
    reason = self._check_pure_state_reusable(state_pure, params_pure, rest_pure)
    if reason is not None:
      self._disable_pure_state(reason)
      return
    self._params_pure, self._rest_pure, self._state_pure = params_pure, rest_pure, state_pure

  def _read_model_pure(self, model: Any) -> tuple[Any, Any]:
    if self._params_pure is not None:
      return self._params_pure, self._rest_pure
    self._model_graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    return params, rest

  def _read_state_pure(self) -> Any:
    if self._state_pure is not None:
      return self._state_pure
    self._state_graphdef, state_pure = nnx.split(self._state)
    return state_pure

  def _publish_model_rest(self, new_rest: Any) -> None:
    if self._params_pure is None:
      return
    if jax.tree.structure(new_rest) != jax.tree.structure(self._rest_pure):
      self._disable_pure_state("fwd_bwd returned a wider non-parameter state than the model was split into")
      return
    self._rest_pure = new_rest
    self._state_pure = self._with_model_state(self._state_pure, nnx.merge_state(self._params_pure, new_rest))

  def _publish_state(self, new_state_pure: Any) -> None:
    if self._params_pure is None:
      return
    if not isinstance(new_state_pure, nnx.State) or _MODEL_STATE_KEY not in new_state_pure:
      self._disable_pure_state("the update kernel returned a state with no model entry")
      return
    params_pure, rest_pure = nnx.split_state(new_state_pure[_MODEL_STATE_KEY], nnx.Param, ...)
    if jax.tree.structure(params_pure) != jax.tree.structure(self._params_pure):
      self._disable_pure_state("the update kernel's output does not partition into the same parameters")
      return
    self._params_pure, self._rest_pure, self._state_pure = params_pure, rest_pure, new_state_pure

  def _note_zero1_declined(self, reason: str) -> None:
    if getattr(self._config, "shard_optimizer_over_data", False) and not self._zero1_warned:
      self._zero1_warned = True
      logging.warning(_ZERO1_DECLINED_WARNING, reason)

  def _zero1_shardings_for(self, params_pure: Any, params_shardings: Any) -> Any:
    declined = _zero1_active(self._config, self._mesh)
    if declined is not None:
      self._note_zero1_declined(declined)
      return None

    def target(leaf, base):
      sharded = _zero1_sharding(self._mesh, leaf, base)
      return base if sharded is None else sharded

    return jax.tree.map(target, params_pure, params_shardings)

  def _place_leaf(self, leaf: Any, target: jax.sharding.Sharding) -> Any:
    return jax.device_put(leaf, target)

  def _place_state_on_mesh(self) -> None:
    if self._mesh is None or self._state is None:
      return
    moved = False

    def place(leaf):
      nonlocal moved
      if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
        return leaf
      leaf_sharding = getattr(leaf, "sharding", None)
      if isinstance(leaf_sharding, jax.sharding.NamedSharding) and leaf_sharding.mesh == self._mesh:
        return leaf
      moved = True
      return self._place_leaf(leaf, self._mesh_sharding(leaf))

    placed = jax.tree.map(place, self._read_state_pure())
    if not moved:
      return
    with self._sharding_ctx():
      nnx.update(self._state, placed)
    self._invalidate_pure_state()
    self._refresh_pure_state()

  def _shard_optimizer_state_over_data(self) -> None:
    if _zero1_active(self._config, self._mesh) is not None:
      return
    state_pure = self._read_state_pure()
    if _OPTIMIZER_STATE_KEY not in state_pure:
      return
    moved = False

    def place(leaf):
      nonlocal moved
      target = _zero1_sharding(self._mesh, leaf, self._mesh_sharding(leaf))
      if target is None:
        return leaf
      moved = True
      return self._place_leaf(leaf, target)

    optimizer_pure = jax.tree.map(place, state_pure[_OPTIMIZER_STATE_KEY])
    if not moved:
      return
    with self._sharding_ctx():
      nnx.update(self._state, nnx.State({_OPTIMIZER_STATE_KEY: optimizer_pure.raw_mapping}))
    self._invalidate_pure_state()
    self._refresh_pure_state()

  def _reshard_model_params(self, state_pure: Any, params_shardings: Any) -> Any:
    params_pure, rest_pure = nnx.split_state(state_pure[_MODEL_STATE_KEY], nnx.Param, ...)
    params_pure = jax.tree.map(jax.sharding.reshard, params_pure, params_shardings)
    return self._with_model_state(state_pure, nnx.merge_state(params_pure, rest_pure))

  def _fwd_bwd_kernel(self, params, rest, batch, acc_grads=None, acc_denom=None):
    loss_callable = self._loss_fn if self._loss_fn is not None else maxtext_train.loss_fn

    def diff_wrapper(params_state, rest_state, inputs):
      model = nnx.merge(self._model_graphdef, params_state, rest_state, copy=True)
      if self._gen_model_input_fn is not None:
        if not isinstance(inputs, dict):
          raise TypeError(
              "gen_model_input_fn must return a dict of loss-fn keyword arguments, got " f"{type(inputs).__name__}."
          )
        out = loss_callable(model, **inputs)
      else:
        out = loss_callable(model, self._config, inputs, None, None, is_train=True)
      _, _, new_rest = nnx.split(model, nnx.Param, ...)
      loss_out = _normalize_loss_output(out, self._has_aux)
      return loss_out.primary_loss.unreduced_sum, (loss_out, new_rest)

    if self._reduced_params_shardings is not None:
      params = jax.tree.map(jax.sharding.reshard, params, self._reduced_params_shardings)
    grad_func = jax.value_and_grad(diff_wrapper, argnums=0, has_aux=True)
    (_, (loss_out, new_rest)), micro_grads = grad_func(params, rest, batch)
    micro_grads = jax.tree.map(
        lambda grad: (
            grad.astype(self._config.grad_dtype)
            if hasattr(grad, "dtype") and jnp.issubdtype(grad.dtype, jnp.floating) and grad.dtype != self._config.grad_dtype
            else grad
        ),
        micro_grads,
    )
    denominator = loss_out.primary_loss.denominator.astype(jnp.float32)
    if acc_grads is None:
      return loss_out.primary_loss, loss_out.aux_metrics, new_rest, micro_grads, denominator
    acc_grads = jax.tree.map(jnp.add, acc_grads, micro_grads)
    return loss_out.primary_loss, loss_out.aux_metrics, new_rest, acc_grads, acc_denom + denominator

  def _update_kernel(self, state_pure, accumulated_grads, accumulated_denominator, mean_loss):
    grad_norm = None
    is_skipped_val = None
    if state_pure is not None:
      grad_target = self._zero1_params_shardings
      if grad_target is None:
        grad_target = self._plain_grad_shardings
      if grad_target is not None:
        accumulated_grads = jax.tree.map(jax.sharding.reshard, accumulated_grads, grad_target)
      if self._zero1_params_shardings is not None:
        state_pure = self._reshard_model_params(state_pure, self._zero1_params_shardings)
      has_weights = accumulated_denominator > 0
      safe_denominator = jnp.where(has_weights, accumulated_denominator, 1.0)
      grads = jax.tree.map(
          lambda grad: jnp.where(has_weights, grad / safe_denominator.astype(grad.dtype), jnp.zeros_like(grad)),
          accumulated_grads,
      )
      grad_norm = max_utils.l2norm_pytree(jax.tree.map(lambda grad: grad.astype(jnp.float32), grads))
      if self._config.gradient_clipping_threshold > 0:
        grads = maxtext_utils.apply_gradient_clipping(grads, None, self._config.gradient_clipping_threshold)
      local_state = nnx.merge(self._state_graphdef, state_pure, copy=True)
      if hasattr(local_state, "apply_gradients"):
        if self._config.skip_step_on_spikes:
          local_state.apply_gradients(grads, loss=mean_loss, grad_norm=grad_norm)
          opt_obj = getattr(local_state, "optimizer", self._optimizer)
          if opt_obj is not None:
            opt_state = nnx.to_pure_dict(nnx.state(opt_obj)).get("opt_state", {})
            is_skipped = opt_state.get("is_skipped") if isinstance(opt_state, dict) else None
            if is_skipped is not None:
              is_skipped_val = is_skipped.astype(jnp.float32)
        else:
          local_state.apply_gradients(grads)
      _, new_state_pure = nnx.split(local_state)
      if self._zero1_params_shardings is not None:
        new_state_pure = self._reshard_model_params(new_state_pure, self._gathered_params_shardings)
      return new_state_pure, grad_norm, is_skipped_val
    return state_pure, grad_norm, is_skipped_val

  def _eval_kernel(self, params, rest, batch):
    loss_callable = self._loss_fn if self._loss_fn is not None else maxtext_train.loss_fn
    model = nnx.merge(self._model_graphdef, params, rest, copy=True)
    if self._gen_model_input_fn is not None:
      if not isinstance(batch, dict):
        raise TypeError(
            "gen_model_input_fn must return a dict of loss-fn keyword arguments, got " f"{type(batch).__name__}."
        )
      out = loss_callable(model, **batch)
    else:
      out = loss_callable(model, self._config, batch, None, None, is_train=False)
    loss_out = _normalize_loss_output(out, self._has_aux)
    return loss_out.primary_loss, loss_out.aux_metrics

  def _warn_uncomparable(self, what: str, hint: str, exc: Exception) -> None:
    if self._signature_compare_warned:
      return
    self._signature_compare_warned = True
    logging.warning(_UNCOMPARABLE_SIGNATURE_WARNING, what, exc, hint)

  def _needs_recompile(self, signature: Any, previous: Any) -> bool:
    if previous is None:
      return True
    try:
      if bool(previous[:2] != signature[:2]):
        return True
    except Exception as exc:
      self._warn_uncomparable("batch structure", _UNCOMPARABLE_STRUCTURE_HINT, exc)
      return True
    try:
      return bool(previous[2] != signature[2])
    except Exception as exc:
      self._warn_uncomparable("static loss arguments", _UNCOMPARABLE_STATIC_HINT, exc)
      return True

  def _prepare_batch(self, payload: Any) -> Any:
    if self._gen_model_input_fn is not None:
      if dataclasses.is_dataclass(payload) and "metadata" in payload.__dataclass_fields__:
        payload = dataclasses.replace(payload, metadata={})
      return self._gen_model_input_fn(payload)
    if dataclasses.is_dataclass(payload):
      return {
          key: getattr(payload, key)
          for key in payload.__dataclass_fields__
          if getattr(payload, key) is not None and key != "metadata"
      }
    return payload

  def _mesh_sharding(self, leaf: Any) -> jax.sharding.Sharding | None:
    if leaf is None:
      return None
    leaf_sharding = getattr(leaf, "sharding", None)
    if isinstance(leaf_sharding, jax.sharding.NamedSharding) and leaf_sharding.mesh == self._mesh:
      return leaf_sharding
    return jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec())

  def _batch_axis_size(self, axis_names: Any) -> int:
    if axis_names is None:
      return 1
    names = axis_names if isinstance(axis_names, tuple) else (axis_names,)
    size = 1
    for name in names:
      size *= self._mesh.shape[name]
    return size

  def _input_data_spec(self) -> tuple[Any, ...]:
    return tuple(sharding.get_input_data_sharding(self._config, self._mesh).spec)

  def _leaf_data_sharding(self, leaf: Any, data_spec: tuple[Any, ...]) -> jax.sharding.NamedSharding | None:
    if leaf is None:
      return None
    rank = jnp.ndim(leaf)
    spec = list(data_spec[:rank])
    if spec and spec[0] is not None:
      axis_size = self._batch_axis_size(spec[0])
      if leaf.shape[0] % axis_size:
        if not self._replicated_batch_warned:
          self._replicated_batch_warned = True
          logging.warning(_REPLICATED_BATCH_DIM_WARNING, leaf.shape[0], spec[0], axis_size, axis_size)
        spec[0] = None
    return jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec(*spec))

  def _batch_data_shardings(self, dynamic_batch: Any) -> Any:
    data_spec = self._input_data_spec()
    return jax.tree.map(lambda leaf: self._leaf_data_sharding(leaf, data_spec), dynamic_batch)

  def _compile_for_batch(self, dynamic_batch: Any, static_batch: dict[str, Any]) -> None:
    self._refresh_pure_state()
    self._place_state_on_mesh()
    self._shard_optimizer_state_over_data()
    state_pure = self._read_state_pure()
    params_pure, rest_pure = self._read_model_pure(getattr(self._state, _MODEL_STATE_KEY, self._model))

    def fwd_bwd(params, rest, dynamic):
      batch = {**dynamic, **static_batch} if isinstance(dynamic, dict) else dynamic
      return self._fwd_bwd_kernel(params, rest, batch)

    def fwd_bwd_accum(params, rest, dynamic, acc_grads, acc_denom):
      batch = {**dynamic, **static_batch} if isinstance(dynamic, dict) else dynamic
      return self._fwd_bwd_kernel(params, rest, batch, acc_grads, acc_denom)

    def update(state_pure, accumulated_grads, accumulated_denominator, mean_loss):
      return self._update_kernel(state_pure, accumulated_grads, accumulated_denominator, mean_loss)

    if self._mesh is not None:
      replicated = jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec())
      state_mesh_shardings = jax.tree.map(self._mesh_sharding, state_pure)
      params_shardings = jax.tree.map(self._mesh_sharding, params_pure)
      rest_shardings = jax.tree.map(self._mesh_sharding, rest_pure)
      batch_shardings = self._batch_data_shardings(dynamic_batch)
      self._reduced_params_shardings, self._unreduced_grad_shardings = _deferred_all_reduce_shardings(
          self._config, self._mesh, params_shardings
      )
      self._zero1_params_shardings = self._zero1_shardings_for(params_pure, params_shardings)
      self._gathered_params_shardings = params_shardings if self._zero1_params_shardings is not None else None
      grad_shardings = self._unreduced_grad_shardings
      if grad_shardings is None:
        grad_shardings = params_shardings
        self._plain_grad_shardings = None
      else:
        self._plain_grad_shardings = params_shardings
      first_in_shardings = (params_shardings, rest_shardings, batch_shardings)
      accum_in_shardings = first_in_shardings + (grad_shardings, replicated)
      fwd_bwd_out_shardings = (None, None, rest_shardings, grad_shardings, replicated)
      update_in_shardings = (state_mesh_shardings, grad_shardings, replicated, None)
      update_out_shardings = (state_mesh_shardings, None, None)
      if self._accumulated_grads is not None:
        with self._sharding_ctx():
          self._accumulated_grads = jax.tree.map(_conform_accumulator, self._accumulated_grads, grad_shardings)
    else:
      first_in_shardings = None
      accum_in_shardings = None
      fwd_bwd_out_shardings = None
      update_in_shardings = None
      update_out_shardings = None
      self._reduced_params_shardings = None
      self._unreduced_grad_shardings = None
      self._plain_grad_shardings = None
      self._note_zero1_declined(_zero1_active(self._config, self._mesh))
      self._zero1_params_shardings = None
      self._gathered_params_shardings = None

    self._compiled_fwd_bwd = jax.jit(
        fwd_bwd, in_shardings=first_in_shardings, out_shardings=fwd_bwd_out_shardings
    )
    self._compiled_fwd_bwd_accum = jax.jit(
        fwd_bwd_accum, in_shardings=accum_in_shardings, out_shardings=fwd_bwd_out_shardings, donate_argnums=(3, 4)
    )
    self._compiled_update = jax.jit(
        update, in_shardings=update_in_shardings, out_shardings=update_out_shardings, donate_argnums=(0,)
    )
    self._jitted_kernels = {
        "fwd_bwd": self._compiled_fwd_bwd,
        "fwd_bwd_accum": self._compiled_fwd_bwd_accum,
        "update": self._compiled_update,
    }
    self._compiled_signature = _batch_signature(dynamic_batch, static_batch)
    self._compiled = True

  def _compile_eval_for_batch(self, dynamic_batch: Any, static_batch: dict[str, Any]) -> None:
    self._model_graphdef, params_pure, rest_pure = nnx.split(self._model, nnx.Param, ...)

    def kernel(params, rest, dynamic):
      batch = {**dynamic, **static_batch} if isinstance(dynamic, dict) else dynamic
      return self._eval_kernel(params, rest, batch)

    if self._mesh is not None:
      params_shardings = jax.tree.map(self._mesh_sharding, params_pure)
      rest_shardings = jax.tree.map(self._mesh_sharding, rest_pure)
      eval_in_shardings = (params_shardings, rest_shardings, self._batch_data_shardings(dynamic_batch))
      eval_out_shardings = (None, None)
    else:
      eval_in_shardings = None
      eval_out_shardings = None
    self._compiled_eval = jax.jit(kernel, in_shardings=eval_in_shardings, out_shardings=eval_out_shardings)
    self._compiled_eval_signature = _batch_signature(dynamic_batch, static_batch)

  def compile(
      self, dummy_data: abstract_engine.TrainerPayload | None = None, compiler_options: dict[str, Any] | None = None
  ) -> None:
    def _resolve_dummy_routed_experts(data: Any) -> Any:
      if not isinstance(data, abstract_engine.RLTrainerPayload):
        return data
      routed_experts = data.routed_experts
      if routed_experts is None or routed_experts.size != 0:
        return data
      num_layers = self._config.num_decoder_layers
      top_k = self._config.num_experts_per_tok
      batch_size, seq_len = routed_experts.shape[:2]
      expanded = np.zeros((batch_size, seq_len, num_layers, top_k), dtype=routed_experts.dtype)
      return dataclasses.replace(data, routed_experts=expanded)

    self._compile_requested = True
    if self._compiled:
      return
    if dummy_data is None:
      logging.info(
          "MaxTextTrainingEngine.compile() was called without dummy_data; compiling "
          "against the first fwd_bwd payload instead."
      )
      return
    dummy_data = _resolve_dummy_routed_experts(dummy_data)
    compiled = self.compile_kernels(dummy_data, compiler_options)
    self._compiled_fwd_bwd = compiled["fwd_bwd"]
    self._compiled_fwd_bwd_accum = compiled["fwd_bwd_accum"]
    self._compiled_update = compiled["update"]
    self._compile_eval(dummy_data, compiler_options)

  def compile_kernels(
      self, dummy_data: abstract_engine.TrainerPayload, compiler_options: dict[str, Any] | None = None
  ) -> dict[str, jax.stages.Compiled]:
    options = self._xla_options(compiler_options)
    lowered = self._lower_kernels(dummy_data)
    with self._sharding_ctx():
      return {name: lowered[name].compile(compiler_options=options) for name in KERNEL_NAMES}

  def _xla_options(self, compiler_options: dict[str, Any] | None) -> dict[str, Any] | None:
    if compiler_options is None and getattr(self._config, "compile_xla_flags", ""):
      return max_utils.parse_libtpu_flags_to_dict(self._config.compile_xla_flags)
    return compiler_options

  def _compile_eval(self, dummy_data: abstract_engine.TrainerPayload, compiler_options: dict[str, Any] | None) -> None:
    dynamic_batch, static_batch = _split_static_and_dynamic(self._prepare_batch(dummy_data))
    self._compile_eval_for_batch(dynamic_batch, static_batch)
    params_pure, rest_pure = self._read_model_pure(getattr(self._state, _MODEL_STATE_KEY, self._model))
    with self._sharding_ctx():
      self._compiled_eval = self._compiled_eval.lower(
          jax.tree.map(_to_aval, params_pure), jax.tree.map(_to_aval, rest_pure), jax.tree.map(_to_aval, dynamic_batch)
      ).compile(compiler_options=self._xla_options(compiler_options))

  def _lower_kernels(self, dummy_data: abstract_engine.TrainerPayload) -> dict[str, jax.stages.Lowered]:
    if dummy_data is None:
      raise ValueError(
          "compile_kernels() needs a dummy payload -- unlike compile(), it cannot defer to the first real batch."
      )
    dynamic_batch, static_batch = _split_static_and_dynamic(self._prepare_batch(dummy_data))
    self._compile_for_batch(dynamic_batch, static_batch)
    state_aval = jax.tree.map(_to_aval, self._read_state_pure())
    params_pure, rest_pure = self._read_model_pure(getattr(self._state, _MODEL_STATE_KEY, self._model))
    params_aval = jax.tree.map(_to_aval, params_pure)
    rest_aval = jax.tree.map(_to_aval, rest_pure)
    batch_aval = jax.tree.map(_to_aval, dynamic_batch)
    mean_loss_aval = jax.ShapeDtypeStruct((), jnp.float32) if self._config.skip_step_on_spikes else None
    with self._sharding_ctx():
      fwd_bwd = self._jitted_kernels["fwd_bwd"].lower(params_aval, rest_aval, batch_aval)
      _, _, _, grads_aval, denominator_aval = fwd_bwd.out_info
      return {
          "fwd_bwd": fwd_bwd,
          "fwd_bwd_accum": self._jitted_kernels["fwd_bwd_accum"].lower(
              params_aval, rest_aval, batch_aval, grads_aval, denominator_aval
          ),
          "update": self._jitted_kernels["update"].lower(state_aval, grads_aval, denominator_aval, mean_loss_aval),
      }

  @_profiled_step("fwd_bwd", control_profile=True)
  def fwd_bwd(self, payload: abstract_engine.TrainerPayload) -> None:
    batch = self._prepare_batch(payload)
    model = getattr(self._state, "model", None) if self._state is not None else self._model
    if not isinstance(model, nnx.Module):
      raise TypeError("MaxTextTrainingEngine requires an NNX model (flax.nnx.Module), got" f" {type(model).__name__}")
    self._throttler.wait_for_next()
    if self._state is None:
      self._state = train_state_nnx.TrainStateNNX(self._model, self._optimizer)
    model = getattr(self._state, _MODEL_STATE_KEY, self._model)
    if self._compile_requested:
      dynamic_batch, static_batch = _split_static_and_dynamic(batch)
      signature = _batch_signature(dynamic_batch, static_batch)
      if not self._compiled or self._needs_recompile(signature, self._compiled_signature):
        self._compile_for_batch(dynamic_batch, static_batch)
      params, rest = self._read_model_pure(model)
      with self._sharding_ctx():
        if self._accumulated_grads is None:
          loss, aux, new_rest, acc_grads, acc_denom = self._compiled_fwd_bwd(params, rest, dynamic_batch)
        else:
          loss, aux, new_rest, acc_grads, acc_denom = self._compiled_fwd_bwd_accum(
              params, rest, dynamic_batch, self._accumulated_grads, self._accumulated_denominator
          )
    else:
      params, rest = self._read_model_pure(model)
      with self._sharding_ctx():
        loss, aux, new_rest, acc_grads, acc_denom = self._fwd_bwd_kernel(
            params, rest, batch, self._accumulated_grads, self._accumulated_denominator
        )
    nnx.update(model, new_rest)
    self._publish_model_rest(new_rest)
    self._throttler.add_computation(computation=loss, metrics=None)
    if isinstance(loss, abstract_engine.WeightedMetric):
      self.record_metrics("loss", loss)
    if isinstance(aux, dict):
      for key, value in aux.items():
        if value is not None:
          self.record_metrics(key, value)
    self._cached_losses.append(loss)
    self._accumulated_grads = acc_grads
    self._accumulated_denominator = acc_denom
    self._micro_step_count += 1
    self._total_micro_steps += 1

  @_profiled_step("update")
  def update(self) -> int:
    if self._accumulated_grads is None:
      return self.train_step
    if self._learning_rate_schedule is not None:
      lr = self._learning_rate_schedule(self.train_step)
      self.record_metrics("learning_rate", lr)
    self._throttler.wait_for_next()
    if self._state is None:
      self._state = train_state_nnx.TrainStateNNX(self._model, self._optimizer)
    state_pure = self._read_state_pure()
    if not self._config.skip_step_on_spikes:
      mean_loss = None
    elif self._cached_losses:
      loss_values = [loss.compute() if isinstance(loss, abstract_engine.WeightedMetric) else loss for loss in self._cached_losses]
      mean_loss = jnp.mean(jnp.stack(loss_values)) if len(loss_values) > 1 else loss_values[0]
    else:
      mean_loss = jnp.array(0.0)
    with self._sharding_ctx():
      if self._compiled and hasattr(self, "_compiled_update"):
        new_state_pure, grad_norm, is_skipped = self._compiled_update(
            state_pure, self._accumulated_grads, self._accumulated_denominator, mean_loss
        )
      else:
        new_state_pure, grad_norm, is_skipped = self._update_kernel(
            state_pure, self._accumulated_grads, self._accumulated_denominator, mean_loss
        )
    nnx.update(self._state, new_state_pure)
    self._publish_state(new_state_pure)
    if grad_norm is not None:
      self.record_metrics("gradient_norm", grad_norm)
    if is_skipped is not None:
      self.record_metrics("step_skipped", is_skipped)
    self._throttler.add_computation(
        grad_norm if grad_norm is not None else (self._state if self._state is not None else self._model),
        self._metrics_recorder.get_step_metrics(self.train_step),
    )
    self._cached_losses.clear()
    self._accumulated_grads = None
    self._accumulated_denominator = None
    self._micro_step_count = 0
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
    batch = self._prepare_batch(payload)
    model = getattr(self._state, "model", self._model) if self._state is not None else self._model
    if not isinstance(model, nnx.Module):
      raise TypeError("MaxTextTrainingEngine requires an NNX model (flax.nnx.Module), got" f" {type(model).__name__}")
    self._model_graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    self._throttler.wait_for_next()
    if self._compile_requested:
      dynamic_batch, static_batch = _split_static_and_dynamic(batch)
      signature = _batch_signature(dynamic_batch, static_batch)
      if self._compiled_eval is None or self._needs_recompile(signature, self._compiled_eval_signature):
        self._compile_eval_for_batch(dynamic_batch, static_batch)
      loss, aux = self._compiled_eval(params, rest, dynamic_batch)
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

  def _place_read_only_inputs(self, inputs: Any) -> Any:
    if self._mesh is None:
      return inputs
    data_spec = self._input_data_spec()

    def place(leaf: Any) -> Any:
      if not isinstance(leaf, (jax.Array, np.ndarray)):
        return leaf
      if leaf.ndim == 0 or 0 in leaf.shape:
        return leaf
      if isinstance(leaf, jax.Array) and not leaf.is_fully_addressable:
        return leaf
      target = self._leaf_data_sharding(leaf, data_spec)
      if target is None or getattr(leaf, "sharding", None) == target:
        return leaf
      return jax.device_put(leaf, target)

    return jax.tree.map(place, inputs)

  def fwd_only(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    if self._state is None:
      self._state = train_state_nnx.TrainStateNNX(self._model, self._optimizer)
    model = getattr(self._state, _MODEL_STATE_KEY, self._model)
    if not isinstance(model, nnx.Module):
      raise TypeError("MaxTextTrainingEngine requires an NNX model (flax.nnx.Module), got" f" {type(model).__name__}")
    self._throttler.wait_for_next()
    with self._sharding_ctx():
      args, kwargs = self._place_read_only_inputs((args, kwargs))
      out = fn(model, *args, **kwargs)
    self._throttler.add_computation(
        computation=[leaf for leaf in jax.tree.leaves(out) if isinstance(leaf, jax.Array)], metrics=None
    )
    return out

  def save_checkpoint(self, metadata: Any = None, *, step: int | None = None, force: bool = False) -> None:
    if self._micro_step_count > 0:
      raise ValueError(
          f"Mid-step checkpointing is not supported; micro_step_count must be 0, got {self._micro_step_count}."
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
    self._invalidate_pure_state()
    self.train_step = restored_step
    self._metrics_recorder.cleanup()
    self._cached_losses.clear()
    self._accumulated_grads = None
    self._accumulated_denominator = None
    self._micro_step_count = 0
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
    self._target_state = target_state

  def prepare_weight_sync(self, *, sync_request: Any = None, staging_transport: str = "raiden") -> Any:
    del sync_request
    if staging_transport == "raiden":
      if self._raiden_sync is not None and self._last_staged_step == self.train_step and self._staged_metadata is not None:
        logging.info(
            "Trainer reusing staged weight sync for step %d (%d variables)",
            self.train_step, sum(len(metadata.variables) for metadata in self._staged_metadata),
        )
        return self._staged_metadata
      self._throttler.wait_for_all()
      params_state = self._get_trainable_params_state()
      if self._use_weight_converter:
        converted_state = self._weight_converter.convert(params_state, target_state=self._target_state)
      else:
        params_state = jax.tree_util.tree_map(
            lambda value: value.astype(jnp.bfloat16)
            if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating) else value,
            params_state,
        )
        if self._config.scan_layers:
          converted_state = raiden_unscan.unscan_layers(
              params_state,
              num_layers=self._config.num_decoder_layers,
              scan_axis=self._config.param_scan_axis,
              cycle_interval=self._config.inhomogeneous_layer_cycle_interval,
          )
        else:
          converted_state = params_state
      del params_state
      if self._raiden_sync is None:
        self._raiden_sync = raiden_synchronizer.RaidenSynchronizer(
            job_name="trainer", worker_index=jax.process_index(), auto_h2d=False, parallelism=4
        )
      self._raiden_sync.bind(converted_state)
      del converted_state
      if self._raiden_sync.active:
        self._raiden_sync.d2h()
      verify_weights = is_verify_weights_enabled()
      if verify_weights:
        logging.info("Source weights checksums: %s", self._raiden_sync.checksums())
      all_metadata = self._raiden_sync.work_unit_metadata_all()
      total_variables = sum(len(metadata.variables) for metadata in all_metadata)
      logging.info(
          "Trainer prepared weight sync for step %d: registered %d work unit(s) with %d variables on mesh %s",
          self.train_step, len(all_metadata), total_variables, all_metadata[0].mesh_axes if all_metadata else (),
      )
      self._last_staged_step = self.train_step
      self._staged_metadata = all_metadata
      return all_metadata
    raise ValueError(f"unknown staging_transport {staging_transport!r}; expected 'raiden'.")

  def release_weight_sync(self, *, sync_request: Any = None) -> Any:
    del sync_request
    self._last_staged_step = None
    self._staged_metadata = None
    if self._raiden_sync:
      logging.vlog(1, "Trainer Raiden metrics: %s", self._raiden_sync.metrics())
    return True

  def close(self) -> None:
    if self._profiler is not None:
      self._profiler.close(blocking_object=self._read_state_pure() if self._state is not None else None)
    if self._raiden_sync:
      if hasattr(self._raiden_sync, "close"):
        self._raiden_sync.close()
      self._raiden_sync = None
    self._last_staged_step = None
    self._staged_metadata = None
    if self._config.enable_checkpointing and self._checkpoint_dir() and self._checkpoint_manager:
      if self._micro_step_count > 0:
        logging.warning("Closing mid-step (micro_step_count=%d); skipping the final checkpoint.", self._micro_step_count)
      else:
        self.save_checkpoint(metadata=None, force=True)
      self._checkpoint_manager.close()
    self._throttler.cleanup()
    self._metrics_recorder.cleanup()
    self._metrics_logger.cleanup()
