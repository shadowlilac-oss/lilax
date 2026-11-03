"""Converters that translate MaxText trainer weights into vLLM rollout weights."""

import abc
import dataclasses
import gc
import logging
import re
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, Union

from flax import nnx, traverse_util
import jax
import jax.numpy as jnp

from lilax._backend.integration.vllm.convert_utils import (
    MOE_MLP_WEIGHTS,
    _align_per_axis,
    _apply_dtype_cast,
    _bulk_align_and_unstack,
    _device_ids,
    _fuse_and_unstack_moe,
    _get_n_shards,
    _jit_repeat_axes,
    _jit_unstack,
    _scanned_sharding_from_per_layer,
    _sharding_summary,
    normalize_dtype,
    pad_to_tpu_lanes,
    resolve_rollout_tp,
)
from lilax._backend.integration.vllm.weight_tree import (
    _NON_WEIGHT_PATH_PREFIXES,
    _is_non_weight_path,
    _non_param_paths,
    _rekey_to_target,
    _strip_root,
    _to_pure_dict,
)
from lilax._backend.integration.vllm.weight_execution import execute_group
from lilax._backend.integration.vllm.weight_plan import build_plan, build_target_free_plan


class Operation(abc.ABC):
  """Base class for weight transformation operations."""

  @abc.abstractmethod
  def __call__(self, tensors: List[Any], **kwargs) -> Any:
    pass


class Transpose(Operation):
  """Transposes the input tensor along specified axes."""

  def __init__(self, axes):
    self.axes = axes

  def __call__(self, tensors, **kwargs):
    @jax.jit
    def _f(t):
      return jnp.transpose(t, self.axes)

    return _f(tensors[0])


class TransposeUnstack(Operation):
  """Transposes the input tensor and unstacks along axis 0."""

  def __init__(self, axes):
    self.axes = axes

  def __call__(self, tensors, **kwargs):
    @jax.jit
    def _f(t):
      return jnp.transpose(t, self.axes)

    return list(jnp.unstack(_f(tensors[0]), axis=0))


class AttentionOut(Operation):
  """Transforms and unstacks attention output projection weights."""

  def __call__(self, tensors, **kwargs):
    @jax.jit
    def _f(t):
      o = jnp.transpose(t, (1, 3, 0, 2))
      return o.reshape(o.shape[0], o.shape[1], -1)

    return list(jnp.unstack(_f(tensors[0]), axis=0))


class AttentionQKV(Operation):
  """Combines and transforms Q, K, V projections into interleaved QKV format."""

  def __call__(self, tensors, **kwargs):
    tp = kwargs.get("tp", 1)

    @jax.jit
    def _f(q_in, k_in, v_in):
      q = jnp.transpose(q_in, (1, 0, 2, 3))
      k = jnp.transpose(k_in, (1, 0, 2, 3))
      v = jnp.transpose(v_in, (1, 0, 2, 3))
      num_layers, d_model, num_q_heads, head_dim = q.shape
      num_kv_heads = k.shape[2]
      actual_tp = min(tp, num_kv_heads)
      kv_per_tp = num_kv_heads // actual_tp
      q_per_tp = num_q_heads // actual_tp
      q_by_tp = q.reshape(num_layers, d_model, actual_tp, q_per_tp, head_dim)
      k_by_tp = k.reshape(num_layers, d_model, actual_tp, kv_per_tp, head_dim)
      v_by_tp = v.reshape(num_layers, d_model, actual_tp, kv_per_tp, head_dim)
      qkv_by_tp = jnp.concatenate([q_by_tp, k_by_tp, v_by_tp], axis=3)
      qkv_flat = qkv_by_tp.reshape(num_layers, d_model, -1)
      return jnp.transpose(qkv_flat, (0, 2, 1))

    res = _f(tensors[0], tensors[1], tensors[2])
    return list(jnp.unstack(res, axis=0))


class MoEExpertDown(Operation):
  """Transposes MoE expert down projection weights across layer axis."""

  def __call__(self, tensors, **kwargs):
    @jax.jit
    def _f(t_in):
      return jnp.transpose(t_in, (1, 0, 2, 3))

    return list(jnp.unstack(_f(tensors[0]), axis=0))


class MoEFuseGateUp(Operation):
  """Fuses separate gate and up MoE projections with TP chunking."""

  def __call__(self, tensors, **kwargs):
    tp = kwargs.get("tp", 1)

    @jax.jit
    def _fuse_all(wi_0, wi_1):
      wi_0 = jnp.transpose(wi_0, (1, 0, 2, 3))
      wi_1 = jnp.transpose(wi_1, (1, 0, 2, 3))

      def _fuse_single(w0, w1):
        w0 = jnp.transpose(w0, (0, 2, 1))
        w1 = jnp.transpose(w1, (0, 2, 1))
        num_experts, d_inner, d_model = w0.shape
        chunk_size = d_inner // tp
        padded_chunk_size = pad_to_tpu_lanes(chunk_size)
        pad_amount = padded_chunk_size - chunk_size
        gate_chunks = w0.reshape(num_experts, tp, chunk_size, d_model)
        up_chunks = w1.reshape(num_experts, tp, chunk_size, d_model)
        if pad_amount > 0:
          gate_chunks = jnp.pad(gate_chunks, ((0, 0), (0, 0), (0, pad_amount), (0, 0)))
          up_chunks = jnp.pad(up_chunks, ((0, 0), (0, 0), (0, pad_amount), (0, 0)))
        combined = jnp.stack([gate_chunks, up_chunks], axis=2)
        res = combined.reshape(num_experts, 2 * padded_chunk_size * tp, d_model)
        return jnp.transpose(res, (0, 2, 1))

      return jax.vmap(_fuse_single)(wi_0, wi_1)

    fused = _fuse_all(tensors[0], tensors[1])
    return list(jnp.unstack(fused, axis=0))


class MoEFuseGateUpPrefused(Operation):
  """Restructures pre-fused gate/up MoE projections with TP chunking."""

  def __call__(self, tensors, **kwargs):
    tp = kwargs.get("tp", 1)

    @jax.jit
    def _fuse_all(wi):
      wi = jnp.transpose(wi, (1, 0, 2, 3))

      def _fuse_single(w):
        w = jnp.transpose(w, (0, 2, 1))
        num_experts, double_d_inner, d_model = w.shape
        d_inner = double_d_inner // 2
        w0 = w[:, :d_inner, :]
        w1 = w[:, d_inner:, :]
        chunk_size = d_inner // tp
        padded_chunk_size = pad_to_tpu_lanes(chunk_size)
        pad_amount = padded_chunk_size - chunk_size
        gate_chunks = w0.reshape(num_experts, tp, chunk_size, d_model)
        up_chunks = w1.reshape(num_experts, tp, chunk_size, d_model)
        if pad_amount > 0:
          gate_chunks = jnp.pad(gate_chunks, ((0, 0), (0, 0), (0, pad_amount), (0, 0)))
          up_chunks = jnp.pad(up_chunks, ((0, 0), (0, 0), (0, pad_amount), (0, 0)))
        combined = jnp.stack([gate_chunks, up_chunks], axis=2)
        res = combined.reshape(num_experts, 2 * padded_chunk_size * tp, d_model)
        return jnp.transpose(res, (0, 2, 1))

      return jax.vmap(_fuse_single)(wi)

    fused = _fuse_all(tensors[0])
    return list(jnp.unstack(fused, axis=0))


class Rule:
  """Unified rule format for converting weights."""

  def __init__(self, source_patterns: Union[str, List[str]], target_pattern: str, operations: List[Operation] = None):
    if isinstance(source_patterns, str):
      self.source_patterns = [source_patterns]
    else:
      self.source_patterns = source_patterns
    self.target_pattern = target_pattern
    self.operations = operations or []


MODEL_TO_CONVERSION_RULES = {
    "qwen3": None,
    "qwen35_moe": None,
    "qwen3_moe": [
        Rule(["base.token_embedder.embedding"], "vllm_model.model.embed_tokens.weight"),
        Rule(["base.decoder.decoder_norm.scale"], "vllm_model.model.norm.weight"),
        Rule(["base.decoder.logits_dense.kernel"], "vllm_model.lm_head.weight", [Transpose(axes=(1, 0))]),
        Rule(["base.decoder.layers.pre_self_attention_layer_norm.scale"],
             "vllm_model.model.layers.{}.input_layernorm.weight", [TransposeUnstack(axes=(1, 0))]),
        Rule(["base.decoder.layers.post_self_attention_layer_norm.scale"],
             "vllm_model.model.layers.{}.post_attention_layernorm.weight", [TransposeUnstack(axes=(1, 0))]),
        Rule(["base.decoder.layers.self_attention.out.kernel"],
             "vllm_model.model.layers.{}.self_attn.o_proj.weight", [AttentionOut()]),
        Rule(["base.decoder.layers.self_attention.query.kernel", "base.decoder.layers.self_attention.key.kernel",
              "base.decoder.layers.self_attention.value.kernel"],
             "vllm_model.model.layers.{}.self_attn.qkv_proj.weight", [AttentionQKV()]),
        Rule(["base.decoder.layers.self_attention.query_norm.scale"],
             "vllm_model.model.layers.{}.self_attn.q_norm.weight", [TransposeUnstack(axes=(1, 0))]),
        Rule(["base.decoder.layers.self_attention.key_norm.scale"],
             "vllm_model.model.layers.{}.self_attn.k_norm.weight", [TransposeUnstack(axes=(1, 0))]),
        Rule(["base.decoder.layers.moe_block.gate.kernel"],
             "vllm_model.model.layers.{}.mlp.gate.weight", [TransposeUnstack(axes=(1, 2, 0))]),
        Rule(["base.decoder.layers.moe_block.wo"], "vllm_model.model.layers.{}.mlp.experts.w2_weight",
             [MoEExpertDown()]),
        Rule(["base.decoder.layers.moe_block.wi_0", "base.decoder.layers.moe_block.wi_1"],
             "vllm_model.model.layers.{}.mlp.experts.w13_weight", [MoEFuseGateUp()]),
        Rule(["base.decoder.layers.moe_block.wi"], "vllm_model.model.layers.{}.mlp.experts.w13_weight",
             [MoEFuseGateUpPrefused()]),
    ],
}


class WeightConverter:
  """Converts MaxText trainer weights into the rollout engine's layout."""

  def __init__(
      self,
      rules: Optional[List[Rule]] = None,
      tp: int = 1,
      kv_tp_size: Optional[int] = None,
      moe_mlp_tp_size: Optional[int] = None,
      num_kv_heads: Optional[int] = None,
      head_dim: Optional[int] = None,
      config: Any = None,
      trainer_config: Any = None,
      rollout_backend: str = "maxtext",
      moe_fused_layout: Optional[str] = None,
      allow_unused_source_keys: Tuple[str, ...] = (),
      debug: bool = False,
      prefuse_moe_weights: Optional[bool] = None,
      target_dtype: Optional[Any] = None,
  ):
    config = trainer_config if config is None else config
    if rules is not None and not rules:
      raise ValueError(
          "WeightConverter(rules=[]) would convert nothing and leave the "
          "rollout on dummy weights. Pass rules=None for the direct "
          "MaxText-to-MaxText path, or a non-empty rule list."
      )
    self.rules = rules
    self.tp = resolve_rollout_tp(config, tp)
    self.kv_tp_size = kv_tp_size if kv_tp_size is not None else (getattr(config, "kv_tp_size", None) or self.tp)
    self.moe_mlp_tp_size = (
        moe_mlp_tp_size if moe_mlp_tp_size is not None else (getattr(config, "moe_mlp_tp_size", None) or self.tp)
    )
    self.debug = debug
    self.rollout_backend = rollout_backend
    self._direct: Optional["MaxTextToMaxTextConverter"] = None
    if rollout_backend == "maxtext" and rules is None:
      if config is None:
        raise ValueError(
            "WeightConverter(rules=None) needs `config` to derive the "
            "scanned-to-unrolled layer plan (num_decoder_layers, "
            "inhomogeneous_layer_cycle_interval, param_scan_axis)."
        )
      self._direct = MaxTextToMaxTextConverter(
          config=config,
          tp=self.tp,
          kv_tp_size=self.kv_tp_size,
          moe_mlp_tp_size=self.moe_mlp_tp_size,
          moe_fused_layout=(moe_fused_layout or MoEFusedLayout.PER_SHARD_INTERLEAVE),
          allow_unused_source_keys=allow_unused_source_keys,
          debug=debug,
          prefuse_moe_weights=prefuse_moe_weights,
          target_dtype=target_dtype,
      )
      logging.info("WeightConverter: direct MaxText-to-MaxText mode (debug=%s).", debug)
    else:
      if self.rules is None and config is not None:
        keys = [
            config.model_name,
            config.decoder_block.value if hasattr(config.decoder_block, "value") else config.decoder_block,
        ]
        for candidate_key in keys:
          if (
              candidate_key
              and candidate_key in MODEL_TO_CONVERSION_RULES
              and MODEL_TO_CONVERSION_RULES[candidate_key] is not None
          ):
            self.rules = MODEL_TO_CONVERSION_RULES[candidate_key]
            break
      logging.info("WeightConverter: torchax rule mode (tp=%d, %d rules).", self.tp, len(self.rules) if self.rules else 0)

  def convert(self, src_pytree: Any, target_state: Any = None) -> Any:
    """Converts source weights pytree into target format using rules or direct converter."""
    if self.rollout_backend == "maxtext" and self.rules is None:
      return self._direct.convert(src_pytree, target_state=target_state)
    if self.rules is None:
      raise ValueError("WeightConverter in torchax mode requires conversion rules.")

    flat_src = traverse_util.flatten_dict(_to_pure_dict(src_pytree), sep=".")
    gc.collect()
    result = {}
    unfired = []
    for rule in self.rules:
      tensors = [flat_src.pop(src_pat) for src_pat in rule.source_patterns if src_pat in flat_src]
      if not tensors:
        unfired.append(rule.target_pattern)
        continue
      out = tensors
      for op in rule.operations:
        out = op(out, tp=self.tp)
      if isinstance(out, list) and len(out) > 1 and "{}" in rule.target_pattern:
        for i, tensor in enumerate(out):
          result[rule.target_pattern.format(i)] = tensor
      elif isinstance(out, list) and len(out) == 1:
        result[rule.target_pattern] = out[0]
      else:
        result[rule.target_pattern] = out
      del out, tensors
      gc.collect()
    if not result:
      raise ValueError(
          "No conversion rule matched the trainer state; the rollout would "
          f"keep its dummy weights. Rule targets: {unfired}"
      )
    if unfired:
      logging.info("Conversion rules that did not fire: %s", unfired)
    return _rekey_to_target(result, target_state)

  def convert_streaming(
      self, src_pytree: Any, target_state: Any = None, *, groups_per_piece: int = 1
  ) -> Iterator[Dict[str, Any]]:
    """Yields converted weight pieces incrementally in direct MaxText-to-MaxText mode."""
    if self.rollout_backend == "maxtext" and self.rules is None:
      return self._direct.convert_streaming(src_pytree, target_state=target_state, groups_per_piece=groups_per_piece)
    raise NotImplementedError(
        "convert_streaming is only supported in direct MaxText-to-MaxText mode (rollout_backend='maxtext' and rules=None)."
    )


_UNROLLED_LAYER_TOKEN = re.compile(r"^layers?_(\d+)$")
_SUPPORTED_TARGET_FREE_DTYPES = frozenset({jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16), jnp.dtype(jnp.float32)})


def _cast_target_free_floating_leaf(val: Any, target_dtype: Any, path: str) -> Any:
  """Casts floating-point leaves to `target_dtype`, leaving non-floating leaves untouched."""
  if target_dtype is None or not hasattr(val, "dtype"):
    return val
  if not jnp.issubdtype(val.dtype, jnp.floating):
    return val
  return _apply_dtype_cast(val, target_dtype, path)


class MoEFusedLayout:
  """Layouts supported for pre-fused MoE `wi` kernels."""

  PER_SHARD_INTERLEAVE = "per_shard_interleave"
  CONCAT = "concat"


@dataclasses.dataclass(frozen=True)
class _PlanEntry:
  target_key: Tuple[Any, ...]
  source_keys: Tuple[Tuple[Any, ...], ...]
  slice_index: Optional[int]
  op: str
  local_index: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class _PlanGroup:
  source_keys: Tuple[Tuple[Any, ...], ...]
  op: str
  targets: Tuple[Tuple[Optional[int], Tuple[Any, ...]], ...]
  source_path: str
  local_index: Optional[int] = None


def _group_plan(plan: List[_PlanEntry]) -> List[_PlanGroup]:
  grouped: Dict[Tuple[Any, ...], List[_PlanEntry]] = {}
  order: List[Tuple[Any, ...]] = []
  for entry in plan:
    key = (entry.source_keys, entry.op, entry.local_index)
    if key not in grouped:
      grouped[key] = []
      order.append(key)
    grouped[key].append(entry)
  groups: List[_PlanGroup] = []
  for key in order:
    entries = grouped[key]
    source_keys, op, local_index = key
    groups.append(
        _PlanGroup(
            source_keys=source_keys,
            op=op,
            local_index=local_index,
            targets=tuple(
                (entry.slice_index, entry.target_key)
                for entry in sorted(entries, key=lambda entry: -1 if entry.slice_index is None else entry.slice_index)
            ),
            source_path=".".join(map(str, source_keys[0])),
        )
    )
  return groups


def _last_group_reading(groups: List[_PlanGroup]) -> Dict[Tuple[Any, ...], int]:
  last: Dict[Tuple[Any, ...], int] = {}
  for idx, group in enumerate(groups):
    for key in group.source_keys:
      last[key] = idx
  return last


class ConversionPlanError(ValueError):
  """Raised when the source and target trees cannot be fully reconciled."""


def _find_unrolled_layer(key_tuple: Tuple[Any, ...]) -> Tuple[int, int, int]:
  for i, part in enumerate(key_tuple):
    if isinstance(part, str):
      match = _UNROLLED_LAYER_TOKEN.match(part)
      if match:
        return int(match.group(1)), i, i + 1
      if part.isdigit() and i > 0 and key_tuple[i - 1] in ("layers", "layer"):
        return int(part), i - 1, i + 1
    elif isinstance(part, int) and i > 0 and key_tuple[i - 1] in ("layers", "layer"):
      return part, i - 1, i + 1
  return -1, -1, -1


def _summarize_tree_placement(flat: Mapping[Tuple[Any, ...], Any]) -> str:
  ids = set()
  for value in flat.values():
    ids.update(_device_ids(value))
  if not ids:
    return "[unknown]"
  return f"[{min(ids)}..{max(ids)}]({len(ids)})"


class MaxTextToMaxTextConverter:
  """Structural converter for the direct MaxText-to-MaxText sync path."""

  _plan_entry_type = _PlanEntry
  _plan_error_type = ConversionPlanError
  _find_unrolled_layer = staticmethod(_find_unrolled_layer)

  def __init__(
      self,
      config: Any,
      tp: int = 1,
      moe_fused_layout: str = MoEFusedLayout.PER_SHARD_INTERLEAVE,
      allow_unused_source_keys: Tuple[str, ...] = (),
      debug: bool = False,
      prefuse_moe_weights: Optional[bool] = None,
      target_dtype: Optional[Any] = None,
      kv_tp_size: Optional[int] = None,
      moe_mlp_tp_size: Optional[int] = None,
  ):
    self.config = config
    self.tp = resolve_rollout_tp(config, tp)
    self.kv_tp_size = kv_tp_size if kv_tp_size is not None else (getattr(config, "kv_tp_size", None) or self.tp)
    self.moe_mlp_tp_size = (
        moe_mlp_tp_size if moe_mlp_tp_size is not None else (getattr(config, "moe_mlp_tp_size", None) or self.tp)
    )
    self.moe_fused_layout = moe_fused_layout
    self.allow_unused_source_keys = allow_unused_source_keys
    self.debug = debug
    if prefuse_moe_weights is not None:
      self.prefuse_moe_weights = prefuse_moe_weights
    else:
      self.prefuse_moe_weights = getattr(config, "prefuse_moe_weights", False)
    self.padded_base_moe_mlp_dim = getattr(config, "padded_base_moe_mlp_dim", None)
    cfg_rollout_dtype = getattr(config, "rollout_weight_dtype", None) or None
    if target_dtype is not None:
      self.target_dtype = target_dtype
    elif cfg_rollout_dtype is not None:
      self.target_dtype = cfg_rollout_dtype
    else:
      self.target_dtype = getattr(config, "weight_dtype", None)
    self.base_num_kv_heads = getattr(config, "base_num_kv_heads", None) or getattr(config, "num_kv_heads", None)
    self.kv_replication = 1
    if self.base_num_kv_heads is not None and self.kv_tp_size > self.base_num_kv_heads:
      if self.kv_tp_size % self.base_num_kv_heads != 0:
        raise ValueError(
            f"kv_tp_size ({self.kv_tp_size}) must be divisible by base_num_kv_heads ({self.base_num_kv_heads})."
        )
      self.kv_replication = self.kv_tp_size // self.base_num_kv_heads
    self.cycle = int(getattr(config, "inhomogeneous_layer_cycle_interval", 1) or 1)
    self.num_decoder_layers = int(config.num_decoder_layers)
    if self.num_decoder_layers % self.cycle:
      raise ConversionPlanError(
          f"num_decoder_layers ({self.num_decoder_layers}) is not a multiple of "
          f"inhomogeneous_layer_cycle_interval ({self.cycle}); the scanned "
          "block structure cannot be unrolled unambiguously."
      )
    self.num_blocks = self.num_decoder_layers // self.cycle
    self.scan_axis = int(getattr(config, "param_scan_axis", 1))
    self._plan: Optional[List[_PlanEntry]] = None
    self._groups: Optional[List[_PlanGroup]] = None
    logging.info(
        "MaxTextToMaxTextConverter: %d layers, cycle=%d, %d scanned blocks, "
        "scan_axis=%d, moe_fused_layout=%s, prefuse_moe=%s, padded_moe_dim=%s, "
        "kv_tp_size=%d, moe_mlp_tp_size=%d, kv_replication=%d",
        self.num_decoder_layers, self.cycle, self.num_blocks, self.scan_axis, self.moe_fused_layout,
        self.prefuse_moe_weights, self.padded_base_moe_mlp_dim, self.kv_tp_size, self.moe_mlp_tp_size,
        self.kv_replication,
    )

  def _resolve_target_dtype(self):
    return normalize_dtype(self.target_dtype)

  def _execute(self, group, src_flat, tgt_flat):
    return execute_group(
        group, src_flat, tgt_flat, scan_axis=self.scan_axis, cycle=self.cycle, num_blocks=self.num_blocks,
        moe_fused_layout=self.moe_fused_layout, tp=self.tp, kv_tp_size=self.kv_tp_size,
        moe_mlp_tp_size=self.moe_mlp_tp_size, padded_base_moe_mlp_dim=self.padded_base_moe_mlp_dim,
        kv_replication=self.kv_replication, base_num_kv_heads=self.base_num_kv_heads,
        target_dtype=self._resolve_target_dtype() if tgt_flat is None else self.target_dtype,
        error_type=ConversionPlanError,
    )

  def convert(self, src_pytree: Any, target_state: Any = None) -> Dict[str, Any]:
    """Returns a nested dict of rollout weights, keyed by target paths."""
    if target_state is None:
      flat_result = {}
      for piece in self.convert_streaming(src_pytree, target_state=None):
        flat_result.update(traverse_util.flatten_dict(piece))
      return traverse_util.unflatten_dict(flat_result)
    src_flat = traverse_util.flatten_dict(_to_pure_dict(src_pytree))
    src_flat, _ = _strip_root(src_flat, "base")
    skip_paths = _non_param_paths(target_state)
    tgt_flat = traverse_util.flatten_dict(_to_pure_dict(target_state))
    tgt_flat, tgt_root = _strip_root(tgt_flat, "model")
    if tgt_root:
      depth = len(tgt_root)
      skip_paths = frozenset(path[depth:] for path in skip_paths if path[:depth] == tgt_root)
    if self._plan is None:
      self._plan = build_plan(
          src_flat, tgt_flat, skip_paths, self.cycle, self.num_decoder_layers, self.allow_unused_source_keys,
          self._plan_entry_type, self._plan_error_type, self._find_unrolled_layer,
      )
      self._groups = _group_plan(self._plan)
    if self.debug:
      src_span = _summarize_tree_placement(src_flat)
      tgt_span = _summarize_tree_placement(tgt_flat)
      logging.info(
          "weight_sync_debug: converting %d entries in %d source groups | "
          "source tree on devices %s | target tree on devices %s",
          len(self._plan), len(self._groups), src_span, tgt_span,
      )
    result: Dict[Tuple[Any, ...], Any] = {}
    last_reader = _last_group_reading(self._groups)
    for group_idx, group in enumerate(self._groups):
      if self.debug:
        for key in group.source_keys:
          logging.info(
              "weight_sync_debug: op=%s source=%s (%d targets) | src %s | tgt %s",
              group.op, ".".join(map(str, key)), len(group.targets), _sharding_summary(src_flat[key]),
              _sharding_summary(tgt_flat[group.targets[0][1]]),
          )
        try:
          outs = self._execute(group, src_flat, tgt_flat)
          jax.block_until_ready([out for _, out in outs])
        except BaseException:
          logging.exception(
              "weight_sync_debug: FAILED on op=%s sources=%s -> %d "
              "targets (first: %s). Output of the preceding groups "
              "completed successfully.",
              group.op, ", ".join(".".join(map(str, key)) for key in group.source_keys),
              len(group.targets), ".".join(map(str, group.targets[0][1])),
          )
          raise
        logging.info(
            "weight_sync_debug: ok op=%s source=%s -> %d outputs, first %s",
            group.op, group.source_path, len(outs), _sharding_summary(outs[0][1]),
        )
      else:
        outs = self._execute(group, src_flat, tgt_flat)
      for key in group.source_keys:
        if last_reader[key] == group_idx:
          src_flat.pop(key, None)
      for tgt_key, out in outs:
        tgt_val = tgt_flat[tgt_key]
        if out.shape != tgt_val.shape:
          raise ConversionPlanError(
              f"Shape mismatch after conversion for "
              f"{'.'.join(map(str, tgt_key))}: produced {out.shape}, "
              f"rollout expects {tgt_val.shape}."
          )
        result[tgt_root + tgt_key] = out
      del outs
    del src_flat, tgt_flat
    gc.collect()
    nested = traverse_util.unflatten_dict(result)
    del result
    return jax.tree_util.tree_map(nnx.Param, nested)

  def convert_streaming(
      self, src_pytree: Any, target_state: Any = None, *, groups_per_piece: int = 1
  ) -> Iterator[Dict[str, Any]]:
    """Yields converted rollout weight pieces incrementally for target-free conversion."""
    if target_state is not None:
      raise NotImplementedError("convert_streaming only supports target-free conversion (target_state=None).")
    resolved_dtype = self._resolve_target_dtype()
    if resolved_dtype is not None and jnp.dtype(resolved_dtype) not in _SUPPORTED_TARGET_FREE_DTYPES:
      raise ConversionPlanError(
          f"Unsupported target_dtype={resolved_dtype!r} for target-free weight conversion; "
          "only unquantized floating-point dtypes (bfloat16, float16, float32) are supported."
      )
    src_flat = traverse_util.flatten_dict(_to_pure_dict(src_pytree))
    src_flat, src_root = _strip_root(src_flat, "base")
    if self._plan is None:
      self._plan = build_target_free_plan(
          src_flat, self.cycle, self.num_blocks, self.prefuse_moe_weights, self._plan_entry_type,
          self._plan_error_type,
      )
      self._groups = _group_plan(self._plan)
    groups_per_piece = max(1, groups_per_piece)
    last_reader = _last_group_reading(self._groups)
    for i in range(0, len(self._groups), groups_per_piece):
      piece_groups = self._groups[i : i + groups_per_piece]
      piece_result: Dict[Tuple[Any, ...], Any] = {}
      for group_idx, group in enumerate(piece_groups, start=i):
        outs = self._execute(group, src_flat, None)
        for key in group.source_keys:
          if last_reader[key] == group_idx:
            src_flat.pop(key, None)
        for tgt_key, out in outs:
          piece_result[src_root + tgt_key] = out
        del outs
      nested = traverse_util.unflatten_dict(piece_result)
      del piece_result
      yield jax.tree_util.tree_map(nnx.Param, nested)
    del src_flat
