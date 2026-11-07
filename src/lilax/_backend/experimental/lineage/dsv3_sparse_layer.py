"""Single DeepSeekV3 sparse layer."""

from collections.abc import Callable, Mapping
import dataclasses
import functools
import math
from typing import Any, Protocol, overload

import jax
from jax.experimental import layout as jax_layout
from jax.experimental.overlap import program_order
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

from lilax._backend.experimental.lineage import dsv3_expert_shuffle
from lilax._backend.experimental.lineage import dsv3_experts
from lilax._backend.experimental.lineage import dsv3_mla
from lilax._backend.experimental.lineage import dsv3_router
from lilax._backend.experimental.lineage import dsv3_types
from lilax._backend.experimental.lineage import microbatching
from lilax._backend.experimental.lineage import quantization
from lilax._backend.experimental.lineage.scan import weight_scan_prologue_epilogue
from lilax._backend.experimental.lineage.sharding import axis_collectives
from lilax._backend.experimental.lineage.sharding import partition_specs


class NormFn(Protocol):
  def __call__(self, x: jt.Num[jax.Array, "*input_dims"], scale: jt.Num[jax.Array, "..."] | None = None) -> jt.Num[jax.Array, "*input_dims"]:
    ...


class RopeFn(Protocol):
  def __call__(
      self, x: jt.Num[jax.Array, "*BT N D"],
      freqs: tuple[jt.Num[jax.Array, "*BT 1 D"], jt.Num[jax.Array, "*BT 1 D"]],
  ) -> jt.Num[jax.Array, "*BT N D"]:
    ...


GmmFn = dsv3_experts.GmmFn


def _get_local_dim_size(dim_size: int, dim_pspec: Any, mesh: jax.sharding.Mesh) -> int:
  if dim_pspec is None:
    return dim_size
  if isinstance(dim_pspec, str):
    return dim_size // mesh.shape[dim_pspec]
  if isinstance(dim_pspec, tuple):
    return dim_size // math.prod(mesh.shape[name] for name in dim_pspec)
  return dim_size


@overload
def _offload_to_host(x: jax.Array) -> jax.Array: ...


@overload
def _offload_to_host(x: None) -> None: ...


def _offload_to_host(x: jax.Array | None) -> jax.Array | None:
  if x is None:
    return None
  return jax.device_put(x, jax.typeof(x).sharding.with_memory_kind("pinned_host"))


@overload
def _load_to_device(x: jax.Array) -> jax.Array: ...


@overload
def _load_to_device(x: None) -> None: ...


def _load_to_device(x: jax.Array | None) -> jax.Array | None:
  if x is None:
    return None
  return jax.device_put(x, jax.typeof(x).sharding.with_memory_kind("device"))


_MLA_OFFLOAD_FIELDS: tuple[str, ...] = ()
_MOE_OFFLOAD_FIELDS: tuple[str, ...] = ()


def _offload_mla(mla: dsv3_types.DSv3MLAResiduals) -> dsv3_types.DSv3MLAResiduals:
  return mla._replace(**{f: _offload_to_host(getattr(mla, f)) for f in _MLA_OFFLOAD_FIELDS})


def _load_mla(mla: dsv3_types.DSv3MLAResiduals) -> dsv3_types.DSv3MLAResiduals:
  return mla._replace(**{f: _load_to_device(getattr(mla, f)) for f in _MLA_OFFLOAD_FIELDS})


def _offload_moe(moe: dsv3_types.DSv3MoEResiduals) -> dsv3_types.DSv3MoEResiduals:
  return moe._replace(**{f: _offload_to_host(getattr(moe, f)) for f in _MOE_OFFLOAD_FIELDS})


def _load_moe(moe: dsv3_types.DSv3MoEResiduals) -> dsv3_types.DSv3MoEResiduals:
  return moe._replace(**{f: _load_to_device(getattr(moe, f)) for f in _MOE_OFFLOAD_FIELDS})


def _offload_mb(mb: dsv3_types.DSv3MicrobatchResiduals) -> dsv3_types.DSv3MicrobatchResiduals:
  mla = _offload_mla(mb.mla) if mb.mla is not None else None
  moe = _offload_moe(mb.moe) if mb.moe is not None else None
  return dsv3_types.DSv3MicrobatchResiduals(mla=mla, moe=moe)


def _load_mb(mb: dsv3_types.DSv3MicrobatchResiduals) -> dsv3_types.DSv3MicrobatchResiduals:
  mla = _load_mla(mb.mla) if mb.mla is not None else None
  moe = _load_moe(mb.moe) if mb.moe is not None else None
  return dsv3_types.DSv3MicrobatchResiduals(mla=mla, moe=moe)


def offload_residuals(res: dsv3_types.DSv3LayerResiduals) -> dsv3_types.DSv3LayerResiduals:
  return dsv3_types.DSv3LayerResiduals(
      mb0=_offload_mb(res.mb0), mb1=_offload_mb(res.mb1) if res.mb1 is not None else None,
      bank=res.bank, bank_offset=res.bank_offset,
  )


def load_residuals(res: dsv3_types.DSv3LayerResiduals) -> dsv3_types.DSv3LayerResiduals:
  return dsv3_types.DSv3LayerResiduals(
      mb0=_load_mb(res.mb0), mb1=_load_mb(res.mb1) if res.mb1 is not None else None,
      bank=res.bank, bank_offset=res.bank_offset,
  )


def _merge_router_aux(
    aux0: dsv3_types.DSv3RouterAux[Any], aux1: dsv3_types.DSv3RouterAux[Any], *, mesh: jax.sharding.Mesh,
) -> dsv3_types.DSv3RouterAux[Any]:
  group_sizes = None
  if aux0.group_sizes is not None and aux1.group_sizes is not None:
    group_sizes = aux0.group_sizes + aux1.group_sizes
  elif aux0.group_sizes is not None:
    group_sizes = aux0.group_sizes
  elif aux1.group_sizes is not None:
    group_sizes = aux1.group_sizes
  return dsv3_types.DSv3RouterAux(
      group_sizes=group_sizes, selected_experts=microbatching.merge_microbatches(aux0.selected_experts, aux1.selected_experts, mesh=mesh),
      logits=microbatching.merge_microbatches(aux0.logits, aux1.logits, mesh=mesh),
  )


@jax.named_call
def ubatch_wait(*args: Any) -> Any:
  if len(args) == 1:
    return args[0]
  return args


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def ubatch_mla(
    x: jt.Num[jax.Array, "B T D"], pre_attn_norm_scale: jt.Num[jax.Array, "..."] | None,
    w_mla: dsv3_types.DSv3MLAWeightsPytree,
    yarn_freqs: tuple[jt.Num[jax.Array, "B T 1 R"], jt.Num[jax.Array, "B T 1 R"]], splash_kernel: Any,
    *, qk_head_dim: int, rope_head_dim: int, num_query_heads: int, mscale: float, kv_lora_rank: int,
    max_position_embeddings: int, original_max_position_embeddings: int, rope_factor: int, mesh: jax.sharding.Mesh,
    norm_fn: NormFn, rope_fn: RopeFn, axis_mapping: Mapping[str, str | tuple[str, ...]],
    segment_ids: jt.Num[jax.Array, "B T"] | None = None, quant_rule: quantization.GmmQuantRule | None = None,
) -> tuple[tuple[jt.Num[jax.Array, "B T D"], jt.Num[jax.Array, "B T D"]], dsv3_types.DSv3MLAResiduals]:
  orig_x = x
  with jax.named_scope("pre_attn_norm"):
    norm_x = norm_fn(x, pre_attn_norm_scale)
  out, mla_residuals = dsv3_mla.dsv3_mla_fwd(
      norm_x, yarn_freqs, splash_kernel, w_mla, segment_ids=segment_ids, kv_lora_rank=kv_lora_rank,
      qk_head_dim=qk_head_dim, rope_head_dim=rope_head_dim, num_query_heads=num_query_heads,
      max_position_embeddings=max_position_embeddings, original_max_position_embeddings=original_max_position_embeddings,
      rope_factor=rope_factor, mscale=mscale, norm_fn=norm_fn, rope_fn=rope_fn, mesh=mesh,
      axis_mapping=axis_mapping, quant_rule=quant_rule,
  )
  res = dsv3_types.DSv3MLAResiduals(
      orig_x=orig_x, q_down=mla_residuals[0], kv_down=mla_residuals[1], context=mla_residuals[2], splash_out=mla_residuals[3],
  )
  return (out, orig_x), res


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def ubatch_norm_and_metadata(
    out_proj_out: jt.Num[jax.Array, "B T D"], orig_x: jt.Num[jax.Array, "B T D"],
    post_attn_norm_scale: jt.Num[jax.Array, "..."] | None, w_router: dsv3_types.DSv3MoERouterWeightsPytree,
    *, norm_fn: NormFn, num_experts: int, num_experts_per_tok: int, routed_scaling_factor: float,
    n_routing_groups: int, topk_routing_group: int, topk_in_group: int, expert_axis_name: str = "expert",
    axis_mapping: Mapping[str, str | tuple[str, ...]], mesh: jax.sharding.Mesh, capacity_factor: float,
    router_dtype: jax.typing.DTypeLike | None = None, expert_permutations: jt.Int[jax.Array, "E"] | None = None,
) -> tuple[Any, Any]:
  mla_out = out_proj_out + orig_x
  with jax.named_scope("post_attn_norm"):
    moe_in = norm_fn(mla_out, post_attn_norm_scale)
  router_metadata, _, aux = dsv3_router.dsv3_routing_metadata(
      moe_in, w_router, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok,
      routed_scaling_factor=routed_scaling_factor, n_routing_groups=n_routing_groups,
      topk_routing_group=topk_routing_group, topk_in_group=topk_in_group, expert_axis_name=expert_axis_name,
      axis_mapping=axis_mapping, mesh=mesh, capacity_factor=capacity_factor,
      router_dtype=router_dtype, expert_permutations=expert_permutations,
  )
  router_metadata = dataclasses.replace(router_metadata, sorted_coeffs=jnp.reshape(router_metadata.sorted_coeffs, (-1,)))
  return ((router_metadata, moe_in, mla_out), aux), (out_proj_out, orig_x)


def _quantize_act(x: jax.Array, quant_rule: quantization.GmmQuantRule | None) -> jax.Array:
  if quant_rule is None:
    return x
  return quantization.static_quantize(x, quant_rule.act_qtype, quant_rule.act_scale)


def _quantize_routed_weights(
    w_routed: dsv3_types.DSv3MoERoutedExpertWeightsPytree, quant_rule: quantization.GmmQuantRule,
) -> dsv3_types.DSv3MoERoutedExpertWeightsPytree:
  return jax.tree.map(
      lambda w: w if w.dtype == quant_rule.weight_qtype
      else quantization.static_quantize(w, quant_rule.weight_qtype, quant_rule.weight_scale), w_routed,
  )


def quantize_weights(
    w: dsv3_types.DSv3SparseLayerWeightsPytree, *, quant: quantization.QuantConfig,
) -> dsv3_types.DSv3SparseLayerWeightsPytree:
  if quant.routed_experts is not None:
    w = dataclasses.replace(w, moe=dataclasses.replace(w.moe, routed=_quantize_routed_weights(w.moe.routed, quant.routed_experts)))
  if quant.mla is not None:
    w = dataclasses.replace(w, mla=dsv3_mla.quantize_mla_weights(w.mla, quant.mla))
  return w


def _dispatch_bwd(
    moe_in: jt.Num[jax.Array, "B T D"], grad_out: jt.Num[jax.Array, "B T D"], *, expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]], mesh: jax.sharding.Mesh, gmm_fn: GmmFn,
    quant_rule: quantization.GmmQuantRule | None,
) -> tuple[jt.Num[jax.Array, "NBT D0 D1"], jt.Num[jax.Array, "NBT D0 D1"], jt.Num[jax.Array, "num_token_shards 1"] | None]:
  moe_in_q = _quantize_act(moe_in, quant_rule)
  if quant_rule is None or not hasattr(gmm_fn, "linear_bwd_gather"):
    x_ag, grad_y_ag = dsv3_router.dsv3_aggregated_dispatches(
        moe_in_q, grad_out, expert_axis_name=expert_axis_name, axis_mapping=axis_mapping, mesh=mesh,
    )
    return x_ag, grad_y_ag, None
  physical_expert_axis = axis_mapping.get(expert_axis_name, expert_axis_name)
  token_axis = dsv3_router.dsv3_get_token_axis(jax.typeof(moe_in_q).sharding.spec, num_model_dims=1) or (
      dsv3_router.dsv3_get_token_axis(jax.typeof(grad_out).sharding.spec, num_model_dims=1)
  )
  token_spec_3d = partition_specs.physical_pspec(jax.sharding.PartitionSpec(token_axis, None, None), axis_mapping)
  scale_spec = partition_specs.physical_pspec(jax.sharding.PartitionSpec(token_axis, None), axis_mapping)

  def _impl(m_in, g_out):
    local_amax = quantization.ragged_absmax(g_out)
    amax = jax.lax.pmax(local_amax, axis_name=physical_expert_axis)
    scale = quantization.absmax_to_scale(amax, quant_rule.bwd_qtype)
    g_out_q = quantization.quantize_with_scale(g_out, scale, quant_rule.bwd_qtype)
    x_ag, grad_y_ag = dsv3_router.dsv3_aggregated_dispatches_impl(
        m_in, g_out_q, expert_axis_name=expert_axis_name, axis_mapping=axis_mapping,
    )
    return x_ag, grad_y_ag, scale
  return jax.shard_map(_impl, mesh=mesh, out_specs=(token_spec_3d, token_spec_3d, scale_spec))(moe_in_q, grad_out)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def ubatch_dispatch(
    router_metadata: dsv3_router.RouterMetadata[jax.Array], moe_in: jt.Num[jax.Array, "B T D"],
    mla_out: jt.Num[jax.Array, "B T D"], *, expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]], mesh: jax.sharding.Mesh,
    quant_rule: quantization.GmmQuantRule | None = None,
) -> tuple[Any, tuple[()]]:
  x_ag = dsv3_router.dsv3_dispatch(
      _quantize_act(moe_in, quant_rule), expert_axis_name=expert_axis_name, axis_mapping=axis_mapping, mesh=mesh,
  )
  return (x_ag, router_metadata, moe_in, mla_out), ()


def constrain_activation_bank(bank: jt.Num[jax.Array, "*bank_dims gate_dim"]) -> jt.Num[jax.Array, "*bank_dims gate_dim"]:
  tiling = ((8, 128), (2, 1)) if jnp.dtype(bank.dtype).itemsize == 2 else ((8, 128),)
  layout = jax_layout.Layout(major_to_minor=tuple(range(bank.ndim)))
  return jax_layout.with_layout_constraint(bank, layout)


@jt.jaxtyped(typechecker=typeguard.typechecked)
def init_activation_bank(
    mb0: jt.Num[jax.Array, "B T D"], w: dsv3_types.DSv3SparseLayerWeightsPytree,
    *, num_layers: int, num_experts_per_tok: int, ragged_buffer_factor: float, capacity_factor: float,
    expert_axis_name: str, axis_mapping: Mapping[str, str | tuple[str, ...]], mesh: jax.sharding.Mesh,
) -> tuple[Any, Any]:
  x_pspec = jax.typeof(mb0).sharding.spec
  local_batch_size = _get_local_dim_size(mb0.shape[0], x_pspec[0], mesh)
  local_seq_length = _get_local_dim_size(mb0.shape[1], x_pspec[1], mesh)
  perfectly_balanced_tokens_per_layer_per_device = local_batch_size * local_seq_length * num_experts_per_tok
  bank_capacity_per_device = int(ragged_buffer_factor * num_layers * perfectly_balanced_tokens_per_layer_per_device)
  expert_axes = axis_mapping.get(expert_axis_name, expert_axis_name)
  if isinstance(expert_axes, str):
    expert_axes = (expert_axes,)
  capacity = dsv3_router.chunk_capacity(
      perfectly_balanced_tokens_per_layer_per_device, math.prod(mesh.shape[a] for a in expert_axes), capacity_factor,
  )
  bank_capacity_per_device = dsv3_experts.bank_aligned(bank_capacity_per_device) + dsv3_experts.bank_aligned(capacity)
  assert w.moe.routed.gate is not None
  gate_dim = w.moe.routed.gate.shape[-1]
  token_axis = dsv3_router.dsv3_get_token_axis(x_pspec, num_model_dims=1)
  bank_pspec = partition_specs.physical_pspec(jax.sharding.PartitionSpec(token_axis, None, None), axis_mapping)
  offset_pspec = partition_specs.physical_pspec(jax.sharding.PartitionSpec(token_axis), axis_mapping)

  def _init_local(_):
    def _init():
      bank = jax.lax.empty((bank_capacity_per_device, gate_dim), mb0.dtype)
      return _bank_to_carry(bank), jnp.zeros((1,), dtype=jnp.int32)
    return _init(), _init()
  (bank_0, offset_0), (bank_1, offset_1) = jax.shard_map(
      _init_local, mesh=mesh, in_specs=x_pspec, out_specs=((bank_pspec, offset_pspec), (bank_pspec, offset_pspec)),
  )(mb0)
  return (constrain_activation_bank(bank_0), offset_0), (constrain_activation_bank(bank_1), offset_1)


def _chunk_rows(x: jt.Num[jax.Array, "rows ..."], start: jax.Array | int, capacity: int) -> jt.Num[jax.Array, "capacity ..."]:
  x = jnp.pad(x, ((0, capacity),) + ((0, 0),) * (x.ndim - 1))
  return jax.lax.dynamic_slice_in_dim(x, start, capacity)


def _bank_to_carry(bank: jt.Num[jax.Array, "bank_tokens gate_dim"]) -> jt.Num[jax.Array, "bank_tiles 8 gate_dim"]:
  return bank.reshape(bank.shape[0] // 8, 8, bank.shape[1])


def _bank_from_carry(bank: jt.Num[jax.Array, "bank_tiles 8 gate_dim"]) -> jt.Num[jax.Array, "bank_tokens gate_dim"]:
  return constrain_activation_bank(bank.reshape(-1, bank.shape[-1]))


def routed_w_to_tiles(w: dsv3_types.DSv3MoERoutedExpertWeightsPytree) -> dsv3_types.DSv3MoERoutedExpertWeightsPytree:
  return jax.tree.map(lambda a: a.reshape(*a.shape[:-2], a.shape[-2] // 8, 8, a.shape[-1]), w)


def routed_w_from_tiles(w: dsv3_types.DSv3MoERoutedExpertWeightsPytree) -> dsv3_types.DSv3MoERoutedExpertWeightsPytree:
  return jax.tree.map(lambda a: a.reshape(*a.shape[:-3], a.shape[-3] * 8, a.shape[-1]), w)


def _routed_experts_chunked_impl(
    x_ag: jt.Num[jax.Array, "NBT D0 D1"], metadata: dsv3_router.RouterMetadata[jax.Array],
    w_routed: dsv3_types.DSv3MoERoutedExpertWeightsPytree, bank: jt.Num[jax.Array, "bank_tiles 8 gate_dim"],
    bank_offset: jt.Num[jax.Array, "1"], *, num_experts_per_tok: int, gmm_fn: GmmFn,
    quant_rule: quantization.GmmQuantRule | None = None,
) -> tuple[jt.Num[jax.Array, "NBT D0 D1"], jt.Num[jax.Array, "bank_tiles 8 gate_dim"], jt.Num[jax.Array, "1"]]:
  capacity = metadata.capacity
  num_local_tokens = jnp.sum(metadata.group_sizes, dtype=jnp.int32)
  gather_in_gate = (
      quant_rule is not None and getattr(gmm_fn, "supports_fused_glu", False)
      and hasattr(gmm_fn, "gather_gate") and hasattr(gmm_fn, "linear_combine")
  )
  token_indices = None
  if gather_in_gate:
    token_indices = metadata.sort_indices // num_experts_per_tok

  def _chunk(start, reduce_metadata, bank, bank_offset, y_ag_acc=None):
    num_tokens = dsv3_router.chunk_num_tokens(num_local_tokens, start, capacity)
    group_sizes = dsv3_router.chunk_group_sizes(metadata.group_sizes, start, num_tokens)
    sorted_coeffs = _chunk_rows(metadata.sorted_coeffs, start, capacity)
    if gather_in_gate:
      assert quant_rule is not None and token_indices is not None
      y_ag, bank = dsv3_experts.dsv3_routed_experts_gather_impl(
          x_ag, token_indices, start, routed_w_from_tiles(w_routed), group_sizes, sorted_coeffs,
          bank, bank_offset[0], reduce_metadata, y_ag_acc, gmm_fn=gmm_fn, quant_rule=quant_rule,
      )
    else:
      routed = dsv3_router.permute_chunk(
          x_ag, metadata.sort_indices, start, num_tokens, capacity=capacity, num_experts_per_tok=num_experts_per_tok,
      )
      routed_out, bank = dsv3_experts.dsv3_routed_experts_impl(
          routed, routed_w_from_tiles(w_routed), group_sizes, sorted_coeffs, bank, bank_offset[0],
          gmm_fn=gmm_fn, quant_rule=quant_rule,
      )
      y_ag = dsv3_router.unpermute_chunk(
          routed_out, reduce_metadata, num_out_tokens=x_ag.shape[0], num_experts_per_tok=num_experts_per_tok,
          out=y_ag_acc, zero_initialized=(y_ag_acc is None),
      )
    bank = constrain_activation_bank(bank)
    return y_ag, bank, bank_offset + dsv3_experts.bank_aligned(num_tokens)

  y_ag, bank, bank_offset = _chunk(0, metadata.reduce_metadata, _bank_from_carry(bank), bank_offset)

  def _cond(carry):
    _, remaining, *_ = carry
    return remaining > capacity

  def _body(carry):
    start, remaining, y_ag, bank, bank_offset = carry
    start = start + capacity
    remaining = remaining - capacity
    reduce_metadata = dsv3_router.chunk_reduce_metadata(
        metadata.sort_indices, start, dsv3_router.chunk_num_tokens(num_local_tokens, start, capacity),
        capacity=capacity, num_experts_per_tok=num_experts_per_tok,
    )
    y_ag, bank, bank_offset = _chunk(start, reduce_metadata, _bank_from_carry(bank), bank_offset, y_ag_acc=y_ag)
    return start, remaining, y_ag, _bank_to_carry(bank), bank_offset

  _, _, y_ag, bank, bank_offset = jax.lax.while_loop(
      _cond, _body, (jnp.int32(0), num_local_tokens, y_ag, _bank_to_carry(bank), bank_offset),
  )
  return y_ag, bank, bank_offset


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def ubatch_expert(
    x_ag: jt.Num[jax.Array, "NBT D0 D1"], router_metadata: dsv3_router.RouterMetadata[jax.Array],
    moe_in: jt.Num[jax.Array, "B T D"], mla_out: jt.Num[jax.Array, "B T D"],
    bank: jt.Num[jax.Array, "bank_tiles 8 gate_dim"], bank_offset: jt.Num[jax.Array, "num_token_shards"],
    w_routed: dsv3_types.DSv3MoERoutedExpertWeightsPytree, w_shared: dsv3_types.DSv3MoESharedExpertWeightsPytree | None = None,
    *, num_experts_per_tok: int = 8, gmm_fn: GmmFn, mesh: jax.sharding.Mesh,
    quant_rule: quantization.GmmQuantRule | None = None,
) -> tuple[Any, dsv3_router.RouterMetadata[jax.Array]]:
  x_ag_pspec = jax.typeof(x_ag).sharding.spec
  bank_pspec = jax.typeof(bank).sharding.spec
  offset_pspec = jax.typeof(bank_offset).sharding.spec
  y_ag, bank, bank_offset = jax.shard_map(
      functools.partial(_routed_experts_chunked_impl, num_experts_per_tok=num_experts_per_tok, gmm_fn=gmm_fn, quant_rule=quant_rule),
      mesh=mesh, out_specs=(x_ag_pspec, bank_pspec, offset_pspec),
  )(x_ag, router_metadata, w_routed, bank, bank_offset)
  shared_or_moe_in = moe_in if w_shared is None else dsv3_experts.dsv3_shared_expert(moe_in, w_shared)
  return ((y_ag, shared_or_moe_in, mla_out), (bank, bank_offset)), router_metadata


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def ubatch_combine(
    y_ag: jt.Num[jax.Array, "NBT D0 D1"], shared_or_moe_in: jt.Num[jax.Array, "B T D"],
    mla_out: jt.Num[jax.Array, "B T D"], *, w_shared: dsv3_types.DSv3MoESharedExpertWeightsPytree | None = None,
    local_seq_length: int, expert_axis_name: str, axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh, out_specs: jax.sharding.PartitionSpec,
    combine_qtype: Any | None = None, combine_per_hop_scaling: bool = False,
) -> tuple[Any, tuple[()]]:
  combined_x = dsv3_router.dsv3_combine(
      y_ag, local_seq_length=local_seq_length, expert_axis_name=expert_axis_name, axis_mapping=axis_mapping,
      mesh=mesh, out_specs=out_specs, combine_qtype=combine_qtype, combine_per_hop_scaling=combine_per_hop_scaling,
  )
  shared_x = shared_or_moe_in if w_shared is None else dsv3_experts.dsv3_shared_expert(shared_or_moe_in, w_shared)
  return (combined_x, shared_x, mla_out), ()


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def ubatch_residual_add(
    combined_x: jt.Num[jax.Array, "B T D"], shared_x: jt.Num[jax.Array, "B T D"], mla_out: jt.Num[jax.Array, "B T D"],
) -> tuple[tuple[jt.Num[jax.Array, "B T D"]], tuple[()]]:
  moe_out = combined_x + shared_x
  out = mla_out + moe_out
  return (out,), ()


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def ubatch_expert_bwd(
    x_ag: jt.Num[jax.Array, "NBT D0 D1"], grad_y_ag: jt.Num[jax.Array, "NBT D0 D1"],
    router_metadata: dsv3_router.RouterMetadata[jax.Array], bank: jt.Num[jax.Array, "bank_tiles 8 gate_dim"],
    bank_offset: jt.Num[jax.Array, "num_token_shards"], w_routed: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    grad_w_acc: dsv3_types.DSv3MoERoutedExpertWeightsPytree | None = None, *, num_experts_per_tok: int = 8,
    gmm_fn: GmmFn, mesh: jax.sharding.Mesh, quant_rule: quantization.GmmQuantRule | None = None,
    w_routed_q: dsv3_types.DSv3MoERoutedExpertWeightsPytree | None = None,
    grad_y_scale: jt.Num[jax.Array, "num_token_shards 1"] | None = None,
) -> tuple[Any, ...]:
  bank = constrain_activation_bank(bank)
  coeffs_pspec = jax.sharding.PartitionSpec(jax.typeof(router_metadata.sorted_coeffs).sharding.spec[0])
  w_grad_pspecs = jax.tree.map(lambda g: jax.typeof(g).sharding.spec.to_ct_spec(), w_routed)
  grad_x_ag, grad_coeffs, grad_w_routed, bank, bank_offset = jax.shard_map(
      functools.partial(_routed_experts_chunked_bwd_impl, num_experts_per_tok=num_experts_per_tok,
                        gmm_fn=gmm_fn, quant_rule=quant_rule),
      mesh=mesh, out_specs=(jax.typeof(x_ag).sharding.spec, coeffs_pspec, w_grad_pspecs,
                            jax.typeof(bank).sharding.spec, jax.typeof(bank_offset).sharding.spec),
  )(x_ag, grad_y_ag, router_metadata, w_routed, bank, bank_offset, grad_w_acc, w_routed_q, grad_y_scale)
  return grad_x_ag, grad_coeffs, grad_w_routed, constrain_activation_bank(bank), bank_offset


def _routed_experts_chunked_bwd_impl(
    x_ag: jt.Num[jax.Array, "NBT D0 D1"], grad_y_ag: jt.Num[jax.Array, "NBT D0 D1"],
    metadata: dsv3_router.RouterMetadata[jax.Array], w_routed: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    bank: jt.Num[jax.Array, "bank_tiles 8 gate_dim"], bank_offset: jt.Num[jax.Array, "1"],
    grad_w_acc: dsv3_types.DSv3MoERoutedExpertWeightsPytree | None,
    w_routed_q: dsv3_types.DSv3MoERoutedExpertWeightsPytree | None = None,
    grad_y_scale: jt.Num[jax.Array, "1 1"] | None = None, *, num_experts_per_tok: int,
    gmm_fn: GmmFn, quant_rule: quantization.GmmQuantRule | None = None,
) -> tuple[Any, ...]:
  capacity = metadata.capacity
  num_slots = metadata.sort_indices.shape[0]
  num_local_tokens = jnp.sum(metadata.group_sizes, dtype=jnp.int32)
  bank_offset = bank_offset - dsv3_experts.bank_aligned(num_local_tokens)
  if grad_w_acc is not None:
    grad_w_acc = routed_w_from_tiles(grad_w_acc)
  gather_in_linear_bwd = quant_rule is not None and hasattr(gmm_fn, "linear_bwd_gather")
  token_indices = None
  if gather_in_linear_bwd:
    assert quant_rule is not None
    token_indices = metadata.sort_indices // num_experts_per_tok
    if grad_y_scale is None:
      grad_y_ag, grad_y_scale = quantization.absmax_quantize(grad_y_ag, quant_rule.bwd_qtype)

  def _chunk(start, reduce_metadata, bank, grad_w_acc, grad_x_ag_acc=None):
    num_tokens = dsv3_router.chunk_num_tokens(num_local_tokens, start, capacity)
    permute = functools.partial(
        dsv3_router.permute_chunk, sort_indices=metadata.sort_indices, start=start, num_tokens=num_tokens,
        capacity=capacity, num_experts_per_tok=num_experts_per_tok,
    )
    group_sizes = dsv3_router.chunk_group_sizes(metadata.group_sizes, start, num_tokens)
    sorted_coeffs = _chunk_rows(metadata.sorted_coeffs, start, capacity)
    if gather_in_linear_bwd:
      assert quant_rule is not None and token_indices is not None
      assert w_routed_q is not None
      grad_x_sorted, grad_w_routed, grad_coeffs, bank = dsv3_experts.dsv3_routed_experts_gather_bwd_impl(
          grad_y_ag, token_indices, start, permute(x_ag), routed_w_from_tiles(w_routed), group_sizes, sorted_coeffs,
          bank, bank_offset[0] + start, grad_w_acc, gmm_fn=gmm_fn, quant_rule=quant_rule,
          w_routed_q=routed_w_from_tiles(w_routed_q), grad_y_scale=grad_y_scale,
      )
    else:
      grad_x_sorted, grad_w_routed, grad_coeffs, bank = dsv3_experts.dsv3_routed_experts_bwd_impl(
          permute(grad_y_ag), permute(x_ag), routed_w_from_tiles(w_routed), group_sizes, sorted_coeffs,
          bank, bank_offset[0] + start, grad_w_acc, gmm_fn=gmm_fn, quant_rule=quant_rule,
          w_routed_q=None if w_routed_q is None else routed_w_from_tiles(w_routed_q),
      )
    grad_x_ag = dsv3_router.unpermute_chunk(
        grad_x_sorted, reduce_metadata, num_out_tokens=x_ag.shape[0], num_experts_per_tok=num_experts_per_tok,
        out=grad_x_ag_acc, zero_initialized=(grad_x_ag_acc is None),
    )
    return grad_x_ag, grad_w_routed, jnp.reshape(grad_coeffs, (-1,)), constrain_activation_bank(bank)

  grad_x_ag, grad_w_routed, grad_coeffs, bank = _chunk(0, metadata.reduce_metadata, _bank_from_carry(bank), grad_w_acc)
  grad_sorted_coeffs = jnp.pad(grad_coeffs, (0, num_slots))

  def _cond(carry):
    _, remaining, *_ = carry
    return remaining > capacity

  def _body(carry):
    start, remaining, grad_x_ag, grad_w_routed, grad_sorted_coeffs, bank = carry
    start = start + capacity
    remaining = remaining - capacity
    reduce_metadata = dsv3_router.chunk_reduce_metadata(
        metadata.sort_indices, start, dsv3_router.chunk_num_tokens(num_local_tokens, start, capacity),
        capacity=capacity, num_experts_per_tok=num_experts_per_tok,
    )
    grad_x_ag, grad_w_routed, grad_coeffs, bank = _chunk(
        start, reduce_metadata, _bank_from_carry(bank), grad_w_routed, grad_x_ag_acc=grad_x_ag,
    )
    grad_sorted_coeffs = jax.lax.dynamic_update_slice_in_dim(grad_sorted_coeffs, grad_coeffs, start, axis=0)
    return start, remaining, grad_x_ag, grad_w_routed, grad_sorted_coeffs, _bank_to_carry(bank)

  _, _, grad_x_ag, grad_w_routed, grad_sorted_coeffs, bank = jax.lax.while_loop(
      _cond, _body, (jnp.int32(0), num_local_tokens, grad_x_ag, grad_w_routed, grad_sorted_coeffs, _bank_to_carry(bank)),
  )
  return grad_x_ag, grad_sorted_coeffs[:num_slots], routed_w_to_tiles(grad_w_routed), bank, bank_offset


def _w_dcn_axes() -> dsv3_types.DSv3SparseLayerWeightsPytree:
  axes = jax.tree.map(lambda _: "dcn", _w_ici_axes())
  return dataclasses.replace(
      axes, moe=dataclasses.replace(axes.moe, router=dsv3_types.DSv3MoERouterWeightsPytree(kernel="dcn", bias=())),
  )


@jt.jaxtyped(typechecker=typeguard.typechecked)
def collect_w_dcn(
    w: dsv3_types.DSv3SparseLayerWeightsPytree, *, axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> dsv3_types.DSv3SparseLayerWeightsPytree:
  return axis_collectives.collect_along_axis(w, _w_dcn_axes(), axis_mapping)


@jt.jaxtyped(typechecker=typeguard.typechecked)
def reduce_w_dcn(
    grad_w: dsv3_types.DSv3SparseLayerWeightsPytree, like: dsv3_types.DSv3SparseLayerWeightsPytree,
    *, axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> dsv3_types.DSv3SparseLayerWeightsPytree:
  return axis_collectives.reduce_along_axis(grad_w, _w_dcn_axes(), axis_mapping, like)


_SC1_KWARGS = dict(compute_type="tpu_sparsecore", compiler_options={"sparse_core_config": {"core_ids": [1]}})
_SC0_KWARGS = dict(compute_type="tpu_sparsecore", compiler_options={"sparse_core_config": {"core_ids": [0]}})


def _sc1_reduce_axes(axis_mapping: Mapping[str, str | tuple[str, ...]]) -> frozenset[str]:
  fsdp_moe = axis_mapping["fsdp_moe"]
  return frozenset((fsdp_moe,) if isinstance(fsdp_moe, str) else fsdp_moe)


def _w_ici_axes() -> dsv3_types.DSv3SparseLayerWeightsPytree:
  return dsv3_types.DSv3SparseLayerWeightsPytree(
      pre_attn_norm_scale=("fsdp_attention", "attention"),
      mla=dsv3_types.DSv3MLAWeightsPytree(
          q_down=("fsdp_attention", "attention"), q_up=("fsdp_attention",), q_norm_scale=("fsdp_attention",),
          kv_down=("fsdp_attention", "attention"), k_up=("fsdp_attention",), v_up=("fsdp_attention",),
          kv_norm_scale=("fsdp_attention",), out=("fsdp_attention",),
      ),
      post_attn_norm_scale=("fsdp_attention", "attention"),
      moe=dsv3_types.DSv3MoEWeightsPytree(
          router=dsv3_types.DSv3MoERouterWeightsPytree(kernel=("expert", "fsdp_moe"), bias=()),
          routed=dsv3_types.DSv3MoERoutedExpertWeightsPytree(gate=("fsdp_moe",), linear=("fsdp_moe",)),
          shared=dsv3_types.DSv3MoESharedExpertWeightsPytree(
              gate_0=("expert", "fsdp_moe"), gate_1=("expert", "fsdp_moe"), linear=("expert", "fsdp_moe"),
          ),
      ),
  )


@jt.jaxtyped(typechecker=typeguard.typechecked)
def shuffle_w_routed(
    w: dsv3_types.DSv3SparseLayerWeightsPytree, expert_permutations: jt.Int[jax.Array, "E"],
    *, axis_mapping: Mapping[str, str | tuple[str, ...]], expert_axis_name: str = "expert",
) -> dsv3_types.DSv3SparseLayerWeightsPytree:
  with jax.named_scope("shuffle_experts"):
    routed = dsv3_expert_shuffle.shuffle_experts(
        w.moe.routed, expert_permutations, expert_axis_name=expert_axis_name, axis_mapping=axis_mapping,
    )
  return dataclasses.replace(w, moe=dataclasses.replace(w.moe, routed=routed))


def unshuffle_w_routed(
    grad_w: dsv3_types.DSv3SparseLayerWeightsPytree, expert_permutations: jt.Int[jax.Array, "E"],
    *, axis_mapping: Mapping[str, str | tuple[str, ...]], expert_axis_name: str = "expert",
) -> dsv3_types.DSv3SparseLayerWeightsPytree:
  return shuffle_w_routed(
      grad_w, dsv3_expert_shuffle.inverse_permutation(expert_permutations),
      axis_mapping=axis_mapping, expert_axis_name=expert_axis_name,
  )


@jt.jaxtyped(typechecker=typeguard.typechecked)
def collect_w_ici(
    w: dsv3_types.DSv3SparseLayerWeightsPytree, *, axis_mapping: Mapping[str, str | tuple[str, ...]],
    split: bool = True, expert_axis_name: str = "expert", expert_permutations: jt.Int[jax.Array, "E"] | None = None,
) -> dsv3_types.DSv3SparseLayerWeightsPytree:
  w_axes = _w_ici_axes()
  if expert_permutations is not None:
    w = shuffle_w_routed(w, expert_permutations, axis_mapping=axis_mapping, expert_axis_name=expert_axis_name)
  assert w.moe.routed.gate is not None
  flat_routed = w.moe.routed.gate.ndim == 2
  w = dataclasses.replace(w, moe=dataclasses.replace(w.moe, routed=routed_w_to_tiles(w.moe.routed)))
  if split:
    pre_attn_norm_scale, mla = axis_collectives.collect_along_axis(
        (w.pre_attn_norm_scale, w.mla), (w_axes.pre_attn_norm_scale, w_axes.mla), axis_mapping, **_SC1_KWARGS,
    )
    post_attn_norm_scale, moe = axis_collectives.collect_along_axis(
        (w.post_attn_norm_scale, w.moe), (w_axes.post_attn_norm_scale, w_axes.moe), axis_mapping, **_SC1_KWARGS,
    )
    collected = dsv3_types.DSv3SparseLayerWeightsPytree(
        pre_attn_norm_scale=pre_attn_norm_scale, mla=mla, post_attn_norm_scale=post_attn_norm_scale, moe=moe,
    )
  else:
    collected = axis_collectives.collect_along_axis(w, w_axes, axis_mapping, **_SC1_KWARGS)
  if not flat_routed:
    return collected
  assert w.moe.router.kernel is not None
  num_experts = w.moe.router.kernel.shape[-1]
  routed = jax.tree.map(lambda a: a.reshape(num_experts, -1, *a.shape[-2:]), collected.moe.routed)
  return dataclasses.replace(collected, moe=dataclasses.replace(collected.moe, routed=routed))


def _split_w_attention_moe(
    w: dsv3_types.DSv3SparseLayerWeightsPytree,
) -> tuple[dsv3_types.DSv3SparseLayerWeightsPytree, dsv3_types.DSv3SparseLayerWeightsPytree]:
  masked = jax.tree.map(lambda _: None, w)
  attention = dataclasses.replace(masked, pre_attn_norm_scale=w.pre_attn_norm_scale, mla=w.mla)
  moe = dataclasses.replace(masked, post_attn_norm_scale=w.post_attn_norm_scale, moe=w.moe)
  return attention, moe


def _w_ici_split_scatters() -> dsv3_types.DSv3SparseLayerWeightsPytree:
  return dsv3_types.DSv3SparseLayerWeightsPytree(
      pre_attn_norm_scale=False,
      mla=dsv3_types.DSv3MLAWeightsPytree(
          q_down=True, q_up=False, q_norm_scale=False, kv_down=True, k_up=True, v_up=True,
          kv_norm_scale=False, out=False,
      ),
      post_attn_norm_scale=False,
      moe=dsv3_types.DSv3MoEWeightsPytree(
          router=dsv3_types.DSv3MoERouterWeightsPytree(kernel=False, bias=False),
          routed=dsv3_types.DSv3MoERoutedExpertWeightsPytree(gate=False, linear=False),
          shared=dsv3_types.DSv3MoESharedExpertWeightsPytree(gate_0=True, gate_1=True, linear=True),
      ),
  )


@jt.jaxtyped(typechecker=typeguard.typechecked)
def reduce_w_ici_sc1(
    grad_w: dsv3_types.DSv3SparseLayerWeightsPytree, like: dsv3_types.DSv3SparseLayerWeightsPytree,
    *, axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> dsv3_types.DSv3SparseLayerWeightsPytree:
  flat_routed = like.moe.routed.gate is not None and like.moe.routed.gate.ndim == 2
  routed = (
      jax.tree.map(lambda a: a.reshape(-1, a.shape[-1]), grad_w.moe.routed)
      if flat_routed else routed_w_from_tiles(grad_w.moe.routed)
  )
  grad_w = dataclasses.replace(grad_w, moe=dataclasses.replace(grad_w.moe, routed=routed))
  return axis_collectives.reduce_along_axis(
      grad_w, _w_ici_axes(), axis_mapping, like, axes=_sc1_reduce_axes(axis_mapping),
      split_scatters=_w_ici_split_scatters(), **_SC1_KWARGS,
  )


@jt.jaxtyped(typechecker=typeguard.typechecked)
def reduce_w_ici_sc0(
    grad_w: dsv3_types.DSv3SparseLayerWeightsPytree, like: dsv3_types.DSv3SparseLayerWeightsPytree,
    *, axis_mapping: Mapping[str, str | tuple[str, ...]], defer: bool = False,
) -> dsv3_types.DSv3SparseLayerWeightsPytree | tuple[axis_collectives.DeferredReduce, axis_collectives.DeferredReduce]:
  mesh = jax.typeof(jax.tree.leaves(like)[0]).sharding.mesh
  axes = frozenset(mesh.axis_names) - _sc1_reduce_axes(axis_mapping)
  if not defer:
    return axis_collectives.reduce_along_axis(grad_w, _w_ici_axes(), axis_mapping, like, axes=axes, **_SC0_KWARGS)
  return tuple(
      axis_collectives.reduce_along_axis(grad_w_half, _w_ici_axes(), axis_mapping, like_half, axes=axes, defer=True)
      for grad_w_half, like_half in zip(_split_w_attention_moe(grad_w), _split_w_attention_moe(like))
  )


@jax.named_call
def dsv3_sparse_layer_prologue_fwd(
    x: tuple[jt.Num[jax.Array, "B T D"], jt.Num[jax.Array, "B T D"]],
    w: dsv3_types.DSv3SparseLayerWeightsPytree, yarn_freqs: tuple[Any, Any], splash_kernel: Any,
    *, qk_head_dim: int, rope_head_dim: int, num_query_heads: int, mscale: float, kv_lora_rank: int,
    max_position_embeddings: int, original_max_position_embeddings: int, rope_factor: int,
    mesh: jax.sharding.Mesh, norm_fn: NormFn, rope_fn: RopeFn, axis_mapping: Mapping[str, str | tuple[str, ...]],
    banks: tuple[jax.Array, jax.Array], bank_offsets: tuple[jax.Array, jax.Array],
    segment_ids: tuple[jax.Array | None, jax.Array | None] = (None, None),
    quant: quantization.QuantConfig | None = None,
) -> tuple[Any, dsv3_types.DSv3LayerResiduals]:
  mb0, mb1 = x
  bank_0, bank_1 = banks
  bank_offset_0, bank_offset_1 = bank_offsets
  yarn_freqs_0, _ = yarn_freqs
  segment_ids_0, _ = segment_ids
  mla_rule = quant.mla if quant else None
  outputs_mla_0, res_mla = ubatch_mla(
      mb0, w.pre_attn_norm_scale, w.mla, yarn_freqs_0, splash_kernel, segment_ids=segment_ids_0,
      qk_head_dim=qk_head_dim, rope_head_dim=rope_head_dim, num_query_heads=num_query_heads, mscale=mscale,
      kv_lora_rank=kv_lora_rank, max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings, rope_factor=rope_factor,
      mesh=mesh, norm_fn=norm_fn, rope_fn=rope_fn, axis_mapping=axis_mapping, quant_rule=mla_rule,
  )
  mb1_wait = ubatch_wait(mb1)
  res = dsv3_types.DSv3LayerResiduals(mb0=dsv3_types.DSv3MicrobatchResiduals(mla=res_mla), mb1=None)
  return ((outputs_mla_0, bank_0, bank_offset_0), (mb1_wait, bank_1, bank_offset_1)), res


@jax.named_call
def dsv3_sparse_layer_prologue_bwd(
    res: dsv3_types.DSv3LayerResiduals, grad_outputs: Any, w: dsv3_types.DSv3SparseLayerWeightsPytree,
    *, yarn_freqs: tuple[Any, Any], splash_kernel: Any, qk_head_dim: int, rope_head_dim: int,
    num_query_heads: int, mscale: float, kv_lora_rank: int, max_position_embeddings: int,
    original_max_position_embeddings: int, rope_factor: int, mesh: jax.sharding.Mesh,
    norm_fn: NormFn, rope_fn: RopeFn, axis_mapping: Mapping[str, str | tuple[str, ...]],
    segment_ids: tuple[jax.Array | None, jax.Array | None] = (None, None),
    quant: quantization.QuantConfig | None = None,
) -> tuple[Any, dsv3_types.DSv3SparseLayerWeightsPytree]:
  (grad_outputs_mla_0, _, _), (grad_mb1, _, _) = grad_outputs
  grad_out_proj_out, grad_orig_x = grad_outputs_mla_0
  yarn_freqs_0, _ = yarn_freqs
  segment_ids_0, _ = segment_ids
  mla_rule = quant.mla if quant else None
  assert res.mb0.mla is not None
  with jax.named_scope("pre_attn_norm"):
    norm_x = norm_fn(res.mb0.mla.orig_x, w.pre_attn_norm_scale)
  grad_norm_x, grad_w_mla, _ = dsv3_mla.dsv3_mla_bwd(
      grad_out_proj_out, norm_x, res.mb0.mla.q_down, res.mb0.mla.kv_down, res.mb0.mla.context,
      res.mb0.mla.splash_out, w.mla, yarn_freqs_0, splash_kernel, segment_ids=segment_ids_0,
      kv_lora_rank=kv_lora_rank, qk_head_dim=qk_head_dim, rope_head_dim=rope_head_dim,
      num_query_heads=num_query_heads, max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings, rope_factor=rope_factor,
      mscale=mscale, norm_fn=norm_fn, rope_fn=rope_fn, mesh=mesh, axis_mapping=axis_mapping, quant_rule=mla_rule,
  )

  def _fwd_pre_norm(x, pre_attn_norm_scale):
    with jax.named_scope("pre_attn_norm"):
      return norm_fn(x, pre_attn_norm_scale)
  _, vjp_pre_norm = jax.vjp(_fwd_pre_norm, res.mb0.mla.orig_x, w.pre_attn_norm_scale)
  grad_x_norm, grad_pre_attn_norm_scale = vjp_pre_norm(grad_norm_x)
  grad_mb0 = grad_x_norm + grad_orig_x
  grad_mb1_out = ubatch_wait(grad_mb1)
  grad_w = dsv3_types.DSv3SparseLayerWeightsPytree(
      pre_attn_norm_scale=grad_pre_attn_norm_scale, mla=grad_w_mla, post_attn_norm_scale=None,
      moe=dsv3_types.DSv3MoEWeightsPytree(
          router=dsv3_types.DSv3MoERouterWeightsPytree(), routed=dsv3_types.DSv3MoERoutedExpertWeightsPytree(),
          shared=dsv3_types.DSv3MoESharedExpertWeightsPytree(),
      ),
  )
  return (grad_mb0, grad_mb1_out), grad_w


@jax.named_call
def dsv3_sparse_layer_epilogue_fwd(
    outputs_mla: Any, w: dsv3_types.DSv3SparseLayerWeightsPytree, yarn_freqs: tuple[Any, Any], splash_kernel: Any,
    *, num_experts: int, num_experts_per_tok: int, routed_scaling_factor: float, n_routing_groups: int,
    topk_routing_group: int, topk_in_group: int, qk_head_dim: int, rope_head_dim: int, num_query_heads: int,
    mscale: float, kv_lora_rank: int, max_position_embeddings: int, original_max_position_embeddings: int,
    rope_factor: int, mesh: jax.sharding.Mesh, norm_fn: NormFn, rope_fn: RopeFn, gmm_fn: GmmFn,
    axis_mapping: Mapping[str, str | tuple[str, ...]], expert_axis_name: str = "expert", capacity_factor: float = 2.0,
    router_dtype: jax.typing.DTypeLike | None = None, segment_ids: tuple[jax.Array | None, jax.Array | None] = (None, None),
    quant: quantization.QuantConfig | None = None, expert_permutations: jt.Int[jax.Array, "E"] | None = None,
) -> tuple[Any, dsv3_types.DSv3LayerResiduals]:
  quant_rule = quant.routed_experts if quant else None
  mla_rule = quant.mla if quant else None
  combine_qtype = quant.combine_qtype if quant else None
  combine_per_hop_scaling = quant.combine_per_hop_scaling if quant else False
  w_routed = w.moe.routed if quant_rule is None else _quantize_routed_weights(w.moe.routed, quant_rule)
  (outputs_mla_curr_0, bank_0, bank_offset_0), (mb1, bank_1, bank_offset_1) = outputs_mla
  out_proj_out_curr_0, _ = outputs_mla_curr_0
  x_pspec = jax.typeof(out_proj_out_curr_0).sharding.spec
  local_seq_length = _get_local_dim_size(out_proj_out_curr_0.shape[1], x_pspec[1], mesh)
  _, yarn_freqs_1 = yarn_freqs
  _, segment_ids_1 = segment_ids

  @jax.named_call
  @program_order(enforce=False)
  def phase1(outputs_mla_curr_0):
    out_proj_out_curr_0, orig_x_curr_0 = outputs_mla_curr_0
    (outputs_metadata_curr_0, aux_curr_0), res_metadata_curr_0 = ubatch_norm_and_metadata(
        out_proj_out_curr_0, orig_x_curr_0, w.post_attn_norm_scale, w.moe.router, norm_fn=norm_fn,
        num_experts=num_experts, num_experts_per_tok=num_experts_per_tok, routed_scaling_factor=routed_scaling_factor,
        n_routing_groups=n_routing_groups, topk_routing_group=topk_routing_group, topk_in_group=topk_in_group,
        expert_axis_name=expert_axis_name, axis_mapping=axis_mapping, mesh=mesh, capacity_factor=capacity_factor,
        router_dtype=router_dtype, expert_permutations=expert_permutations,
    )
    return (outputs_metadata_curr_0, aux_curr_0), res_metadata_curr_0

  @jax.named_call
  @program_order(enforce=False)
  def phase2(outputs_metadata_curr_0, mb1):
    outputs_dispatch_curr_0, _ = ubatch_dispatch(
        *outputs_metadata_curr_0, expert_axis_name=expert_axis_name, axis_mapping=axis_mapping, mesh=mesh, quant_rule=quant_rule,
    )
    outputs_mla_curr_1, res_mla_curr_1 = ubatch_mla(
        mb1, w.pre_attn_norm_scale, w.mla, yarn_freqs_1, splash_kernel, segment_ids=segment_ids_1,
        qk_head_dim=qk_head_dim, rope_head_dim=rope_head_dim, num_query_heads=num_query_heads, mscale=mscale,
        kv_lora_rank=kv_lora_rank, max_position_embeddings=max_position_embeddings,
        original_max_position_embeddings=original_max_position_embeddings, rope_factor=rope_factor,
        mesh=mesh, norm_fn=norm_fn, rope_fn=rope_fn, axis_mapping=axis_mapping, quant_rule=mla_rule,
    )
    return (outputs_dispatch_curr_0, outputs_mla_curr_1), res_mla_curr_1

  @jax.named_call
  @program_order(enforce=False)
  def phase3(outputs_dispatch_curr_0, outputs_mla_curr_1):
    outputs_dispatch_wait_0 = ubatch_wait(outputs_dispatch_curr_0)
    out_proj_out_curr_1, orig_x_curr_1 = outputs_mla_curr_1
    (outputs_metadata_curr_1, aux_curr_1), res_metadata_curr_1 = ubatch_norm_and_metadata(
        out_proj_out_curr_1, orig_x_curr_1, w.post_attn_norm_scale, w.moe.router, norm_fn=norm_fn,
        num_experts=num_experts, num_experts_per_tok=num_experts_per_tok, routed_scaling_factor=routed_scaling_factor,
        n_routing_groups=n_routing_groups, topk_routing_group=topk_routing_group, topk_in_group=topk_in_group,
        expert_axis_name=expert_axis_name, axis_mapping=axis_mapping, mesh=mesh, capacity_factor=capacity_factor,
        router_dtype=router_dtype, expert_permutations=expert_permutations,
    )
    return (outputs_dispatch_wait_0, outputs_metadata_curr_1, aux_curr_1), res_metadata_curr_1

  @jax.named_call
  @program_order(enforce=False)
  def phase4(outputs_dispatch_curr_0, outputs_metadata_curr_1, bank_0, bank_offset_0):
    (outputs_expert_curr_0, (bank_0, bank_offset_0)), res_expert_curr_0 = ubatch_expert(
        *outputs_dispatch_curr_0, bank=bank_0, bank_offset=bank_offset_0, w_routed=w_routed,
        num_experts_per_tok=num_experts_per_tok, gmm_fn=gmm_fn, mesh=mesh, quant_rule=quant_rule,
    )
    outputs_dispatch_curr_1, _ = ubatch_dispatch(
        *outputs_metadata_curr_1, expert_axis_name=expert_axis_name, axis_mapping=axis_mapping, mesh=mesh, quant_rule=quant_rule,
    )
    return (outputs_expert_curr_0, outputs_dispatch_curr_1, bank_0, bank_offset_0), res_expert_curr_0

  @jax.named_call
  @program_order(enforce=False)
  def phase5(outputs_expert_curr_0, outputs_dispatch_curr_1, bank_1, bank_offset_1):
    outputs_combine_curr_0, _ = ubatch_combine(
        *outputs_expert_curr_0, w_shared=w.moe.shared, local_seq_length=local_seq_length,
        expert_axis_name=expert_axis_name, axis_mapping=axis_mapping, mesh=mesh, out_specs=x_pspec,
        combine_qtype=combine_qtype, combine_per_hop_scaling=combine_per_hop_scaling,
    )
    (outputs_expert_curr_1, (bank_1, bank_offset_1)), res_expert_curr_1 = ubatch_expert(
        *outputs_dispatch_curr_1, bank=bank_1, bank_offset=bank_offset_1, w_routed=w_routed,
        num_experts_per_tok=num_experts_per_tok, gmm_fn=gmm_fn, mesh=mesh, quant_rule=quant_rule,
    )
    return (outputs_combine_curr_0, outputs_expert_curr_1, bank_1, bank_offset_1), res_expert_curr_1

  @jax.named_call
  @program_order(enforce=False)
  def phase6(outputs_combine_curr_0, outputs_expert_curr_1):
    outputs_res_add_curr_0, _ = ubatch_residual_add(*outputs_combine_curr_0)
    (out_mb0,) = outputs_res_add_curr_0
    outputs_expert_wait_1 = ubatch_wait(outputs_expert_curr_1)
    return (out_mb0, outputs_expert_wait_1), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase7(out_mb0, outputs_expert_curr_1):
    out_mb0_wait = ubatch_wait(out_mb0)
    outputs_combine_curr_1, _ = ubatch_combine(
        *outputs_expert_curr_1, w_shared=w.moe.shared, local_seq_length=local_seq_length,
        expert_axis_name=expert_axis_name, axis_mapping=axis_mapping, mesh=mesh, out_specs=x_pspec,
        combine_qtype=combine_qtype, combine_per_hop_scaling=combine_per_hop_scaling,
    )
    return (out_mb0_wait, outputs_combine_curr_1), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase8(out_mb0_wait, outputs_combine_curr_1):
    out_mb0_final = ubatch_wait(out_mb0_wait)
    outputs_res_add_curr_1, _ = ubatch_residual_add(*outputs_combine_curr_1)
    (out_mb1,) = outputs_res_add_curr_1
    return (out_mb0_final, out_mb1), ()

  @program_order(enforce=True)
  def _epilogue_fwd(outputs_mla_curr_0, mb1, bank_0, bank_offset_0, bank_1, bank_offset_1):
    (outputs_metadata_curr_0, aux_curr_0), res_phase1_curr_0 = phase1(outputs_mla_curr_0)
    (outputs_dispatch_curr_0, outputs_mla_curr_1), res_mla_curr_1 = phase2(outputs_metadata_curr_0, mb1)
    (outputs_dispatch_wait_0, outputs_metadata_curr_1, aux_curr_1), res_phase3_curr_1 = phase3(
        outputs_dispatch_curr_0, outputs_mla_curr_1,
    )
    (outputs_expert_curr_0, outputs_dispatch_curr_1, bank_0, bank_offset_0), res_expert_curr_0 = phase4(
        outputs_dispatch_wait_0, outputs_metadata_curr_1, bank_0, bank_offset_0,
    )
    (outputs_combine_curr_0, outputs_expert_curr_1, bank_1, bank_offset_1), res_expert_curr_1 = phase5(
        outputs_expert_curr_0, outputs_dispatch_curr_1, bank_1, bank_offset_1,
    )
    (out_mb0, outputs_expert_wait_1), _ = phase6(outputs_combine_curr_0, outputs_expert_curr_1)
    (out_mb0_wait, outputs_combine_curr_1), _ = phase7(out_mb0, outputs_expert_wait_1)
    (out_mb0_final, out_mb1), _ = phase8(out_mb0_wait, outputs_combine_curr_1)
    aux = _merge_router_aux(aux_curr_0, aux_curr_1, mesh=mesh)
    return (((out_mb0_final, out_mb1), aux), (bank_0, bank_1), (bank_offset_0, bank_offset_1)), (
        res_phase1_curr_0, res_expert_curr_0, res_mla_curr_1, res_phase3_curr_1, res_expert_curr_1,
    )

  (((out_mb0_final, out_mb1), aux), (bank_0, bank_1), (bank_offset_0, bank_offset_1)), (
      res_phase1_curr_0, res_expert_curr_0, res_mla_curr_1, res_phase3_curr_1, res_expert_curr_1,
  ) = _epilogue_fwd(outputs_mla_curr_0, mb1, bank_0, bank_offset_0, bank_1, bank_offset_1)
  out_proj_out_curr_0, orig_x_curr_0 = res_phase1_curr_0
  out_proj_out_curr_1, orig_x_curr_1 = res_phase3_curr_1
  moe_curr_0 = dsv3_types.DSv3MoEResiduals(
      out_proj_out=out_proj_out_curr_0, orig_x=orig_x_curr_0, router_metadata=res_expert_curr_0,
  )
  moe_curr_1 = dsv3_types.DSv3MoEResiduals(
      out_proj_out=out_proj_out_curr_1, orig_x=orig_x_curr_1, router_metadata=res_expert_curr_1,
  )
  res = dsv3_types.DSv3LayerResiduals(
      mb0=dsv3_types.DSv3MicrobatchResiduals(moe=moe_curr_0),
      mb1=dsv3_types.DSv3MicrobatchResiduals(mla=res_mla_curr_1, moe=moe_curr_1),
      bank=(bank_0, bank_1), bank_offset=(bank_offset_0, bank_offset_1),
  )
  return ((out_mb0_final, out_mb1), aux), res


@jax.named_call
def dsv3_sparse_layer_epilogue_bwd(
    res: dsv3_types.DSv3LayerResiduals, grad_outputs: Any, w: dsv3_types.DSv3SparseLayerWeightsPytree,
    *, yarn_freqs: tuple[Any, Any], splash_kernel: Any, num_experts: int, num_experts_per_tok: int,
    routed_scaling_factor: float, n_routing_groups: int, topk_routing_group: int, topk_in_group: int,
    qk_head_dim: int, rope_head_dim: int, num_query_heads: int, mscale: float, kv_lora_rank: int,
    max_position_embeddings: int, original_max_position_embeddings: int, rope_factor: int, mesh: jax.sharding.Mesh,
    norm_fn: NormFn, rope_fn: RopeFn, gmm_fn: GmmFn, axis_mapping: Mapping[str, str | tuple[str, ...]],
    expert_axis_name: str = "expert", capacity_factor: float = 2.0, router_dtype: jax.typing.DTypeLike | None = None,
    segment_ids: tuple[jax.Array | None, jax.Array | None] = (None, None),
    quant: quantization.QuantConfig | None = None,
) -> tuple[Any, dsv3_types.DSv3SparseLayerWeightsPytree]:
  quant_rule = quant.routed_experts if quant else None
  mla_rule = quant.mla if quant else None
  w_routed_q = None if quant_rule is None else _quantize_routed_weights(w.moe.routed, quant_rule)
  grad_out, grad_aux = grad_outputs
  grad_out_mb0, grad_out_mb1 = grad_out
  assert res.bank is not None
  assert res.bank_offset is not None
  bank_0, bank_1 = res.bank
  bank_0 = constrain_activation_bank(bank_0)
  bank_1 = constrain_activation_bank(bank_1)
  bank_offset_0, bank_offset_1 = res.bank_offset
  assert res.mb0.moe is not None
  assert res.mb1 is not None
  assert res.mb1.mla is not None
  assert res.mb1.moe is not None
  x_pspec = jax.typeof(res.mb0.moe.out_proj_out).sharding.spec
  local_seq_length = _get_local_dim_size(res.mb0.moe.out_proj_out.shape[1], x_pspec[1], mesh)
  _, yarn_freqs_1 = yarn_freqs
  _, segment_ids_1 = segment_ids
  if getattr(grad_aux, "logits", None) is not None:
    logits_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*x_pspec[:-1], None))
    grad_logits = grad_aux.logits
    if jax.typeof(grad_logits).sharding != logits_sharding:
      grad_logits = jax.reshard(grad_logits, logits_sharding)
    grad_logits_0, grad_logits_1 = microbatching.split_microbatches(grad_logits, mesh=mesh)
    grad_aux_0 = dsv3_types.DSv3RouterAux[Any](
        group_sizes=getattr(grad_aux, "group_sizes", None),
        selected_experts=getattr(grad_aux, "selected_experts", None), logits=grad_logits_0,
    )
    grad_aux_1 = dsv3_types.DSv3RouterAux[Any](
        group_sizes=getattr(grad_aux, "group_sizes", None),
        selected_experts=getattr(grad_aux, "selected_experts", None), logits=grad_logits_1,
    )
  else:
    grad_aux_0 = grad_aux
    grad_aux_1 = grad_aux
  mla_out_curr_0 = res.mb0.moe.out_proj_out if res.mb0.moe.orig_x is None else res.mb0.moe.out_proj_out + res.mb0.moe.orig_x
  with jax.named_scope("post_attn_norm"):
    moe_in_curr_0 = norm_fn(mla_out_curr_0, w.post_attn_norm_scale)
  mla_out_curr_1 = res.mb1.moe.out_proj_out + res.mb1.mla.orig_x
  with jax.named_scope("post_attn_norm"):
    moe_in_curr_1 = norm_fn(mla_out_curr_1, w.post_attn_norm_scale)

  @jax.named_call
  @program_order(enforce=False)
  def phase1(grad_out_mb0, grad_out_mb1):
    grad_out_mb0_wait = ubatch_wait(grad_out_mb0)
    grad_shared_x_1 = grad_out_mb1
    grad_mla_out_unpermute_1 = grad_out_mb1
    return (grad_out_mb0_wait, grad_out_mb1, grad_shared_x_1, grad_mla_out_unpermute_1), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase2(grad_out_mb0_wait, grad_out_mb1):
    grad_shared_x_0 = grad_out_mb0_wait
    grad_mla_out_unpermute_0 = grad_out_mb0_wait
    x_ag_1, grad_y_ag_1, grad_y_scale_1 = _dispatch_bwd(
        moe_in_curr_1, grad_out_mb1, expert_axis_name=expert_axis_name, axis_mapping=axis_mapping,
        mesh=mesh, gmm_fn=gmm_fn, quant_rule=quant_rule,
    )
    return ((grad_out_mb0_wait, grad_shared_x_0, grad_mla_out_unpermute_0), (x_ag_1, grad_y_ag_1, grad_y_scale_1)), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase3(mb0_state, mb1_state):
    return ubatch_wait((mb0_state, mb1_state)), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase4(grad_out_mb0, x_ag_1, grad_y_ag_1, grad_y_scale_1, grad_shared_x_1, bank_1, bank_offset_1):
    assert res.mb1 is not None
    assert res.mb1.moe is not None
    x_ag_0, grad_y_ag_0, grad_y_scale_0 = _dispatch_bwd(
        moe_in_curr_0, grad_out_mb0, expert_axis_name=expert_axis_name, axis_mapping=axis_mapping,
        mesh=mesh, gmm_fn=gmm_fn, quant_rule=quant_rule,
    )

    def _shared_expert_fwd_1(moe_in, w_shared):
      return dsv3_experts.dsv3_shared_expert(moe_in, w_shared)
    _, vjp_shared_1 = jax.vjp(_shared_expert_fwd_1, moe_in_curr_1, w.moe.shared)
    grad_moe_in_shared_1, grad_w_shared_1 = vjp_shared_1(grad_shared_x_1)
    grad_x_ag_1, grad_sorted_coeffs_1, grad_w_routed_1, bank_1, bank_offset_1 = ubatch_expert_bwd(
        x_ag_1, grad_y_ag_1, res.mb1.moe.router_metadata, bank=bank_1, bank_offset=bank_offset_1,
        w_routed=w.moe.routed, num_experts_per_tok=num_experts_per_tok, gmm_fn=gmm_fn, mesh=mesh,
        quant_rule=quant_rule, w_routed_q=w_routed_q, grad_y_scale=grad_y_scale_1,
    )
    return (
        (x_ag_0, grad_y_ag_0, grad_y_scale_0),
        (grad_x_ag_1, grad_sorted_coeffs_1, grad_w_routed_1, grad_moe_in_shared_1, grad_w_shared_1),
        (bank_1, bank_offset_1),
    ), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase5(x_ag_0, grad_y_ag_0, grad_y_scale_0, grad_x_ag_1, grad_sorted_coeffs_1, grad_w_routed_1, bank_0, bank_offset_0):
    assert res.mb0.moe is not None
    assert res.mb1 is not None
    assert res.mb1.moe is not None
    grad_coeffs_ag_1 = dsv3_router.dsv3_unsort_coeff_grads(
        grad_sorted_coeffs_1, res.mb1.moe.router_metadata, expert_axis_name=expert_axis_name, axis_mapping=axis_mapping, mesh=mesh,
    )
    grad_coeffs_ag_1, x_ag_0, grad_y_ag_0, grad_y_scale_0 = jax.lax.optimization_barrier(
        (grad_coeffs_ag_1, x_ag_0, grad_y_ag_0, grad_y_scale_0)
    )
    grad_x_ag_0, grad_sorted_coeffs_0, grad_w_routed, bank_0, bank_offset_0 = ubatch_expert_bwd(
        x_ag_0, grad_y_ag_0, res.mb0.moe.router_metadata, bank=bank_0, bank_offset=bank_offset_0,
        w_routed=w.moe.routed, grad_w_acc=grad_w_routed_1, num_experts_per_tok=num_experts_per_tok,
        gmm_fn=gmm_fn, mesh=mesh, quant_rule=quant_rule, w_routed_q=w_routed_q, grad_y_scale=grad_y_scale_0,
    )
    grad_moe_in_routed_1, grad_coeffs_1 = dsv3_router.dsv3_combine_with_coeff_grads(
        grad_x_ag_1, grad_coeffs_ag_1, res.mb1.moe.router_metadata, local_seq_length=local_seq_length,
        num_experts_per_tok=num_experts_per_tok, expert_axis_name=expert_axis_name, axis_mapping=axis_mapping,
        mesh=mesh, out_specs=x_pspec,
    )
    return ((grad_x_ag_0, grad_sorted_coeffs_0, grad_w_routed), (grad_moe_in_routed_1, grad_coeffs_1), (bank_0, bank_offset_0)), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase6(grad_x_ag_0, grad_sorted_coeffs_0, grad_shared_x_0, grad_moe_in_routed_1, grad_moe_in_shared_1, grad_coeffs_1, grad_aux_1):
    assert res.mb0.moe is not None
    grad_coeffs_ag_0 = dsv3_router.dsv3_unsort_coeff_grads(
        grad_sorted_coeffs_0, res.mb0.moe.router_metadata, expert_axis_name=expert_axis_name, axis_mapping=axis_mapping, mesh=mesh,
    )
    grad_coeffs_ag_0, grad_shared_x_0, grad_moe_in_shared_1 = jax.lax.optimization_barrier(
        (grad_coeffs_ag_0, grad_shared_x_0, grad_moe_in_shared_1)
    )
    grad_moe_in_routed_0, grad_coeffs_0 = dsv3_router.dsv3_combine_with_coeff_grads(
        grad_x_ag_0, grad_coeffs_ag_0, res.mb0.moe.router_metadata, local_seq_length=local_seq_length,
        num_experts_per_tok=num_experts_per_tok, expert_axis_name=expert_axis_name, axis_mapping=axis_mapping,
        mesh=mesh, out_specs=x_pspec,
    )

    def _shared_expert_fwd_0(moe_in, w_shared):
      return dsv3_experts.dsv3_shared_expert(moe_in, w_shared)
    _, vjp_shared_0 = jax.vjp(_shared_expert_fwd_0, moe_in_curr_0, w.moe.shared)
    grad_moe_in_shared_0, grad_w_shared_0 = vjp_shared_0(grad_shared_x_0)
    grad_moe_in_expert_1 = grad_moe_in_routed_1 + grad_moe_in_shared_1
    assert res.mb1 is not None
    assert res.mb1.moe is not None
    grad_moe_in_router_1, grad_w_router_1 = dsv3_router.dsv3_routing_bwd(
        grad_coeffs_1, grad_aux_1.logits, moe_in_curr_1, w.moe.router, res.mb1.moe.router_metadata,
        routed_scaling_factor=routed_scaling_factor, mesh=mesh,
    )
    grad_moe_in_total_1 = grad_moe_in_expert_1 + grad_moe_in_router_1

    def _fwd_post_norm_1(mla_out, post_attn_norm_scale):
      with jax.named_scope("post_attn_norm"):
        return norm_fn(mla_out, post_attn_norm_scale)
    _, vjp_post_norm_1 = jax.vjp(_fwd_post_norm_1, mla_out_curr_1, w.post_attn_norm_scale)
    grad_mla_out_norm_1, grad_post_attn_norm_scale_1 = vjp_post_norm_1(grad_moe_in_total_1)
    return ((grad_moe_in_routed_0, grad_coeffs_0, grad_moe_in_shared_0, grad_w_shared_0),
            (grad_mla_out_norm_1, grad_post_attn_norm_scale_1, grad_w_router_1)), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase7(mb0_state, grad_mla_out_total_1):
    assert res.mb1 is not None
    assert res.mb1.mla is not None
    mb0_wait = ubatch_wait(mb0_state)
    with jax.named_scope("pre_attn_norm"):
      norm_x_1 = norm_fn(res.mb1.mla.orig_x, w.pre_attn_norm_scale)
    grad_norm_x_1, grad_w_mla_1, grad_yarn_freqs_1 = dsv3_mla.dsv3_mla_bwd(
        grad_mla_out_total_1, norm_x_1, res.mb1.mla.q_down, res.mb1.mla.kv_down, res.mb1.mla.context,
        res.mb1.mla.splash_out, w.mla, yarn_freqs_1, splash_kernel, segment_ids=segment_ids_1,
        kv_lora_rank=kv_lora_rank, qk_head_dim=qk_head_dim, rope_head_dim=rope_head_dim,
        num_query_heads=num_query_heads, max_position_embeddings=max_position_embeddings,
        original_max_position_embeddings=original_max_position_embeddings, rope_factor=rope_factor, mscale=mscale,
        norm_fn=norm_fn, rope_fn=rope_fn, mesh=mesh, axis_mapping=axis_mapping, quant_rule=mla_rule,
    )

    def _fwd_pre_norm_1(x, pre_attn_norm_scale):
      with jax.named_scope("pre_attn_norm"):
        return norm_fn(x, pre_attn_norm_scale)
    _, vjp_pre_norm_1 = jax.vjp(_fwd_pre_norm_1, res.mb1.mla.orig_x, w.pre_attn_norm_scale)
    grad_x_norm_1, grad_pre_attn_norm_scale_1 = vjp_pre_norm_1(grad_norm_x_1)
    grad_mb1 = grad_x_norm_1 + grad_mla_out_total_1
    return (mb0_wait, (grad_mb1, grad_pre_attn_norm_scale_1, grad_w_mla_1, grad_yarn_freqs_1)), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase8(grad_moe_in_routed_0, grad_moe_in_shared_0, grad_coeffs_0, grad_aux_0, grad_mb1):
    grad_moe_in_expert_0 = grad_moe_in_routed_0 + grad_moe_in_shared_0
    assert res.mb0.moe is not None
    grad_moe_in_router_0, grad_w_router_0 = dsv3_router.dsv3_routing_bwd(
        grad_coeffs_0, grad_aux_0.logits, moe_in_curr_0, w.moe.router, res.mb0.moe.router_metadata,
        routed_scaling_factor=routed_scaling_factor, mesh=mesh,
    )
    grad_moe_in_total_0 = grad_moe_in_expert_0 + grad_moe_in_router_0

    def _fwd_post_norm_0(mla_out, post_attn_norm_scale):
      with jax.named_scope("post_attn_norm"):
        return norm_fn(mla_out, post_attn_norm_scale)
    _, vjp_post_norm_0 = jax.vjp(_fwd_post_norm_0, mla_out_curr_0, w.post_attn_norm_scale)
    grad_mla_out_norm_0, grad_post_attn_norm_scale_0 = vjp_post_norm_0(grad_moe_in_total_0)
    grad_mb1_wait = ubatch_wait(grad_mb1)
    return ((grad_mla_out_norm_0, grad_post_attn_norm_scale_0, grad_w_router_0), grad_mb1_wait), ()

  @program_order(enforce=True)
  def _epilogue_bwd(grad_out_mb0, grad_out_mb1, bank_0, bank_offset_0, bank_1, bank_offset_1):
    (grad_out_mb0_wait, grad_out_mb1_p1, grad_shared_x_1, grad_mla_out_unpermute_1), _ = phase1(grad_out_mb0, grad_out_mb1)
    ((grad_out_mb0_p2, grad_shared_x_0, grad_mla_out_unpermute_0), (x_ag_1, grad_y_ag_1, grad_y_scale_1)), _ = phase2(
        grad_out_mb0_wait, grad_out_mb1_p1,
    )
    ((grad_out_mb0_wait3, grad_shared_x_0_wait, grad_mla_out_unpermute_0_wait),
     (x_ag_1_wait, grad_y_ag_1_wait, grad_y_scale_1_wait)), _ = phase3(
         (grad_out_mb0_p2, grad_shared_x_0, grad_mla_out_unpermute_0), (x_ag_1, grad_y_ag_1, grad_y_scale_1),
     )
    ((x_ag_0, grad_y_ag_0, grad_y_scale_0),
     (grad_x_ag_1, grad_sorted_coeffs_1, grad_w_routed_1, grad_moe_in_shared_1, grad_w_shared_1),
     (bank_1, bank_offset_1)), _ = phase4(
         grad_out_mb0_wait3, x_ag_1_wait, grad_y_ag_1_wait, grad_y_scale_1_wait, grad_shared_x_1, bank_1, bank_offset_1,
     )
    ((grad_x_ag_0, grad_sorted_coeffs_0, grad_w_routed), (grad_moe_in_routed_1, grad_coeffs_1),
     (bank_0, bank_offset_0)), _ = phase5(
         x_ag_0, grad_y_ag_0, grad_y_scale_0, grad_x_ag_1, grad_sorted_coeffs_1, grad_w_routed_1, bank_0, bank_offset_0,
     )
    ((grad_moe_in_routed_0, grad_coeffs_0, grad_moe_in_shared_0, grad_w_shared_0),
     (grad_mla_out_norm_1, grad_post_attn_norm_scale_1, grad_w_router_1)), _ = phase6(
         grad_x_ag_0, grad_sorted_coeffs_0, grad_shared_x_0_wait, grad_moe_in_routed_1,
         grad_moe_in_shared_1, grad_coeffs_1, grad_aux_1,
     )
    grad_mla_out_total_1 = grad_mla_out_unpermute_1 + grad_mla_out_norm_1
    (mb0_wait_7, (grad_mb1, grad_pre_attn_norm_scale_1, grad_w_mla_1, grad_yarn_freqs_1)), _ = phase7(
        (grad_moe_in_routed_0, grad_moe_in_shared_0, grad_coeffs_0, grad_mla_out_unpermute_0_wait), grad_mla_out_total_1,
    )
    grad_moe_in_routed_0_w, grad_moe_in_shared_0_w, grad_coeffs_0_w, grad_mla_out_unpermute_0_w = mb0_wait_7
    ((grad_mla_out_norm_0, grad_post_attn_norm_scale_0, grad_w_router_0), grad_mb1_wait), _ = phase8(
        grad_moe_in_routed_0_w, grad_moe_in_shared_0_w, grad_coeffs_0_w, grad_aux_0, grad_mb1,
    )
    grad_mla_out_total_0 = grad_mla_out_unpermute_0_w + grad_mla_out_norm_0
    return (
        grad_mla_out_total_0, grad_mb1_wait, bank_0, bank_offset_0, bank_1, bank_offset_1,
        grad_pre_attn_norm_scale_1, grad_w_mla_1, grad_yarn_freqs_1,
        grad_post_attn_norm_scale_0, grad_post_attn_norm_scale_1, grad_w_router_0, grad_w_router_1,
        grad_w_routed, grad_w_shared_0, grad_w_shared_1,
    )

  (
      grad_mla_out_total_0, grad_mb1_wait, bank_0, bank_offset_0, bank_1, bank_offset_1,
      grad_pre_attn_norm_scale_1, grad_w_mla_1, _, grad_post_attn_norm_scale_0, grad_post_attn_norm_scale_1,
      grad_w_router_0, grad_w_router_1, grad_w_routed, grad_w_shared_0, grad_w_shared_1,
  ) = _epilogue_bwd(grad_out_mb0, grad_out_mb1, bank_0, bank_offset_0, bank_1, bank_offset_1)
  grad_carry = (((grad_mla_out_total_0, grad_mla_out_total_0), bank_0, bank_offset_0), (grad_mb1_wait, bank_1, bank_offset_1))
  grad_w = dsv3_types.DSv3SparseLayerWeightsPytree(
      pre_attn_norm_scale=grad_pre_attn_norm_scale_1, mla=grad_w_mla_1,
      post_attn_norm_scale=grad_post_attn_norm_scale_0 + grad_post_attn_norm_scale_1,
      moe=dsv3_types.DSv3MoEWeightsPytree(
          router=jax.tree.map(jnp.add, grad_w_router_0, grad_w_router_1), routed=grad_w_routed,
          shared=jax.tree.map(jnp.add, grad_w_shared_0, grad_w_shared_1),
      ),
  )
  return grad_carry, grad_w


@jax.named_call
def dsv3_sparse_layer_scan_body_fwd(
    outputs_mla_curr: Any, w_curr: dsv3_types.DSv3SparseLayerWeightsPytree,
    w_next: dsv3_types.DSv3SparseLayerWeightsPytree, yarn_freqs: tuple[Any, Any], splash_kernel: Any,
    *, num_experts: int, num_experts_per_tok: int, routed_scaling_factor: float, n_routing_groups: int,
    topk_routing_group: int, topk_in_group: int, qk_head_dim: int, rope_head_dim: int, num_query_heads: int,
    mscale: float, kv_lora_rank: int, max_position_embeddings: int, original_max_position_embeddings: int,
    rope_factor: int, mesh: jax.sharding.Mesh, norm_fn: NormFn, rope_fn: RopeFn, gmm_fn: GmmFn,
    axis_mapping: Mapping[str, str | tuple[str, ...]], expert_axis_name: str = "expert", capacity_factor: float = 2.0,
    router_dtype: jax.typing.DTypeLike | None = None, segment_ids: tuple[jax.Array | None, jax.Array | None] = (None, None),
    quant: quantization.QuantConfig | None = None, expert_permutations: jt.Int[jax.Array, "E"] | None = None,
) -> tuple[Any, dsv3_types.DSv3LayerResiduals]:
  quant_rule = quant.routed_experts if quant else None
  mla_rule = quant.mla if quant else None
  combine_qtype = quant.combine_qtype if quant else None
  combine_per_hop_scaling = quant.combine_per_hop_scaling if quant else False
  w_routed = w_curr.moe.routed if quant_rule is None else _quantize_routed_weights(w_curr.moe.routed, quant_rule)
  (outputs_mla_curr_0, bank_0, bank_offset_0), (orig_x_curr_1, bank_1, bank_offset_1) = outputs_mla_curr
  out_proj_out_curr_0, _ = outputs_mla_curr_0
  x_pspec = jax.typeof(out_proj_out_curr_0).sharding.spec
  local_seq_length = _get_local_dim_size(out_proj_out_curr_0.shape[1], x_pspec[1], mesh)
  yarn_freqs_0, yarn_freqs_1 = yarn_freqs
  segment_ids_0, segment_ids_1 = segment_ids

  @jax.named_call
  @program_order(enforce=False)
  def phase1(outputs_mla_curr_0):
    out_proj_out_curr_0, orig_x_curr_0 = outputs_mla_curr_0
    (outputs_metadata_curr_0, aux_curr_0), res_metadata_curr_0 = ubatch_norm_and_metadata(
        out_proj_out_curr_0, orig_x_curr_0, w_curr.post_attn_norm_scale, w_curr.moe.router, norm_fn=norm_fn,
        num_experts=num_experts, num_experts_per_tok=num_experts_per_tok, routed_scaling_factor=routed_scaling_factor,
        n_routing_groups=n_routing_groups, topk_routing_group=topk_routing_group, topk_in_group=topk_in_group,
        expert_axis_name=expert_axis_name, axis_mapping=axis_mapping, mesh=mesh, capacity_factor=capacity_factor,
        router_dtype=router_dtype, expert_permutations=expert_permutations,
    )
    return (outputs_metadata_curr_0, aux_curr_0), res_metadata_curr_0

  @jax.named_call
  @program_order(enforce=False)
  def phase2(outputs_metadata_curr_0, mb1):
    outputs_dispatch_curr_0, _ = ubatch_dispatch(
        *outputs_metadata_curr_0, expert_axis_name=expert_axis_name, axis_mapping=axis_mapping, mesh=mesh, quant_rule=quant_rule,
    )
    outputs_mla_curr_1, res_mla_curr_1 = ubatch_mla(
        mb1, w_curr.pre_attn_norm_scale, w_curr.mla, yarn_freqs_1, splash_kernel, segment_ids=segment_ids_1,
        qk_head_dim=qk_head_dim, rope_head_dim=rope_head_dim, num_query_heads=num_query_heads, mscale=mscale,
        kv_lora_rank=kv_lora_rank, max_position_embeddings=max_position_embeddings,
        original_max_position_embeddings=original_max_position_embeddings, rope_factor=rope_factor,
        mesh=mesh, norm_fn=norm_fn, rope_fn=rope_fn, axis_mapping=axis_mapping, quant_rule=mla_rule,
    )
    return (outputs_dispatch_curr_0, outputs_mla_curr_1), res_mla_curr_1

  @jax.named_call
  @program_order(enforce=False)
  def phase3(outputs_dispatch_curr_0, outputs_mla_curr_1):
    outputs_dispatch_wait_0 = ubatch_wait(outputs_dispatch_curr_0)
    out_proj_out_curr_1, orig_x_curr_1 = outputs_mla_curr_1
    (outputs_metadata_curr_1, aux_curr_1), res_metadata_curr_1 = ubatch_norm_and_metadata(
        out_proj_out_curr_1, orig_x_curr_1, w_curr.post_attn_norm_scale, w_curr.moe.router, norm_fn=norm_fn,
        num_experts=num_experts, num_experts_per_tok=num_experts_per_tok, routed_scaling_factor=routed_scaling_factor,
        n_routing_groups=n_routing_groups, topk_routing_group=topk_routing_group, topk_in_group=topk_in_group,
        expert_axis_name=expert_axis_name, axis_mapping=axis_mapping, mesh=mesh, capacity_factor=capacity_factor,
        router_dtype=router_dtype, expert_permutations=expert_permutations,
    )
    return (outputs_dispatch_wait_0, outputs_metadata_curr_1, aux_curr_1), res_metadata_curr_1

  @jax.named_call
  @program_order(enforce=False)
  def phase4(outputs_dispatch_curr_0, outputs_metadata_curr_1, bank_0, bank_offset_0):
    (outputs_expert_curr_0, (bank_0, bank_offset_0)), res_expert_curr_0 = ubatch_expert(
        *outputs_dispatch_curr_0, w_routed=w_routed, num_experts_per_tok=num_experts_per_tok,
        gmm_fn=gmm_fn, mesh=mesh, bank=bank_0, bank_offset=bank_offset_0, quant_rule=quant_rule,
    )
    outputs_dispatch_curr_1, _ = ubatch_dispatch(
        *outputs_metadata_curr_1, expert_axis_name=expert_axis_name, axis_mapping=axis_mapping, mesh=mesh, quant_rule=quant_rule,
    )
    return (outputs_expert_curr_0, outputs_dispatch_curr_1, bank_0, bank_offset_0), res_expert_curr_0

  @jax.named_call
  @program_order(enforce=False)
  def phase5(outputs_expert_curr_0, outputs_dispatch_curr_1, bank_1, bank_offset_1):
    outputs_combine_curr_0, _ = ubatch_combine(
        *outputs_expert_curr_0, w_shared=w_curr.moe.shared, local_seq_length=local_seq_length,
        expert_axis_name=expert_axis_name, axis_mapping=axis_mapping, mesh=mesh, out_specs=x_pspec,
        combine_qtype=combine_qtype, combine_per_hop_scaling=combine_per_hop_scaling,
    )
    (outputs_expert_curr_1, (bank_1, bank_offset_1)), res_expert_curr_1 = ubatch_expert(
        *outputs_dispatch_curr_1, w_shared=w_curr.moe.shared, w_routed=w_routed, num_experts_per_tok=num_experts_per_tok,
        gmm_fn=gmm_fn, mesh=mesh, bank=bank_1, bank_offset=bank_offset_1, quant_rule=quant_rule,
    )
    return (outputs_combine_curr_0, outputs_expert_curr_1, bank_1, bank_offset_1), res_expert_curr_1

  @jax.named_call
  @program_order(enforce=False)
  def phase6(outputs_combine_curr_0, outputs_expert_curr_1):
    outputs_res_add_curr_0, _ = ubatch_residual_add(*outputs_combine_curr_0)
    (x_next_0,) = outputs_res_add_curr_0
    outputs_expert_wait_1 = ubatch_wait(outputs_expert_curr_1)
    return (x_next_0, outputs_expert_wait_1), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase7(x_next_0, outputs_expert_curr_1):
    outputs_mla_next_0, res_mla_next_0 = ubatch_mla(
        x_next_0, w_next.pre_attn_norm_scale, w_next.mla, yarn_freqs_0, splash_kernel, segment_ids=segment_ids_0,
        qk_head_dim=qk_head_dim, rope_head_dim=rope_head_dim, num_query_heads=num_query_heads, mscale=mscale,
        kv_lora_rank=kv_lora_rank, max_position_embeddings=max_position_embeddings,
        original_max_position_embeddings=original_max_position_embeddings, rope_factor=rope_factor,
        mesh=mesh, norm_fn=norm_fn, rope_fn=rope_fn, axis_mapping=axis_mapping, quant_rule=mla_rule,
    )
    outputs_combine_curr_1, _ = ubatch_combine(
        *outputs_expert_curr_1, local_seq_length=local_seq_length, expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping, mesh=mesh, out_specs=x_pspec,
        combine_qtype=combine_qtype, combine_per_hop_scaling=combine_per_hop_scaling,
    )
    return (outputs_mla_next_0, outputs_combine_curr_1), res_mla_next_0

  @jax.named_call
  @program_order(enforce=False)
  def phase8(outputs_mla_next_0, outputs_combine_curr_1):
    outputs_mla_next_0_wait = ubatch_wait(outputs_mla_next_0)
    outputs_res_add_curr_1, _ = ubatch_residual_add(*outputs_combine_curr_1)
    (x_next_1,) = outputs_res_add_curr_1
    return (outputs_mla_next_0_wait, x_next_1), ()

  @program_order(enforce=True)
  def _scan_body_fwd(outputs_mla_curr_0, mb1, bank_0, bank_offset_0, bank_1, bank_offset_1):
    (outputs_metadata_curr_0, aux_curr_0), res_phase1_curr_0 = phase1(outputs_mla_curr_0)
    (outputs_dispatch_curr_0, outputs_mla_curr_1), res_mla_curr_1 = phase2(outputs_metadata_curr_0, mb1)
    (outputs_dispatch_wait_0, outputs_metadata_curr_1, aux_curr_1), res_phase3_curr_1 = phase3(
        outputs_dispatch_curr_0, outputs_mla_curr_1,
    )
    (outputs_expert_curr_0, outputs_dispatch_curr_1, bank_0, bank_offset_0), res_expert_curr_0 = phase4(
        outputs_dispatch_wait_0, outputs_metadata_curr_1, bank_0, bank_offset_0,
    )
    (outputs_combine_curr_0, outputs_expert_curr_1, bank_1, bank_offset_1), res_expert_curr_1 = phase5(
        outputs_expert_curr_0, outputs_dispatch_curr_1, bank_1, bank_offset_1,
    )
    (x_next_0, outputs_expert_wait_1), _ = phase6(outputs_combine_curr_0, outputs_expert_curr_1)
    (outputs_mla_next_0, outputs_combine_curr_1), res_mla_next_0 = phase7(x_next_0, outputs_expert_wait_1)
    (outputs_mla_next_0_wait, x_next_1), _ = phase8(outputs_mla_next_0, outputs_combine_curr_1)
    aux_curr = _merge_router_aux(aux_curr_0, aux_curr_1, mesh=mesh)
    return (((outputs_mla_next_0_wait, bank_0, bank_offset_0), (x_next_1, bank_1, bank_offset_1)), aux_curr), (
        res_phase1_curr_0, res_mla_curr_1, res_phase3_curr_1, res_expert_curr_0, res_expert_curr_1, res_mla_next_0,
    )

  (((outputs_mla_next_0_wait, bank_0, bank_offset_0), (x_next_1, bank_1, bank_offset_1)), aux_curr), (
      res_phase1_curr_0, res_mla_curr_1, res_phase3_curr_1, res_expert_curr_0, res_expert_curr_1, res_mla_next_0,
  ) = _scan_body_fwd(outputs_mla_curr_0, orig_x_curr_1, bank_0, bank_offset_0, bank_1, bank_offset_1)
  out_proj_out_curr_0, orig_x_curr_0 = res_phase1_curr_0
  out_proj_out_curr_1, orig_x_curr_1 = res_phase3_curr_1
  moe_curr_0 = dsv3_types.DSv3MoEResiduals(
      out_proj_out=out_proj_out_curr_0, orig_x=orig_x_curr_0, router_metadata=res_expert_curr_0,
  )
  moe_curr_1 = dsv3_types.DSv3MoEResiduals(
      out_proj_out=out_proj_out_curr_1, orig_x=orig_x_curr_1, router_metadata=res_expert_curr_1,
  )
  res = dsv3_types.DSv3LayerResiduals(
      mb0=dsv3_types.DSv3MicrobatchResiduals(mla=res_mla_next_0, moe=moe_curr_0),
      mb1=dsv3_types.DSv3MicrobatchResiduals(mla=res_mla_curr_1, moe=moe_curr_1),
  )
  return (((outputs_mla_next_0_wait, bank_0, bank_offset_0), (x_next_1, bank_1, bank_offset_1)), aux_curr), res


@jax.named_call
def dsv3_sparse_layer_scan_body_bwd(
    res: dsv3_types.DSv3LayerResiduals, grad_outputs: Any,
    w_curr: dsv3_types.DSv3SparseLayerWeightsPytree, w_next: dsv3_types.DSv3SparseLayerWeightsPytree,
    w_next_expert_grad_unreduced: dsv3_types.DSv3SparseLayerWeightsPytree,
    reduce_ici_first_fn: Callable[[dsv3_types.DSv3SparseLayerWeightsPytree], dsv3_types.DSv3SparseLayerWeightsPytree],
    w_pending_grad_partial: dsv3_types.DSv3SparseLayerWeightsPytree | None,
    reduce_ici_second_fn: Callable[..., tuple[axis_collectives.DeferredReduce, axis_collectives.DeferredReduce]],
    *, yarn_freqs: tuple[Any, Any], splash_kernel: Any, num_experts: int, num_experts_per_tok: int,
    routed_scaling_factor: float, n_routing_groups: int, topk_routing_group: int, topk_in_group: int,
    qk_head_dim: int, rope_head_dim: int, num_query_heads: int, mscale: float, kv_lora_rank: int,
    max_position_embeddings: int, original_max_position_embeddings: int, rope_factor: int, mesh: jax.sharding.Mesh,
    norm_fn: NormFn, rope_fn: RopeFn, gmm_fn: GmmFn, axis_mapping: Mapping[str, str | tuple[str, ...]],
    expert_axis_name: str = "expert", capacity_factor: float = 2.0, router_dtype: jax.typing.DTypeLike | None = None,
    segment_ids: tuple[jax.Array | None, jax.Array | None] = (None, None), quant: quantization.QuantConfig | None = None,
) -> tuple[Any, dsv3_types.DSv3SparseLayerWeightsPytree, dsv3_types.DSv3SparseLayerWeightsPytree, Any]:
  quant_rule = quant.routed_experts if quant else None
  mla_rule = quant.mla if quant else None
  w_routed_q = None if quant_rule is None else _quantize_routed_weights(w_curr.moe.routed, quant_rule)
  grad_carry, grad_aux_curr = grad_outputs
  (grad_outputs_mla_next_0, bank_0, bank_offset_0), (grad_x_next_1, bank_1, bank_offset_1) = grad_carry
  assert res.mb0.moe is not None
  assert res.mb1 is not None
  assert res.mb1.mla is not None
  assert res.mb1.moe is not None
  mla_out_curr_0 = res.mb0.moe.out_proj_out if res.mb0.moe.orig_x is None else res.mb0.moe.out_proj_out + res.mb0.moe.orig_x
  with jax.named_scope("post_attn_norm"):
    moe_in_curr_0 = norm_fn(mla_out_curr_0, w_curr.post_attn_norm_scale)
  mla_out_curr_1 = res.mb1.moe.out_proj_out + res.mb1.mla.orig_x
  with jax.named_scope("post_attn_norm"):
    moe_in_curr_1 = norm_fn(mla_out_curr_1, w_curr.post_attn_norm_scale)
  x_pspec = jax.typeof(moe_in_curr_0).sharding.spec
  local_seq_length = _get_local_dim_size(moe_in_curr_0.shape[1], x_pspec[1], mesh)
  yarn_freqs_0, yarn_freqs_1 = yarn_freqs
  segment_ids_0, segment_ids_1 = segment_ids
  if getattr(grad_aux_curr, "logits", None) is not None:
    logits_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*x_pspec[:-1], None))
    grad_logits_curr = grad_aux_curr.logits
    if jax.typeof(grad_logits_curr).sharding != logits_sharding:
      grad_logits_curr = jax.reshard(grad_logits_curr, logits_sharding)
    grad_logits_0, grad_logits_1 = microbatching.split_microbatches(grad_logits_curr, mesh=mesh)
    grad_aux_0 = dsv3_types.DSv3RouterAux[Any](
        group_sizes=grad_aux_curr.group_sizes, selected_experts=grad_aux_curr.selected_experts, logits=grad_logits_0,
    )
    grad_aux_1 = dsv3_types.DSv3RouterAux[Any](
        group_sizes=grad_aux_curr.group_sizes, selected_experts=grad_aux_curr.selected_experts, logits=grad_logits_1,
    )
  else:
    grad_aux_0 = grad_aux_curr
    grad_aux_1 = grad_aux_curr

  @jax.named_call
  @program_order(enforce=False)
  def phase1(grad_outputs_mla_next_0, grad_x_next_1):
    grad_outputs_mla_next_0_wait = ubatch_wait(grad_outputs_mla_next_0)
    grad_shared_x_1 = grad_x_next_1
    grad_mla_out_unpermute_1 = grad_x_next_1
    return (grad_outputs_mla_next_0_wait, grad_x_next_1, grad_shared_x_1, grad_mla_out_unpermute_1), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase2(grad_outputs_mla_next_0_wait, grad_x_next_1, w_next_expert_grad_unreduced):
    assert res.mb0.mla is not None
    grad_out_proj_out_next_0, grad_orig_x_next_0 = grad_outputs_mla_next_0_wait
    with jax.named_scope("pre_attn_norm"):
      norm_x_next_0 = norm_fn(res.mb0.mla.orig_x, w_next.pre_attn_norm_scale)
    grad_norm_x_next_0, grad_w_mla_next_0, grad_yarn_freqs_next_0 = dsv3_mla.dsv3_mla_bwd(
        grad_out_proj_out_next_0, norm_x_next_0, res.mb0.mla.q_down, res.mb0.mla.kv_down,
        res.mb0.mla.context, res.mb0.mla.splash_out, w_next.mla, yarn_freqs_0, splash_kernel,
        segment_ids=segment_ids_0, kv_lora_rank=kv_lora_rank, qk_head_dim=qk_head_dim,
        rope_head_dim=rope_head_dim, num_query_heads=num_query_heads, max_position_embeddings=max_position_embeddings,
        original_max_position_embeddings=original_max_position_embeddings, rope_factor=rope_factor, mscale=mscale,
        norm_fn=norm_fn, rope_fn=rope_fn, mesh=mesh, axis_mapping=axis_mapping, quant_rule=mla_rule,
    )

    def _fwd_pre_norm_next_0(x, pre_attn_norm_scale):
      with jax.named_scope("pre_attn_norm"):
        return norm_fn(x, pre_attn_norm_scale)
    _, vjp_pre_norm_next_0 = jax.vjp(_fwd_pre_norm_next_0, res.mb0.mla.orig_x, w_next.pre_attn_norm_scale)
    grad_x_norm_next_0, grad_pre_attn_norm_scale_next_0 = vjp_pre_norm_next_0(grad_norm_x_next_0)
    grad_x_next_0 = grad_x_norm_next_0 + grad_orig_x_next_0
    w_next_grad_unreduced = weight_scan_prologue_epilogue.combine_w_grads(
        dsv3_types.DSv3SparseLayerWeightsPytree(
            pre_attn_norm_scale=grad_pre_attn_norm_scale_next_0, mla=grad_w_mla_next_0,
            post_attn_norm_scale=None, moe=dsv3_types.DSv3MoEWeightsPytree(),
        ), w_next_expert_grad_unreduced,
    )
    x_ag_1, grad_y_ag_1, grad_y_scale_1 = _dispatch_bwd(
        moe_in_curr_1, grad_x_next_1, expert_axis_name=expert_axis_name, axis_mapping=axis_mapping,
        mesh=mesh, gmm_fn=gmm_fn, quant_rule=quant_rule,
    )
    return ((grad_x_next_0, w_next_grad_unreduced, grad_yarn_freqs_next_0), (x_ag_1, grad_y_ag_1, grad_y_scale_1)), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase3(grad_x_next_0, x_ag_1, grad_y_ag_1, grad_y_scale_1):
    grad_shared_x_0 = grad_x_next_0
    grad_mla_out_unpermute_0 = grad_x_next_0
    x_ag_1_wait, grad_y_ag_1_wait, grad_y_scale_1_wait = ubatch_wait((x_ag_1, grad_y_ag_1, grad_y_scale_1))
    return ((grad_x_next_0, grad_shared_x_0, grad_mla_out_unpermute_0),
            (x_ag_1_wait, grad_y_ag_1_wait, grad_y_scale_1_wait)), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase4(grad_x_next_0, x_ag_1, grad_y_ag_1, grad_y_scale_1, grad_shared_x_1, bank_1, bank_offset_1):
    assert res.mb1 is not None
    assert res.mb1.moe is not None
    x_ag_0, grad_y_ag_0, grad_y_scale_0 = _dispatch_bwd(
        moe_in_curr_0, grad_x_next_0, expert_axis_name=expert_axis_name, axis_mapping=axis_mapping,
        mesh=mesh, gmm_fn=gmm_fn, quant_rule=quant_rule,
    )

    def _shared_expert_fwd_1(moe_in, w_shared):
      return dsv3_experts.dsv3_shared_expert(moe_in, w_shared)
    _, vjp_shared_1 = jax.vjp(_shared_expert_fwd_1, moe_in_curr_1, w_curr.moe.shared)
    grad_moe_in_shared_1, grad_w_shared_1 = vjp_shared_1(grad_shared_x_1)
    grad_x_ag_1, grad_sorted_coeffs_1, grad_w_routed_1, bank_1, bank_offset_1 = ubatch_expert_bwd(
        x_ag_1, grad_y_ag_1, res.mb1.moe.router_metadata, bank=bank_1, bank_offset=bank_offset_1,
        w_routed=w_curr.moe.routed, num_experts_per_tok=num_experts_per_tok, gmm_fn=gmm_fn, mesh=mesh,
        quant_rule=quant_rule, w_routed_q=w_routed_q, grad_y_scale=grad_y_scale_1,
    )
    return ((x_ag_0, grad_y_ag_0, grad_y_scale_0),
            (grad_x_ag_1, grad_sorted_coeffs_1, grad_w_routed_1, grad_moe_in_shared_1, grad_w_shared_1),
            (bank_1, bank_offset_1)), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase5(x_ag_0, grad_y_ag_0, grad_y_scale_0, grad_shared_x_0, grad_x_ag_1,
             grad_sorted_coeffs_1, grad_w_routed_1, bank_0, bank_offset_0, w_pending_grad_partial):
    assert res.mb0.moe is not None
    assert res.mb1 is not None
    assert res.mb1.moe is not None
    grad_coeffs_ag_1 = dsv3_router.dsv3_unsort_coeff_grads(
        grad_sorted_coeffs_1, res.mb1.moe.router_metadata, expert_axis_name=expert_axis_name, axis_mapping=axis_mapping, mesh=mesh,
    )
    grad_coeffs_ag_1, x_ag_0, grad_y_ag_0, grad_y_scale_0, grad_shared_x_0 = jax.lax.optimization_barrier(
        (grad_coeffs_ag_1, x_ag_0, grad_y_ag_0, grad_y_scale_0, grad_shared_x_0)
    )

    def _shared_expert_fwd_0(moe_in, w_shared):
      return dsv3_experts.dsv3_shared_expert(moe_in, w_shared)
    _, vjp_shared_0 = jax.vjp(_shared_expert_fwd_0, moe_in_curr_0, w_curr.moe.shared)
    grad_moe_in_shared_0, grad_w_shared_0 = vjp_shared_0(grad_shared_x_0)
    grad_x_ag_0, grad_sorted_coeffs_0, grad_w_routed, bank_0, bank_offset_0 = ubatch_expert_bwd(
        x_ag_0, grad_y_ag_0, res.mb0.moe.router_metadata, bank=bank_0, bank_offset=bank_offset_0,
        w_routed=w_curr.moe.routed, grad_w_acc=grad_w_routed_1, num_experts_per_tok=num_experts_per_tok,
        gmm_fn=gmm_fn, mesh=mesh, quant_rule=quant_rule, w_routed_q=w_routed_q, grad_y_scale=grad_y_scale_0,
    )
    grad_moe_in_routed_1, grad_coeffs_1, w_pending_grad_attention = dsv3_router.dsv3_combine_with_coeff_grads_and_reduce(
        grad_x_ag_1, grad_coeffs_ag_1, res.mb1.moe.router_metadata,
        None if w_pending_grad_partial is None else reduce_ici_second_fn(w_pending_grad_partial, defer=True)[0],
        local_seq_length=local_seq_length, num_experts_per_tok=num_experts_per_tok,
        expert_axis_name=expert_axis_name, axis_mapping=axis_mapping, mesh=mesh, out_specs=x_pspec,
    )
    return ((grad_x_ag_0, grad_sorted_coeffs_0, grad_w_routed, grad_moe_in_shared_0, grad_w_shared_0),
            (grad_moe_in_routed_1, grad_coeffs_1), (bank_0, bank_offset_0), w_pending_grad_attention), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase6(mb0_state, grad_moe_in_routed_1, grad_moe_in_shared_1, grad_coeffs_1, grad_aux_1):
    mb0_wait = ubatch_wait(mb0_state)
    grad_moe_in_expert_1 = grad_moe_in_routed_1 + grad_moe_in_shared_1
    assert res.mb1 is not None
    assert res.mb1.moe is not None
    grad_moe_in_router_1, grad_w_router_1 = dsv3_router.dsv3_routing_bwd(
        grad_coeffs_1, grad_aux_1.logits, moe_in_curr_1, w_curr.moe.router, res.mb1.moe.router_metadata,
        routed_scaling_factor=routed_scaling_factor, mesh=mesh,
    )
    grad_moe_in_total_1 = grad_moe_in_expert_1 + grad_moe_in_router_1

    def _fwd_post_norm_1(mla_out, post_attn_norm_scale):
      with jax.named_scope("post_attn_norm"):
        return norm_fn(mla_out, post_attn_norm_scale)
    _, vjp_post_norm_1 = jax.vjp(_fwd_post_norm_1, mla_out_curr_1, w_curr.post_attn_norm_scale)
    grad_mla_out_norm_1, grad_post_attn_norm_scale_1 = vjp_post_norm_1(grad_moe_in_total_1)
    return (mb0_wait, (grad_mla_out_norm_1, grad_post_attn_norm_scale_1, grad_w_router_1)), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase7(grad_x_ag_0, grad_sorted_coeffs_0, grad_mla_out_total_1, w_pending_grad_partial):
    assert res.mb1 is not None
    assert res.mb1.mla is not None
    assert res.mb0.moe is not None
    grad_coeffs_ag_0 = dsv3_router.dsv3_unsort_coeff_grads(
        grad_sorted_coeffs_0, res.mb0.moe.router_metadata, expert_axis_name=expert_axis_name, axis_mapping=axis_mapping, mesh=mesh,
    )
    grad_moe_in_routed_0, grad_coeffs_0, w_pending_grad_moe = dsv3_router.dsv3_combine_with_coeff_grads_and_reduce(
        grad_x_ag_0, grad_coeffs_ag_0, res.mb0.moe.router_metadata,
        None if w_pending_grad_partial is None else reduce_ici_second_fn(w_pending_grad_partial, defer=True)[1],
        local_seq_length=local_seq_length, num_experts_per_tok=num_experts_per_tok,
        expert_axis_name=expert_axis_name, axis_mapping=axis_mapping, mesh=mesh, out_specs=x_pspec,
    )
    with jax.named_scope("pre_attn_norm"):
      norm_x_curr_1 = norm_fn(res.mb1.mla.orig_x, w_curr.pre_attn_norm_scale)
    grad_norm_x_1, grad_w_mla_1, grad_yarn_freqs_1 = dsv3_mla.dsv3_mla_bwd(
        grad_mla_out_total_1, norm_x_curr_1, res.mb1.mla.q_down, res.mb1.mla.kv_down, res.mb1.mla.context,
        res.mb1.mla.splash_out, w_curr.mla, yarn_freqs_1, splash_kernel, segment_ids=segment_ids_1,
        kv_lora_rank=kv_lora_rank, qk_head_dim=qk_head_dim, rope_head_dim=rope_head_dim,
        num_query_heads=num_query_heads, max_position_embeddings=max_position_embeddings,
        original_max_position_embeddings=original_max_position_embeddings, rope_factor=rope_factor, mscale=mscale,
        norm_fn=norm_fn, rope_fn=rope_fn, mesh=mesh, axis_mapping=axis_mapping, quant_rule=mla_rule,
    )

    def _fwd_pre_norm_1(x, pre_attn_norm_scale):
      with jax.named_scope("pre_attn_norm"):
        return norm_fn(x, pre_attn_norm_scale)
    _, vjp_pre_norm_1 = jax.vjp(_fwd_pre_norm_1, res.mb1.mla.orig_x, w_curr.pre_attn_norm_scale)
    grad_x_norm_1, grad_pre_attn_norm_scale_1 = vjp_pre_norm_1(grad_norm_x_1)
    grad_mb1 = grad_x_norm_1 + grad_mla_out_total_1
    return ((grad_moe_in_routed_0, grad_coeffs_0),
            (grad_mb1, grad_pre_attn_norm_scale_1, grad_w_mla_1, grad_yarn_freqs_1), w_pending_grad_moe), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase8(grad_moe_in_routed_0, grad_moe_in_shared_0, grad_coeffs_0, grad_aux_0, grad_mb1):
    grad_moe_in_expert_0 = grad_moe_in_routed_0 + grad_moe_in_shared_0
    assert res.mb0.moe is not None
    grad_moe_in_router_0, grad_w_router_0 = dsv3_router.dsv3_routing_bwd(
        grad_coeffs_0, grad_aux_0.logits, moe_in_curr_0, w_curr.moe.router, res.mb0.moe.router_metadata,
        routed_scaling_factor=routed_scaling_factor, mesh=mesh,
    )
    grad_moe_in_total_0 = grad_moe_in_expert_0 + grad_moe_in_router_0

    def _fwd_post_norm_0(mla_out, post_attn_norm_scale):
      with jax.named_scope("post_attn_norm"):
        return norm_fn(mla_out, post_attn_norm_scale)
    _, vjp_post_norm_0 = jax.vjp(_fwd_post_norm_0, mla_out_curr_0, w_curr.post_attn_norm_scale)
    grad_mla_out_norm_0, grad_post_attn_norm_scale_0 = vjp_post_norm_0(grad_moe_in_total_0)
    grad_mb1_wait = ubatch_wait(grad_mb1)
    return ((grad_mla_out_norm_0, grad_post_attn_norm_scale_0, grad_w_router_0), grad_mb1_wait), ()

  @program_order(enforce=True)
  def _scan_body_bwd(
      grad_outputs_mla_next_0, grad_x_next_1, bank_0, bank_offset_0, bank_1, bank_offset_1,
      w_next_expert_grad_unreduced, w_pending_grad_partial,
  ):
    (grad_outputs_mla_next_0_wait, grad_x_next_1_p1, grad_shared_x_1, grad_mla_out_unpermute_1), _ = phase1(
        grad_outputs_mla_next_0, grad_x_next_1,
    )
    ((grad_x_next_0, w_next_grad_unreduced, _), (x_ag_1, grad_y_ag_1, grad_y_scale_1)), _ = phase2(
        grad_outputs_mla_next_0_wait, grad_x_next_1_p1, w_next_expert_grad_unreduced,
    )
    ((grad_x_next_0_p3, grad_shared_x_0, grad_mla_out_unpermute_0),
     (x_ag_1_wait, grad_y_ag_1_wait, grad_y_scale_1_wait)), _ = phase3(grad_x_next_0, x_ag_1, grad_y_ag_1, grad_y_scale_1)
    ((x_ag_0, grad_y_ag_0, grad_y_scale_0),
     (grad_x_ag_1, grad_sorted_coeffs_1, grad_w_routed_1, grad_moe_in_shared_1, grad_w_shared_1),
     (bank_1, bank_offset_1)), _ = phase4(
         grad_x_next_0_p3, x_ag_1_wait, grad_y_ag_1_wait, grad_y_scale_1_wait, grad_shared_x_1, bank_1, bank_offset_1,
     )
    ((grad_x_ag_0, grad_sorted_coeffs_0, grad_w_routed, grad_moe_in_shared_0, grad_w_shared_0),
     (grad_moe_in_routed_1, grad_coeffs_1), (bank_0, bank_offset_0), w_pending_grad_attention), _ = phase5(
         x_ag_0, grad_y_ag_0, grad_y_scale_0, grad_shared_x_0, grad_x_ag_1, grad_sorted_coeffs_1,
         grad_w_routed_1, bank_0, bank_offset_0, w_pending_grad_partial,
     )
    (mb0_wait_6, (grad_mla_out_norm_1, grad_post_attn_norm_scale_1, grad_w_router_1)), _ = phase6(
        (grad_x_ag_0, grad_sorted_coeffs_0, grad_moe_in_shared_0, grad_mla_out_unpermute_0),
        grad_moe_in_routed_1, grad_moe_in_shared_1, grad_coeffs_1, grad_aux_1,
    )
    grad_x_ag_0_w, grad_sorted_coeffs_0_w, grad_moe_in_shared_0_w, grad_mla_out_unpermute_0_w = mb0_wait_6
    grad_mla_out_total_1 = grad_mla_out_unpermute_1 + grad_mla_out_norm_1
    ((grad_moe_in_routed_0, grad_coeffs_0), (grad_mb1, grad_pre_attn_norm_scale_1, grad_w_mla_1, _),
     w_pending_grad_moe), _ = phase7(grad_x_ag_0_w, grad_sorted_coeffs_0_w, grad_mla_out_total_1, w_pending_grad_partial)
    ((grad_mla_out_norm_0, grad_post_attn_norm_scale_0, grad_w_router_0), grad_mb1_wait), _ = phase8(
        grad_moe_in_routed_0, grad_moe_in_shared_0_w, grad_coeffs_0, grad_aux_0, grad_mb1,
    )
    grad_mla_out_total_0 = grad_mla_out_unpermute_0_w + grad_mla_out_norm_0
    return (
        w_next_grad_unreduced, grad_mla_out_total_0, grad_mb1_wait, bank_0, bank_offset_0, bank_1, bank_offset_1,
        grad_pre_attn_norm_scale_1, grad_w_mla_1, grad_post_attn_norm_scale_0, grad_post_attn_norm_scale_1,
        grad_w_router_0, grad_w_router_1, grad_w_routed, grad_w_shared_0, grad_w_shared_1,
        w_pending_grad_attention, w_pending_grad_moe,
    )

  (
      w_next_grad_unreduced, grad_mla_out_total_0, grad_mb1_wait, bank_0, bank_offset_0, bank_1, bank_offset_1,
      grad_pre_attn_norm_scale_1, grad_w_mla_1, grad_post_attn_norm_scale_0, grad_post_attn_norm_scale_1,
      grad_w_router_0, grad_w_router_1, grad_w_routed, grad_w_shared_0, grad_w_shared_1,
      w_pending_grad_attention, w_pending_grad_moe,
  ) = _scan_body_bwd(
      grad_outputs_mla_next_0, grad_x_next_1, bank_0, bank_offset_0, bank_1, bank_offset_1,
      w_next_expert_grad_unreduced, w_pending_grad_partial,
  )
  w_next_grad_partial = reduce_ici_first_fn(w_next_grad_unreduced)
  w_pending_grad = (
      None if w_pending_grad_partial is None
      else weight_scan_prologue_epilogue.combine_w_grads(w_pending_grad_attention, w_pending_grad_moe)
  )
  grad_outputs_mla_curr = (
      ((grad_mla_out_total_0, grad_mla_out_total_0), bank_0, bank_offset_0), (grad_mb1_wait, bank_1, bank_offset_1),
  )
  grad_w_curr = dsv3_types.DSv3SparseLayerWeightsPytree(
      pre_attn_norm_scale=grad_pre_attn_norm_scale_1, mla=grad_w_mla_1,
      post_attn_norm_scale=grad_post_attn_norm_scale_0 + grad_post_attn_norm_scale_1,
      moe=dsv3_types.DSv3MoEWeightsPytree(
          router=jax.tree.map(jnp.add, grad_w_router_0, grad_w_router_1), routed=grad_w_routed,
          shared=jax.tree.map(jnp.add, grad_w_shared_0, grad_w_shared_1),
      ),
  )
  return grad_outputs_mla_curr, grad_w_curr, w_next_grad_partial, w_pending_grad


@functools.partial(jax.custom_vjp, nondiff_argnums=tuple(range(6, 30)))
def _dsv3_sparse_layer_vjp(
    x, w, yarn_freqs, splash_kernel, segment_ids, expert_permutations, num_experts, num_experts_per_tok,
    routed_scaling_factor, n_routing_groups, topk_routing_group, topk_in_group, qk_head_dim, rope_head_dim,
    num_query_heads, mscale, kv_lora_rank, max_position_embeddings, original_max_position_embeddings,
    rope_factor, mesh, norm_fn, rope_fn, gmm_fn, axis_mapping, expert_axis_name, capacity_factor,
    router_dtype, ragged_buffer_factor, quant,
):
  return _dsv3_sparse_layer_fwd(
      x, w, yarn_freqs, splash_kernel, segment_ids, expert_permutations, num_experts, num_experts_per_tok,
      routed_scaling_factor, n_routing_groups, topk_routing_group, topk_in_group, qk_head_dim, rope_head_dim,
      num_query_heads, mscale, kv_lora_rank, max_position_embeddings, original_max_position_embeddings,
      rope_factor, mesh, norm_fn, rope_fn, gmm_fn, axis_mapping, expert_axis_name, capacity_factor,
      router_dtype, ragged_buffer_factor, quant,
  )[0]


def _dsv3_sparse_layer_fwd(
    x, w, yarn_freqs, splash_kernel, segment_ids, expert_permutations, num_experts, num_experts_per_tok,
    routed_scaling_factor, n_routing_groups, topk_routing_group, topk_in_group, qk_head_dim, rope_head_dim,
    num_query_heads, mscale, kv_lora_rank, max_position_embeddings, original_max_position_embeddings,
    rope_factor, mesh, norm_fn, rope_fn, gmm_fn, axis_mapping, expert_axis_name, capacity_factor,
    router_dtype, ragged_buffer_factor, quant,
):
  mb0, mb1 = microbatching.split_microbatches(x, mesh=mesh)
  yarn_freqs_0, yarn_freqs_1 = microbatching.split_microbatches(yarn_freqs, mesh=mesh)
  segment_ids_0, segment_ids_1 = microbatching.split_microbatches(segment_ids, mesh=mesh)
  (bank_0, bank_offset_0), (bank_1, bank_offset_1) = init_activation_bank(
      mb0, w, num_layers=1, num_experts_per_tok=num_experts_per_tok, ragged_buffer_factor=ragged_buffer_factor,
      capacity_factor=capacity_factor, expert_axis_name=expert_axis_name, axis_mapping=axis_mapping, mesh=mesh,
  )
  outputs_mla, res_prologue = dsv3_sparse_layer_prologue_fwd(
      (mb0, mb1), w, (yarn_freqs_0, yarn_freqs_1), splash_kernel,
      banks=(bank_0, bank_1), bank_offsets=(bank_offset_0, bank_offset_1),
      qk_head_dim=qk_head_dim, rope_head_dim=rope_head_dim, num_query_heads=num_query_heads, mscale=mscale,
      kv_lora_rank=kv_lora_rank, max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings, rope_factor=rope_factor,
      mesh=mesh, norm_fn=norm_fn, rope_fn=rope_fn, axis_mapping=axis_mapping,
      segment_ids=(segment_ids_0, segment_ids_1), quant=quant,
  )
  ((out_mb0, out_mb1), aux), res_epilogue = dsv3_sparse_layer_epilogue_fwd(
      outputs_mla, w, (yarn_freqs_0, yarn_freqs_1), splash_kernel,
      num_experts=num_experts, num_experts_per_tok=num_experts_per_tok, routed_scaling_factor=routed_scaling_factor,
      n_routing_groups=n_routing_groups, topk_routing_group=topk_routing_group, topk_in_group=topk_in_group,
      qk_head_dim=qk_head_dim, rope_head_dim=rope_head_dim, num_query_heads=num_query_heads, mscale=mscale,
      kv_lora_rank=kv_lora_rank, max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings, rope_factor=rope_factor,
      mesh=mesh, norm_fn=norm_fn, rope_fn=rope_fn, gmm_fn=gmm_fn, axis_mapping=axis_mapping,
      expert_axis_name=expert_axis_name, capacity_factor=capacity_factor, router_dtype=router_dtype,
      segment_ids=(segment_ids_0, segment_ids_1), quant=quant, expert_permutations=expert_permutations,
  )
  out = microbatching.merge_microbatches(out_mb0, out_mb1, mesh=mesh)
  res = (res_prologue, res_epilogue, w, yarn_freqs, splash_kernel, segment_ids)
  return (out, aux), res


def _dsv3_sparse_layer_bwd(
    num_experts, num_experts_per_tok, routed_scaling_factor, n_routing_groups, topk_routing_group,
    topk_in_group, qk_head_dim, rope_head_dim, num_query_heads, mscale, kv_lora_rank,
    max_position_embeddings, original_max_position_embeddings, rope_factor, mesh, norm_fn, rope_fn,
    gmm_fn, axis_mapping, expert_axis_name, capacity_factor, router_dtype, ragged_buffer_factor, quant, res, grad_outputs,
):
  del ragged_buffer_factor
  res_prologue, res_epilogue, w, yarn_freqs, splash_kernel, segment_ids = res
  grad_out, grad_aux = grad_outputs
  grad_mb0, grad_mb1 = microbatching.split_microbatches(grad_out, mesh=mesh)
  yarn_freqs_0, yarn_freqs_1 = microbatching.split_microbatches(yarn_freqs, mesh=mesh)
  segment_ids_0, segment_ids_1 = microbatching.split_microbatches(segment_ids, mesh=mesh)
  grad_carry, grad_w_epilogue = dsv3_sparse_layer_epilogue_bwd(
      res_epilogue, ((grad_mb0, grad_mb1), grad_aux), w, yarn_freqs=(yarn_freqs_0, yarn_freqs_1),
      splash_kernel=splash_kernel, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok,
      routed_scaling_factor=routed_scaling_factor, n_routing_groups=n_routing_groups,
      topk_routing_group=topk_routing_group, topk_in_group=topk_in_group, qk_head_dim=qk_head_dim,
      rope_head_dim=rope_head_dim, num_query_heads=num_query_heads, mscale=mscale, kv_lora_rank=kv_lora_rank,
      max_position_embeddings=max_position_embeddings, original_max_position_embeddings=original_max_position_embeddings,
      rope_factor=rope_factor, mesh=mesh, norm_fn=norm_fn, rope_fn=rope_fn, gmm_fn=gmm_fn,
      axis_mapping=axis_mapping, expert_axis_name=expert_axis_name, capacity_factor=capacity_factor,
      router_dtype=router_dtype, segment_ids=(segment_ids_0, segment_ids_1), quant=quant,
  )
  (grad_mb0_in, grad_mb1_in), grad_w_prologue = dsv3_sparse_layer_prologue_bwd(
      res_prologue, grad_carry, w, yarn_freqs=(yarn_freqs_0, yarn_freqs_1), splash_kernel=splash_kernel,
      qk_head_dim=qk_head_dim, rope_head_dim=rope_head_dim, num_query_heads=num_query_heads, mscale=mscale,
      kv_lora_rank=kv_lora_rank, max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings, rope_factor=rope_factor,
      mesh=mesh, norm_fn=norm_fn, rope_fn=rope_fn, axis_mapping=axis_mapping,
      segment_ids=(segment_ids_0, segment_ids_1), quant=quant,
  )
  grad_x = microbatching.merge_microbatches(grad_mb0_in, grad_mb1_in, mesh=mesh)
  grad_w = weight_scan_prologue_epilogue.combine_w_grads(grad_w_prologue, grad_w_epilogue)
  return grad_x, grad_w, None, None, None, None


_dsv3_sparse_layer_vjp.defvjp(_dsv3_sparse_layer_fwd, _dsv3_sparse_layer_bwd)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_sparse_layer(
    x: jt.Num[jax.Array, "B T D"], w: dsv3_types.DSv3SparseLayerWeightsPytree,
    yarn_freqs: tuple[jt.Num[jax.Array, "B T 1 R"], jt.Num[jax.Array, "B T 1 R"]], splash_kernel: Any,
    *, num_experts: int, num_experts_per_tok: int, routed_scaling_factor: float, n_routing_groups: int,
    topk_routing_group: int, topk_in_group: int, qk_head_dim: int, rope_head_dim: int, num_query_heads: int,
    mscale: float, kv_lora_rank: int, max_position_embeddings: int, original_max_position_embeddings: int,
    rope_factor: int, mesh: jax.sharding.Mesh, norm_fn: NormFn, rope_fn: RopeFn, gmm_fn: GmmFn,
    axis_mapping: Mapping[str, str | tuple[str, ...]], expert_axis_name: str = "expert", capacity_factor: float = 2.0,
    router_dtype: jax.typing.DTypeLike | None = None, segment_ids: jt.Num[jax.Array, "B T"] | None = None,
    ragged_buffer_factor: float, quant: quantization.QuantConfig | None = None,
    expert_permutations: jt.Int[jax.Array, "E"] | None = None,
) -> tuple[jt.Num[jax.Array, "B T D"], dsv3_types.DSv3RouterAux[jax.Array]]:
  return _dsv3_sparse_layer_vjp(
      x, w, yarn_freqs, splash_kernel, segment_ids, expert_permutations, num_experts, num_experts_per_tok,
      routed_scaling_factor, n_routing_groups, topk_routing_group, topk_in_group, qk_head_dim, rope_head_dim,
      num_query_heads, mscale, kv_lora_rank, max_position_embeddings, original_max_position_embeddings,
      rope_factor, mesh, norm_fn, rope_fn, gmm_fn, axis_mapping, expert_axis_name, capacity_factor,
      router_dtype, ragged_buffer_factor, quant,
  )
