"""DeepSeek MoE experts implementation."""

import functools
from typing import Protocol

import jax
from jax.experimental import layout as jax_layout
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

from lilax._backend.experimental.lineage import dsv3_types
from lilax._backend.experimental.lineage import ops
from lilax._backend.experimental.lineage import quantization


class GmmFn(Protocol):
  """Per-expert grouped matmuls on the three-dimensional token layout."""

  def gate(
      self, x: jt.Num[jax.Array, "BT D0 D1"], w: jt.Num[jax.Array, "E D F"],
      group_sizes: jt.Num[jax.Array, "E"], out: jt.Num[jax.Array, "R F"],
      out_offset: jt.Int[jax.Array, ""] | int, out_scale: jt.Float[jax.Array, "1 1"] | None = None,
  ) -> jt.Num[jax.Array, "R F"]:
    ...

  def gate_bwd(
      self, x: jt.Num[jax.Array, "BT D0 D1"], w: jt.Num[jax.Array, "E D F"],
      group_sizes: jt.Num[jax.Array, "E"], dgate: jt.Num[jax.Array, "BT F"],
      dw_acc: jt.Num[jax.Array, "E D F"] | None = None, *, w_q: jt.Num[jax.Array, "E D F"] | None = None,
      dx_scale: jt.Float[jax.Array, "1 1"] | None = None, dw_scale: jt.Float[jax.Array, "1 1"] | None = None,
      dw_dtype: jnp.dtype | None = None,
  ) -> tuple[jt.Num[jax.Array, "BT D0 D1"], jt.Num[jax.Array, "E D F"]]:
    ...

  def linear(
      self, x: jt.Num[jax.Array, "BT F"], w: jt.Num[jax.Array, "E F D"],
      group_sizes: jt.Num[jax.Array, "E"], *, x_scale: jt.Float[jax.Array, "1 1"] | None = None,
      x_qtype: jnp.dtype | None = None, out_scale: jt.Float[jax.Array, "1 1"] | None = None,
  ) -> jt.Num[jax.Array, "BT D0 D1"]:
    ...

  def linear_bwd(
      self, x: jt.Num[jax.Array, "BT F"], w: jt.Num[jax.Array, "E F D"],
      group_sizes: jt.Num[jax.Array, "E"], dout: jt.Num[jax.Array, "BT D0 D1"],
      dw_acc: jt.Num[jax.Array, "E F D"] | None = None, *, w_q: jt.Num[jax.Array, "E F D"] | None = None,
      x_scale: jt.Float[jax.Array, "1 1"] | None = None, x_qtype: jnp.dtype | None = None,
      dx_scale: jt.Float[jax.Array, "1 1"] | None = None, dw_scale: jt.Float[jax.Array, "1 1"] | None = None,
      dw_dtype: jnp.dtype | None = None,
  ) -> tuple[jt.Num[jax.Array, "BT F"], jt.Num[jax.Array, "E F D"]]:
    ...


_BANK_ROW_ALIGN = 32


def bank_aligned(num_rows: jax.Array | int) -> jax.Array | int:
  return -(-num_rows // _BANK_ROW_ALIGN) * _BANK_ROW_ALIGN


def bank_chunk_row(
    bank: jt.Num[jax.Array, "bank_rows D_gate"], capacity: int, offset: jt.Int[jax.Array, ""],
    num_tokens: jt.Int[jax.Array, ""],
) -> tuple[jt.Int[jax.Array, ""], jt.Bool[jax.Array, ""]]:
  checkpoint_rows = bank.shape[0] - bank_aligned(capacity)
  is_overflow = (offset < 0) | (offset + num_tokens > checkpoint_rows)
  return jnp.where(is_overflow, checkpoint_rows, offset), is_overflow


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_shared_expert(
    x: jt.Num[jax.Array, "*BT D"], w: dsv3_types.DSv3MoESharedExpertWeightsPytree,
) -> jt.Num[jax.Array, "*BT D"]:
  assert w.gate_0 is not None
  assert w.gate_1 is not None
  assert w.linear is not None
  dot = functools.partial(jnp.tensordot, axes=1)
  return dot(jax.nn.silu(dot(x, w.gate_0)) * dot(x, w.gate_1), w.linear)


def _scale(value: float) -> jt.Float[jax.Array, "1 1"]:
  return jnp.full((1, 1), value, jnp.float32)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routed_experts_impl(
    x: jt.Num[jax.Array, "BT D0 D1"], w: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    group_sizes: jt.Num[jax.Array, "E"], sorted_coeffs: jt.Num[jax.Array, "BT 1"] | jt.Num[jax.Array, "BT"],
    bank: jt.Num[jax.Array, "bank_rows D_gate"], bank_offset: jt.Int[jax.Array, ""],
    *, gmm_fn: GmmFn, quant_rule: quantization.GmmQuantRule | None = None,
) -> tuple[jt.Num[jax.Array, "BT D0 D1"], jt.Num[jax.Array, "bank_rows D_gate"]]:
  assert w.gate is not None
  assert w.linear is not None
  if sorted_coeffs.ndim == 1:
    sorted_coeffs = sorted_coeffs[:, None]
  gate_scale = act_scale = w_scale = act_qtype = None
  if quant_rule is not None:
    sa, sw = quant_rule.act_scale, quant_rule.weight_scale
    gate_scale, act_scale, w_scale = _scale(sa * sw), _scale(sa), _scale(sw)
    act_qtype = quant_rule.act_qtype
  num_tokens = jnp.sum(group_sizes, dtype=jnp.int32)
  row, _ = bank_chunk_row(bank, x.shape[0], bank_offset, num_tokens)
  if quant_rule is not None and getattr(gmm_fn, "supports_fused_glu", False):
    with jax.named_scope("gate"):
      bank, act = getattr(gmm_fn, "gate")(
          x, w.gate, group_sizes, bank, row, out_scale=gate_scale, glu_coeffs=sorted_coeffs,
          glu_out_dtype=act_qtype, glu_out_scale=act_scale,
      )
  else:
    with jax.named_scope("gate"):
      bank = gmm_fn.gate(x, w.gate, group_sizes, bank, row, out_scale=gate_scale)
    with jax.named_scope("silu"):
      act = ops.ragged_silu_mul_fwd(
          bank, row, sorted_coeffs, num_tokens, block_size_tokens=1024, block_size_hidden=2048,
      )
  with jax.named_scope("linear"):
    routed_out = gmm_fn.linear(act, w.linear, group_sizes, x_scale=act_scale, x_qtype=act_qtype, out_scale=w_scale)
  return routed_out, bank


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routed_experts_gather_impl(
    x_ag: jt.Num[jax.Array, "NBT D0 D1"], token_indices: jt.Int[jax.Array, "num_indices"],
    start: jt.Int[jax.Array, ""] | int, w: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    group_sizes: jt.Num[jax.Array, "E"], sorted_coeffs: jt.Num[jax.Array, "BT 1"] | jt.Num[jax.Array, "BT"],
    bank: jt.Num[jax.Array, "bank_rows D_gate"], bank_offset: jt.Int[jax.Array, ""],
    reduce_metadata: ops.RaggedGatherReduceMetadata, y_ag_acc: jt.Num[jax.Array, "NBT D0 D1"] | None = None,
    *, gmm_fn: GmmFn, quant_rule: quantization.GmmQuantRule,
) -> tuple[jt.Num[jax.Array, "NBT D0 D1"], jt.Num[jax.Array, "bank_rows D_gate"]]:
  assert w.gate is not None
  assert w.linear is not None
  if sorted_coeffs.ndim == 1:
    sorted_coeffs = sorted_coeffs[:, None]
  sa, sw = quant_rule.act_scale, quant_rule.weight_scale
  capacity = sorted_coeffs.shape[0]
  num_tokens = jnp.sum(group_sizes, dtype=jnp.int32)
  row, _ = bank_chunk_row(bank, capacity, bank_offset, num_tokens)
  with jax.named_scope("gate"):
    bank, act = getattr(gmm_fn, "gather_gate")(
        x_ag, token_indices, start, w.gate, group_sizes, bank, row, out_scale=_scale(sa * sw), size_m=capacity,
        glu_coeffs=sorted_coeffs, glu_out_dtype=quant_rule.act_qtype, glu_out_scale=_scale(sa),
    )
  with jax.named_scope("linear_combine"):
    y_ag = getattr(gmm_fn, "linear_combine")(
        act, w.linear, group_sizes, token_indices, start, reduce_metadata.tokens,
        out_scale=_scale(sa * sw), num_out_tokens=x_ag.shape[0], out=y_ag_acc,
    )
  return y_ag, bank


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routed_experts(
    x: jt.Num[jax.Array, "BT D0 D1"], w: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    group_sizes: jt.Num[jax.Array, "E"], sorted_coeffs: jt.Num[jax.Array, "BT 1"] | jt.Num[jax.Array, "BT"],
    bank: jt.Num[jax.Array, "B_cap D_gate"], bank_offset: jt.Num[jax.Array, "num_token_shards"],
    *, gmm_fn: GmmFn, mesh: jax.sharding.Mesh,
) -> tuple[jt.Num[jax.Array, "BT D0 D1"], jt.Num[jax.Array, "B_cap D_gate"], jt.Num[jax.Array, "num_token_shards"]]:
  bank_pspec = jax.typeof(bank).sharding.spec
  offset_pspec = jax.typeof(bank_offset).sharding.spec

  def _fwd_push(x, w, group_sizes, sorted_coeffs, bank, bank_offset):
    routed_out, bank = dsv3_routed_experts_impl(x, w, group_sizes, sorted_coeffs, bank, bank_offset[0], gmm_fn=gmm_fn)
    num_tokens = jnp.sum(group_sizes, dtype=jnp.int32)
    return routed_out, bank, bank_offset + bank_aligned(num_tokens)

  return jax.shard_map(_fwd_push, mesh=mesh, out_specs=(jax.typeof(x).sharding.spec, bank_pspec, offset_pspec))(
      x, w, group_sizes, sorted_coeffs, bank, bank_offset,
  )


def _retrieve_or_remat_gate_out(
    routed: jt.Num[jax.Array, "BT D0 D1"], w_gate: jt.Num[jax.Array, "E D F"],
    group_sizes: jt.Num[jax.Array, "E"], bank: jt.Num[jax.Array, "bank_rows D_gate"],
    row: jt.Int[jax.Array, ""], is_overflow: jt.Bool[jax.Array, ""],
    *, gmm_fn: GmmFn, out_scale: jt.Float[jax.Array, "1 1"] | None,
) -> jt.Num[jax.Array, "bank_rows D_gate"]:
  @jax.named_call
  def _remat_gate_out(bank):
    with jax.named_scope("gate"):
      return gmm_fn.gate(routed, w_gate, group_sizes, bank, row, out_scale=out_scale)
  with jax.named_scope("retrieve_or_remat_gate_out"):
    return jax.lax.cond(is_overflow, _remat_gate_out, lambda b: b, bank)


def _gate_bwd(
    routed: jt.Num[jax.Array, "BT D0 D1"], w_gate: jt.Num[jax.Array, "E D F"],
    group_sizes: jt.Num[jax.Array, "E"], grad_gate_out: jt.Num[jax.Array, "BT F"],
    grad_gate_amax: jt.Float[jax.Array, "1 1"] | None, grad_w_acc: jt.Num[jax.Array, "E D F"] | None,
    *, gmm_fn: GmmFn, quant_rule: quantization.GmmQuantRule | None, w_gate_q: jt.Num[jax.Array, "E D F"] | None,
    num_tokens: jt.Int[jax.Array, ""], dgate_tile_exps: jt.Int[jax.Array, "max_gd"] | None = None,
) -> tuple[jt.Num[jax.Array, "BT D0 D1"], jt.Num[jax.Array, "E D F"]]:
  with jax.named_scope("gate"):
    dx_scale = dw_scale = dw_dtype = None
    extra_kwargs = {}
    if quant_rule is not None:
      if dgate_tile_exps is not None:
        dx_scale = _scale(quant_rule.weight_scale)
        dw_scale = _scale(quant_rule.act_scale)
        extra_kwargs["dgate_tile_exps"] = dgate_tile_exps
      else:
        if grad_gate_amax is not None:
          s_dg = quantization.absmax_to_scale(grad_gate_amax, quant_rule.bwd_qtype)
          grad_gate_out = quantization.quantize_with_scale(grad_gate_out, s_dg, quant_rule.bwd_qtype, num_tokens)
        else:
          grad_gate_out, s_dg = quantization.absmax_quantize(grad_gate_out, quant_rule.bwd_qtype, num_tokens)
        dx_scale = s_dg * quant_rule.weight_scale
        dw_scale = s_dg * quant_rule.act_scale
      dw_dtype = quant_rule.grad_dtype
    return gmm_fn.gate_bwd(
        routed, w_gate, group_sizes, grad_gate_out, grad_w_acc, w_q=w_gate_q,
        dx_scale=dx_scale, dw_scale=dw_scale, dw_dtype=dw_dtype, **extra_kwargs,
    )


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routed_experts_bwd_impl(
    grad_routed: jt.Num[jax.Array, "BT D0 D1"], routed: jt.Num[jax.Array, "BT D0 D1"],
    w_routed: dsv3_types.DSv3MoERoutedExpertWeightsPytree, group_sizes: jt.Num[jax.Array, "E"],
    sorted_coeffs: jt.Num[jax.Array, "BT 1"] | jt.Num[jax.Array, "BT"],
    bank: jt.Num[jax.Array, "bank_rows D_gate"], bank_offset: jt.Int[jax.Array, ""],
    grad_w_acc: dsv3_types.DSv3MoERoutedExpertWeightsPytree | None = None, *, gmm_fn: GmmFn,
    quant_rule: quantization.GmmQuantRule | None = None,
    w_routed_q: dsv3_types.DSv3MoERoutedExpertWeightsPytree | None = None,
    grad_routed_amax: jt.Float[jax.Array, "1 1"] | None = None,
) -> tuple[
    jt.Num[jax.Array, "BT D0 D1"], dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    jt.Num[jax.Array, "BT 1"] | jt.Num[jax.Array, "BT"], jt.Num[jax.Array, "bank_rows D_gate"],
]:
  """Runs the routed expert backward pass using the activation bank."""
  assert w_routed.gate is not None
  assert w_routed.linear is not None
  orig_coeffs_shape = sorted_coeffs.shape
  if sorted_coeffs.ndim == 1:
    sorted_coeffs = sorted_coeffs[:, None]
  if grad_w_acc is None:
    grad_w_acc = dsv3_types.DSv3MoERoutedExpertWeightsPytree()
  num_tokens = jnp.sum(group_sizes, dtype=jnp.int32)
  row, is_overflow = bank_chunk_row(bank, routed.shape[0], bank_offset, num_tokens)
  w_gate_q = w_linear_q = gate_scale = act_scale = w_scale = act_qtype = dw_dtype = None
  if quant_rule is not None:
    assert w_routed_q is not None
    sa, sw = quant_rule.act_scale, quant_rule.weight_scale
    w_gate_q, w_linear_q = w_routed_q.gate, w_routed_q.linear
    gate_scale, act_scale, w_scale = _scale(sa * sw), _scale(sa), _scale(sw)
    act_qtype = quant_rule.act_qtype
    dw_dtype = quant_rule.grad_dtype
  bank = _retrieve_or_remat_gate_out(
      routed, w_routed.gate if w_gate_q is None else w_gate_q, group_sizes, bank, row, is_overflow,
      gmm_fn=gmm_fn, out_scale=gate_scale,
  )
  fuse_glu = quant_rule is not None and getattr(gmm_fn, "supports_fused_glu", False)
  dyn_grad_scale = getattr(gmm_fn, "supports_dynamic_grad_scale", False)
  with jax.named_scope("silu"):
    act = ops.ragged_silu_mul_fwd(
        bank, row, sorted_coeffs, num_tokens, block_size_tokens=1024, block_size_hidden=2048,
        out_dtype=act_qtype if fuse_glu else None, out_scale=act_scale if fuse_glu else None,
    )
  with jax.named_scope("linear"):
    if quant_rule is not None and dyn_grad_scale:
      if grad_routed_amax is None:
        grad_routed_amax = quantization.ragged_absmax(grad_routed, num_tokens)
      s_dy = quantization.absmax_to_scale(grad_routed_amax, quant_rule.bwd_qtype)
      grad_act, grad_w_linear = getattr(gmm_fn, "linear_bwd")(
          act, w_routed.linear, group_sizes, grad_routed, grad_w_acc.linear, w_q=w_linear_q,
          x_scale=act_scale, x_qtype=act_qtype, dout_scale=s_dy, dout_qtype=quant_rule.bwd_qtype,
          dx_scale=w_scale, dw_scale=None, dw_dtype=dw_dtype,
      )
    else:
      dact_scale = dw_linear_scale = None
      if quant_rule is not None:
        grad_routed, s_dy = quantization.absmax_quantize(grad_routed, quant_rule.bwd_qtype, num_tokens)
        dact_scale, dw_linear_scale = s_dy * quant_rule.weight_scale, s_dy
      grad_act, grad_w_linear = gmm_fn.linear_bwd(
          act, w_routed.linear, group_sizes, grad_routed, grad_w_acc.linear, w_q=w_linear_q,
          x_scale=act_scale, x_qtype=act_qtype, dx_scale=dact_scale, dw_scale=dw_linear_scale, dw_dtype=dw_dtype,
      )
    act_max = None
    if quant_rule is not None:
      act_max = quant_rule.act_scale * float(jnp.finfo(act_qtype).max)
      if not fuse_glu:
        grad_act = jnp.where(jnp.abs(act) <= act_max, grad_act, 0).astype(grad_act.dtype)
  with jax.named_scope("silu"):
    grad_gate_amax = None
    if quant_rule is not None and dyn_grad_scale:
      grad_gate_out, grad_sorted_coeffs, grad_gate_amax = ops.ragged_silu_mul_bwd(
          grad_act, bank, row, sorted_coeffs, num_tokens, block_size_tokens=1024, block_size_hidden=1024,
          g_mask_max=act_max if fuse_glu else None, with_absmax=True,
      )
    else:
      grad_gate_out, grad_sorted_coeffs = ops.ragged_silu_mul_bwd(
          grad_act, bank, row, sorted_coeffs, num_tokens, block_size_tokens=1024, block_size_hidden=1024,
          g_mask_max=act_max if fuse_glu else None,
      )
  grad_routed_in, grad_w_gate = _gate_bwd(
      routed, w_routed.gate, group_sizes, grad_gate_out, grad_gate_amax, grad_w_acc.gate,
      gmm_fn=gmm_fn, quant_rule=quant_rule, w_gate_q=w_gate_q, num_tokens=num_tokens,
  )
  grad_w_routed = dsv3_types.DSv3MoERoutedExpertWeightsPytree(gate=grad_w_gate, linear=grad_w_linear)
  grad_sorted_coeffs = jnp.reshape(grad_sorted_coeffs, orig_coeffs_shape)
  return grad_routed_in, grad_w_routed, grad_sorted_coeffs, bank


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routed_experts_gather_bwd_impl(
    grad_y_ag: jt.Num[jax.Array, "NBT D0 D1"], token_indices: jt.Int[jax.Array, "num_indices"],
    start: jt.Int[jax.Array, ""] | int, routed: jt.Num[jax.Array, "BT D0 D1"],
    w_routed: dsv3_types.DSv3MoERoutedExpertWeightsPytree, group_sizes: jt.Num[jax.Array, "E"],
    sorted_coeffs: jt.Num[jax.Array, "BT 1"] | jt.Num[jax.Array, "BT"],
    bank: jt.Num[jax.Array, "bank_rows D_gate"], bank_offset: jt.Int[jax.Array, ""],
    grad_w_acc: dsv3_types.DSv3MoERoutedExpertWeightsPytree | None = None, *, gmm_fn: GmmFn,
    quant_rule: quantization.GmmQuantRule, w_routed_q: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    grad_y_scale: jt.Float[jax.Array, "1 1"] | None = None,
) -> tuple[
    jt.Num[jax.Array, "BT D0 D1"], dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    jt.Num[jax.Array, "BT 1"] | jt.Num[jax.Array, "BT"], jt.Num[jax.Array, "bank_rows D_gate"],
]:
  """Runs FP8 expert backward with in-kernel gathering and activation replay."""
  assert w_routed.gate is not None and w_routed.linear is not None
  assert w_routed_q.gate is not None and w_routed_q.linear is not None
  orig_coeffs_shape = sorted_coeffs.shape
  if sorted_coeffs.ndim == 1:
    sorted_coeffs = sorted_coeffs[:, None]
  if grad_w_acc is None:
    grad_w_acc = dsv3_types.DSv3MoERoutedExpertWeightsPytree()
  sa, sw = quant_rule.act_scale, quant_rule.weight_scale
  num_tokens = jnp.sum(group_sizes, dtype=jnp.int32)
  row, is_overflow = bank_chunk_row(bank, routed.shape[0], bank_offset, num_tokens)
  bank = _retrieve_or_remat_gate_out(
      routed, w_routed_q.gate, group_sizes, bank, row, is_overflow, gmm_fn=gmm_fn, out_scale=_scale(sa * sw),
  )
  with jax.named_scope("linear_silu"):
    grad_gate_out, grad_gate_amax, grad_gate_tile_exps, grad_sorted_coeffs, grad_w_linear = getattr(
        gmm_fn, "linear_bwd_gather"
    )(
        grad_y_ag, token_indices, start, w_routed.linear, group_sizes, bank, row, sorted_coeffs, grad_w_acc.linear,
        w_q=w_routed_q.linear, x_scale=sa, x_qtype=quant_rule.act_qtype, dout_qtype=quant_rule.bwd_qtype,
        dout_scale=grad_y_scale, w_scale=sw, dw_dtype=quant_rule.grad_dtype,
    )
  grad_routed_in, grad_w_gate = _gate_bwd(
      routed, w_routed.gate, group_sizes, grad_gate_out, grad_gate_amax, grad_w_acc.gate,
      gmm_fn=gmm_fn, quant_rule=quant_rule, w_gate_q=w_routed_q.gate,
      num_tokens=num_tokens, dgate_tile_exps=grad_gate_tile_exps,
  )
  grad_w_routed = dsv3_types.DSv3MoERoutedExpertWeightsPytree(gate=grad_w_gate, linear=grad_w_linear)
  grad_sorted_coeffs = jnp.reshape(grad_sorted_coeffs.astype(sorted_coeffs.dtype), orig_coeffs_shape)
  return grad_routed_in, grad_w_routed, grad_sorted_coeffs, bank


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routed_experts_bwd(
    grad_routed: jt.Num[jax.Array, "BT D0 D1"], routed: jt.Num[jax.Array, "BT D0 D1"],
    w_routed: dsv3_types.DSv3MoERoutedExpertWeightsPytree, group_sizes: jt.Num[jax.Array, "E"],
    sorted_coeffs: jt.Num[jax.Array, "BT 1"] | jt.Num[jax.Array, "BT"],
    ragged_bank: jt.Num[jax.Array, "B_cap D_gate"], bank_offset: jt.Num[jax.Array, "num_token_shards"],
    *, gmm_fn: GmmFn, mesh: jax.sharding.Mesh,
) -> tuple[
    jt.Num[jax.Array, "BT D0 D1"], dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    jt.Num[jax.Array, "B_cap D_gate"], jt.Num[jax.Array, "num_token_shards"],
    jt.Num[jax.Array, "BT 1"] | jt.Num[jax.Array, "BT"],
]:
  tiling = ((8, 128), (2, 1)) if jnp.dtype(ragged_bank.dtype).itemsize == 2 else ((8, 128),)
  bank_layout = jax_layout.Layout(major_to_minor=(0, 1))
  ragged_bank = jax_layout.with_layout_constraint(ragged_bank, bank_layout)
  x_pspec = jax.typeof(routed).sharding.spec
  offset_pspec = jax.typeof(bank_offset).sharding.spec
  out_specs = (
      x_pspec, jax.tree_util.tree_map(lambda g: jax.typeof(g).sharding.spec.to_ct_spec(), w_routed),
      jax.typeof(ragged_bank).sharding.spec, offset_pspec, jax.typeof(sorted_coeffs).sharding.spec,
  )

  def _bwd_pop(grad_routed, routed, w_routed, group_sizes, sorted_coeffs, ragged_bank, bank_offset):
    new_bank_offset = bank_offset - bank_aligned(jnp.sum(group_sizes, dtype=jnp.int32))
    grad_routed_in, grad_w_routed, grad_sorted_coeffs, ragged_bank = dsv3_routed_experts_bwd_impl(
        grad_routed, routed, w_routed, group_sizes, sorted_coeffs, ragged_bank, new_bank_offset[0], gmm_fn=gmm_fn,
    )
    return grad_routed_in, grad_w_routed, ragged_bank, new_bank_offset, grad_sorted_coeffs

  return jax.shard_map(_bwd_pop, mesh=mesh, out_specs=out_specs)(
      grad_routed, routed, w_routed, group_sizes, sorted_coeffs, ragged_bank, bank_offset,
  )
