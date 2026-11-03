# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Differentiable 3D-layout expert GMMs for the DSv3 routed-expert MLP."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp

from lilax._backend.experimental.lineage.gmmv2 import gate_bwd_fused
from lilax._backend.experimental.lineage.gmmv2 import gmmv2_kernel
from lilax._backend.experimental.lineage.gmmv2 import linear_bwd_split
from lilax._backend.experimental.lineage.gmmv2 import linear_combine_fwd
from lilax._backend.experimental.lineage.gmmv2 import tgmm_kernel

TileSizes = gmmv2_kernel.TileSizes
LANES = 128


@dataclasses.dataclass(frozen=True)
class ExpertMlpConfig:
  """Static configuration for the six expert-MLP kernels."""

  gate_fwd: TileSizes | None = None
  gate_dx: TileSizes | None = None
  gate_dw: TileSizes | None = None
  linear_fwd: TileSizes | None = None
  linear_dact: TileSizes | None = None
  linear_dw: TileSizes | None = None
  disable_multi_core_mode: bool = False
  zero_initialize: bool = True
  flat_weights: bool = True
  fuse_gate_bwd: bool = True
  linear_bwd_dgrad: linear_bwd_split.DgradTiles = linear_bwd_split.DEFAULT_DGRAD_TILES
  linear_bwd_wgrad: linear_bwd_split.WgradTiles = linear_bwd_split.DEFAULT_WGRAD_TILES


def _check_3d(name: str, x: jax.Array) -> None:
  if x.ndim != 3 or x.shape[-1] != LANES:
    raise ValueError(f"{name} must be 3D [M, D // 128, 128], got shape {x.shape}.")
  if x.dtype != jnp.bfloat16 and not gmmv2_kernel.is_fp8(x.dtype):
    raise ValueError(f"{name} must be bfloat16 or fp8, got {x.dtype}.")


_FP8_OUT_DTYPE = jnp.bfloat16


def _gmm(cfg: ExpertMlpConfig, tiles: TileSizes | None, *args, **kwargs):
  if tiles is not None:
    kwargs["tile_info"] = tiles
  return gmmv2_kernel.gmm_v2(
      *args,
      disable_multi_core_mode=cfg.disable_multi_core_mode,
      zero_initialize=cfg.zero_initialize,
      flat_rhs=cfg.flat_weights,
      **kwargs,
  )


def _tgmm(cfg: ExpertMlpConfig, tiles: TileSizes | None, *args, **kwargs):
  if tiles is not None:
    kwargs["tile_info"] = tiles
  return tgmm_kernel.tgmm_v2(*args, disable_multi_core_mode=cfg.disable_multi_core_mode, **kwargs)


def _psum_axes(ct: jax.Array, primal: jax.Array) -> frozenset[str]:
  """Axes over which `_match_vma(ct, primal)` needs a real psum."""
  primal_mat = jax.typeof(primal).manual_axis_type
  return jax.typeof(ct).manual_axis_type.varying - primal_mat.varying - primal_mat.reduced


def _match_vma(ct: jax.Array, primal: jax.Array) -> jax.Array:
  """Gives cotangent `ct` the manual-axis type JAX expects for `primal`."""
  ct_mat = jax.typeof(ct).manual_axis_type
  primal_mat = jax.typeof(primal).manual_axis_type
  if ct_mat.unreduced or ct_mat.reduced:
    raise NotImplementedError(f"Kernel cotangent has unreduced/reduced axes: {ct_mat}.")
  if primal_mat.unreduced:
    raise NotImplementedError(f"Unreduced primal inputs are not supported: {primal_mat}.")
  want_varying = primal_mat.varying
  want_unreduced = primal_mat.reduced
  have_varying = ct_mat.varying
  if extra := _psum_axes(ct, primal):
    ct = jax.lax.psum(ct, tuple(sorted(extra)))
  if want_unreduced:
    missing_var = want_unreduced - have_varying
    if missing_var:
      ct = jax.lax.pcast(ct, tuple(sorted(missing_var)), to="varying")
    ct = jax.lax.pcast(ct, tuple(sorted(want_unreduced)), to="unreduced")
  if missing := want_varying - jax.typeof(ct).manual_axis_type.varying:
    ct = jax.lax.pcast(ct, tuple(sorted(missing)), to="varying")
  return ct


def _tgmm_dw(
    cfg: ExpertMlpConfig,
    tiles: TileSizes | None,
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    w: jax.Array,
    dw_acc: jax.Array | None,
    dw_dtype: jnp.dtype | None = None,
    **kwargs,
) -> jax.Array:
  """Returns `dw_acc` plus the weight gradient `lhs^T @ rhs` of `w`."""
  tgmm = functools.partial(
      _tgmm, cfg, tiles, lhs, rhs, group_sizes, w.shape[0],
      preferred_element_type=w.dtype if dw_dtype is None else dw_dtype, **kwargs,
  )
  if dw_acc is not None and cfg.disable_multi_core_mode and not _psum_axes(lhs, w):
    return tgmm(acc=dw_acc)
  dw = _match_vma(tgmm(), w)
  return dw if dw_acc is None else dw_acc + dw


def _gate_fwd_kwargs(
    out: jax.Array,
    out_scale: jax.Array | None,
    glu_coeffs: jax.Array | None,
    glu_out_dtype: jnp.dtype | None,
    glu_out_scale: jax.Array | None,
) -> dict[str, Any]:
  """Returns the fp8 / fused GLU `gmm_v2` arguments of the gate GMM."""
  kwargs: dict[str, Any] = {}
  if out_scale is not None:
    kwargs.update(out_scale=out_scale, acc_dtype=jnp.float32, preferred_element_type=out.dtype)
  if glu_coeffs is not None:
    kwargs["glu_coeffs"] = glu_coeffs
    kwargs["glu_out_dtype"] = glu_out_dtype
    kwargs["glu_out_scale"] = glu_out_scale
  return kwargs


def gate_gmm(
    cfg: ExpertMlpConfig,
    x3d: jax.Array,
    w_gate: jax.Array,
    group_sizes: jax.Array,
    out: jax.Array,
    out_offset: jax.Array | int,
    out_scale: jax.Array | None = None,
    *,
    glu_coeffs: jax.Array | None = None,
    glu_out_dtype: jnp.dtype | None = None,
    glu_out_scale: jax.Array | None = None,
) -> Any:
  """Writes x[m] @ w_gate[expert(m)] to out[out_offset + m], x in 3D layout."""
  _check_3d("x3d", x3d)
  return _gmm(
      cfg, cfg.gate_fwd, x3d, w_gate, group_sizes, out=out, out_offset=out_offset,
      **_gate_fwd_kwargs(out, out_scale, glu_coeffs, glu_out_dtype, glu_out_scale),
  )


def gather_gate_gmm(
    cfg: ExpertMlpConfig,
    x3d: jax.Array,
    indices: jax.Array,
    start: jax.Array | int,
    w_gate: jax.Array,
    group_sizes: jax.Array,
    out: jax.Array,
    out_offset: jax.Array | int,
    out_scale: jax.Array | None = None,
    *,
    size_m: int,
    glu_coeffs: jax.Array | None = None,
    glu_out_dtype: jnp.dtype | None = None,
    glu_out_scale: jax.Array | None = None,
) -> Any:
  """`gate_gmm` of the rows `x3d[indices[start + m]]`, m < size_m."""
  _check_3d("x3d", x3d)
  return _gmm(
      cfg, cfg.gate_fwd, x3d, w_gate, group_sizes, out=out, out_offset=out_offset,
      lhs_gather_indices=indices, lhs_gather_start=start, lhs_gather_size_m=size_m,
      **_gate_fwd_kwargs(out, out_scale, glu_coeffs, glu_out_dtype, glu_out_scale),
  )


def gate_gmm_bwd(
    cfg: ExpertMlpConfig,
    x3d: jax.Array,
    w_gate: jax.Array,
    group_sizes: jax.Array,
    dgate: jax.Array,
    dw_acc: jax.Array | None = None,
    *,
    w_q: jax.Array | None = None,
    dgate_scale: jax.Array | None = None,
    dgate_qtype: jnp.dtype | None = None,
    dgate_tile_exps: jax.Array | None = None,
    dgrad_tile_rows: int | None = None,
    dx_scale: jax.Array | None = None,
    dw_scale: jax.Array | None = None,
    dw_dtype: jnp.dtype | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Backward pass for gate_gmm given the cotangent of its BT output rows."""
  dx_kwargs: dict[str, Any]
  dw_kwargs: dict[str, Any]
  if w_q is None:
    dgate = dgate.astype(x3d.dtype)
    dx_kwargs, dw_kwargs = {}, {}
  else:
    target_dw_dtype = w_gate.dtype if dw_dtype is None else dw_dtype
    if (
        dgate_scale is None and cfg.disable_multi_core_mode and cfg.fuse_gate_bwd
        and dx_scale is not None and dw_scale is not None
        and gate_bwd_fused.is_supported(x3d, w_q, group_sizes, dgate, dw_dtype=target_dw_dtype)
    ):
      can_alias_acc = dw_acc is not None and not _psum_axes(x3d, w_gate)
      dx3d, dw_gate = gate_bwd_fused.gate_bwd_fused(
          x3d, w_q, group_sizes, dgate, dx_scale=dx_scale, dw_scale=dw_scale,
          dgate_tile_exps=dgate_tile_exps,
          dgrad_tile_rows=(cfg.linear_bwd_dgrad.tile_rows if dgrad_tile_rows is None else dgrad_tile_rows),
          dw_acc=dw_acc if can_alias_acc else None, dw_dtype=target_dw_dtype,
      )
      if not can_alias_acc:
        dw_gate = _match_vma(dw_gate, w_gate)
        if dw_acc is not None:
          dw_gate = dw_acc + dw_gate
      return _match_vma(dx3d, x3d), dw_gate
    dx_kwargs = dict(out_scale=dx_scale, acc_dtype=jnp.float32, preferred_element_type=_FP8_OUT_DTYPE)
    dw_kwargs = dict(out_scale=dw_scale, acc_dtype=jnp.float32)
    if dgate_scale is not None:
      dx_kwargs["lhs_scale"] = dgate_scale
      dx_kwargs["lhs_quant_dtype"] = dgate_qtype
      dw_kwargs["rhs_quant_scale"] = dgate_scale
      dw_kwargs["rhs_quant_dtype"] = dgate_qtype
  dx3d = _gmm(
      cfg, cfg.gate_dx, dgate, w_gate if w_q is None else w_q, group_sizes,
      transpose_rhs=True, out_is_3d=True, **dx_kwargs,
  )
  dw_gate = _tgmm_dw(cfg, cfg.gate_dw, x3d, dgate, group_sizes, w_gate, dw_acc, dw_dtype, **dw_kwargs)
  return _match_vma(dx3d, x3d), dw_gate


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def linear_gmm(cfg: ExpertMlpConfig, act: jax.Array, w_linear: jax.Array, group_sizes: jax.Array) -> jax.Array:
  """out[m] = act[m] @ w_linear[expert(m)], produced in the 3D layout."""
  return _linear_gmm_fwd(cfg, act, w_linear, group_sizes)[0]


def _linear_gmm_fwd(cfg, act, w_linear, group_sizes):
  if act.ndim != 2:
    raise ValueError(f"act must be 2D [M, F], got shape {act.shape}.")
  out3d = _gmm(cfg, cfg.linear_fwd, act, w_linear, group_sizes, out_is_3d=True)
  return out3d, (act, w_linear, group_sizes)


def linear_gmm_fp8(
    cfg: ExpertMlpConfig,
    act: jax.Array,
    w_q: jax.Array,
    group_sizes: jax.Array,
    *,
    act_scale: jax.Array,
    act_qtype: jnp.dtype,
    out_scale: jax.Array,
) -> jax.Array:
  """fp8 `linear_gmm` (forward only; see `linear_gmm_bwd(w_q=...)`)."""
  if act.ndim != 2:
    raise ValueError(f"act must be 2D [M, F], got shape {act.shape}.")
  if gmmv2_kernel.is_fp8(act.dtype):
    return _gmm(
        cfg, cfg.linear_fwd, act, w_q, group_sizes, out_scale=out_scale * act_scale,
        acc_dtype=jnp.float32, preferred_element_type=_FP8_OUT_DTYPE, out_is_3d=True,
    )
  return _gmm(
      cfg, cfg.linear_fwd, act, w_q, group_sizes, lhs_scale=act_scale, lhs_quant_dtype=act_qtype,
      out_scale=out_scale, acc_dtype=jnp.float32, preferred_element_type=_FP8_OUT_DTYPE, out_is_3d=True,
  )


def linear_gmm_bwd(
    cfg: ExpertMlpConfig,
    act: jax.Array,
    w_linear: jax.Array,
    group_sizes: jax.Array,
    dout3d: jax.Array,
    dw_acc: jax.Array | None = None,
    *,
    w_q: jax.Array | None = None,
    act_scale: jax.Array | None = None,
    act_qtype: jnp.dtype | None = None,
    dout_scale: jax.Array | None = None,
    dout_qtype: jnp.dtype | None = None,
    dact_scale: jax.Array | None = None,
    dw_scale: jax.Array | None = None,
    dw_dtype: jnp.dtype | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Backward pass for linear_gmm given the cotangent of its output."""
  dact_kwargs: dict[str, Any]
  dw_kwargs: dict[str, Any]
  if w_q is None:
    dout3d = dout3d.astype(act.dtype)
    dact_kwargs, dw_kwargs = {}, {}
  else:
    dact_kwargs = dict(out_scale=dact_scale, acc_dtype=jnp.float32, preferred_element_type=_FP8_OUT_DTYPE)
    if gmmv2_kernel.is_fp8(act.dtype):
      assert act_scale is not None
      dw_out_scale = act_scale if dw_scale is None else dw_scale * act_scale
      dw_kwargs = dict(out_scale=dw_out_scale, acc_dtype=jnp.float32)
    else:
      dw_kwargs = dict(lhs_scale=act_scale, lhs_quant_dtype=act_qtype, out_scale=dw_scale, acc_dtype=jnp.float32)
    if dout_scale is not None:
      dact_kwargs["lhs_scale"] = dout_scale
      dact_kwargs["lhs_quant_dtype"] = dout_qtype
      dw_kwargs["rhs_quant_scale"] = dout_scale
      dw_kwargs["rhs_quant_dtype"] = dout_qtype
  _check_3d("dout3d", dout3d)
  dact = _gmm(
      cfg, cfg.linear_dact, dout3d, w_linear if w_q is None else w_q, group_sizes,
      transpose_rhs=True, **dact_kwargs,
  )
  dw_linear = _tgmm_dw(cfg, cfg.linear_dw, act, dout3d, group_sizes, w_linear, dw_acc, dw_dtype, **dw_kwargs)
  return _match_vma(dact, act), dw_linear


def _linear_gmm_bwd(cfg, res, dout3d):
  act, w_linear, group_sizes = res
  dact, dw_linear = linear_gmm_bwd(cfg, act, w_linear, group_sizes, dout3d)
  return dact, dw_linear, None


linear_gmm.defvjp(_linear_gmm_fwd, _linear_gmm_bwd)


def linear_gmm_bwd_gather(
    cfg: ExpertMlpConfig,
    dout_ag: jax.Array,
    indices: jax.Array,
    start: jax.Array | int,
    w_linear: jax.Array,
    w_q: jax.Array,
    group_sizes: jax.Array,
    bank: jax.Array,
    bank_row: jax.Array | int,
    coeffs: jax.Array,
    dw_acc: jax.Array | None = None,
    *,
    act_scale: float,
    act_qtype: jnp.dtype,
    dout_qtype: jnp.dtype,
    dout_scale: jax.Array | None = None,
    quantize_grad_gate: bool | None = None,
    w_scale: float,
    dw_dtype: jnp.dtype | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array | None, jax.Array, jax.Array]:
  """FP8 backward of SiLU + linear_gmm of the gathered rows of `dout_ag`."""
  dw_dtype = w_linear.dtype if dw_dtype is None else jnp.dtype(dw_dtype)
  if quantize_grad_gate is None:
    quantize_grad_gate = cfg.disable_multi_core_mode and cfg.fuse_gate_bwd
  dgrad = linear_bwd_split.linear_bwd_dgrad(
      dout_ag, indices, start, w_q, group_sizes, bank, bank_row, coeffs,
      act_scale=act_scale, w_scale=w_scale, act_qtype=act_qtype, dy_qtype=dout_qtype,
      dy_scale=dout_scale, grad_gate_qtype=dout_qtype if quantize_grad_gate else None,
      tiles=cfg.linear_bwd_dgrad,
  )
  can_alias_acc = dw_acc is not None and not _psum_axes(dgrad.act_q, w_linear)
  dw = linear_bwd_split.linear_bwd_wgrad(
      dgrad.act_q, dgrad.dy_q, dgrad.tile_exps, group_sizes, act_scale=act_scale, dy_scale=dout_scale,
      dgrad_tile_rows=cfg.linear_bwd_dgrad.tile_rows, dw_acc=dw_acc if can_alias_acc else None,
      dw_dtype=dw_dtype, tiles=cfg.linear_bwd_wgrad,
  )
  if not can_alias_acc:
    dw = _match_vma(dw, w_linear)
    if dw_acc is not None:
      dw = dw_acc + dw
  return dgrad.grad_gate, dgrad.grad_gate_amax, dgrad.grad_gate_tile_exps, dgrad.grad_coeffs, dw


@dataclasses.dataclass(frozen=True)
class ExpertGmm3d:
  """3D-layout expert GMMs implementing `dsv3_experts.GmmFn`."""

  cfg: ExpertMlpConfig = ExpertMlpConfig(disable_multi_core_mode=True)
  supports_fused_glu: bool = True
  supports_dynamic_grad_scale: bool = True

  def gate(
      self, x: jax.Array, w: jax.Array, group_sizes: jax.Array, out: jax.Array, out_offset: jax.Array | int,
      out_scale: jax.Array | None = None, *, glu_coeffs: jax.Array | None = None,
      glu_out_dtype: jnp.dtype | None = None, glu_out_scale: jax.Array | None = None,
  ) -> Any:
    return gate_gmm(
        self.cfg, x, w, group_sizes, out, out_offset, out_scale=out_scale, glu_coeffs=glu_coeffs,
        glu_out_dtype=glu_out_dtype, glu_out_scale=glu_out_scale,
    )

  def gather_gate(
      self, x: jax.Array, indices: jax.Array, start: jax.Array | int, w: jax.Array, group_sizes: jax.Array,
      out: jax.Array, out_offset: jax.Array | int, out_scale: jax.Array | None = None, *, size_m: int,
      glu_coeffs: jax.Array | None = None, glu_out_dtype: jnp.dtype | None = None,
      glu_out_scale: jax.Array | None = None,
  ) -> Any:
    return gather_gate_gmm(
        self.cfg, x, indices, start, w, group_sizes, out, out_offset, out_scale=out_scale, size_m=size_m,
        glu_coeffs=glu_coeffs, glu_out_dtype=glu_out_dtype, glu_out_scale=glu_out_scale,
    )

  def gate_bwd(
      self, x: jax.Array, w: jax.Array, group_sizes: jax.Array, dgate: jax.Array,
      dw_acc: jax.Array | None = None, *, w_q: jax.Array | None = None, dgate_scale: jax.Array | None = None,
      dgate_qtype: jnp.dtype | None = None, dgate_tile_exps: jax.Array | None = None,
      dgrad_tile_rows: int | None = None, dx_scale: jax.Array | None = None, dw_scale: jax.Array | None = None,
      dw_dtype: jnp.dtype | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    return gate_gmm_bwd(
        self.cfg, x, w, group_sizes, dgate, dw_acc, w_q=w_q, dgate_scale=dgate_scale, dgate_qtype=dgate_qtype,
        dgate_tile_exps=dgate_tile_exps,
        dgrad_tile_rows=(self.cfg.linear_bwd_dgrad.tile_rows if dgrad_tile_rows is None else dgrad_tile_rows),
        dx_scale=dx_scale, dw_scale=dw_scale, dw_dtype=dw_dtype,
    )

  def linear_bwd(
      self, x: jax.Array, w: jax.Array, group_sizes: jax.Array, dout: jax.Array, dw_acc: jax.Array | None = None,
      *, w_q: jax.Array | None = None, x_scale: jax.Array | None = None, x_qtype: jnp.dtype | None = None,
      dout_scale: jax.Array | None = None, dout_qtype: jnp.dtype | None = None, dx_scale: jax.Array | None = None,
      dw_scale: jax.Array | None = None, dw_dtype: jnp.dtype | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    return linear_gmm_bwd(
        self.cfg, x, w, group_sizes, dout, dw_acc, w_q=w_q, act_scale=x_scale, act_qtype=x_qtype,
        dout_scale=dout_scale, dout_qtype=dout_qtype, dact_scale=dx_scale, dw_scale=dw_scale, dw_dtype=dw_dtype,
    )

  def linear_combine(
      self, x: jax.Array, w: jax.Array, group_sizes: jax.Array, token_indices: jax.Array,
      start: jax.Array | int, sorted_tokens: jax.Array, *, out_scale: jax.Array, num_out_tokens: int,
      out: jax.Array | None = None,
  ) -> jax.Array:
    """fp8 `linear` of x fused with the top-k combine of its rows."""
    assert self.cfg.disable_multi_core_mode
    return linear_combine_fwd.linear_combine_fwd(
        x, w, group_sizes, token_indices, start, None if out is not None else sorted_tokens,
        out_scale=out_scale, num_out_tokens=num_out_tokens, out=out,
    )

  def linear_bwd_gather(
      self, dout_ag: jax.Array, indices: jax.Array, start: jax.Array | int, w: jax.Array,
      group_sizes: jax.Array, bank: jax.Array, bank_row: jax.Array | int, coeffs: jax.Array,
      dw_acc: jax.Array | None = None, *, w_q: jax.Array, x_scale: float, x_qtype: jnp.dtype,
      dout_qtype: jnp.dtype, dout_scale: jax.Array | None = None, quantize_grad_gate: bool | None = None,
      w_scale: float, dw_dtype: jnp.dtype | None = None,
  ) -> tuple[jax.Array, jax.Array, jax.Array | None, jax.Array, jax.Array]:
    """FP8 SiLU + `linear` backward of gathered cotangent rows."""
    return linear_gmm_bwd_gather(
        self.cfg, dout_ag, indices, start, w, w_q, group_sizes, bank, bank_row, coeffs, dw_acc,
        act_scale=x_scale, act_qtype=x_qtype, dout_qtype=dout_qtype, dout_scale=dout_scale,
        quantize_grad_gate=quantize_grad_gate, w_scale=w_scale, dw_dtype=dw_dtype,
    )

  def linear(
      self, x: jax.Array, w: jax.Array, group_sizes: jax.Array, *, x_scale: jax.Array | None = None,
      x_qtype: jnp.dtype | None = None, out_scale: jax.Array | None = None,
  ) -> jax.Array:
    if out_scale is None:
      return linear_gmm(self.cfg, x, w, group_sizes)
    assert x_scale is not None and x_qtype is not None
    return linear_gmm_fp8(self.cfg, x, w, group_sizes, act_scale=x_scale, act_qtype=x_qtype, out_scale=out_scale)


DSV3_V7X_CONFIG = ExpertMlpConfig(
    gate_fwd=TileSizes(tile_m=768, tile_k=1024, tile_n=4096, bucket_base=256),
    gate_dx=TileSizes(tile_m=1024, tile_k=4096, tile_n=1024, bucket_base=128),
    gate_dw=TileSizes(tile_m=768, tile_k=1024, tile_n=4096, bucket_base=256),
    linear_fwd=TileSizes(tile_m=1024, tile_k=2048, tile_n=1024, bucket_base=128),
    linear_dact=TileSizes(tile_m=768, tile_k=1024, tile_n=2048, bucket_base=256),
    linear_dw=TileSizes(tile_m=1024, tile_k=2048, tile_n=1024, bucket_base=256),
    disable_multi_core_mode=True,
    zero_initialize=False,
)
_DSV3_EMB_DIM = 7168


def make_gmm_fn(emb_dim: int) -> ExpertGmm3d:
  """Returns the `gmm_fn` for the DSv3 model and MTP layers."""
  if emb_dim == _DSV3_EMB_DIM:
    return ExpertGmm3d(DSV3_V7X_CONFIG)
  return ExpertGmm3d()
