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

"""Split Pallas kernels for the fp8 MLA QKV-up backward."""

import dataclasses
import functools
from typing import Any

import jax
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

_E5M2 = jnp.float8_e5m2
_VMEM_LIMIT_BYTES = int(0.9 * 64 * 1024 * 1024)
_NN = (((1,), (0,)), ((), ()))
_TN = (((0,), (0,)), ((), ()))


@dataclasses.dataclass(frozen=True)
class QkvUpBwdDims:
  """Static dimensions and constants of the QKV-up backward kernels."""

  qk_head_dim: int
  rope_head_dim: int
  v_head_dim: int
  softmax_scale: float
  eps: float
  act_scale: float
  act_qtype: Any
  weight_scale: float

  @property
  def q_head_dim(self) -> int:
    return self.qk_head_dim + self.rope_head_dim

  @property
  def act_max(self) -> float:
    return self.act_scale * float(jnp.finfo(self.act_qtype).max)


def _e5m2_scale(amax):
  return jnp.where(amax > 0, amax / float(jnp.finfo(_E5M2).max), 1.0)


def _to_e5m2(x, inv_scale):
  return (x.astype(jnp.float32) * inv_scale).astype(_E5M2)


def _abs_max(x):
  return jnp.max(jnp.max(jnp.abs(x), axis=0, keepdims=True).astype(jnp.float32))


def _rms_norm(x, scale, eps):
  x = x.astype(jnp.float32)
  mean2 = jnp.mean(x * x, axis=-1, keepdims=True)
  y = (x * jax.lax.rsqrt(mean2 + eps)).astype(jnp.bfloat16)
  return (y.astype(jnp.float32) * scale.astype(jnp.float32)).astype(jnp.bfloat16)


def _static_quantize(x, dims: QkvUpBwdDims):
  fmax = float(jnp.finfo(dims.act_qtype).max)
  return jnp.clip(x.astype(jnp.float32) / dims.act_scale, -fmax, fmax).astype(dims.act_qtype)


def _row_chunks(num_rows, chunk, body):
  chunk = min(chunk, num_rows)

  def loop_body(i, carry):
    body(pl.ds(pl.multiple_of(i * chunk, chunk), chunk))
    return carry

  jax.lax.fori_loop(0, num_rows // chunk, loop_body, 0)


def _dgrad_kernel(
    dq_ref, dk_ref, dv_ref, kv_down_ref, kv_norm_scale_ref, cos_sin_ref, wq_ref, wk_ref, wv_ref,
    grad_cq_ref, grad_kv_ref, gq_ref, dk_q_ref, dv_q_ref, scales_ref, rope_ref, acc_q_ref,
    acc_k_ref, acc_v_ref, acc_pe_ref, *, dims: QkvUpBwdDims, chunk: int,
):
  n = pl.program_id(2)
  num_heads, block_t = dq_ref.shape[:2]
  qk, r = dims.qk_head_dim, dims.rope_head_dim
  h = r // 2
  cs = cos_sin_ref[...].astype(jnp.float32)
  cos1, cos2, sin1, sin2 = (cs[:, i * h : (i + 1) * h] for i in range(4))
  amax_q = jnp.zeros((), jnp.float32)
  ropes = []
  for i in range(num_heads):
    amax_q = jnp.maximum(amax_q, _abs_max(dq_ref[i, :, :qk]))
    x = dq_ref[i, :, qk:].astype(jnp.float32)
    x1, x2 = x[:, :h], x[:, h:]
    rope = jnp.concatenate([x1 * cos1 + x2 * sin2, x2 * cos2 - x1 * sin1], axis=1).astype(jnp.bfloat16)
    amax_q = jnp.maximum(amax_q, _abs_max(rope))
    ropes.append(rope)
  rope_ref[...] = jnp.concatenate(ropes, axis=1)
  amax_k = jnp.zeros((), jnp.float32)
  amax_v = jnp.zeros((), jnp.float32)
  for i in range(num_heads):
    amax_k = jnp.maximum(amax_k, _abs_max(dk_ref[i, :, :qk]))
    amax_v = jnp.maximum(amax_v, _abs_max(dv_ref[i]))
  s_q, s_k, s_v = map(_e5m2_scale, (amax_q, amax_k, amax_v))
  inv_q, inv_k, inv_v = 1.0 / s_q, 1.0 / s_k, 1.0 / s_v
  row = jax.lax.broadcasted_iota(jnp.int32, scales_ref.shape, 0)
  scales_ref[...] = jnp.where(row == 0, s_q, jnp.where(row == 1, s_v, s_k))
  q_dequant = s_q * (dims.softmax_scale * dims.weight_scale)
  k_dequant = s_k * dims.weight_scale
  v_dequant = s_v * dims.weight_scale

  def accumulate(acc_ref, rows, x, cols=slice(None)):
    acc_ref[rows, cols] = jnp.where(n > 0, acc_ref[rows, cols], 0.0) + x

  def rows_body(rows):
    heads = []
    for i in range(num_heads):
      heads.append(dq_ref[i, rows, :qk].astype(jnp.float32))
      heads.append(rope_ref[rows, i * r : (i + 1) * r].astype(jnp.float32))
    gq_ref[rows] = _to_e5m2(jnp.concatenate(heads, axis=1), inv_q)
    accumulate(
        acc_q_ref, rows,
        jax.lax.dot_general(gq_ref[rows], wq_ref[...], _NN, preferred_element_type=jnp.float32) * q_dequant,
    )
    dk_q_ref[rows] = _to_e5m2(jnp.concatenate([dk_ref[i, rows, :qk] for i in range(num_heads)], axis=1), inv_k)
    accumulate(
        acc_k_ref, rows,
        jax.lax.dot_general(dk_q_ref[rows], wk_ref[...], _NN, preferred_element_type=jnp.float32) * k_dequant,
    )
    dk_pe = dk_ref[0, rows, qk:].astype(jnp.float32)
    for i in range(1, num_heads):
      dk_pe += dk_ref[i, rows, qk:].astype(jnp.float32)
    accumulate(acc_pe_ref, rows, dk_pe, slice(None, r))
    dv_q_ref[rows] = _to_e5m2(jnp.concatenate([dv_ref[i, rows] for i in range(num_heads)], axis=1), inv_v)
    accumulate(
        acc_v_ref, rows,
        jax.lax.dot_general(dv_q_ref[rows], wv_ref[...], _NN, preferred_element_type=jnp.float32) * v_dequant,
    )

  _row_chunks(block_t, chunk, rows_body)

  @pl.when(n == pl.num_programs(2) - 1)
  def _():
    ckv_dim = acc_k_ref.shape[1]

    def write_body(rows):
      ckv = _rms_norm(kv_down_ref[rows, :ckv_dim], kv_norm_scale_ref[...], dims.eps)
      keep = jnp.abs(ckv.astype(jnp.float32)) <= dims.act_max
      grad_ckv = jnp.where(keep, acc_k_ref[rows] + acc_v_ref[rows], 0.0)
      grad_cq_ref[:, rows] = acc_q_ref[rows].T.astype(grad_cq_ref.dtype)
      grad_kv_ref[:ckv_dim, rows] = grad_ckv.T.astype(grad_kv_ref.dtype)
      grad_kv_ref[ckv_dim:, rows] = acc_pe_ref[rows].T[:r].astype(grad_kv_ref.dtype)

    _row_chunks(block_t, chunk, write_body)


def _wgrad_kernel(
    q_down_ref, ckv_ref, q_norm_scale_ref, gq_ref, dk_q_ref, dv_q_ref, scales_ref,
    grad_wq_ref, grad_wk_ref, grad_wv_ref, cq_q_ref, acc_wq_ref, acc_wk_ref, acc_wv_ref,
    *, dims: QkvUpBwdDims, chunk: int, cache_cq: bool = True,
):
  n, b, t = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  block_t = dk_q_ref.shape[0]
  t_rows = pl.ds(pl.multiple_of(t * block_t, block_t), block_t)
  first = (b == 0) & (t == 0)
  last = (b == pl.num_programs(1) - 1) & (t == pl.num_programs(2) - 1)

  if cache_cq:
    @pl.when(n == 0)
    def stage_cq():
      def stage_body(rows):
        cache_rows = pl.ds(pl.multiple_of(t * block_t + rows.start, chunk), chunk)
        cq = _rms_norm(q_down_ref[rows], q_norm_scale_ref[...], dims.eps)
        cq_q_ref[b, cache_rows] = _static_quantize(cq, dims)

      _row_chunks(block_t, chunk, stage_body)

    cq_q = cq_q_ref[b, t_rows]
  else:
    def stage_body(rows):
      cq = _rms_norm(q_down_ref[rows], q_norm_scale_ref[...], dims.eps)
      cq_q_ref[rows] = _static_quantize(cq, dims)

    _row_chunks(block_t, chunk, stage_body)
    cq_q = cq_q_ref[...]

  def tn(x, y):
    return jax.lax.dot_general(x, y, _TN, preferred_element_type=jnp.float32)

  def accumulate(acc_ref, x):
    acc_ref[...] = jnp.where(first, 0.0, acc_ref[...]) + x

  s_q, s_v = scales_ref[0:1, 0:1], scales_ref[1:2, 0:1]
  s_k = scales_ref[2:3, 0:1]
  accumulate(acc_wq_ref, tn(gq_ref[...], cq_q) * (s_q * (dims.softmax_scale * dims.act_scale)))
  ckv_q = _static_quantize(ckv_ref[...], dims)
  accumulate(acc_wk_ref, tn(dk_q_ref[...], ckv_q) * (s_k * dims.act_scale))
  accumulate(acc_wv_ref, tn(dv_q_ref[...], ckv_q) * (s_v * dims.act_scale))

  @pl.when(last)
  def write_grads():
    grad_wq_ref[...] = acc_wq_ref[...].astype(grad_wq_ref.dtype)
    grad_wk_ref[...] = acc_wk_ref[...].astype(grad_wk_ref.dtype)
    grad_wv_ref[...] = acc_wv_ref[...].astype(grad_wv_ref.dtype)


def _out_struct(shape, dtype, like):
  return jax.ShapeDtypeStruct(shape, dtype, manual_axis_type=jax.typeof(like).manual_axis_type)


def _nbytes(*xs):
  return sum(x.size * x.dtype.itemsize for x in xs)


def qkv_up_dgrad(
    dq: jax.Array, dk: jax.Array, dv: jax.Array, kv_down: jax.Array, kv_norm_scale: jax.Array,
    cos_sin: jax.Array, wq_up: jax.Array, wk_up: jax.Array, wv_up: jax.Array, *,
    dims: QkvUpBwdDims, block_t: int, block_n: int, chunk: int = 256,
) -> tuple[jax.Array, ...]:
  """Compute head-unreduced activation gradients of the fp8 QKV-up projections."""
  batch, num_heads, num_tokens, d = dq.shape
  if num_tokens % block_t or num_heads % block_n or block_n % 2:
    raise ValueError(f"Bad {block_t=}, {block_n=} for {dq.shape=}.")
  qk, r, v = dims.qk_head_dim, dims.rope_head_dim, dims.v_head_dim
  cq_dim, ckv_dim, kv_width = wq_up.shape[1], wk_up.shape[1], kv_down.shape[2]
  grid = (batch, num_tokens // block_t, num_heads // block_n)

  def heads_spec(head_dim):
    return pl.BlockSpec((None, block_n, block_t, head_dim), lambda b, t, n: (b, n, t, 0))

  def tokens_spec(width):
    return pl.BlockSpec((None, block_t, width), lambda b, t, n: (b, t, 0))

  def head_block_spec(cols, head_dim):
    return pl.BlockSpec((block_n * head_dim, cols), lambda b, t, n: (n, 0))

  def tokens_t_spec(rows):
    return pl.BlockSpec((None, rows, block_t), lambda b, t, n: (b, 0, t))

  def tile_spec(head_dim):
    return pl.BlockSpec((None, block_t, block_n * head_dim), lambda b, t, n: (b, t, n))

  flops = 2 * batch * num_tokens * num_heads * (d * cq_dim + qk * ckv_dim + v * ckv_dim)
  bytes_accessed = (
      _nbytes(dq, dk, dv, kv_down, cos_sin) + batch * grid[1] * _nbytes(wq_up, wk_up, wv_up)
      + batch * num_tokens * (num_heads * (d + qk + v) + 2 * (cq_dim + kv_width))
  )
  return pl.pallas_call(
      functools.partial(_dgrad_kernel, dims=dims, chunk=chunk),
      out_shape=(
          _out_struct((batch, cq_dim, num_tokens), jnp.bfloat16, dq),
          _out_struct((batch, kv_width, num_tokens), jnp.bfloat16, dq),
          _out_struct((batch, num_tokens, num_heads * d), _E5M2, dq),
          _out_struct((batch, num_tokens, num_heads * qk), _E5M2, dq),
          _out_struct((batch, num_tokens, num_heads * v), _E5M2, dq),
          _out_struct((*grid, 8, 128), jnp.float32, dq),
      ),
      grid=grid,
      in_specs=[
          heads_spec(d), heads_spec(d), heads_spec(v), tokens_spec(kv_width),
          pl.BlockSpec((1, ckv_dim), lambda b, t, n: (0, 0)), tokens_spec(cos_sin.shape[2]),
          head_block_spec(cq_dim, d), head_block_spec(ckv_dim, qk), head_block_spec(ckv_dim, v),
      ],
      out_specs=(
          tokens_t_spec(cq_dim), tokens_t_spec(kv_width), tile_spec(d), tile_spec(qk), tile_spec(v),
          pl.BlockSpec((None, None, None, 8, 128), lambda b, t, n: (b, t, n, 0, 0)),
      ),
      scratch_shapes=[
          pltpu.VMEM((block_t, block_n * r), jnp.bfloat16), pltpu.VMEM((block_t, cq_dim), jnp.float32),
          pltpu.VMEM((block_t, ckv_dim), jnp.float32), pltpu.VMEM((block_t, ckv_dim), jnp.float32),
          pltpu.VMEM((block_t, 128), jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL, pltpu.ARBITRARY),
          vmem_limit_bytes=_VMEM_LIMIT_BYTES,
      ),
      cost_estimate=pl.CostEstimate(flops=flops, transcendentals=0, bytes_accessed=bytes_accessed),
      name="qkv_up_dgrad",
  )(dq, dk, dv, kv_down, kv_norm_scale, cos_sin, wq_up, wk_up, wv_up)


def qkv_up_wgrad(
    q_down: jax.Array, ckv: jax.Array, q_norm_scale: jax.Array, gq: jax.Array, dk_q: jax.Array,
    dv_q: jax.Array, scales: jax.Array, *, dims: QkvUpBwdDims, block_t: int, block_n: int,
    grad_dtype: Any = jnp.bfloat16, chunk: int = 256,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Compute weight gradients of the fp8 QKV-up projections, summed over tokens."""
  qk, v, d = dims.qk_head_dim, dims.v_head_dim, dims.q_head_dim
  batch, num_tokens = dk_q.shape[:2]
  num_heads = dk_q.shape[2] // qk
  cq_dim, ckv_dim = q_down.shape[2], ckv.shape[2]
  grid = (num_heads // block_n, batch, num_tokens // block_t)
  assert scales.shape[:3] == (batch, grid[2], grid[0]), scales.shape

  def tile_spec(head_dim):
    return pl.BlockSpec((None, block_t, block_n * head_dim), lambda n, b, t: (b, t, n))

  def grad_spec(cols, head_dim):
    return pl.BlockSpec((block_n * head_dim, cols), lambda n, b, t: (n, 0))

  cache_cq = batch * num_tokens * cq_dim <= 2 * 4096 * 1536
  if cache_cq:
    q_down_spec = pl.BlockSpec(
        (None, block_t, cq_dim),
        lambda n, b, t: (jnp.where(n == 0, b, batch - 1), jnp.where(n == 0, t, grid[2] - 1), 0),
        pipeline_mode=pl.Buffered(1),
    )
    cq_scratch = pltpu.VMEM((batch, num_tokens, cq_dim), dims.act_qtype)
  else:
    q_down_spec = pl.BlockSpec((None, block_t, cq_dim), lambda n, b, t: (b, t, 0))
    cq_scratch = pltpu.VMEM((block_t, cq_dim), dims.act_qtype)
  flops = 2 * batch * num_tokens * num_heads * (d * cq_dim + qk * ckv_dim + v * ckv_dim)
  bytes_accessed = (
      _nbytes(q_down, gq, dk_q, dv_q) + grid[0] * _nbytes(ckv)
      + (0 if cache_cq else (grid[0] - 1) * _nbytes(q_down))
      + num_heads * (d * cq_dim + (qk + v) * ckv_dim) * 2
  )
  chunk = min(chunk, block_t)
  return pl.pallas_call(
      functools.partial(_wgrad_kernel, dims=dims, chunk=chunk, cache_cq=cache_cq),
      out_shape=(
          _out_struct((num_heads * d, cq_dim), grad_dtype, dk_q),
          _out_struct((num_heads * qk, ckv_dim), grad_dtype, dk_q),
          _out_struct((num_heads * v, ckv_dim), grad_dtype, dk_q),
      ),
      grid=grid,
      in_specs=[
          q_down_spec, pl.BlockSpec((None, block_t, ckv_dim), lambda n, b, t: (b, t, 0)),
          pl.BlockSpec((1, cq_dim), lambda n, b, t: (0, 0)), tile_spec(d), tile_spec(qk), tile_spec(v),
          pl.BlockSpec((None, None, None, 8, 128), lambda n, b, t: (b, t, n, 0, 0)),
      ],
      out_specs=(grad_spec(cq_dim, d), grad_spec(ckv_dim, qk), grad_spec(ckv_dim, v)),
      scratch_shapes=[
          cq_scratch, pltpu.VMEM((block_n * d, cq_dim), jnp.float32),
          pltpu.VMEM((block_n * qk, ckv_dim), jnp.float32), pltpu.VMEM((block_n * v, ckv_dim), jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(pltpu.ARBITRARY, pltpu.ARBITRARY, pltpu.ARBITRARY),
          vmem_limit_bytes=_VMEM_LIMIT_BYTES,
      ),
      cost_estimate=pl.CostEstimate(flops=flops, transcendentals=0, bytes_accessed=bytes_accessed),
      name="qkv_up_wgrad",
  )(q_down, ckv, q_norm_scale, gq, dk_q, dv_q, scales)
