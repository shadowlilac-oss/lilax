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

"""Fused fp8 MLA q / kv up projection forward Pallas kernel."""

import functools
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def _rms_norm(x: jax.Array, scale: jax.Array, eps: float) -> jax.Array:
  x = x.astype(jnp.float32)
  mean2 = jnp.mean(x * x, axis=-1, keepdims=True)
  return x * jax.lax.rsqrt(mean2 + eps) * scale.astype(jnp.float32)


def _static_quantize(x: jax.Array, qtype: Any, scale: float) -> jax.Array:
  fmax = float(jnp.finfo(qtype).max)
  return jnp.clip(x.astype(jnp.float32) / scale, -fmax, fmax).astype(qtype)


def _nt_dot(x: jax.Array, y: jax.Array) -> jax.Array:
  return jax.lax.dot_general(x, y, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)


def _transpose(x: jax.Array) -> jax.Array:
  r = x.shape[1]
  eye = (jax.lax.broadcasted_iota(jnp.int32, (r, r), 0) == jax.lax.broadcasted_iota(jnp.int32, (r, r), 1)).astype(x.dtype)
  return _nt_dot(eye, x)


def _rotate_half_rows(x: jax.Array) -> jax.Array:
  half = x.shape[0] // 2
  return jnp.concatenate([-x[half:], x[:half]], axis=0)


def _qkv_up_kernel(
    q_down_ref, kv_down_ref, q_norm_scale_ref, kv_norm_scale_ref, cos_ref, sin_ref,
    wq_ref, wk_ref, wv_ref, q_ref, k_ref, v_ref, cq_q_scratch, ckv_q_scratch,
    cos_scratch, sin_scratch, k_pe_scratch, *, eps: float, act_qtype: Any,
    act_scale: float, dequant_scale: float, q_scale: float, qk_head_dim: int,
    kv_lora_rank: int, head_group: int,
):
  block_n, qk_rope_dim, block_t = q_ref.shape[1:]
  v_head_dim = v_ref.shape[3]
  rope_head_dim = cos_ref.shape[2]
  half = rope_head_dim // 2
  rows = pl.ds(pl.multiple_of(pl.program_id(2) * block_t, block_t), block_t)

  @pl.when(pl.program_id(1) == 0)
  def _():
    cq_q_scratch[rows, :] = _static_quantize(
        _rms_norm(q_down_ref[0], q_norm_scale_ref[...], eps), act_qtype, act_scale,
    )
    kv_down = kv_down_ref[0]
    ckv = _rms_norm(kv_down[:, :kv_lora_rank], kv_norm_scale_ref[...], eps)
    ckv_q_scratch[rows, :] = _static_quantize(ckv, act_qtype, act_scale)
    cos = _transpose(cos_ref[0])
    sin = _transpose(sin_ref[0])
    k_pe = _transpose(kv_down[:, kv_lora_rank:])
    k_pe_scratch[:, rows] = (k_pe * cos + _rotate_half_rows(k_pe) * sin).astype(k_pe_scratch.dtype)
    cos_scratch[:, rows] = cos * q_scale
    sin_scratch[:, rows] = sin * q_scale

  cq_q = cq_q_scratch[rows, :]
  ckv_q = ckv_q_scratch[rows, :]
  cos = cos_scratch[:, rows]
  sin = sin_scratch[:, rows]
  k_pe = k_pe_scratch[:, rows]

  for g in range(block_n // head_group):
    h0 = g * head_group
    q = _nt_dot(wq_ref[h0 * qk_rope_dim : (h0 + head_group) * qk_rope_dim, :], cq_q)
    q = q.astype(jnp.bfloat16).astype(jnp.float32)
    k = _nt_dot(wk_ref[h0 * qk_head_dim : (h0 + head_group) * qk_head_dim, :], ckv_q)
    v = jax.lax.dot_general(
        ckv_q, wv_ref[:, h0 * v_head_dim : (h0 + head_group) * v_head_dim],
        (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32,
    )
    for j in range(head_group):
      h = h0 + j
      qh = q[j * qk_rope_dim : (j + 1) * qk_rope_dim]
      x1 = qh[qk_head_dim : qk_head_dim + half]
      x2 = qh[qk_head_dim + half :]
      q_ref[0, h, :qk_head_dim, :] = (qh[:qk_head_dim] * q_scale).astype(q_ref.dtype)
      q_ref[0, h, qk_head_dim : qk_head_dim + half, :] = (x1 * cos[:half] - x2 * sin[:half]).astype(q_ref.dtype)
      q_ref[0, h, qk_head_dim + half :, :] = (x2 * cos[half:] + x1 * sin[half:]).astype(q_ref.dtype)
      k_ref[0, h, :qk_head_dim, :] = (k[j * qk_head_dim : (j + 1) * qk_head_dim] * dequant_scale).astype(k_ref.dtype)
      k_ref[0, h, qk_head_dim:, :] = k_pe
      v_ref[0, h] = (v[:, j * v_head_dim : (j + 1) * v_head_dim] * dequant_scale).astype(v_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=(
        "qk_head_dim", "kv_lora_rank", "softmax_scale", "norm_epsilon", "act_qtype",
        "act_scale", "weight_scale", "block_t", "block_n", "head_group",
    ),
)
def qkv_up_fwd(
    q_down: jax.Array, kv_down: jax.Array, wq_up: jax.Array, wk_up: jax.Array,
    wv_up: jax.Array, wq_norm_scale: jax.Array, wkv_norm_scale: jax.Array,
    yarn_freqs: tuple[jax.Array, jax.Array], *, qk_head_dim: int, kv_lora_rank: int,
    softmax_scale: float, norm_epsilon: float, act_qtype: Any, act_scale: float,
    weight_scale: float, block_t: int = 1024, block_n: int = 8, head_group: int = 4,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Compute per-device fused fp8 query, key and value up projections."""
  batch, seq_len, q_lora_rank = q_down.shape
  num_heads = wq_up.shape[1]
  qk_rope_dim = wq_up.shape[2]
  v_head_dim = wv_up.shape[2]
  rope_head_dim = qk_rope_dim - qk_head_dim
  kv_dim = kv_down.shape[2]
  block_t = min(block_t, seq_len)
  block_n = min(block_n, num_heads)
  assert seq_len % block_t == 0, (seq_len, block_t)
  assert num_heads % block_n == 0, (num_heads, block_n)
  head_group = min(head_group, block_n)
  assert block_n % head_group == 0, (block_n, head_group)

  wq_t = jnp.transpose(wq_up, (1, 2, 0)).reshape(-1, q_lora_rank)
  wk_t = jnp.transpose(wk_up, (1, 2, 0)).reshape(-1, kv_lora_rank)
  wv_2d = wv_up.reshape(kv_lora_rank, -1)
  cos, sin = (f.reshape(batch, seq_len, rope_head_dim) for f in yarn_freqs)

  dequant_scale = act_scale * weight_scale
  kernel = functools.partial(
      _qkv_up_kernel, eps=norm_epsilon, act_qtype=act_qtype, act_scale=act_scale,
      dequant_scale=dequant_scale, q_scale=dequant_scale * softmax_scale,
      qk_head_dim=qk_head_dim, kv_lora_rank=kv_lora_rank, head_group=head_group,
  )
  num_t = seq_len // block_t
  grid = (batch, num_heads // block_n, num_t)
  tok = lambda b, n, t: (b, jnp.where(n == 0, t, num_t - 1), 0)
  weight = lambda b, n, t: (n, 0)
  in_specs = [
      pl.BlockSpec((1, block_t, q_lora_rank), tok),
      pl.BlockSpec((1, block_t, kv_dim), tok),
      pl.BlockSpec((1, q_lora_rank), lambda b, n, t: (0, 0)),
      pl.BlockSpec((1, kv_lora_rank), lambda b, n, t: (0, 0)),
      pl.BlockSpec((1, block_t, rope_head_dim), tok),
      pl.BlockSpec((1, block_t, rope_head_dim), tok),
      pl.BlockSpec((block_n * qk_rope_dim, q_lora_rank), weight),
      pl.BlockSpec((block_n * qk_head_dim, kv_lora_rank), weight),
      pl.BlockSpec((kv_lora_rank, block_n * v_head_dim), lambda b, n, t: (0, n)),
  ]
  seq_minor = lambda b, n, t: (b, n, 0, t)
  out_specs = [
      pl.BlockSpec((1, block_n, qk_rope_dim, block_t), seq_minor),
      pl.BlockSpec((1, block_n, qk_rope_dim, block_t), seq_minor),
      pl.BlockSpec((1, block_n, block_t, v_head_dim), lambda b, n, t: (b, n, t, 0)),
  ]
  out_shape = [
      jax.ShapeDtypeStruct((batch, num_heads, qk_rope_dim, seq_len), jnp.bfloat16),
      jax.ShapeDtypeStruct((batch, num_heads, qk_rope_dim, seq_len), jnp.bfloat16),
      jax.ShapeDtypeStruct((batch, num_heads, seq_len, v_head_dim), jnp.bfloat16),
  ]
  flops = 2 * batch * seq_len * num_heads * (q_lora_rank * qk_rope_dim + kv_lora_rank * (qk_head_dim + v_head_dim))
  bytes_accessed = sum(x.size * x.dtype.itemsize for x in (q_down, kv_down, wq_up, wk_up, wv_up)) + sum(
      s.size * 2 for s in out_shape
  )
  q, k, v = pl.pallas_call(
      kernel, grid=grid, in_specs=in_specs, out_specs=out_specs, out_shape=out_shape,
      scratch_shapes=[
          pltpu.VMEM((seq_len, q_lora_rank), act_qtype),
          pltpu.VMEM((seq_len, kv_lora_rank), act_qtype),
          pltpu.VMEM((rope_head_dim, seq_len), jnp.float32),
          pltpu.VMEM((rope_head_dim, seq_len), jnp.float32),
          pltpu.VMEM((rope_head_dim, seq_len), jnp.bfloat16),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary", "arbitrary", "arbitrary"), vmem_limit_bytes=63 * 1024 * 1024,
      ),
      cost_estimate=pl.CostEstimate(flops=flops, transcendentals=0, bytes_accessed=bytes_accessed),
      name="mla_qkv_up_fwd",
  )(
      q_down, kv_down, wq_norm_scale.reshape(1, q_lora_rank), wkv_norm_scale.reshape(1, kv_lora_rank),
      cos, sin, wq_t, wk_t, wv_2d,
  )
  return q, k, v
