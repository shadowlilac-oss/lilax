"""Fused Pallas TPU kernel for DeepSeek-V4 CSA StreamIndex Score Computation.

Computes:
  index_scores = sum_h(ReLU(q_h @ comp^T) * softmax_scale * w_h)

The kernel fuses the dot product, ReLU activation, softmax scaling and
head-weight contraction so the intermediate [B, H, S, W] tensor stays in VMEM.
Masking is the caller's responsibility; see `csa_streamindex_score_kernel`.
"""

import functools
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


DEFAULT_BLOCK_Q_LARGE_H = 128
DEFAULT_BLOCK_W_LARGE_H = 1024
DEFAULT_BLOCK_Q_SMALL_H = 256
DEFAULT_BLOCK_W_SMALL_H = 2048

DEFAULT_BWD_HEAD_CHUNK = 1


def csa_streamindex_score_kernel(
    q_ref,
    k_ref,
    w_ref,
    out_ref,
    *,
    softmax_scale: float,
):
  """Fused Pallas TPU kernel with 2D MXU matmul.

  Masking is deliberately NOT done here: the kernel has no `position_ids` and
  would have to synthesize positions from grid indices, which is wrong for
  packed sequences. The caller masks, using the `future_mask` it already builds.
  """
  num_heads, block_q, head_dim = q_ref.shape
  block_w, _ = k_ref.shape

  q_2d = q_ref[...].reshape(num_heads * block_q, head_dim)
  k_2d = k_ref[...]

  s_2d = jnp.einsum("nd,md->nm", q_2d, k_2d, preferred_element_type=jnp.float32)

  s = s_2d.reshape(num_heads, block_q, block_w)
  s = jnp.maximum(s, 0.0)

  w = w_ref[...].astype(jnp.float32).transpose(1, 0)[:, :, None]
  s_weighted = jnp.sum(s * w, axis=0) * softmax_scale

  out_ref[...] = s_weighted.astype(out_ref.dtype)


def _csa_streamindex_score_pallas_fwd(
    q: jax.Array,
    compressed: jax.Array,
    weights: jax.Array,
    *,
    softmax_scale: float,
    block_q: int | None = None,
    block_w: int | None = None,
    interpret: bool = False,
) -> jax.Array:
  """Forward implementation using fused Pallas TPU kernel for head-major [B, H, S, D] q."""
  batch_size, num_heads, seq_len, head_dim = q.shape
  _, compressed_len, comp_head_dim = compressed.shape
  assert comp_head_dim == head_dim, f"{comp_head_dim=} != {head_dim=}"
  assert weights.shape == (batch_size, seq_len, num_heads), f"{weights.shape=} != {(batch_size, seq_len, num_heads)=}"

  if block_q is None:
    block_q = DEFAULT_BLOCK_Q_LARGE_H if num_heads >= 32 else DEFAULT_BLOCK_Q_SMALL_H
  if block_w is None:
    block_w = DEFAULT_BLOCK_W_LARGE_H if num_heads >= 32 else DEFAULT_BLOCK_W_SMALL_H

  padded_s = ((seq_len + block_q - 1) // block_q) * block_q
  padded_w = ((compressed_len + block_w - 1) // block_w) * block_w

  if padded_s > seq_len:
    pad_s = padded_s - seq_len
    q = jnp.pad(q, ((0, 0), (0, 0), (0, pad_s), (0, 0)))
    weights = jnp.pad(weights, ((0, 0), (0, pad_s), (0, 0)))
  if padded_w > compressed_len:
    pad_w = padded_w - compressed_len
    compressed = jnp.pad(compressed, ((0, 0), (0, pad_w), (0, 0)))

  grid = (batch_size, padded_s // block_q, padded_w // block_w)

  in_specs = [
      pl.BlockSpec((None, num_heads, block_q, head_dim), lambda b, i, j: (b, 0, i, 0)),
      pl.BlockSpec((None, block_w, head_dim), lambda b, i, j: (b, j, 0)),
      pl.BlockSpec((None, block_q, num_heads), lambda b, i, j: (b, i, 0)),
  ]
  out_specs = pl.BlockSpec((None, block_q, block_w), lambda b, i, j: (b, i, j))

  out = pl.pallas_call(
      functools.partial(
          csa_streamindex_score_kernel,
          softmax_scale=softmax_scale,
      ),
      in_specs=in_specs,
      out_specs=out_specs,
      grid=grid,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel"),
      ),
      out_shape=jax.ShapeDtypeStruct((batch_size, padded_s, padded_w), jnp.float32),
      interpret=interpret,
  )(q, compressed, weights)

  return out[:, :seq_len, :compressed_len]


def csa_indexer_scores_jax(
    q: jax.Array,
    compressed: jax.Array,
    weights: jax.Array,
    *,
    softmax_scale: float,
    precision: jax.lax.PrecisionLike = None,
) -> jax.Array:
  """Pure-JAX CSA indexer scores. The definition the Pallas kernel must match.

  Also the function `DeepseekV4Indexer` calls on its non-kernel path, so the two
  cannot drift. Masking is applied by the caller, not here.

  Args:
    q: Query, `[B, H, S, D]`.
    compressed: Compressed KV blocks, `[B, W, D]`.
    weights: Per-head indexer weights, `[B, S, H]`.
    softmax_scale: Scalar applied after the ReLU.
    precision: Matmul precision for both einsums. The layer passes
      `config.matmul_precision`; the kernel's VJP leaves it at the default.

  Returns:
    Index scores, `[B, S, W]`.
  """
  scores = jnp.einsum("bhsd,bwd->bhsw", q.astype(jnp.float32), compressed.astype(jnp.float32), precision=precision)
  scores = jax.nn.relu(scores) * softmax_scale
  return jnp.einsum("bhsw,bsh->bsw", scores, weights.astype(jnp.float32), precision=precision)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7))
def csa_streamindex_score(
    q: jax.Array,
    compressed: jax.Array,
    weights: jax.Array,
    softmax_scale: float,
    block_q: int | None = None,
    block_w: int | None = None,
    interpret: bool = False,
    head_chunk_size: int = DEFAULT_BWD_HEAD_CHUNK,
) -> jax.Array:
  """Computes CSA StreamIndex scores using head-major [B, H, S, D] q layout with 2D MXU matmul."""
  del head_chunk_size
  return _csa_streamindex_score_pallas_fwd(
      q,
      compressed,
      weights,
      softmax_scale=softmax_scale,
      block_q=block_q,
      block_w=block_w,
      interpret=interpret,
  )


def _csa_streamindex_score_fwd(
    q: jax.Array,
    compressed: jax.Array,
    weights: jax.Array,
    softmax_scale: float,
    block_q: int | None = None,
    block_w: int | None = None,
    interpret: bool = False,
    head_chunk_size: int = DEFAULT_BWD_HEAD_CHUNK,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
  """custom_vjp forward rule. Saves inputs, not the `[B, H, S, W]` scores, so the backward can recompute per chunk."""
  del head_chunk_size
  out = _csa_streamindex_score_pallas_fwd(
      q,
      compressed,
      weights,
      softmax_scale=softmax_scale,
      block_q=block_q,
      block_w=block_w,
      interpret=interpret,
  )
  return out, (q, compressed, weights)


def _csa_streamindex_score_bwd(
    softmax_scale: float,
    block_q: int | None,
    block_w: int | None,
    interpret: bool,
    head_chunk_size: int,
    res: tuple[jax.Array, jax.Array, jax.Array],
    g: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Head-chunked backward pass.

  The forward fuses `[B, H, S, W]` away, but reverse mode must keep it, so
  whole-head autodiff materializes it in HBM. Scanning over `head_chunk_size`
  heads bounds it to `[B, head_chunk_size, S, W]`; without this the backward
  OOMs at 32k context on v6e. `d(compressed)` accumulates across chunks; `dq`
  and `dweights` are per-head and get stacked back into head-major order.
  """
  del block_q, block_w, interpret
  q, compressed, weights = res
  num_heads = q.shape[1]

  chunk = min(head_chunk_size, num_heads) if head_chunk_size > 0 else num_heads
  if num_heads % chunk != 0:
    chunk = num_heads
  num_chunks = num_heads // chunk

  q_chunks = jnp.moveaxis(q, 1, 0).reshape(num_chunks, chunk, *q.shape[0:1], *q.shape[2:])
  q_chunks = jnp.moveaxis(q_chunks, 1, 2)
  w_chunks = jnp.moveaxis(weights, 2, 0).reshape(num_chunks, chunk, *weights.shape[0:2])
  w_chunks = jnp.moveaxis(w_chunks, 1, 3)

  def scan_body(dcomp_acc, xs):
    q_c, w_c = xs["q"], xs["w"]
    _, vjp_fn = jax.vjp(
        functools.partial(csa_indexer_scores_jax, softmax_scale=softmax_scale),
        q_c,
        compressed,
        w_c,
    )
    dq_c, dcomp_c, dw_c = vjp_fn(g)
    return dcomp_acc + dcomp_c, (dq_c, dw_c)

  dcomp_init = jnp.zeros(compressed.shape, dtype=jnp.float32)
  dcompressed, (dq_stacked, dw_stacked) = jax.lax.scan(scan_body, dcomp_init, {"q": q_chunks, "w": w_chunks})

  dq = jnp.moveaxis(dq_stacked, 2, 1).reshape(num_heads, *q.shape[0:1], *q.shape[2:])
  dq = jnp.moveaxis(dq, 0, 1)
  dw = jnp.moveaxis(dw_stacked, 0, 2).reshape(weights.shape[0], weights.shape[1], num_heads)

  return dq.astype(q.dtype), dcompressed.astype(compressed.dtype), dw.astype(weights.dtype)


csa_streamindex_score.defvjp(_csa_streamindex_score_fwd, _csa_streamindex_score_bwd)
