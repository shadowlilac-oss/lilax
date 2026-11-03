"""Qwen3-Next Gated Delta Rule recurrences."""

import jax
from jax import lax
import jax.numpy as jnp
from lilax._backend.common.common_types import Array
from lilax._backend.kernels.attention import gdn_cp
from lilax._backend.layers.normalizations import l2norm


def naive_jax_chunk_gated_delta_rule(
    query, key, value, g, beta, chunk_size=64, initial_state=None, use_qk_norm_in_gdn=False
):
  """Naive implementation of the Gated Delta Rule in jax."""
  initial_dtype = query.dtype
  if use_qk_norm_in_gdn:
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)

  query = jnp.transpose(query, (0, 2, 1, 3)).astype(jnp.float32)
  key = jnp.transpose(key, (0, 2, 1, 3)).astype(jnp.float32)
  value = jnp.transpose(value, (0, 2, 1, 3)).astype(jnp.float32)
  beta = jnp.transpose(beta, (0, 2, 1)).astype(jnp.float32)
  g = jnp.transpose(g, (0, 2, 1)).astype(jnp.float32)

  batch_size, num_heads, sequence_length, k_head_dim = key.shape
  v_head_dim = value.shape[-1]
  pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

  if pad_size > 0:
    query = jnp.pad(query, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
    key = jnp.pad(key, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
    value = jnp.pad(value, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
    beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_size)))
    g = jnp.pad(g, ((0, 0), (0, 0), (0, pad_size)))

  total_sequence_length = sequence_length + pad_size
  scale = jax.lax.rsqrt(jnp.array(query.shape[-1]).astype(jnp.float32))
  query = query * scale

  v_beta = value * jnp.expand_dims(beta, -1)
  k_beta = key * jnp.expand_dims(beta, -1)

  num_chunks = total_sequence_length // chunk_size
  query_c = query.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
  key_c = key.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
  k_beta_c = k_beta.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
  v_beta_c = v_beta.reshape(batch_size, num_heads, num_chunks, chunk_size, v_head_dim)
  g_c = g.reshape(batch_size, num_heads, num_chunks, chunk_size)

  mask = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=0)

  g_cumsum = jnp.cumsum(g_c, axis=-1)
  g_diff = jnp.expand_dims(g_cumsum, -1) - jnp.expand_dims(g_cumsum, -2)
  g_diff_tril = jnp.tril(g_diff)
  g_diff_exp = jnp.exp(g_diff_tril).astype(jnp.float32)
  decay_mask = g_diff_exp

  prec = jax.lax.Precision.HIGHEST
  attn = -jnp.matmul(k_beta_c, jnp.swapaxes(key_c, -1, -2), precision=prec) * decay_mask
  attn = jnp.where(mask, 0.0, attn)

  def inner_attn_body(i, attn_val):
    indices = jnp.arange(chunk_size)
    col_mask = indices < i
    row = attn_val[..., i, :] * col_mask
    sub_mask = jnp.expand_dims(indices < i, -1) & (indices < i)
    sub = attn_val * sub_mask
    row_exp = jnp.expand_dims(row, -1)
    term = row_exp * sub
    summed = jnp.sum(term, axis=-2)
    update_val = row + summed
    original_row = attn_val[..., i, :]
    new_row = jnp.where(col_mask, update_val, original_row)
    return attn_val.at[..., i, :].set(new_row)

  attn = jax.lax.fori_loop(1, chunk_size, inner_attn_body, attn)
  attn = attn + jnp.eye(chunk_size, dtype=attn.dtype)
  value_intra = jnp.matmul(attn, v_beta_c, precision=prec)
  k_cumdecay = jnp.matmul(attn, (k_beta_c * jnp.expand_dims(jnp.exp(g_cumsum), -1)), precision=prec)

  output_final_state = initial_state is not None
  if initial_state is None:
    last_recurrent_state = jnp.zeros((batch_size, num_heads, k_head_dim, v_head_dim), dtype=value_intra.dtype)
  else:
    last_recurrent_state = initial_state.astype(value_intra.dtype)

  mask_inter = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=1)

  query_scan = jnp.transpose(query_c, (2, 0, 1, 3, 4))
  key_scan = jnp.transpose(key_c, (2, 0, 1, 3, 4))
  value_scan = jnp.transpose(value_intra, (2, 0, 1, 3, 4))
  k_cumdecay_scan = jnp.transpose(k_cumdecay, (2, 0, 1, 3, 4))
  g_scan = jnp.transpose(g_cumsum, (2, 0, 1, 3))
  decay_mask_scan = jnp.transpose(decay_mask, (2, 0, 1, 3, 4))

  xs = (query_scan, key_scan, value_scan, k_cumdecay_scan, g_scan, decay_mask_scan)

  def scan_body(prev_state, x):
    q_i, k_i, v_i, k_cumdecay_i, g_i, decay_mask_i = x
    last_recurrent_state = prev_state
    prec = jax.lax.Precision.HIGHEST

    attn_i = jnp.matmul(q_i, jnp.swapaxes(k_i, -1, -2), precision=prec) * decay_mask_i
    attn_i = jnp.where(mask_inter, 0.0, attn_i)

    v_prime = jnp.matmul(k_cumdecay_i, last_recurrent_state, precision=prec)
    v_new = v_i - v_prime

    g_i_exp = jnp.exp(g_i)
    attn_inter = jnp.matmul(q_i * jnp.expand_dims(g_i_exp, -1), last_recurrent_state, precision=prec)

    core_attn_out_i = attn_inter + jnp.matmul(attn_i, v_new, precision=prec)

    g_i_last_exp = jnp.exp(g_i[..., -1, None, None])
    new_last_recurrent_state = last_recurrent_state * g_i_last_exp

    g_diff_exp = jnp.expand_dims(jnp.exp(jnp.expand_dims(g_i[..., -1], -1) - g_i), -1)
    k_i_g_diff = k_i * g_diff_exp

    update_term = jnp.matmul(jnp.swapaxes(k_i_g_diff, -1, -2), v_new, precision=prec)
    new_last_recurrent_state = new_last_recurrent_state + update_term

    return new_last_recurrent_state, core_attn_out_i

  final_state, core_attn_out_stacked = jax.lax.scan(scan_body, last_recurrent_state, xs)

  core_attn_out = jnp.transpose(core_attn_out_stacked, (1, 2, 0, 3, 4))
  core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, v_head_dim)
  core_attn_out = core_attn_out[:, :, :sequence_length, :]
  core_attn_out = jnp.transpose(core_attn_out, (0, 2, 1, 3)).astype(initial_dtype)

  return core_attn_out, final_state if output_final_state else None


def jax_chunk_gated_delta_rule(
    query: Array,
    key: Array,
    value: Array,
    g: Array,
    beta: Array,
    chunk_size: int = 64,
    initial_state: None | Array = None,
    use_qk_norm_in_gdn: bool = False,
    cp_axis: None | str = None,
    compute_dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[Array, None | Array]:
  """Optimized JAX implementation of Gated Delta Rule."""
  # =========================================================================
  # STAGE 1: PREPARATION & PADDING
  # =========================================================================
  initial_dtype = query.dtype

  if use_qk_norm_in_gdn:
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)

  g = g.astype(jnp.float32)

  # 2. Cast inputs to the requested compute_dtype (cfg.dtype) to save memory/compute
  query = query.astype(compute_dtype)
  key = key.astype(compute_dtype)
  value = value.astype(compute_dtype)
  beta = beta.astype(compute_dtype)

  # Scale Query (keep in compute_dtype)
  scale = jax.lax.rsqrt(jnp.array(query.shape[-1], dtype=jnp.float32)).astype(compute_dtype)
  query = query * scale

  B, seq_len, H, K_dim = key.shape
  V_dim = value.shape[-1]

  pad_len = (chunk_size - (seq_len % chunk_size)) % chunk_size
  if pad_len > 0:

    def pad_fn(x, val=0.0):
      return jnp.pad(x, ((0, 0), (0, pad_len)) + ((0, 0),) * (x.ndim - 2), constant_values=val)

    query = pad_fn(query)
    key = pad_fn(key)
    value = pad_fn(value)
    g = pad_fn(g)
    beta = pad_fn(beta)

  num_chunks = query.shape[1] // chunk_size

  # Helper: (B, S, H, D) -> (B, N, H, C, D)
  def to_chunk(x):
    return x.reshape(B, num_chunks, chunk_size, H, -1).transpose(0, 1, 3, 2, 4)

  # Helper for scalars: (B, S, H) -> (B, N, H, C)
  def to_chunk_scalar(x):
    return x.reshape(B, num_chunks, chunk_size, H).transpose(0, 1, 3, 2)

  q_c = to_chunk(query)
  k_c = to_chunk(key)
  v_c = to_chunk(value)
  g_c = to_chunk_scalar(g)
  beta_c = to_chunk_scalar(beta)

  # =========================================================================
  # STAGE 2: INTRA-CHUNK PRE-COMPUTATION (Parallel)
  # =========================================================================

  # Cumulative decay (Must be float32)
  g_cumsum = jnp.cumsum(g_c, axis=-1)
  k_beta = k_c * beta_c[..., None]

  # S Matrix Calculation
  S = jnp.matmul(k_beta, k_c.swapaxes(-1, -2), precision=jax.lax.Precision.HIGHEST)
  S = S.astype(jnp.float32)

  # Apply mask BEFORE exp to prevent 'inf' gradients
  g_diff = g_cumsum[..., :, None] - g_cumsum[..., None, :]
  mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=-1)
  g_diff = jnp.where(mask, g_diff, -1e30)

  S = S * jnp.exp(g_diff)
  S = jnp.where(mask, S, 0.0)

  # Inversion (A) - Strictly float32
  identity = jnp.eye(chunk_size, dtype=jnp.float32)
  identity_broadcasted = jnp.broadcast_to(identity, S.shape)

  A = jax.scipy.linalg.solve_triangular(identity + S, identity_broadcasted, lower=True, unit_diagonal=True)

  # 5. WY Factors
  v_beta = v_c * beta_c[..., None]
  u_chunks = jnp.matmul(A, v_beta.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)
  u_chunks = u_chunks.astype(compute_dtype)

  k_beta_g = k_beta.astype(jnp.float32) * jnp.exp(g_cumsum)[..., None]
  w_chunks = jnp.matmul(A, k_beta_g, precision=jax.lax.Precision.HIGHEST)
  w_chunks = w_chunks.astype(compute_dtype)

  # =========================================================================
  # STAGE 3: INTER-CHUNK RECURRENCE (Scan)
  # =========================================================================
  scan_perm_vec = (1, 0, 2, 3, 4)
  scan_perm_scl = (1, 0, 2, 3)

  w_scan = w_chunks.transpose(scan_perm_vec)
  u_scan = u_chunks.transpose(scan_perm_vec)
  k_scan = k_c.transpose(scan_perm_vec)
  q_scan = q_c.transpose(scan_perm_vec)
  g_scan = g_cumsum.transpose(scan_perm_scl)

  if initial_state is None:
    h_init = jnp.zeros((B, H, K_dim, V_dim), dtype=jnp.float32)
  else:
    h_init = initial_state.astype(jnp.float32)

  xs = (w_scan, u_scan, q_scan, k_scan, g_scan)

  def scan_body(h, args):
    w, u, q, k, g = args
    prec = jax.lax.Precision.HIGHEST

    # --- Output Computation ---
    # 1. Inter-chunk: q(dtype) * exp(g)(f32) -> f32
    q_g = q.astype(jnp.float32) * jnp.exp(g)[..., None]
    attn_inter = jnp.matmul(q_g, h, precision=prec)

    # 2. Delta Rule Subtraction (v_prime and v_new)
    # w serves as k_cumdecay, u serves as value_intra
    v_prime = jnp.matmul(w.astype(jnp.float32), h, precision=prec)
    v_new = u.astype(jnp.float32) - v_prime

    # 3. Intra-chunk: q(dtype) @ k(dtype) -> f32
    attn = jnp.matmul(q, k.swapaxes(-1, -2), precision=prec)
    attn = attn.astype(jnp.float32)

    # Mask before exp
    g_diff = g[..., :, None] - g[..., None, :]
    mask_intra = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool))
    g_diff = jnp.where(mask_intra, g_diff, -1e30)

    attn_i = attn * jnp.exp(g_diff)
    attn_i = jnp.where(mask_intra, attn_i, 0.0)

    # Note: We do NOT multiply attn_i by beta here. The Delta rule mathematically
    # absorbed beta inside v_new (via u).

    # 4. Combine Core Output
    term2 = jnp.matmul(attn_i, v_new, precision=prec)
    o_c = attn_inter + term2

    # --- State Update ---
    g_i_last_exp = jnp.exp(g[..., -1, None, None])
    h_new = h * g_i_last_exp

    # Apply Delta Rule K decay to state
    g_diff_exp_state = jnp.exp(g[..., -1, None] - g)[..., None]
    k_i_g_diff = k.astype(jnp.float32) * g_diff_exp_state

    update_term = jnp.matmul(k_i_g_diff.swapaxes(-1, -2), v_new, precision=prec)
    h_new = h_new + update_term

    return h_new, o_c

  if cp_axis is None:
    final_h, o_chunks = lax.scan(scan_body, h_init, xs)
  else:
    # Sequence is sharded over cp_axis, so a sequential scan over chunks is not
    # available. Fold the local chunks into one affine map, exchange those, then
    # replay locally from the correct incoming state. See kernels/attention/gdn_cp.py.
    A_loc, B_loc = gdn_cp.compose_local(w_scan, u_scan, k_scan, g_scan)
    h_in, final_h = gdn_cp.incoming_state(A_loc, B_loc, h_init, cp_axis)
    _, o_chunks = lax.scan(scan_body, h_in, xs)

  # =========================================================================
  # STAGE 4: FINALIZATION
  # =========================================================================
  o = o_chunks.transpose(1, 0, 3, 2, 4)
  o = o.reshape(B, -1, H, V_dim)

  if pad_len > 0:
    o = o[:, :seq_len, :, :]

  o = o.astype(initial_dtype)

  return o, (final_h.astype(compute_dtype) if initial_state is not None else None)


def jax_ar_gated_delta_rule(
    query: Array,
    key: Array,
    value: Array,
    g: Array,
    beta: Array,
    initial_state: Array,
    use_qk_norm_in_gdn: bool = False,
    compute_dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[Array, Array]:
  """Highly optimized step for Autoregressive Decoding (seq_len == 1)."""
  # Shapes: q, k (B, 1, H, K_dim) | v (B, 1, H, V_dim) | g, beta (B, 1, H)
  initial_dtype = query.dtype

  if use_qk_norm_in_gdn:
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)

  query = query.astype(compute_dtype)
  key = key.astype(compute_dtype)
  value = value.astype(compute_dtype)
  beta = beta.astype(compute_dtype)
  g = g.astype(jnp.float32)

  # Strip the seq_len=1 dimension to avoid broadcast overhead
  q = query.squeeze(1)
  k = key.squeeze(1)
  v = value.squeeze(1)
  g = g.squeeze(1)
  beta = beta.squeeze(1)

  scale = jax.lax.rsqrt(jnp.array(q.shape[-1], dtype=jnp.float32)).astype(compute_dtype)
  q = q * scale

  g_exp = jnp.exp(g)[..., None]
  beta_exp = beta[..., None]

  k_beta = k * beta_exp
  v_beta = v * beta_exp

  state = initial_state.astype(jnp.float32)

  # v_prime = state @ (k_beta * exp(g))
  k_cumdecay = (k_beta.astype(jnp.float32) * g_exp)[..., None, :]  # (B, H, 1, K)
  v_prime = jnp.matmul(k_cumdecay, state, precision=jax.lax.Precision.HIGHEST).squeeze(-2)

  v_new = v_beta.astype(jnp.float32) - v_prime

  # Core Output
  q_g = (q.astype(jnp.float32) * g_exp)[..., None, :]  # (B, H, 1, K)
  attn_inter = jnp.matmul(q_g, state, precision=jax.lax.Precision.HIGHEST).squeeze(-2)

  attn_intra = jnp.sum(q.astype(jnp.float32) * k.astype(jnp.float32), axis=-1, keepdims=True)
  core_attn_out = attn_inter + attn_intra * v_new

  # State Update: new_state = state * exp(g) + k^T @ v_new
  new_state = state * g_exp[..., None] + jnp.matmul(
      k.astype(jnp.float32)[..., None], v_new[..., None, :], precision=jax.lax.Precision.HIGHEST
  )

  # Restore sequence dimension
  core_attn_out = core_attn_out[:, None, :, :].astype(initial_dtype)

  return core_attn_out, new_state
