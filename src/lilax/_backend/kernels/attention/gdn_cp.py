"""Context-parallel evaluation of the GatedDeltaNet inter-chunk recurrence.

The recurrence h_new = A @ h + B is affine in the state, so it composes
associatively and can be split across a sharded sequence:

    A_i = exp(g_last).I - k_g^T.w_i      B_i = k_g^T.u_i

Each device folds its local chunks into one (A, B) pair, the pairs compose
across devices with a prefix scan, and each device then replays its chunks from
the state that reaches it. Composition is associative but not commutative, so
device order is the sequence order. `context_parallel_load_balance` permutes
that order and `configs/types.py` rejects the combination.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

_PREC = jax.lax.Precision.HIGHEST


def compose(left, right):
  """(A_r, B_r) . (A_l, B_l) = (A_r @ A_l, A_r @ B_l + B_r)."""
  A_l, B_l = left
  A_r, B_r = right
  return (
      jnp.matmul(A_r, A_l, precision=_PREC),
      jnp.matmul(A_r, B_l, precision=_PREC) + B_r,
  )


def compose_local(w, u, k, g):
  """Fold this device's chunks into one affine map, in O(1) memory."""
  k_dim = k.shape[-1]
  eye = jnp.eye(k_dim, dtype=jnp.float32)

  @jax.checkpoint
  def body(carry, x):
    w_c, u_c, k_c, g_c = x
    g_last = g_c[..., -1]
    decay = jnp.exp(g_last)[..., None, None]
    k_g = k_c.astype(jnp.float32) * jnp.exp(g_last[..., None] - g_c)[..., None]
    k_g_T = k_g.swapaxes(-1, -2)
    A_i = decay * eye - jnp.matmul(k_g_T, w_c.astype(jnp.float32), precision=_PREC)
    B_i = jnp.matmul(k_g_T, u_c.astype(jnp.float32), precision=_PREC)
    return compose(carry, (A_i, B_i)), None

  lead = w.shape[1:-2]
  init = (
      jnp.broadcast_to(eye, lead + (k_dim, k_dim)).astype(jnp.float32),
      jnp.zeros(lead + (k_dim, u.shape[-1]), jnp.float32),
  )
  (A_loc, B_loc), _ = lax.scan(body, init, (w, u, k, g))
  return A_loc, B_loc


def incoming_state(A_loc, B_loc, h_init, cp_axis):
  """State entering this device, plus the final state after all devices.

  log2(D) exchanges of one small matrix pair are the only cross-device traffic
  in the scheme. Must be called inside a shard_map over `cp_axis`.
  """
  D = lax.axis_size(cp_axis)
  idx = lax.axis_index(cp_axis)

  a_run, b_run = A_loc, B_loc
  step = 1
  while step < D:
    fwd = [(i, i + step) for i in range(D - step)]
    a_recv = lax.ppermute(a_run, cp_axis, fwd)
    b_recv = lax.ppermute(b_run, cp_axis, fwd)
    a_cmp, b_cmp = compose((a_recv, b_recv), (a_run, b_run))
    live = idx >= step
    a_run = jnp.where(live, a_cmp, a_run)
    b_run = jnp.where(live, b_cmp, b_run)
    step *= 2

  shift1 = [(i, i + 1) for i in range(D - 1)]
  a_ex = lax.ppermute(a_run, cp_axis, shift1)
  b_ex = lax.ppermute(b_run, cp_axis, shift1)
  carried = jnp.matmul(a_ex, h_init, precision=_PREC) + b_ex
  h_in = jnp.where(idx == 0, h_init, carried)

  last = idx == (D - 1)
  a_tot = lax.psum(jnp.where(last, a_run, jnp.zeros_like(a_run)), cp_axis)
  b_tot = lax.psum(jnp.where(last, b_run, jnp.zeros_like(b_run)), cp_axis)
  final_h = jnp.matmul(a_tot, h_init, precision=_PREC) + b_tot
  return h_in, final_h
