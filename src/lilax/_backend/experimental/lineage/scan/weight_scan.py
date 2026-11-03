"""Pipelined scan with weight collection."""

from collections.abc import Callable
import functools
from typing import Any

import jax
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

AnyJaxArray = jt.Num[jax.Array, "..."]
AnyJaxArrayOrPyTree = AnyJaxArray | jt.PyTree[AnyJaxArray]


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4, 5))
def w_collect_pipelined_scan(
    scan_fn: Callable[[AnyJaxArray, AnyJaxArrayOrPyTree], tuple[AnyJaxArray, AnyJaxArrayOrPyTree]],
    x: AnyJaxArray,
    w: AnyJaxArrayOrPyTree,
    collect_fn_dcn: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
    collect_fn_ici: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
    reduce_fn_ici: Callable[[AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree] | None = None,
) -> tuple[AnyJaxArray, AnyJaxArrayOrPyTree]:
  """Pipelines weight collection for scanning over scan_fn."""
  del reduce_fn_ici
  w_first = collect_fn_ici(collect_fn_dcn(jax.tree.map(lambda _w: _w[0], w)))
  carry = (x, w_first)

  def body(carry, w_n_sharded):
    x, w = carry
    w_n = collect_fn_ici(collect_fn_dcn(w_n_sharded))
    x, aux_step = scan_fn(x, w)
    return (x, w_n), aux_step

  (x, w_last), aux_scanned = jax.lax.scan(body, carry, jax.tree.map(lambda _w: _w[1:], w))
  x, aux_last = scan_fn(x, w_last)
  aux = jax.tree.map(lambda s, l: jnp.concatenate([s, l[jnp.newaxis]], axis=0), aux_scanned, aux_last)
  return x, aux


@jt.jaxtyped(typechecker=typeguard.typechecked)
def w_collect_pipelined_scan_fwd(
    scan_fn: Callable[[AnyJaxArray, AnyJaxArrayOrPyTree], tuple[AnyJaxArray, AnyJaxArrayOrPyTree]],
    x: AnyJaxArray,
    w: AnyJaxArrayOrPyTree,
    collect_fn_dcn: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
    collect_fn_ici: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
    reduce_fn_ici: Callable[[AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree] | None,
) -> tuple[tuple[AnyJaxArray, AnyJaxArrayOrPyTree], Any]:
  """Forward pass of w_collect_pipelined_scan which does not save collected weights."""
  assert jax.tree.leaves(w)[0].shape[0] >= 2, (
      "there must be at least two scan iterations to perform a pipelined scan" " and its transpose"
  )
  w_first_dcn, reduce_fn_dcn = jax.vjp(collect_fn_dcn, jax.tree.map(lambda _w: _w[0], w))
  getattr(reduce_fn_dcn, "args_res")[0] = None
  if reduce_fn_ici is None:
    w_first, reduce_fn_ici_vjp = jax.vjp(collect_fn_ici, w_first_dcn)
    getattr(reduce_fn_ici_vjp, "args_res")[0] = None
  else:
    w_first = collect_fn_ici(w_first_dcn)
    reduce_fn_ici_vjp = None
  (x, aux_first), scan_fn_vjp_first = jax.vjp(scan_fn, x, w_first)
  getattr(scan_fn_vjp_first, "args_res")[1] = None
  w_second = collect_fn_ici(collect_fn_dcn(jax.tree.map(lambda _w: _w[1], w)))
  carry = (x, w_second)

  def body_fwd(carry, w_n_sharded):
    x, w = carry
    w_n = collect_fn_ici(collect_fn_dcn(w_n_sharded))
    (x, aux_step), scan_fn_vjp = jax.vjp(scan_fn, x, w)
    getattr(scan_fn_vjp, "args_res")[1] = None
    return (x, w_n), (scan_fn_vjp, aux_step)

  (x, w_last), (scan_fn_vjps, aux_scanned) = jax.lax.scan(body_fwd, carry, jax.tree.map(lambda _w: _w[2:], w))
  (x, aux_last), scan_fn_vjp_last = jax.vjp(scan_fn, x, w_last)
  getattr(scan_fn_vjp_last, "args_res")[1] = None
  aux = jax.tree.map(
      lambda first, scanned, last: jnp.concatenate([first[jnp.newaxis], scanned, last[jnp.newaxis]], axis=0),
      aux_first, aux_scanned, aux_last,
  )
  return (x, aux), (scan_fn_vjp_first, scan_fn_vjps, scan_fn_vjp_last, w, reduce_fn_dcn, reduce_fn_ici_vjp)


@jt.jaxtyped(typechecker=typeguard.typechecked)
def w_collect_pipelined_scan_bwd(
    _,
    collect_fn_dcn: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
    collect_fn_ici: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
    reduce_fn_ici: Callable[[AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree] | None,
    res: Any,
    grad_outputs: tuple[AnyJaxArray, Any],
) -> tuple[AnyJaxArray, AnyJaxArrayOrPyTree]:
  """Backward pass of w_collect_pipelined_scan which re-collects weights."""
  scan_fn_vjp_first, scan_fn_vjps, scan_fn_vjp_last, w, reduce_fn_dcn, reduce_fn_ici_vjp = res
  x_grad, aux_grad = grad_outputs
  aux_step_last = jax.tree.map(lambda g: g[-1], aux_grad)
  step_grad_last = (x_grad, aux_step_last)
  w_last_dcn = collect_fn_dcn(jax.tree.map(lambda _w: _w[-1], w))
  w_last = collect_fn_ici(w_last_dcn)
  if reduce_fn_ici is None:
    assert reduce_fn_ici_vjp is not None
    _reduce_ici = lambda grad: reduce_fn_ici_vjp(grad)[0]
  else:
    _reduce_ici = lambda grad: reduce_fn_ici(grad, w_last_dcn)
  getattr(scan_fn_vjp_last, "args_res")[1] = w_last
  x_grad, w_last_grad_unreduced = scan_fn_vjp_last(step_grad_last)
  w_second_last = collect_fn_ici(collect_fn_dcn(jax.tree.map(lambda _w: _w[-2], w)))
  carry = (x_grad, w_second_last, w_last_grad_unreduced)
  aux_scanned_grad = jax.tree.map(lambda g: g[1:-1], aux_grad)
  scan_xs = (scan_fn_vjps, jax.tree.map(lambda _w: _w[:-2], w), aux_scanned_grad)

  def body_bwd(carry, xs):
    x_grad, w_n, w_prev_grad_unreduced = carry
    scan_fn_vjp, w_next_sharded, aux_step_grad = xs
    w_next = collect_fn_ici(collect_fn_dcn(w_next_sharded))
    w_prev_grad_reduced = reduce_fn_dcn(_reduce_ici(w_prev_grad_unreduced))[0]
    getattr(scan_fn_vjp, "args_res")[1] = w_n
    x_grad, w_n_grad_unreduced = scan_fn_vjp((x_grad, aux_step_grad))
    return (x_grad, w_next, w_n_grad_unreduced), w_prev_grad_reduced

  (x_grad, w_first, w_second_grad_unreduced), w_grads_reduced = jax.lax.scan(
      body_bwd, carry, scan_xs, reverse=True,
  )
  w_second_grad_reduced = reduce_fn_dcn(_reduce_ici(w_second_grad_unreduced))[0]
  getattr(scan_fn_vjp_first, "args_res")[1] = w_first
  aux_step_first = jax.tree.map(lambda g: g[0], aux_grad)
  step_grad_first = (x_grad, aux_step_first)
  x_grad, w_first_grad_unreduced = scan_fn_vjp_first(step_grad_first)
  w_first_grad_reduced = reduce_fn_dcn(_reduce_ici(w_first_grad_unreduced))[0]
  w_grad = jax.tree.map(
      lambda x, y, z: jnp.concatenate([x[jnp.newaxis], y[jnp.newaxis], z], axis=0),
      w_first_grad_reduced, w_second_grad_reduced, w_grads_reduced,
  )
  return x_grad, w_grad


w_collect_pipelined_scan.defvjp(w_collect_pipelined_scan_fwd, w_collect_pipelined_scan_bwd)
