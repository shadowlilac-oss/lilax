"""Pipelined scan with weight collection and peeled prologue and epilogue layers."""

from collections.abc import Callable
import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

AnyJaxArray = jt.Num[jax.Array, "..."]
AnyJaxArrayOrPyTree = AnyJaxArray | jt.PyTree[AnyJaxArray]


class PipelinedScanLayer(NamedTuple):
  fwd: Callable[..., tuple[Any, Any]]
  bwd: Callable[..., Any]


def combine_w_grads(w_mla: AnyJaxArrayOrPyTree, w_expert: AnyJaxArrayOrPyTree) -> AnyJaxArrayOrPyTree:
  """Combines MLA/norm/router gradients and expert gradients for a layer."""
  def _add(a, b):
    if a is None:
      return b
    if b is None:
      return a
    return a + b

  return jax.tree.map(_add, w_mla, w_expert, is_leaf=lambda x: x is None)


def _interleave_arrays(a0: Any, a1: Any) -> Any:
  """Interleaves two arrays along axis 1 and merges leading dimensions."""
  if a0 is None or a1 is None:
    return None
  stacked = jnp.stack([a0, a1], axis=1)
  new_shape = (a0.shape[0] * 2,) + a0.shape[1:]
  s = getattr(jax.typeof(a0), "sharding", None)
  if s is not None:
    return jax.lax.reshape(stacked, new_shape, out_sharding=s)
  return stacked.reshape(new_shape)


def _layer_slice(tree: Any, idx: int | slice) -> Any:
  """Slices the layer dimension of every leaf; None trees stay None."""
  return jax.tree.map(lambda t: t[idx], tree)


def _layer_at(tree: Any, idx: int | jax.Array | None) -> Any:
  """Returns one layer of every leaf."""
  if tree is None or idx is None:
    return None
  if isinstance(idx, int):
    return _layer_slice(tree, idx)
  return jax.tree.map(lambda t: jax.lax.dynamic_index_in_dim(t, idx, keepdims=False), tree)


def _w_aux_kwargs(w_aux_layer: Any) -> dict[str, Any]:
  """Forwards a layer's w_aux slice, if any."""
  return {} if w_aux_layer is None else {"w_aux": w_aux_layer}


class _LayerCollect(NamedTuple):
  """Per-layer weight collection helpers of the pipelined scans."""

  collect: Callable[[Any, Any], Any]
  pre: Callable[[Any, Any], Any]
  collect_ici: Callable[[Any, Any], Any]
  pre_carry: Callable[[int | jax.Array | None], Any]


def _layer_collect(
    collect_fn_dcn: Callable[..., Any],
    collect_fn_ici: Callable[..., Any],
    collect_fn_pre: Callable[..., Any] | None,
    w: Any,
    w_aux: Any,
) -> _LayerCollect:
  """Builds the _LayerCollect helpers for a pipelined scan pass."""
  def pre(w_dcn, w_aux_layer):
    if collect_fn_pre is None:
      return w_dcn
    return collect_fn_pre(w_dcn, **_w_aux_kwargs(w_aux_layer))

  def collect_ici(w_pre, w_aux_layer):
    return collect_fn_ici(w_pre, **_w_aux_kwargs(w_aux_layer))

  def collect(w_layer, w_aux_layer):
    return collect_ici(pre(collect_fn_dcn(w_layer), w_aux_layer), w_aux_layer)

  def pre_carry(idx):
    if collect_fn_pre is None or idx is None:
      return None
    w_aux_layer = _layer_at(w_aux, idx)
    return pre(collect_fn_dcn(_layer_at(w, idx)), w_aux_layer), w_aux_layer

  return _LayerCollect(collect=collect, pre=pre, collect_ici=collect_ici, pre_carry=pre_carry)


def _shifted_indices(layers: slice | int, shift: int, num_layers: int) -> Any:
  """Returns shifted layer indices clamped to the layer range."""
  if isinstance(layers, slice):
    idx = jnp.arange(layers.start, layers.stop, layers.step) + shift
    return jnp.clip(idx, 0, num_layers - 1)
  idx = layers + shift
  return idx if 0 <= idx < num_layers else None


def _fwd_step_xs(w: Any, w_aux: Any, layers: slice | int, shifted: bool) -> tuple[Any, Any, Any]:
  """Returns weight and auxiliary inputs for forward scan steps."""
  if shifted:
    num_layers = jax.tree.leaves(w)[0].shape[0]
    return None, None, _shifted_indices(layers, 1, num_layers)
  return _layer_slice(w, layers), _layer_slice(w_aux, layers), None


@functools.partial(
    jax.custom_vjp, nondiff_argnums=(0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16),
)
def w_collect_pipelined_scan_prologue_epilogue(
    prologue_fn: PipelinedScanLayer,
    scan_body_fn: PipelinedScanLayer,
    epilogue_fn: PipelinedScanLayer,
    x: AnyJaxArrayOrPyTree,
    w: AnyJaxArrayOrPyTree,
    collect_fn_dcn: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
    collect_fn_ici_fwd: Callable[..., AnyJaxArrayOrPyTree],
    reduce_fn_dcn: Callable[[AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
    reduce_fn_ici_first: Callable[..., AnyJaxArrayOrPyTree],
    reduce_fn_ici_second: Callable[..., AnyJaxArrayOrPyTree],
    collect_fn_ici_bwd: Callable[..., AnyJaxArrayOrPyTree] | None = None,
    quantize_fn: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree] | None = None,
    offload_fn: Callable[[Any], Any] = lambda res: res,
    load_fn: Callable[[Any], Any] = lambda res: res,
    w_aux: Any = None,
    collect_fn_pre: Callable[..., AnyJaxArrayOrPyTree] | None = None,
    reduce_fn_pre: Callable[..., AnyJaxArrayOrPyTree] | None = None,
) -> tuple[AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree]:
  """Pipelines weight collection for scanning with prologue and epilogue."""
  del offload_fn, load_fn, collect_fn_ici_bwd, reduce_fn_dcn
  del reduce_fn_ici_first, reduce_fn_ici_second, reduce_fn_pre
  num_layers = jax.tree.leaves(w)[0].shape[0]
  assert num_layers >= 3, (
      "there must be at least three layers to perform a pipelined scan with a" " peeled prologue and epilogue"
  )
  if quantize_fn is not None:
    w = quantize_fn(w)
  shifted = collect_fn_pre is not None
  fns = _layer_collect(collect_fn_dcn, collect_fn_ici_fwd, collect_fn_pre, w, w_aux)
  step_xs = functools.partial(_fwd_step_xs, w, w_aux, shifted=shifted)
  w_aux_0 = _layer_slice(w_aux, 0)
  w_first = fns.collect(_layer_slice(w, 0), w_aux_0)
  carry = prologue_fn.fwd(x, w_first, **_w_aux_kwargs(w_aux_0))[0]
  w_aux_1 = _layer_slice(w_aux, 1)
  w_second = fns.collect(_layer_slice(w, 1), w_aux_1)
  pre_carry = fns.pre_carry(2 if shifted else None)
  carry, aux_first = scan_body_fn.fwd(carry, w_first, w_second, **_w_aux_kwargs(w_aux_0))[0]

  def _scan_step(carry_tuple, xs):
    w_n_sharded, w_aux_n, pre_idx = xs
    activations, w_curr, w_aux_curr, pre_carry = carry_tuple
    if shifted:
      w_pre_n, w_aux_n = pre_carry
      w_next = fns.collect_ici(w_pre_n, w_aux_n)
    else:
      w_next = fns.collect(w_n_sharded, w_aux_n)
    pre_carry = fns.pre_carry(pre_idx)
    activations, aux_step = scan_body_fn.fwd(activations, w_curr, w_next, **_w_aux_kwargs(w_aux_curr))[0]
    return (activations, w_next, w_aux_n, pre_carry), aux_step

  def body(carry_tuple, xs_pair):
    xs_0, xs_1 = xs_pair
    carry_tuple, aux_0 = _scan_step(carry_tuple, xs_0)
    (carry_tuple, aux_0), xs_1 = jax.lax.optimization_barrier(((carry_tuple, aux_0), xs_1))
    carry_tuple, aux_1 = _scan_step(carry_tuple, xs_1)
    return carry_tuple, (aux_0, aux_1)

  num_scanned = num_layers - 3
  num_pairs = num_scanned // 2
  even = slice(2, 2 + 2 * num_pairs, 2)
  odd = slice(3, 2 + 2 * num_pairs, 2)
  (carry, w_curr, w_aux_curr, pre_carry), (aux_0, aux_1) = jax.lax.scan(
      body, (carry, w_second, w_aux_1, pre_carry), (step_xs(even), step_xs(odd)),
  )
  aux_parts = [jax.tree.map(lambda x: x[jnp.newaxis], aux_first)]
  if num_pairs > 0:
    aux_scanned = jax.tree.map(_interleave_arrays, aux_0, aux_1)
    aux_parts.append(aux_scanned)
  if num_scanned % 2 == 1:
    rem = 2 + 2 * num_pairs
    (carry, w_curr, w_aux_curr, pre_carry), aux_rem = _scan_step((carry, w_curr, w_aux_curr, pre_carry), step_xs(rem))
    aux_parts.append(jax.tree.map(lambda x: x[jnp.newaxis], aux_rem))
  (carry, w_last, w_aux_last, _), aux_last_sb = _scan_step(
      (carry, w_curr, w_aux_curr, pre_carry), step_xs(num_layers - 1),
  )
  aux_parts.append(jax.tree.map(lambda x: x[jnp.newaxis], aux_last_sb))
  out, aux_last = epilogue_fn.fwd(carry, w_last, **_w_aux_kwargs(w_aux_last))[0]
  aux_parts.append(jax.tree.map(lambda x: x[jnp.newaxis], aux_last))
  aux = jax.tree.map(lambda *parts: jnp.concatenate(parts, axis=0), *aux_parts)
  return out, aux


def _is_pinned_host(x: Any) -> bool:
  sharding = getattr(jax.typeof(x), "sharding", None)
  return getattr(sharding, "memory_kind", None) == "pinned_host"


@jt.jaxtyped(typechecker=typeguard.typechecked)
def w_collect_pipelined_scan_prologue_epilogue_fwd(
    prologue_fn: PipelinedScanLayer, scan_body_fn: PipelinedScanLayer, epilogue_fn: PipelinedScanLayer,
    x: AnyJaxArrayOrPyTree, w: AnyJaxArrayOrPyTree,
    collect_fn_dcn: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
    collect_fn_ici_fwd: Callable[..., AnyJaxArrayOrPyTree],
    reduce_fn_dcn: Callable[[AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
    reduce_fn_ici_first: Callable[..., AnyJaxArrayOrPyTree],
    reduce_fn_ici_second: Callable[..., AnyJaxArrayOrPyTree],
    collect_fn_ici_bwd: Callable[..., AnyJaxArrayOrPyTree] | None = None,
    quantize_fn: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree] | None = None,
    offload_fn: Callable[[Any], Any] = lambda res: res,
    load_fn: Callable[[Any], Any] = lambda res: res,
    w_aux: Any = None,
    collect_fn_pre: Callable[..., AnyJaxArrayOrPyTree] | None = None,
    reduce_fn_pre: Callable[..., AnyJaxArrayOrPyTree] | None = None,
) -> tuple[tuple[AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree], Any]:
  """Forward pass of w_collect_pipelined_scan_prologue_epilogue."""
  del load_fn, collect_fn_ici_bwd, reduce_fn_dcn
  del reduce_fn_ici_first, reduce_fn_ici_second
  num_layers = jax.tree.leaves(w)[0].shape[0]
  assert num_layers >= 3, (
      "there must be at least three layers to perform a pipelined scan and its"
      " transpose, which peel a scan body step at each end of the scan"
  )
  assert (collect_fn_pre is None) == (reduce_fn_pre is None), "collect_fn_pre and reduce_fn_pre must be given together"
  if quantize_fn is not None:
    w = quantize_fn(w)
  shifted = collect_fn_pre is not None
  fns = _layer_collect(collect_fn_dcn, collect_fn_ici_fwd, collect_fn_pre, w, w_aux)
  step_xs = functools.partial(_fwd_step_xs, w, w_aux, shifted=shifted)
  w_aux_0 = _layer_slice(w_aux, 0)
  with jax.named_scope("prologue_fwd"):
    w_first = fns.collect(_layer_slice(w, 0), w_aux_0)
    carry, res_prologue = prologue_fn.fwd(x, w_first, **_w_aux_kwargs(w_aux_0))
  w_aux_1 = _layer_slice(w_aux, 1)
  with jax.named_scope("scan_step_fwd_0"):
    w_second = fns.collect(_layer_slice(w, 1), w_aux_1)
    pre_carry = fns.pre_carry(2 if shifted else None)
    res_prologue_host = offload_fn(res_prologue)
    (carry, aux_first), res_sb_0 = scan_body_fn.fwd(carry, w_first, w_second, **_w_aux_kwargs(w_aux_0))
  res_sb_0_shaped = jax.eval_shape(offload_fn, res_sb_0)
  has_host_offload = any(_is_pinned_host(s) for s in jax.tree.leaves(res_sb_0_shaped))

  def _split_for_offload(res):
    if not has_host_offload:
      return None, res
    res_to_offload = jax.tree.map(
        lambda x, s: x if _is_pinned_host(s) else None, res, res_sb_0_shaped, is_leaf=lambda x: x is None,
    )
    res_device = jax.tree.map(
        lambda x, s: None if _is_pinned_host(s) else x, res, res_sb_0_shaped, is_leaf=lambda x: x is None,
    )
    return res_to_offload, res_device

  def _offload(res_to_offload):
    return None if res_to_offload is None else offload_fn(res_to_offload)

  res_sb_0_offload, res_sb_0_device = _split_for_offload(res_sb_0)

  def _scan_step_fwd(carry_tuple, xs):
    w_n_sharded, w_aux_n, pre_idx = xs
    activations, w_curr, w_aux_curr, pre_carry, res_prev_offload = carry_tuple
    res_prev_host = _offload(res_prev_offload)
    if shifted:
      w_pre_n, w_aux_n = pre_carry
      w_next = fns.collect_ici(w_pre_n, w_aux_n)
    else:
      w_next = fns.collect(w_n_sharded, w_aux_n)
    pre_carry = fns.pre_carry(pre_idx)
    (activations, aux_step), res_curr = scan_body_fn.fwd(activations, w_curr, w_next, **_w_aux_kwargs(w_aux_curr))
    res_curr_offload, res_curr_device = _split_for_offload(res_curr)
    return (activations, w_next, w_aux_n, pre_carry, res_curr_offload), (res_prev_host, res_curr_device, aux_step)

  @jax.named_call
  def body_fwd(carry_tuple, xs_pair):
    xs_0, xs_1 = xs_pair
    carry_tuple, (res_prev_0_host, res_curr_0_device, aux_0) = jax.named_call(_scan_step_fwd, name="unrolled_0")(
        carry_tuple, xs_0
    )
    (carry_tuple, aux_0), xs_1 = jax.lax.optimization_barrier(((carry_tuple, aux_0), xs_1))
    carry_tuple, (res_prev_1_host, res_curr_1_device, aux_1) = jax.named_call(_scan_step_fwd, name="unrolled_1")(
        carry_tuple, xs_1
    )
    return carry_tuple, ((res_prev_0_host, res_prev_1_host), (res_curr_0_device, res_curr_1_device), (aux_0, aux_1))

  num_scanned = num_layers - 3
  num_pairs = num_scanned // 2
  even = slice(2, 2 + 2 * num_pairs, 2)
  odd = slice(3, 2 + 2 * num_pairs, 2)
  (carry, w_curr, w_aux_curr, pre_carry, res_curr_sb_offload), (
      (res_prev_0_scanned_host, res_prev_1_scanned_host),
      (res_curr_0_scanned_device, res_curr_1_scanned_device), (aux_0, aux_1),
  ) = jax.lax.scan(body_fwd, (carry, w_second, w_aux_1, pre_carry, res_sb_0_offload), (step_xs(even), step_xs(odd)))
  aux_parts = [jax.tree.map(lambda x: x[jnp.newaxis], aux_first)]
  if num_pairs > 0:
    aux_scanned = jax.tree.map(_interleave_arrays, aux_0, aux_1)
    aux_parts.append(aux_scanned)
  res_rem_host = None
  res_rem_device = None
  if num_scanned % 2 == 1:
    rem = 2 + 2 * num_pairs
    (carry, w_curr, w_aux_curr, pre_carry, res_curr_sb_offload), (res_rem_host, res_rem_device, aux_rem) = (
        _scan_step_fwd((carry, w_curr, w_aux_curr, pre_carry, res_curr_sb_offload), step_xs(rem))
    )
    aux_parts.append(jax.tree.map(lambda x: x[jnp.newaxis], aux_rem))
  with jax.named_scope("scan_step_fwd_last"):
    (carry, w_last, w_aux_last, _, res_last_sb_offload), (
        res_penultimate_sb_host, res_last_sb_device, aux_last_sb,
    ) = _scan_step_fwd((carry, w_curr, w_aux_curr, pre_carry, res_curr_sb_offload), step_xs(num_layers - 1))
  aux_parts.append(jax.tree.map(lambda x: x[jnp.newaxis], aux_last_sb))
  with jax.named_scope("epilogue_fwd"):
    res_last_sb_host = _offload(res_last_sb_offload)
    (out, aux_last), res_epilogue = epilogue_fn.fwd(carry, w_last, **_w_aux_kwargs(w_aux_last))
  res_epilogue_host = offload_fn(res_epilogue)
  aux_parts.append(jax.tree.map(lambda x: x[jnp.newaxis], aux_last))
  aux = jax.tree.map(lambda *parts: jnp.concatenate(parts, axis=0), *aux_parts)
  res = (
      res_prologue_host, res_sb_0_device,
      ((res_prev_0_scanned_host, res_prev_1_scanned_host), (res_curr_0_scanned_device, res_curr_1_scanned_device),
       res_rem_host, res_rem_device),
      res_penultimate_sb_host, res_last_sb_host, res_last_sb_device, res_epilogue_host, w, w_aux,
  )
  return (out, aux), res


@jt.jaxtyped(typechecker=typeguard.typechecked)
def w_collect_pipelined_scan_prologue_epilogue_bwd(
    prologue_fn: PipelinedScanLayer, scan_body_fn: PipelinedScanLayer, epilogue_fn: PipelinedScanLayer,
    collect_fn_dcn: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
    collect_fn_ici_fwd: Callable[..., AnyJaxArrayOrPyTree],
    reduce_fn_dcn: Callable[[AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
    reduce_fn_ici_first: Callable[..., AnyJaxArrayOrPyTree],
    reduce_fn_ici_second: Callable[..., AnyJaxArrayOrPyTree],
    collect_fn_ici_bwd: Callable[..., AnyJaxArrayOrPyTree] | None,
    quantize_fn: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree] | None,
    offload_fn: Callable[[Any], Any], load_fn: Callable[[Any], Any],
    collect_fn_pre: Callable[..., AnyJaxArrayOrPyTree] | None,
    reduce_fn_pre: Callable[..., AnyJaxArrayOrPyTree] | None,
    res: Any, grad_outputs: tuple[AnyJaxArrayOrPyTree, Any],
) -> tuple[AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree, None]:
  """Backward pass of w_collect_pipelined_scan_prologue_epilogue."""
  del offload_fn, quantize_fn
  if collect_fn_ici_bwd is None:
    collect_fn_ici_bwd = collect_fn_ici_fwd
  del collect_fn_ici_fwd
  (
      res_prologue_host, res_sb_0_device,
      ((res_prev_0_scanned_host, res_prev_1_scanned_host), (res_curr_0_scanned_device, res_curr_1_scanned_device),
       res_rem_host, res_rem_device),
      res_penultimate_sb_host, res_last_sb_host, res_last_sb_device, res_epilogue_host, w, w_aux,
  ) = res
  num_layers = jax.tree.leaves(w)[0].shape[0]
  shifted = collect_fn_pre is not None
  fns = _layer_collect(collect_fn_dcn, collect_fn_ici_bwd, collect_fn_pre, w, w_aux)

  def _load(res_host):
    return None if res_host is None else load_fn(res_host)

  def _merge_offloaded(res_loaded, res_device):
    if res_loaded is None:
      return res_device
    return jax.tree.map(lambda l, d: d if l is None else l, res_loaded, res_device, is_leaf=lambda x: x is None)

  out_grad, aux_grad = grad_outputs
  aux_step_0 = jax.tree.map(lambda g: g[0], aux_grad)
  aux_scanned_grad = jax.tree.map(lambda g: g[1:-2], aux_grad)
  aux_last_sb_grad = jax.tree.map(lambda g: g[-2], aux_grad)
  aux_last_grad = jax.tree.map(lambda g: g[-1], aux_grad)
  res_epilogue_device = load_fn(res_epilogue_host)
  w_last_sharded = _layer_slice(w, -1)
  w_last_dcn = collect_fn_dcn(w_last_sharded)
  w_aux_last = _layer_slice(w_aux, -1)
  w_last = fns.collect_ici(fns.pre(w_last_dcn, w_aux_last), w_aux_last)

  def _reduce_ici_first(w_unreduced, w_aux_layer):
    return reduce_fn_ici_first(w_unreduced, w_last_dcn, **_w_aux_kwargs(w_aux_layer))

  def _reduce_ici_second(w_partial, w_aux_layer, **kwargs):
    return reduce_fn_ici_second(w_partial, w_last_dcn, **_w_aux_kwargs(w_aux_layer), **kwargs)

  def _reduce_pre(w_partial, w_aux_layer):
    if reduce_fn_pre is None:
      return w_partial
    return reduce_fn_pre(w_partial, **_w_aux_kwargs(w_aux_layer))

  def _reduce_dcn(w_ici):
    return reduce_fn_dcn(w_ici, w_last_sharded)

  with jax.named_scope("epilogue_bwd"):
    res_last_sb_loaded = _load(res_last_sb_host)
    w_aux_penultimate = _layer_slice(w_aux, -2)
    w_curr = fns.collect(_layer_slice(w, -2), w_aux_penultimate)
    pre_carry = fns.pre_carry(num_layers - 3 if shifted else None)
    carry_grad, w_last_expert_grad_unreduced = epilogue_fn.bwd(
        res_epilogue_device, (out_grad, aux_last_grad), w_last, **_w_aux_kwargs(w_aux_last),
    )

  def _scan_step_bwd(carry, xs):
    (
        carry_grad, w_prev_expert_grad_unreduced, w_curr, w_next, w_aux_curr, w_aux_next,
        res_curr_loaded, w_partial_pending, pre_carry,
    ) = carry
    res_curr_device, res_prev_host, w_prev_sharded, w_aux_prev, aux_step_grad, pre_idx = xs
    w_partial_pending_grad, w_aux_partial_pending = (None, None) if w_partial_pending is None else w_partial_pending
    if w_partial_pending is not None:
      w_partial_pending_grad = _reduce_pre(w_partial_pending_grad, w_aux_partial_pending)
    res_prev_loaded = _load(res_prev_host)
    res_curr_full = _merge_offloaded(res_curr_loaded, res_curr_device)
    if shifted:
      w_prev_pre, w_aux_prev = pre_carry
    else:
      w_prev_pre = collect_fn_dcn(w_prev_sharded)
    w_prev_collected = fns.collect_ici(w_prev_pre, w_aux_prev)
    pre_carry = fns.pre_carry(pre_idx)

    def _reduce_ici_first_with_collect(w_unreduced):
      w_partial = _reduce_ici_first(w_unreduced, w_aux_next)
      w_partial, _ = jax.lax.optimization_barrier((w_partial, w_prev_collected))
      return w_partial

    (carry_grad, w_curr_expert_grad_unreduced, w_next_grad_partial, w_partial_pending_grad_ici) = scan_body_fn.bwd(
        res_curr_full, (carry_grad, aux_step_grad), w_curr, w_next, w_prev_expert_grad_unreduced,
        _reduce_ici_first_with_collect, w_partial_pending_grad,
        functools.partial(_reduce_ici_second, w_aux_layer=w_aux_partial_pending), **_w_aux_kwargs(w_aux_curr),
    )
    w_pending_reduced = None if w_partial_pending is None else _reduce_dcn(w_partial_pending_grad_ici)
    return (
        carry_grad, w_curr_expert_grad_unreduced, w_prev_collected, w_curr, w_aux_prev, w_aux_curr,
        res_prev_loaded, (w_next_grad_partial, w_aux_next), pre_carry,
    ), w_pending_reduced

  def _step_xs(res_curr_device, res_prev_host, layers, aux_step_grad):
    if shifted:
      pre_idx = _shifted_indices(layers, -1, num_layers)
      return (res_curr_device, res_prev_host, None, None, aux_step_grad, pre_idx)
    return (res_curr_device, res_prev_host, _layer_slice(w, layers), _layer_slice(w_aux, layers), aux_step_grad, None)

  @jax.named_call
  def body_bwd(carry, xs_pair):
    xs_0, xs_1 = xs_pair
    carry_in = carry
    carry, w_1_reduced = jax.named_call(_scan_step_bwd, name="unrolled_1")(carry, xs_1)
    carry_for_barrier = jax.tree.map(lambda x_in, x_out: None if x_out is x_in else x_out, carry_in, carry)
    res_curr_0_device, *xs_0_rest = xs_0
    (carry_for_barrier, w_1_reduced), xs_0_rest = jax.lax.optimization_barrier(
        ((carry_for_barrier, w_1_reduced), xs_0_rest)
    )
    xs_0 = (res_curr_0_device, *xs_0_rest)
    carry = jax.tree.map(
        lambda x_in, x_bar: x_in if x_bar is None else x_bar, carry_in, carry_for_barrier,
        is_leaf=lambda x: x is None,
    )
    carry, w_0_reduced = jax.named_call(_scan_step_bwd, name="unrolled_0")(carry, xs_0)
    return carry, (w_0_reduced, w_1_reduced)

  num_scanned = num_layers - 3
  num_pairs = num_scanned // 2
  carry_bwd = (
      carry_grad, w_last_expert_grad_unreduced, w_curr, w_last, w_aux_penultimate, w_aux_last,
      res_last_sb_loaded, None, pre_carry,
  )
  with jax.named_scope("scan_step_bwd_last"):
    carry_bwd, _ = _scan_step_bwd(
        carry_bwd, _step_xs(res_last_sb_device, res_penultimate_sb_host, num_layers - 3, aux_last_sb_grad),
    )
  w_rem_reduced = None
  if num_scanned % 2 == 1:
    xs_rem = _step_xs(
        res_rem_device, res_rem_host, num_scanned - 1, jax.tree.map(lambda g: g[-1], aux_scanned_grad),
    )
    carry_bwd, w_rem_reduced = _scan_step_bwd(carry_bwd, xs_rem)
  w_grads_reduced = None
  if num_pairs > 0:
    even = slice(0, 2 * num_pairs, 2)
    odd = slice(1, 1 + 2 * num_pairs, 2)
    aux_even_grad = jax.tree.map(lambda g: g[0 : 2 * num_pairs : 2], aux_scanned_grad)
    aux_odd_grad = jax.tree.map(lambda g: g[1 : 2 * num_pairs : 2], aux_scanned_grad)
    xs_0 = _step_xs(res_curr_0_scanned_device, res_prev_0_scanned_host, even, aux_even_grad)
    xs_1 = _step_xs(res_curr_1_scanned_device, res_prev_1_scanned_host, odd, aux_odd_grad)
    carry_bwd, (w_0_reduced_scanned, w_1_reduced_scanned) = jax.lax.scan(
        body_bwd, carry_bwd, (xs_0, xs_1), reverse=True,
    )
    w_grads_reduced = jax.tree.map(_interleave_arrays, w_0_reduced_scanned, w_1_reduced_scanned)
  (
      carry_grad, w_second_expert_grad_unreduced, w_0, w_1, w_aux_0, w_aux_1, res_sb_0_loaded,
      (w_third_grad_partial, w_aux_2), _,
  ) = carry_bwd
  with jax.named_scope("scan_step_bwd_0"):
    w_third_grad_partial = _reduce_pre(w_third_grad_partial, w_aux_2)
    res_prologue_device = load_fn(res_prologue_host)
    res_sb_0_full = _merge_offloaded(res_sb_0_loaded, res_sb_0_device)
    (carry_grad, w_0_expert_grad_unreduced, w_second_grad_partial, w_third_grad_ici) = scan_body_fn.bwd(
        res_sb_0_full, (carry_grad, aux_step_0), w_0, w_1, w_second_expert_grad_unreduced,
        functools.partial(_reduce_ici_first, w_aux_layer=w_aux_1), w_third_grad_partial,
        functools.partial(_reduce_ici_second, w_aux_layer=w_aux_2), **_w_aux_kwargs(w_aux_0),
    )
    w_third_grad_reduced = _reduce_dcn(w_third_grad_ici)
  with jax.named_scope("prologue_bwd"):
    w_second_grad_reduced = _reduce_dcn(_reduce_ici_second(_reduce_pre(w_second_grad_partial, w_aux_1), w_aux_1))
    x_grad, w_0_mla_grad_unreduced = prologue_fn.bwd(res_prologue_device, carry_grad, w_0, **_w_aux_kwargs(w_aux_0))
    w_0_unreduced = combine_w_grads(w_0_mla_grad_unreduced, w_0_expert_grad_unreduced)
    w_first_grad_reduced = _reduce_dcn(
        _reduce_ici_second(_reduce_pre(_reduce_ici_first(w_0_unreduced, w_aux_0), w_aux_0), w_aux_0)
    )
  w_grad_parts = [
      jax.tree.map(lambda x: x[jnp.newaxis], g)
      for g in (w_first_grad_reduced, w_second_grad_reduced, w_third_grad_reduced)
  ]
  if w_grads_reduced is not None:
    w_grad_parts.append(w_grads_reduced)
  if w_rem_reduced is not None:
    w_grad_parts.append(jax.tree.map(lambda x: x[jnp.newaxis], w_rem_reduced))
  w_grad = jax.tree.map(lambda *parts: jnp.concatenate(parts, axis=0), *w_grad_parts)
  return x_grad, w_grad, None


w_collect_pipelined_scan_prologue_epilogue.defvjp(
    w_collect_pipelined_scan_prologue_epilogue_fwd, w_collect_pipelined_scan_prologue_epilogue_bwd,
)
