"""FSDP weight prefetching, one sub-layer at a time, for inhomogeneous scanned blocks (e.g. Qwen3.5).

`fsdp_prefetch` prefetches whole scan steps. When one scan step is a block of sub-layers with
different parameter structures (e.g. Qwen3.5's mix of Gated DeltaNet and global attention layers),
that gathers a whole block's weights at once. A block that defines `prefetch_sublayer_names()` is
instead prefetched one sub-layer at a time here: the all-gather of sub-layer j+1 overlaps sub-layer
j, across block boundaries, in both passes, and each sub-layer runs in a while-loop nested in the
block scan, as in the block's regular nested scan. The gathers, reduce-scatters and remat policy
handling are shared with `fsdp_prefetch`.

See `_custom_vjp_nested_prefetch_pipeline` for the design.
"""

# pylint: disable=protected-access

import functools

from flax import nnx
import jax
from jax.experimental import xla_metadata
import jax.numpy as jnp
from lilax._backend.layers import fsdp_prefetch
from lilax._backend.observability import max_logging


def _index_at(tree, idx):
  """Layer `idx` (a traced index) of a layer-stacked pytree."""
  return jax.tree.map(lambda x: jax.lax.dynamic_index_in_dim(x, idx, axis=0, keepdims=False), tree)


_SKIP_TRIP_COUNT_ONE = {"skip-simplify-while-loops_trip-count-one": "true"}


def _sublayer_scan(body, carry, xs, reverse=False, unroll=1):
  """`jax.lax.scan` over sub-layers that XLA keeps as a while-loop, even with a single trip.

  Each sub-layer run is its own while-loop nested in the block scan, as in the regular nested scan
  of the block (see `Qwen3NextScannableBlock`). The loop boundary keeps a prefetched all-gather
  inside the sub-layer that issues it: XLA cannot finish it early, ahead of that sub-layer's compute.

  With `unroll=2`, each loop trip runs two consecutive steps; an odd remaining step runs in its own
  single-trip loop after them. Unlike `jax.lax.scan(..., unroll=2)`, this keeps a while-loop when
  there is a single 2-step trip, which `jax.lax.scan` would unroll into the parent computation.
  """
  with xla_metadata.set_xla_metadata(**_SKIP_TRIP_COUNT_ONE):
    if unroll == 1:
      return jax.lax.scan(body, carry, xs, reverse=reverse)
    if unroll != 2:
      raise ValueError(f"unroll must be 1 or 2, got {unroll}.")
    length = jax.tree_util.tree_leaves(xs)[0].shape[0]
    trips, rem = divmod(length, 2)
    # Steps run in index order, or in reverse index order with `reverse`; the odd step runs last.
    split = rem if reverse else 2 * trips
    head = jax.tree.map(lambda x: x[:split], xs)
    rest = jax.tree.map(lambda x: x[split:], xs)
    xs_paired, xs_odd = (rest, head) if reverse else (head, rest)

    def two_steps(carry, xs2):
      first, second = (1, 0) if reverse else (0, 1)
      carry, y_a = body(carry, jax.tree.map(lambda x: x[first], xs2))
      carry, y_b = body(carry, jax.tree.map(lambda x: x[second], xs2))
      ys2 = (y_b, y_a) if reverse else (y_a, y_b)
      return carry, jax.tree.map(lambda a, b: jnp.stack((a, b)), *ys2)

    xs_paired = jax.tree.map(lambda x: x.reshape(trips, 2, *x.shape[1:]), xs_paired)
    carry, ys = jax.lax.scan(two_steps, carry, xs_paired, reverse=reverse)
    ys = jax.tree.map(lambda y: y.reshape(2 * trips, *y.shape[2:]), ys)
    if rem:
      carry, ys_odd = jax.lax.scan(body, carry, xs_odd, reverse=reverse)
      ys = _concat0(ys_odd, ys) if reverse else _concat0(ys, ys_odd)
    return carry, ys


def _concat0(*trees):
  """Concatenates layer-stacked pytrees along the leading axis."""
  return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *trees)


def _setup_groups(graphdefs, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, ctxs):
  """`_prefetch_setup` for each sub-layer group; returns per-group tuples (gather_fn, rs_fn, ...).

  Group params/state leaves are stacked `[length, n, ...]`; the setup sees sub-layer 0's
  `[length, ...]` slice, which has the per-sub-layer shardings.
  """
  setups = [
      fsdp_prefetch._prefetch_setup(
          g, tuple(x[:, 0] for x in p), tuple(x[:, 0] for x in s), diff_args_leaves, diff_kwargs_leaves, ctx
      )
      for g, p, s, ctx in zip(graphdefs, params_leaves, state_leaves, ctxs)
  ]
  _, _, gather, rs_fns, layout_fns, fwd_layers, layer_vjp_fns = zip(*setups)
  params = [jax.tree_util.tree_unflatten(ctx.params_treedef, p) for p, ctx in zip(params_leaves, ctxs)]
  state = [jax.tree_util.tree_unflatten(ctx.state_treedef, s) for s, ctx in zip(state_leaves, ctxs)]
  return params, state, gather, rs_fns, layout_fns, fwd_layers, layer_vjp_fns


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 6, 7, 8))
def _custom_vjp_nested_prefetch_pipeline(
    graphdefs, x_in, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, length, ctxs, sizes
):
  out, _ = _custom_vjp_nested_prefetch_pipeline_fwd(
      graphdefs, x_in, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, length, ctxs, sizes
  )
  return out


def _custom_vjp_nested_prefetch_pipeline_fwd(
    graphdefs, x_in, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, length, ctxs, sizes
):
  num_groups = len(graphdefs)
  params, state, gather, _, layout_fns, fwd_layers, _ = _setup_groups(
      graphdefs, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, ctxs
  )
  layouts = {}

  def fwd(i, y, w, layer_state):
    if i not in layouts:
      layouts[i] = layout_fns[i](y, w, layer_state)
    return fwd_layers[i](y, w, layer_state, layouts[i])

  def body(carry, xs):
    y, w = carry
    b, b_state, b_params = xs
    ys, saved = [], []
    for i in range(num_groups):
      n = sizes[i]
      if i + 1 < num_groups:
        nxt_first, gather_nxt = fsdp_prefetch._slice_at(b_params[i + 1], 0), gather[i + 1]
      else:
        nxt_first = fsdp_prefetch._slice_at(_index_at(params[0], (b + 1) % length), 0)
        gather_nxt = gather[0]
      parts_y, parts_saved = [], []
      if n >= 2:
        ordered = n - 1 >= 2

        def same_group(carry, k, i=i, p=b_params[i], s=b_state[i], ordered=ordered):
          y, w = carry
          p_next = _index_at(p, k + 1)
          if ordered:
            y, p_next = jax.lax.optimization_barrier((y, p_next))
          w_next = gather[i](p_next)
          y_out, saved_k = fwd(i, y, w, _index_at(s, k))
          return (y_out, w_next), (y, saved_k)

        (y, w), (ys_k, saved_k) = _sublayer_scan(same_group, (y, w), jnp.arange(n - 1), unroll=2 if ordered else 1)
        parts_y.append(ys_k)
        parts_saved.append(saved_k)

      def last(carry, xs_last, i=i, gather_nxt=gather_nxt):
        y, w = carry
        layer_state, nxt = xs_last
        w_next = gather_nxt(nxt)
        y_out, saved_last = fwd(i, y, w, layer_state)
        return (y_out, w), (w_next, saved_last)

      last_xs = fsdp_prefetch._expand0((fsdp_prefetch._slice_at(b_state[i], n - 1), nxt_first))
      y_last_in = y
      (y, _), (w_next, saved_last) = _sublayer_scan(last, (y, w), last_xs)
      w = fsdp_prefetch._slice_at(w_next, 0)
      parts_y.append(fsdp_prefetch._expand0(y_last_in))
      parts_saved.append(saved_last)
      ys.append(_concat0(*parts_y))
      saved.append(_concat0(*parts_saved))
    return (y, w), (tuple(ys), tuple(saved))

  w = gather[0](fsdp_prefetch._slice_at(fsdp_prefetch._slice_at(params[0], 0), 0))
  xs = (jnp.arange(length), tuple(state), tuple(params))
  (y, _), (ys, saved) = jax.lax.scan(body, (x_in, w), xs)
  residuals = (ys, saved, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves)
  return (y, state_leaves), residuals


def _custom_vjp_nested_prefetch_pipeline_bwd(graphdefs, length, ctxs, sizes, res, g_out):
  if ctxs[0].defer_grad_reduce_scatter:
    return _nested_prefetch_pipeline_bwd_deferred(graphdefs, length, ctxs, sizes, res, g_out)
  ys, saved, params_leaves, state_leaves, d_args, d_kwargs = res
  g, _ = g_out
  num_groups = len(graphdefs)
  params, state, gather, rs_fns, layout_fns, _, layer_vjp_fns = _setup_groups(
      graphdefs, params_leaves, state_leaves, d_args, d_kwargs, ctxs
  )
  g, params = jax.lax.optimization_barrier((g, tuple(params)))
  layouts = {}

  def bwd(i, y, w, layer_state, g, saved_k):
    if i not in layouts:
      layouts[i] = layout_fns[i](y, w, layer_state)
    g, dw = layer_vjp_fns[i](y, w, layer_state, g, saved_k, layouts[i])
    return g, rs_fns[i](dw)

  def body(carry, xs):
    g, w = carry
    b, b_ys, b_saved, b_state, b_params = xs
    dws = [None] * num_groups
    for i in reversed(range(num_groups)):
      n = sizes[i]
      if i > 0:
        prv_last, gather_prv = fsdp_prefetch._slice_at(b_params[i - 1], sizes[i - 1] - 1), gather[i - 1]
      else:
        prv_last = fsdp_prefetch._slice_at(_index_at(params[-1], (b - 1) % length), sizes[-1] - 1)
        gather_prv = gather[-1]
      d_rest = None
      if n >= 2:
        ordered = n - 1 >= 2

        def same_group(carry, k, i=i, p=b_params[i], s=b_state[i], y_=b_ys[i], sv=b_saved[i], ordered=ordered):
          g, w = carry
          p_prev, y_k, saved_k = _index_at(p, k - 1), _index_at(y_, k), _index_at(sv, k)
          if ordered:
            g, y_k, saved_k, p_prev = jax.lax.optimization_barrier((g, y_k, saved_k, p_prev))
          w_prev = gather[i](p_prev)
          g, dw = bwd(i, y_k, w, _index_at(s, k), g, saved_k)
          if ordered:
            g, dw = jax.lax.optimization_barrier((g, dw))
          return (g, w_prev), dw

        (g, w), d_rest = _sublayer_scan(same_group, (g, w), jnp.arange(1, n), reverse=True, unroll=2 if ordered else 1)

      def first(carry, xs_first, i=i, gather_prv=gather_prv):
        g, w = carry
        y0, saved0, layer_state, prv = xs_first
        w_prev = gather_prv(prv)
        g, dw = bwd(i, y0, w, layer_state, g, saved0)
        return (g, w), (w_prev, dw)

      first_xs = fsdp_prefetch._expand0(
          tuple(fsdp_prefetch._slice_at(t, 0) for t in (b_ys[i], b_saved[i], b_state[i])) + (prv_last,)
      )
      (g, _), (w_prev, dw0) = _sublayer_scan(first, (g, w), first_xs)
      w = fsdp_prefetch._slice_at(w_prev, 0)
      dws[i] = dw0 if d_rest is None else _concat0(dw0, d_rest)
    return (g, w), tuple(dws)

  w = gather[-1](fsdp_prefetch._slice_at(fsdp_prefetch._slice_at(params[-1], length - 1), sizes[-1] - 1))
  xs = (jnp.arange(length), ys, saved, tuple(state), tuple(params))
  (g, _), d_params = jax.lax.scan(body, (g, w), xs, reverse=True)
  params_bar_leaves = tuple(tuple(jax.tree_util.tree_leaves(d)) for d in d_params)
  diff_args_bar_leaves = tuple(jnp.zeros_like(x) for x in d_args)
  diff_kwargs_bar_leaves = tuple(jnp.zeros_like(x) for x in d_kwargs)
  return g, params_bar_leaves, None, diff_args_bar_leaves, diff_kwargs_bar_leaves


def _update_at(a, d, b, k):
  return jax.lax.dynamic_update_slice(a, d[None, None].astype(a.dtype), (b, k) + (0,) * d.ndim)


def _nested_prefetch_pipeline_bwd_deferred(graphdefs, length, ctxs, sizes, res, g_out):
  ys, saved, params_leaves, state_leaves, d_args, d_kwargs = res
  g, _ = g_out
  num_groups = len(graphdefs)
  params, state, gather, rs_fns, layout_fns, _, layer_vjp_fns = _setup_groups(
      graphdefs, params_leaves, state_leaves, d_args, d_kwargs, ctxs
  )
  g, params = jax.lax.optimization_barrier((g, tuple(params)))
  layouts = {}

  def bwd(i, y, w, layer_state, g, saved_k):
    if i not in layouts:
      layouts[i] = layout_fns[i](y, w, layer_state)
    g, dw = layer_vjp_fns[i](y, w, layer_state, g, saved_k, layouts[i])
    return g, rs_fns[i](dw, "untagged")

  def deferred(i, dw):
    leaves, treedef = jax.tree_util.tree_flatten(dw)
    flags = rs_fns[i].tag_flags(leaves)
    return jax.tree_util.tree_unflatten(treedef, [x if f else jnp.zeros((0,), x.dtype) for x, f in zip(leaves, flags)])

  def write(d_acc, i, dw, b, k, which):
    acc_leaves, treedef = jax.tree_util.tree_flatten(d_acc[i])
    d_leaves = jax.tree_util.tree_leaves(dw)
    flags = rs_fns[i].tag_flags(d_leaves)
    out = [_update_at(a, d, b, k) if (which == "tagged") == f else a for a, d, f in zip(acc_leaves, d_leaves, flags)]
    return d_acc[:i] + (jax.tree_util.tree_unflatten(treedef, out),) + d_acc[i + 1 :]

  def flush(d_acc, owner, pend):
    i, b, k = owner
    return write(d_acc, i, rs_fns[i](pend, "tagged"), b, k, "tagged")

  def boundary_step(i, k, b, gather_fn, owner):
    def step(carry, xs_step):
      g, w, d_acc, pend, _ = carry
      y_k, saved_k, layer_state, p_prev = xs_step
      d_acc = flush(d_acc, owner, pend)
      w_prev = gather_fn(p_prev)
      g, dw = bwd(i, y_k, w, layer_state, g, saved_k)
      d_acc = write(d_acc, i, dw, b, k, "untagged")
      return (g, w, d_acc, pend, deferred(i, dw)), w_prev
    return step

  pending_avals = {}

  def pending_zeros(i):
    if i not in pending_avals:
      def first_dw(g):
        y0, saved0, state0 = (jax.tree.map(lambda x: x[0, 0], t) for t in (ys[i], saved[i], state[i]))
        w0 = gather[i](jax.tree.map(lambda x: x[0, 0], params[i]))
        _, dw = layer_vjp_fns[i](y0, w0, state0, g, saved0, layout_fns[i](y0, w0, state0))
        return deferred(i, rs_fns[i](dw, "untagged"))
      pending_avals[i] = jax.eval_shape(first_dw, g)
    return jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype, out_sharding=s.sharding), pending_avals[i])

  def body(carry, xs):
    g, w, d_acc, pend = carry
    b, b_ys, b_saved, b_state, b_params = xs
    owner = (0, (b + 1) % length, 0)
    for i in reversed(range(num_groups)):
      n = sizes[i]
      if i > 0:
        prv_last, gather_prv = fsdp_prefetch._slice_at(b_params[i - 1], sizes[i - 1] - 1), gather[i - 1]
      else:
        prv_last = fsdp_prefetch._slice_at(_index_at(params[-1], (b - 1) % length), sizes[-1] - 1)
        gather_prv = gather[-1]

      def run_boundary(carry, k, p_prev, gather_fn, owner, pend, i=i):
        xs_step = fsdp_prefetch._expand0(
            tuple(fsdp_prefetch._slice_at(t, k) for t in (b_ys[i], b_saved[i], b_state[i])) + (p_prev,)
        )
        g, w, d_acc = carry
        carry = (g, w, d_acc, pend, pending_zeros(i))
        (g, _, d_acc, _, new_pend), w_prev = _sublayer_scan(boundary_step(i, k, b, gather_fn, owner), carry, xs_step)
        return (g, fsdp_prefetch._slice_at(w_prev, 0), d_acc), new_pend

      if n >= 2:
        entry_prev, entry_gather = fsdp_prefetch._slice_at(b_params[i], n - 2), gather[i]
      else:
        entry_prev, entry_gather = prv_last, gather_prv
      (g, w, d_acc), pend = run_boundary((g, w, d_acc), n - 1, entry_prev, entry_gather, owner, pend)
      owner = (i, b, n - 1)
      if n >= 3:
        ordered = n - 2 >= 2

        def same_group(carry, k, i=i, p=b_params[i], s=b_state[i], y_=b_ys[i], sv=b_saved[i], ordered=ordered):
          g, w, d_acc, pend = carry
          p_prev, y_k, saved_k = _index_at(p, k - 1), _index_at(y_, k), _index_at(sv, k)
          if ordered:
            g, y_k, saved_k, p_prev = jax.lax.optimization_barrier((g, y_k, saved_k, p_prev))
          d_acc = flush(d_acc, (i, b, k + 1), pend)
          w_prev = gather[i](p_prev)
          g, dw = bwd(i, y_k, w, _index_at(s, k), g, saved_k)
          if ordered:
            g, dw = jax.lax.optimization_barrier((g, dw))
          d_acc = write(d_acc, i, dw, b, k, "untagged")
          return (g, w_prev, d_acc, deferred(i, dw)), None

        (g, w, d_acc, pend), _ = _sublayer_scan(
            same_group, (g, w, d_acc, pend), jnp.arange(1, n - 1), reverse=True, unroll=2 if ordered else 1
        )
        owner = (i, b, 1)
      if n >= 2:
        (g, w, d_acc), pend = run_boundary((g, w, d_acc), 0, prv_last, gather_prv, owner, pend)
        owner = (i, b, 0)
    return (g, w, d_acc, pend), None

  w = gather[-1](fsdp_prefetch._slice_at(fsdp_prefetch._slice_at(params[-1], length - 1), sizes[-1] - 1))
  d_acc = tuple(jax.tree.map(jnp.zeros_like, p) for p in params)
  pend = pending_zeros(0)
  xs = (jnp.arange(length), ys, saved, tuple(state), tuple(params))
  (g, _, d_acc, pend), _ = jax.lax.scan(body, (g, w, d_acc, pend), xs, reverse=True)
  d_acc = flush(d_acc, (0, 0, 0), pend)
  params_bar_leaves = tuple(tuple(jax.tree_util.tree_leaves(d)) for d in d_acc)
  diff_args_bar_leaves = tuple(jnp.zeros_like(x) for x in d_args)
  diff_kwargs_bar_leaves = tuple(jnp.zeros_like(x) for x in d_kwargs)
  return g, params_bar_leaves, None, diff_args_bar_leaves, diff_kwargs_bar_leaves


_custom_vjp_nested_prefetch_pipeline.defvjp(
    _custom_vjp_nested_prefetch_pipeline_fwd, _custom_vjp_nested_prefetch_pipeline_bwd
)


def _prefetch_sublayer_names(layers):
  names_fn = getattr(layers, "prefetch_sublayer_names", None)
  if names_fn is None:
    return None
  names = tuple(names_fn())
  return names if len(names) >= 2 else None


def _unstack_variable(var, num_local: int, var_scan_axis: int):
  if not isinstance(var, nnx.Variable):
    return (var,) * num_local
  val = var.get_value()
  meta = var.get_metadata()
  updates = {}
  for key in ("sharding", "out_sharding", "kernel_axes", "sharding_names"):
    axes = getattr(var, key, None)
    if axes is None:
      axes = meta.get(key)
    if isinstance(axes, (jax.sharding.PartitionSpec, tuple, list)) and "local_layers" in axes:
      filtered = [a for a in axes if a != "local_layers"]
      updates[key] = (
          jax.sharding.PartitionSpec(*filtered) if isinstance(axes, jax.sharding.PartitionSpec) else tuple(filtered)
      )
  ndim = getattr(val, "ndim", 0)
  if meta.get(nnx.PARTITION_NAME) == "local_layers":
    local_axis = var_scan_axis if ndim > var_scan_axis else 0
  else:
    inner_axis = var_scan_axis if ndim - 1 > var_scan_axis else 0
    outer_axis = var_scan_axis if ndim > var_scan_axis else 0
    local_axis = inner_axis + 1 if inner_axis >= outer_axis else inner_axis
  if ndim <= local_axis or val.shape[local_axis] != num_local:
    stripped = var.replace(**updates) if updates else var
    return (stripped,) * num_local
  slices = jnp.unstack(val, axis=local_axis)
  return tuple(var.replace(value=s, **updates) for s in slices)


def _unstack_scanned_sublayers(sublayers, num_local: int, scan_axis: int):
  graphdef, params, state = nnx.split(sublayers, nnx.Param, ...)

  def unstack_state(tree, var_scan_axis):
    leaves, treedef = jax.tree_util.tree_flatten(tree, is_leaf=lambda x: isinstance(x, nnx.Variable))
    if not leaves:
      return [tree] * num_local
    unstacked = [_unstack_variable(leaf, num_local, var_scan_axis) for leaf in leaves]
    return [jax.tree_util.tree_unflatten(treedef, [u[i] for u in unstacked]) for i in range(num_local)]

  params_per_stage = unstack_state(params, scan_axis)
  state_per_stage = unstack_state(state, 0)
  return [(graphdef, p, s) for p, s in zip(params_per_stage, state_per_stage)]


def sublayer_stages(layers):
  num_local = getattr(layers, "num_local", 0)
  num_global = getattr(layers, "num_global", 0)
  local_layers = getattr(layers, "local_layers", None)
  global_layer = getattr(layers, "global_layer", None)
  if (local_layers is not None or global_layer is not None) and num_local + num_global >= 2:
    scan_axis = getattr(getattr(layers, "config", None), "param_scan_axis", 0)
    stages = []
    if local_layers is not None and num_local > 0:
      stages.extend(_unstack_scanned_sublayers(local_layers, num_local, scan_axis))
    if global_layer is not None and num_global == 1:
      stages.append(nnx.split(global_layer, nnx.Param, ...))
  else:
    names = _prefetch_sublayer_names(layers)
    if names is None:
      return None
    stages = [nnx.split(getattr(layers, name), nnx.Param, ...) for name in names]
  if len(stages) < 2:
    return None
  block_params = nnx.state(layers, nnx.Param)
  stage_param_size = sum(x.size for _, p, _ in stages for x in jax.tree_util.tree_leaves(p))
  block_param_size = sum(x.size for x in jax.tree_util.tree_leaves(block_params))
  if stage_param_size != block_param_size:
    max_logging.info("prefetch_fsdp_weights: the block has params outside its sub-layers; prefetching whole blocks.")
    return None
  return stages


def apply_layers_with_fsdp_prefetch(
    layers, stages, x_in, *args, length: int, mesh, config, remat_policy=None, prevent_cse: bool = False, **kwargs
):
  if length == 0:
    return x_in, layers, None
  _, _, _, diff_args_leaves, diff_kwargs_leaves, ctx = fsdp_prefetch._prepare_prefetch_call(
      layers, args, kwargs, length=length, mesh=mesh, config=config, remat_policy=remat_policy, prevent_cse=prevent_cse
  )
  groups = []
  for stage_graphdef, p, s in stages:
    p, s = fsdp_prefetch._layer_major(p, s, config.param_scan_axis, length)
    p_leaves, p_treedef = jax.tree_util.tree_flatten(p)
    s_leaves, s_treedef = jax.tree_util.tree_flatten(s)
    shapes = [x.shape for x in p_leaves + s_leaves]
    if groups:
      g = groups[-1]
      if (
          g[0] == stage_graphdef
          and g[1] == p_treedef
          and g[2] == s_treedef
          and [x.shape for x in g[3][0] + g[4][0]] == shapes
      ):
        g[3].append(p_leaves)
        g[4].append(s_leaves)
        continue
    groups.append([stage_graphdef, p_treedef, s_treedef, [p_leaves], [s_leaves]])

  def stack(per_sublayer):
    return tuple(jnp.stack(xs, axis=1) for xs in zip(*per_sublayer))

  final_y, _ = _custom_vjp_nested_prefetch_pipeline(
      tuple(g[0] for g in groups), x_in, tuple(stack(g[3]) for g in groups), tuple(stack(g[4]) for g in groups),
      diff_args_leaves, diff_kwargs_leaves, length,
      tuple(ctx._replace(params_treedef=g[1], state_treedef=g[2], scan_axis=0) for g in groups),
      tuple(len(g[3]) for g in groups),
  )
  return final_y, layers, None
