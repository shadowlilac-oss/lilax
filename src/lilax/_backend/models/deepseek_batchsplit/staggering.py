"""Staggered microbatch execution for the DeepSeek BF16 batch-split schedule."""

import jax


def staggered_call(fn, xs):
  """Calls a function in a staggered manner while accumulating residuals."""
  res_dicts = []
  for i, x in enumerate(xs):
    x, res = fn(x)
    res_dicts.append(res)
    if i == len(xs) - 1:
      xs[i] = x
    else:
      xs[i], xs[i + 1] = jax.lax.optimization_barrier((x, xs[i + 1]))
  # Convert list of res dicts to dict of lists.
  return xs, jax.tree_util.tree_map(lambda *rs: list(rs), *res_dicts)
