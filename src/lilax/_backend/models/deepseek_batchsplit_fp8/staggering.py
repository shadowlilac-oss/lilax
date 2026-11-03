"""Staggered microbatch execution for the DeepSeek FP8 batch-split schedule."""

import jax


def staggered_call(fn, xs):
  for i, x in enumerate(xs):
    if i == len(xs) - 1:
      xs[i] = fn(x)
    else:
      xs[i], xs[i + 1] = jax.lax.optimization_barrier((fn(x), xs[i + 1]))
  return xs
