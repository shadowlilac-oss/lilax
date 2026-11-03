"""Identity custom VJP that releases its inputs' cotangents together."""

import jax


def tie_cotangents(*args):
  """Return inputs unchanged while releasing their cotangents together."""
  @jax.custom_vjp
  def tied(*a):
    return a

  def tied_fwd(*a):
    return a, None

  def tied_bwd(_, cts):
    return jax.lax.optimization_barrier(cts)

  tied.defvjp(tied_fwd, tied_bwd)
  outputs = tied(*args)
  return outputs[0] if len(args) == 1 else outputs
