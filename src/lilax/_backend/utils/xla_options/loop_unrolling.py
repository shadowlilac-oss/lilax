"""XLA:GPU loop unroll metadata scoping for model loops."""

from contextlib import nullcontext

from jax.experimental.xla_metadata import set_xla_metadata


def model_loop_unroll_context(config):
  """Returns a context that marks the model's loops traced inside it for double buffering.

  `_xla_loop_unroll_strategy` is only read by XLA:GPU (https://github.com/openxla/xla/pull/43468);
  enable it with JAX_OPTIMIZATION_LEVEL=O1 and
  XLA_FLAGS="--xla_gpu_enable_while_loop_unrolling=WHILE_LOOP_UNROLLING_MANUAL_UNROLL". The other
  unrolling modes ignore it. Enter it around the model's forward and backward pass only, so that it
  scopes the model's scans and not an enclosing loop such as the gradient accumulation scan.
  On TPU/CPU it is a no-op: `set_xla_metadata` pre-populates frontend attributes on every traced op,
  causing XLA's CallInliner to drop `_xla_compute_type` from `compute_on("device")` calls and
  XLA:TPU to auto-offload those collectives to SparseCore.
  """
  if getattr(config, "hardware", "tpu") in ("gpu", "gpu_multiprocess"):
    return set_xla_metadata(_xla_loop_unroll_strategy="double-buffer")
  return nullcontext()
