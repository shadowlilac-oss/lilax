"""Masked grouped matmul kernels for the DeepSeek BF16 batch-split routed experts."""

import functools

import jax
import jax.numpy as jnp

from lilax._backend.kernels import megablox
from lilax._backend.layers.quantization.qwix_rules import get_fp8_full_qwix_rule_w_sparsity


# TODO(b/517203548): After integrating with the GMMv2 kernel and `zero_initialize` is set,
# we will not need to mask the GMM token output explicitly in the fwd or bwd passes.
# At that point, we can delete this custom ragged dot function.
@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def ragged_dot_with_mask(x, ws, group_sizes, dtype):
  return ragged_dot_with_mask_fwd(x, ws, group_sizes, dtype)[0]


def ragged_dot_with_mask_fwd(x, ws, group_sizes, dtype):
  """Performs ragged dot and masks unprocessed tokens."""
  out, ragged_dot_bwd = jax.vjp(
      functools.partial(
          jax.lax.ragged_dot,
          precision=jax.lax.Precision.DEFAULT,
          preferred_element_type=dtype,
          group_sizes=group_sizes,
      ),
      x,
      ws,
  )
  mask = jnp.arange(x.shape[0]) < jnp.sum(group_sizes)
  out = jnp.where(mask[:, None], out, 0.0)
  return out, (mask, ragged_dot_bwd)


def ragged_dot_with_mask_bwd(dtype, res, grad):
  """Dual of ragged dot and masks gradients of unprocessed tokens."""
  mask, ragged_dot_bwd = res
  x_grad, ws_grad = ragged_dot_bwd(grad)
  x_grad = jnp.where(mask[:, None], x_grad, 0.0)
  return x_grad, ws_grad, None


ragged_dot_with_mask.defvjp(ragged_dot_with_mask_fwd, ragged_dot_with_mask_bwd)


# TODO(b/517203548): After integrating with the GMMv2 kernel and `zero_initialize` is set,
# we will not need to mask the GMM token output explicitly in the fwd or bwd passes.
# At that point, we can delete this custom ragged dot function.
@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def megablox_gmm_with_mask(x, ws, group_sizes, dtype, qwix_rule):
  return megablox_gmm_with_mask_fwd(x, ws, group_sizes, dtype, qwix_rule)[0]


def megablox_gmm_with_mask_fwd(x, ws, group_sizes, dtype, qwix_rule):
  """Performs Megablox GMM and masks unprocessed tokens."""
  out, megablox_gmm_bwd = jax.vjp(
      functools.partial(
          megablox.gmm,
          preferred_element_type=dtype,
          use_qwix_quantization=True,
          use_tokamax_backend=True,
          qwix_rule=qwix_rule,
          use_manual_quantization=True,
          group_sizes=group_sizes,
      ),
      x,
      ws,
  )
  mask = jnp.arange(x.shape[0]) < jnp.sum(group_sizes)
  out = jnp.where(mask[:, None], out, 0.0)
  return out, (mask, megablox_gmm_bwd)


def megablox_gmm_with_mask_bwd(dtype, qwix_rule, res, grad):
  """Dual of Megablox GMM and masks gradients of unprocessed tokens."""
  mask, megablox_gmm_bwd = res
  x_grad, ws_grad = megablox_gmm_bwd(grad)
  x_grad = jnp.where(mask[:, None], x_grad, 0.0)
  return x_grad, ws_grad, None


megablox_gmm_with_mask.defvjp(megablox_gmm_with_mask_fwd, megablox_gmm_with_mask_bwd)


def gmm(
    inputs,
    kernel,
    group_sizes,
    preferred_element_type,
    config,
):
  """Performs a Grouped Matrix Multiplication (GMM).

  This function can use either a quantized Megablox kernel or a standard
  jax.lax.ragged_dot for the GMM operation, based on the configuration.

  Args:
    inputs: The left-hand side operand of the GMM.
    kernel: The right-hand side operand (kernel) of the GMM.
    group_sizes: An array indicating the size of each group.
    preferred_element_type: The preferred element type for the computation.
    config: Configuration object containing model settings, including
      `use_qwix_quantization` and `merge_gating_gmm`.

  Returns:
    The result of the grouped matrix multiplication.
  """
  if config.quantization:
    output = megablox_gmm_with_mask(
        inputs,
        kernel,
        group_sizes=group_sizes,
        dtype=preferred_element_type,
        qwix_rule=get_fp8_full_qwix_rule_w_sparsity(config)[-1],
    )
  else:
    output = ragged_dot_with_mask(
        inputs,
        kernel,
        group_sizes=group_sizes,
        dtype=preferred_element_type,
    )
  return output
