"""RMS normalization."""

import string
from typing import Sequence, Union

import jax
import jax.numpy as jnp
import jaxtyping as jt
import typeguard


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def rms_norm(
    x: jt.Num[jax.Array, "*input_dims"],
    scale: jt.Num[jax.Array, "..."] | None = None,
    *,
    epsilon: float,
    axis: Union[int, Sequence[int]] = -1,
    norm_dtype: jt.DTypeLike = jnp.float32,
) -> jt.Num[jax.Array, "*input_dims"]:
  """Performs RMS normalization."""
  assert scale is None or scale.shape == x.shape[-scale.ndim :], (
      f"scale.shape must be a suffix of x.shape, but got {scale.shape} and" f" {x.shape}"
  )
  x_dtype = x.dtype
  x = jnp.astype(x, norm_dtype)
  mean2 = jnp.mean(jnp.square(x), axis=axis, keepdims=True)
  y = jnp.asarray(x * jax.lax.rsqrt(mean2 + epsilon), x_dtype)
  if scale is None:
    return y
  scale_dims = string.ascii_lowercase[: scale.ndim]
  return jnp.einsum(f"...{scale_dims},{scale_dims}->...{scale_dims}", y, scale)
