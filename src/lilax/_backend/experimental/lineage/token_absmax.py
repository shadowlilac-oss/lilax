"""Pallas per-token absolute maxima."""

import jax
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

_TOKEN_ABSMAX_VMEM_LIMIT = 48 * 1024 * 1024


def _token_absmax_kernel(x_ref, out_ref):
  m_bf16 = jnp.max(jnp.abs(x_ref[...]), axis=1)
  absmax = jnp.max(m_bf16.astype(jnp.float32), axis=1)
  out_ref[...] = absmax[:, None]


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def token_absmax(x: jt.Num[jax.Array, "num_tokens d0 d1"], *, max_token_block: int = 1024) -> jt.Float[jax.Array, "num_tokens"]:
  """Computes per-token absolute maxima over trailing dimensions."""
  num_tokens, dim0, dim1 = x.shape
  if num_tokens % 8 != 0:
    raise ValueError(f"num_tokens ({num_tokens}) must be divisible by 8.")
  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  if num_tokens % 128 == 0 and dim1 == 128:
    out_rows = num_tokens // 128
    max_rb = max(1, max_token_block // 128)
    row_block = max(
        rb for rb in range(1, min(max_rb, out_rows) + 1) if out_rows % rb == 0 and (rb % 8 == 0 or rb == out_rows)
    )
    token_block = row_block * 128

    def _token_absmax_2d_kernel(x_ref, out_ref):
      m_bf16 = jnp.max(jnp.abs(x_ref[...]), axis=1)
      m_3d = m_bf16.reshape(row_block, 128, 128)
      out_ref[...] = jnp.max(jnp.swapaxes(m_3d, 1, 2), axis=1).astype(jnp.float32)

    out = pl.pallas_call(
        _token_absmax_2d_kernel,
        out_shape=jax.ShapeDtypeStruct((out_rows, 128), jnp.float32, manual_axis_type=manual_axis_type),
        grid=(out_rows // row_block,),
        in_specs=[pl.BlockSpec((token_block, dim0, dim1), lambda i: (i, 0, 0))],
        out_specs=pl.BlockSpec((row_block, 128), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=_TOKEN_ABSMAX_VMEM_LIMIT,
        ),
        name="token_absmax",
    )(x)
    return out.reshape(num_tokens)
  token_block = max(tb for tb in range(8, min(max_token_block, num_tokens) + 1, 8) if num_tokens % tb == 0)
  out = pl.pallas_call(
      _token_absmax_kernel,
      out_shape=jax.ShapeDtypeStruct((num_tokens, 1), jnp.float32, manual_axis_type=manual_axis_type),
      grid=(num_tokens // token_block,),
      in_specs=[pl.BlockSpec((token_block, dim0, dim1), lambda i: (i, 0, 0))],
      out_specs=pl.BlockSpec((token_block, 1), lambda i: (i, 0)),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",), vmem_limit_bytes=_TOKEN_ABSMAX_VMEM_LIMIT,
      ),
      name="token_absmax",
  )(x)
  return out.reshape(num_tokens)
