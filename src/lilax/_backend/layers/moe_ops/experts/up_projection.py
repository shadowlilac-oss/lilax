"""Expert gate and up projections."""

from jax import ad_checkpoint as adc
import jax.numpy as jnp


def up_projection(
    x,
    w0,
    w1,
    w0_bias,
    w1_bias,
    gmm_fn,
    wi_gather_axes,
    wi_tile_size,
    config,
    partial_accum0=None,
    partial_accum1=None,
    mask=None,
    return_weights=False,
):
  """Run the two up-projections (gate + up) and apply the FFN activation."""
  use_te_bias = config.use_te_grouped_gemm and config.mlp_bias

  def _gmm_maybe_bias(x_, w_, bias_, partial_sum=None, *, return_lhs=False, return_rhs=False):
    # With return_lhs / return_rhs, gmm returns (out, [lhs], [rhs]); bias and mask apply to `out` only.
    te_bias = use_te_bias and bias_ is not None
    res = gmm_fn(
        x_,
        w_,
        tiling=wi_tile_size,
        weight_gather_axes=wi_gather_axes,
        partial_sum=partial_sum,
        return_lhs=return_lhs,
        return_rhs=return_rhs,
        bias=bias_ if te_bias else None,
    )
    layer, extras = (res[0], tuple(res[1:])) if return_lhs or return_rhs else (res, ())
    if te_bias:
      if mask is not None:
        layer = jnp.where(mask[:, None], layer, 0)
    elif config.mlp_bias and bias_ is not None:
      layer = layer + bias_
      if mask is not None:
        layer = jnp.where(mask[:, None], layer, 0)
    return (layer, *extras) if extras else layer

  if config.prefuse_moe_weights:
    # Weights are stored as (G,K,2N); w0/w1 are adjacent slices so XLA elides this concat.
    w_fused = jnp.concatenate([w0, w1], axis=-1)
    fused_bias = None
    if use_te_bias and w0_bias is not None and w1_bias is not None:
      fused_bias = jnp.concatenate([w0_bias, w1_bias], axis=-1)
    out = _gmm_maybe_bias(x, w_fused, fused_bias)
    n = out.shape[-1] // 2
    layer_w0, layer_w1 = out[:, :n], out[:, n:]
    if config.mlp_bias and not use_te_bias and w0_bias is not None and w1_bias is not None:
      layer_w0 = layer_w0 + w0_bias
      layer_w1 = layer_w1 + w1_bias
      if mask is not None:
        layer_w0 = jnp.where(mask[:, None], layer_w0, 0)
        layer_w1 = jnp.where(mask[:, None], layer_w1, 0)
    layer_w0 = adc.checkpoint_name(adc.checkpoint_name(layer_w0, "mlpwi_0"), "moe_mlpwi_0")
    layer_w1 = adc.checkpoint_name(layer_w1, "moe_mlpwi_1")
    return layer_w0, layer_w1
  else:
    # With moe_accumulate_wi_dlhs, the wi_0 GMM returns its input and the wi_1 GMM consumes that, so in the
    # backward pass the wi_1 DLHS arrives as the cotangent of the returned input and the wi_0 DLHS GMM
    # accumulates onto it in place (gmm_v2 partial_sum) instead of a separate add of the two DLHS.
    accum_wi_dlhs = config.moe_accumulate_wi_dlhs
    layer_w0 = _gmm_maybe_bias(x, w0, w0_bias, partial_accum0, return_lhs=accum_wi_dlhs, return_rhs=return_weights)
    x_for_w1 = x
    if accum_wi_dlhs and return_weights:
      layer_w0, x_for_w1, w0 = layer_w0
    elif accum_wi_dlhs:
      layer_w0, x_for_w1 = layer_w0
    elif return_weights:
      layer_w0, w0 = layer_w0
    layer_w0 = adc.checkpoint_name(adc.checkpoint_name(layer_w0, "mlpwi_0"), "moe_mlpwi_0")

    layer_w1 = _gmm_maybe_bias(x_for_w1, w1, w1_bias, partial_accum1, return_rhs=return_weights)
    if return_weights:
      layer_w1, w1 = layer_w1
    layer_w1 = adc.checkpoint_name(layer_w1, "moe_mlpwi_1")
    if return_weights:
      return layer_w0, layer_w1, w0, w1
    return layer_w0, layer_w1
