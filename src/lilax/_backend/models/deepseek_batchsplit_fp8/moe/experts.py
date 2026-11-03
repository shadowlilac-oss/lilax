"""Routed and shared expert MLPs for the DeepSeek FP8 batch-split schedule."""

import functools

import jax
import jax.numpy as jnp
from lilax._backend.models.deepseek_batchsplit_fp8.dot import dot
from lilax._backend.models.deepseek_batchsplit_fp8.moe.grouped_matmul.kernel import gmm
from lilax._backend.models.deepseek_batchsplit_fp8.moe.grouped_matmul.tiling import gmm_tile_sizes
from lilax._backend.models.deepseek_batchsplit_fp8.moe.grouped_matmul.weight_gather import gmm_weight_gather_axes


def shared_expert(inputs, shared_w0, shared_w1, shared_wo, *, quant):
  """Processes tokens through the shared expert MLP."""
  return dot(
      jax.nn.silu(dot(inputs, shared_w0, quant=quant)) * dot(inputs, shared_w1, quant=quant), shared_wo, quant=quant
  )


def compute(x, w0, w1, wo, group_sizes, weights, *, config, mesh):
  """Processes routed tokens through the MLP."""
  gmm_fn = functools.partial(gmm, group_sizes=group_sizes, preferred_element_type=config.dtype, config=config)
  wi_tile_size, wo_tile_size = gmm_tile_sizes(config)
  wi_gather_axes, wo_gather_axes = gmm_weight_gather_axes(config, mesh)
  if config.merge_gating_gmm:
    w01 = jnp.concatenate([w0, w1], axis=-1)
    layer_w01 = gmm_fn(
        x,
        w01,
        tiling=wi_tile_size,
        weight_gather_axes=wi_gather_axes,
    )
    layer_w0, layer_w1 = jnp.split(layer_w01, 2, axis=-1)
  else:
    layer_w0 = gmm_fn(
        x,
        w0,
        tiling=wi_tile_size,
        weight_gather_axes=wi_gather_axes,
    )
    layer_w1 = gmm_fn(
        x,
        w1,
        tiling=wi_tile_size,
        weight_gather_axes=wi_gather_axes,
    )
  layer_w0 = jax.ad_checkpoint.checkpoint_name(layer_w0, "mlpwi_0")
  layer_w1 = jax.ad_checkpoint.checkpoint_name(layer_w1, "mlpwi_1")
  intermediate_layer = jax.nn.silu(layer_w0) * layer_w1
  intermediate_layer *= weights[:, None]
  layer_wo = gmm_fn(
      intermediate_layer,
      wo,
      tiling=wo_tile_size,
      weight_gather_axes=wo_gather_axes,
  )
  return layer_wo
