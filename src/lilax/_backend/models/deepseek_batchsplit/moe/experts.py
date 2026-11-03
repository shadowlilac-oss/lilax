"""Shared and routed expert feed-forward computation for the DeepSeek BF16 batch-split schedule."""

import functools

import jax
import jax.numpy as jnp

from lilax._backend.models.deepseek_batchsplit.dot import dot
from lilax._backend.models.deepseek_batchsplit.moe.grouped_matmul import gmm


def shared_expert(inputs, shared_w0, shared_w1, shared_wo):
  return dot(jax.nn.silu(dot(inputs, shared_w0)) * dot(inputs, shared_w1), shared_wo)


def compute_gating(x, w0, w1, group_sizes, *, dtype, config):
  """Computes the gating GMMs."""
  gmm_fn = functools.partial(
      gmm,
      group_sizes=group_sizes,
      preferred_element_type=dtype,
      config=config,
  )
  if config.merge_gating_gmm:
    w01 = jnp.concatenate([w0, w1], axis=-1)
    layer_w01 = gmm_fn(x, w01)
    layer_w0, layer_w1 = jnp.split(layer_w01, 2, axis=-1)
  else:
    layer_w0 = gmm_fn(x, w0)
    layer_w1 = gmm_fn(x, w1)

  return layer_w0, layer_w1


def compute_linear(layer_w0, layer_w1, wo, group_sizes, weights, *, dtype, config):
  """Combines the outputs of the gating GMMs and computes the final GMM."""
  intermediate_layer = jax.nn.silu(layer_w0) * layer_w1
  intermediate_layer *= weights[:, None]
  gmm_fn = functools.partial(
      gmm,
      group_sizes=group_sizes,
      preferred_element_type=dtype,
      config=config,
  )
  layer_wo = gmm_fn(
      intermediate_layer,
      wo,
  )
  return layer_wo
