"""Gated FFN activation applied between the expert up- and down-projections."""

import jax
import jax.numpy as jnp
from lilax._backend.common import common_types as ctypes


class ExpertActivation:
  @staticmethod
  def apply_ffn_activation(layer_w0, layer_w1, decoder_block, mlp_activations_limit, activation_fn, dtype):
    """Applies FFN activation function."""
    with jax.named_scope("ffn_act"):
      if decoder_block == ctypes.DecoderBlockType.GPT_OSS:
        layer_w0 = jnp.clip(layer_w0, min=None, max=mlp_activations_limit)
        layer_w1 = jnp.clip(
            layer_w1,
            min=-mlp_activations_limit,
            max=mlp_activations_limit,
        )
        layer_act = activation_fn(layer_w0 * 1.702)
        glu = jnp.multiply(layer_w0, layer_act)
        intermediate_layer = jnp.multiply(glu, (layer_w1 + 1))
      elif (
          decoder_block
          in (ctypes.DecoderBlockType.DEEPSEEK, ctypes.DecoderBlockType.DEEPSEEK4, ctypes.DecoderBlockType.HY4)
          and mlp_activations_limit > 0.0
      ):
        layer_w0 = jnp.clip(layer_w0, min=None, max=mlp_activations_limit)
        layer_w1 = jnp.clip(
            layer_w1,
            min=-mlp_activations_limit,
            max=mlp_activations_limit,
        )
        layer_act = activation_fn(layer_w0)
        intermediate_layer = jnp.multiply(layer_act, layer_w1)
      else:
        layer_act = activation_fn(layer_w0)
        intermediate_layer = jnp.multiply(layer_act, layer_w1)
      return intermediate_layer.astype(dtype)
