"""Applying Qwix op interception to a model for the configured quantization recipe."""

import qwix

import jax.numpy as jnp

from flax import nnx

from lilax._backend.layers.quantization.qwix_fp8_providers import NANOOFp8Provider, NvidaFp8Provider
from lilax._backend.layers.quantization.qwix_rules import get_quantization_rule


def get_qt_provider(config):
  """Get quantization rules based on the config."""
  match config.quantization:
    case "int4" | "int8" | "fp4" | "fp4_e2m1" | "fp8" | "fp8_e5m2" | "fp8_e4m3" | "fp8_full":
      return qwix.QtProvider(get_quantization_rule(config))
    case "fp8_gpu":
      return NvidaFp8Provider(get_quantization_rule(config))
    case "fp8_nanoo":
      return NANOOFp8Provider(get_quantization_rule(config))
  return None


def maybe_quantize_model(model, config):
  """Quantize the model if quantization is enabled."""
  # Batch split is not using Qwix's interception feature but manual plumbing
  if config.quantization and config.use_qwix_quantization and not config.use_batch_split_schedule:
    quantization_provider = get_qt_provider(config)
    if quantization_provider:
      input_shape = (config.micro_batch_size_to_train_on, config.max_target_length)
      dummy_tokens = jnp.ones(input_shape, dtype=jnp.int32)
      dummy_positions = jnp.ones(input_shape, dtype=jnp.int32)
      dummy_segment_ids = jnp.ones(input_shape, dtype=jnp.int32)
      # The MTP block reads the decoder targets, so the qwix forward pass needs them.
      dummy_targets = {}
      if config.mtp_num_layers > 0:
        dummy_targets["decoder_target_tokens"] = jnp.ones(input_shape, dtype=jnp.int32)
        dummy_targets["decoder_target_mask"] = jnp.ones(input_shape, dtype=jnp.int32)
      model = qwix.quantize_model(
          model,
          quantization_provider,
          dummy_tokens,
          dummy_positions,
          dummy_segment_ids,
          enable_dropout=False,
          **dummy_targets,
      )
      # Qwix quantization runs a forward pass during tracing, which sows transient nnx.Intermediate variables
      # (e.g. max_logits from QK-Clip, MTP losses) into the model. Popping them here prevents structural mismatches
      # between the initial setup GraphDef/state_mesh_shardings and the stripped states during train steps.
      nnx.pop(model, nnx.Intermediate)
      for _, val in nnx.graph.iter_graph(model):
        if hasattr(val, "__dict__") and "qwix_rngs" in val.__dict__:
          del val.qwix_rngs
  return model
