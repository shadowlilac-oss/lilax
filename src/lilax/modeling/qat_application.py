"""Apply Qwix QAT to a native model graph."""

from flax import nnx
import jax.numpy as jnp
import qwix

from lilax._backend.configs.pyconfig import HyperParameters


def apply_qat(model, config, qat_config, runtime):
  if qat_config.provider == "odml" and any(rule.tile_size is not None for rule in qat_config.rules):
    raise ValueError("Qwix ODML QAT does not support tile_size.")
  rules = [qwix.QuantizationRule(**rule.model_dump()) for rule in qat_config.rules]
  if qat_config.provider == "qt":
    unsupported = []
    if qat_config.disable_per_channel_weights:
      unsupported.append("disable_per_channel_weights")
    if qat_config.fixed_range_for_inputs is not None:
      unsupported.append("fixed_range_for_inputs")
    if qat_config.fixed_range_for_outputs is not None:
      unsupported.append("fixed_range_for_outputs")
    if not qat_config.strict:
      unsupported.append("strict")
    if unsupported:
      raise ValueError(f"Qwix Qt QAT does not support non-default options: {', '.join(unsupported)}.")
    provider = qwix.QtProvider(rules)
  else:
    provider = qwix.OdmlQatProvider(rules, **qat_config.model_dump(exclude={"rules", "provider"}))
  shape = (config.micro_batch_size_to_train_on, config.max_target_length)
  tokens = jnp.ones(shape, dtype=jnp.int32)
  targets = {}
  if config.mtp_num_layers > 0:
    targets = {"decoder_target_tokens": tokens, "decoder_target_mask": tokens}
  with runtime.context():
    model = qwix.quantize_model(
        model, provider, tokens, tokens, tokens, enable_dropout=False, **targets
    )
  nnx.pop(model, nnx.Intermediate)
  for _, node in nnx.graph.iter_graph(model):
    if hasattr(node, "__dict__") and "qwix_rngs" in node.__dict__:
      del node.qwix_rngs
  config = HyperParameters(config._pydantic_config.model_copy(update={"qwix_qat_config": qat_config}))
  return model, config
