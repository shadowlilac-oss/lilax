"""Autoregressive generation for MaxEngine."""

from flax.core.spmd import logical_axis_rules
from jetstream.engine import engine_api
import jax
import jax.numpy as jnp

from lilax._backend.common.common_types import MODEL_MODE_AUTOREGRESSIVE
from lilax._backend.inference import inference_utils
from lilax._backend.inference.maxengine import model as maxengine_model


def generate_step(
    config, mesh, replicated_sharding, kv_cache_shardings, model, graphdef, rest_state,
    params, decode_state, rng, algorithm, topk, nucleus_topp, temperature,
):
  """Run the model and sample one token for each decode slot."""
  previous_token = decode_state["tokens"]
  rng, new_rng = jax.random.split(rng)
  with mesh, logical_axis_rules(config.logical_axis_rules):
    out_logits, new_cache_dict = maxengine_model.run_model(
        abstract_model=model,
        graphdef=graphdef,
        rest_state=rest_state,
        params=params,
        cache_dict=decode_state["cache"],
        decoder_input_tokens=previous_token,
        decoder_positions=decode_state["next_pos"],
        enable_dropout=False,
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
    )
  new_vars = {"cache": new_cache_dict}
  out_logits = jax.lax.with_sharding_constraint(out_logits, replicated_sharding)
  new_cache = jax.lax.with_sharding_constraint(new_vars["cache"], kv_cache_shardings)
  rng, new_rng = jax.random.split(rng)
  new_token = inference_utils.sampling(
      out_logits,
      new_rng,
      algorithm if algorithm is not None else config.decode_sampling_strategy,
      topk=topk if topk is not None else config.decode_sampling_top_k,
      nucleus_topp=nucleus_topp if nucleus_topp is not None else config.decode_sampling_nucleus_p,
      temperature=temperature if temperature is not None else config.decode_sampling_temperature,
  )
  all_valid = jnp.ones(new_token.shape, dtype=jnp.int8)
  if config.return_log_prob:
    token_logp = inference_utils.log_prob_of_chosen_token(out_logits, new_token)
  else:
    token_logp = jnp.zeros(new_token.shape, dtype=jnp.float32)

  next_pos = decode_state["next_pos"] + 1
  generated_tokens = decode_state["generated_tokens"] + 1
  result = engine_api.ResultTokens(
      data=jnp.concatenate((new_token, all_valid, generated_tokens), axis=1),
      tokens_idx=(0, 1),
      valid_idx=(1, 2),
      length_idx=(2, 3),
      log_prob=token_logp,
      samples_per_slot=1,
  )
  return {
      "logits": out_logits,
      "cache": new_cache,
      "next_pos": next_pos,
      "generated_tokens": generated_tokens,
      "tokens": new_token,
      "token_logp": token_logp,
  }, result
