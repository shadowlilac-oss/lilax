"""Condition assembly and layer execution for the Wan transformer."""

from flax import nnx
import jax
import jax.numpy as jnp


def combine_conditioning(encoder_hidden_states, encoder_hidden_states_image, encoder_attention_mask,
                         extend_mask=True):
  if encoder_hidden_states_image is not None:
    encoder_hidden_states = jnp.concatenate([encoder_hidden_states_image, encoder_hidden_states], axis=1)
    if extend_mask and encoder_attention_mask is not None:
      text_mask = jnp.ones((encoder_hidden_states.shape[0],
                            encoder_hidden_states.shape[1] - encoder_hidden_states_image.shape[1]), dtype=jnp.int32)
      encoder_attention_mask = jnp.concatenate([encoder_attention_mask, text_mask], axis=1)
  return encoder_hidden_states, encoder_attention_mask


def compute_layer_kv_cache(blocks, scan_layers, encoder_hidden_states, encoder_attention_mask):
  if scan_layers:
    @nnx.vmap(in_axes=(0, None, None), out_axes=0, transform_metadata={nnx.PARTITION_NAME: None})
    def _compute_kv(block, enc_states, enc_mask):
      return block.compute_kv(enc_states, enc_mask)

    return _compute_kv(blocks, encoder_hidden_states, encoder_attention_mask)
  kv_cache_list = []
  for block in blocks:
    kv_cache_list.append(block.compute_kv(encoder_hidden_states, encoder_attention_mask))
  keys = kv_cache_list[0].keys()
  kv_cache = {}
  for k in keys:
    k_list = [d[k][0] for d in kv_cache_list]
    v_list = [d[k][1] for d in kv_cache_list]
    kv_cache[k] = (jnp.stack(k_list, axis=0), jnp.stack(v_list, axis=0))
  return kv_cache


def run_transformer_blocks(blocks, scan_layers, num_layers, hidden_states, encoder_hidden_states, timestep_proj,
                           rotary_emb, deterministic, rngs, encoder_attention_mask, kv_cache):
  if scan_layers:
    def scan_fn(carry, block_input):
      hidden_states_carry, rngs_carry = carry
      if kv_cache is not None:
        block, layer_kv_cache = block_input
      else:
        block = block_input
        layer_kv_cache = None
      hidden_states = block(hidden_states_carry, encoder_hidden_states, timestep_proj, rotary_emb, deterministic,
                            rngs_carry, encoder_attention_mask, cached_kv=layer_kv_cache)
      new_carry = (hidden_states, rngs_carry)
      return new_carry, None

    initial_carry = (hidden_states, rngs)
    if kv_cache is not None:
      scan_input = (blocks, kv_cache)
    else:
      scan_input = blocks
    final_carry, _ = nnx.scan(scan_fn, length=num_layers, in_axes=(nnx.Carry, 0),
                              out_axes=(nnx.Carry, 0))(initial_carry, scan_input)
    hidden_states, _ = final_carry
  else:
    for i, block in enumerate(blocks):
      layer_kv_cache = None
      if kv_cache is not None:
        layer_kv_cache = jax.tree.map(lambda x: x[i], kv_cache)

      def layer_forward(hidden_states, l_kv):
        return block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, deterministic, rngs,
                     encoder_attention_mask=encoder_attention_mask, cached_kv=l_kv)

      hidden_states = layer_forward(hidden_states, layer_kv_cache)
  return hidden_states
