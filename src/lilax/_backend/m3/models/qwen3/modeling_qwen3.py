"""Qwen3 dense model in m3 format."""

from typing import Any, Optional

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from lilax._backend.common.common_types import Config
from lilax._backend.m3.core.rope import apply_rope


class Qwen3MLP(nnx.Module):
  """Qwen3 SwiGLU feed-forward network."""

  def __init__(self, config: Config, mesh: Optional[Mesh] = None, *, rngs: nnx.Rngs):
    self.config = config
    kw = {"use_bias": False, "dtype": config.dtype, "param_dtype": config.weight_dtype, "rngs": rngs}
    self.wi_0 = nnx.Linear(in_features=config.emb_dim, out_features=config.mlp_dim, **kw)
    self.wi_1 = nnx.Linear(in_features=config.emb_dim, out_features=config.mlp_dim, **kw)
    self.wo = nnx.Linear(in_features=config.mlp_dim, out_features=config.emb_dim, **kw)

  def __call__(self, x: jax.Array) -> jax.Array:
    gate = jax.nn.silu(self.wi_0(x))
    up = self.wi_1(x)
    hidden = (gate * up).astype(self.config.dtype)
    return self.wo(hidden)


class Qwen3Attention(nnx.Module):
  """Qwen3 grouped-query attention with QK normalization and RoPE."""

  def __init__(self, config: Config, mesh: Optional[Mesh] = None, *, rngs: nnx.Rngs):
    self.config = config
    self.num_query_heads = config.num_query_heads
    self.num_kv_heads = config.num_kv_heads
    self.head_dim = config.head_dim
    kw = {"bias_shape": None, "dtype": config.dtype, "param_dtype": config.weight_dtype, "rngs": rngs}
    self.query = nnx.Einsum("...e,ehd->...hd", (config.emb_dim, self.num_query_heads, self.head_dim), **kw)
    self.key = nnx.Einsum("...e,ehd->...hd", (config.emb_dim, self.num_kv_heads, self.head_dim), **kw)
    self.value = nnx.Einsum("...e,ehd->...hd", (config.emb_dim, self.num_kv_heads, self.head_dim), **kw)
    self.out = nnx.Einsum("...hd,hde->...e", (self.num_query_heads, self.head_dim, config.emb_dim), **kw)
    norm_kw = {"epsilon": config.normalization_layer_epsilon, "dtype": config.dtype,
               "param_dtype": config.weight_dtype, "rngs": rngs}
    self.query_norm = nnx.RMSNorm(num_features=self.head_dim, **norm_kw)
    self.key_norm = nnx.RMSNorm(num_features=self.head_dim, **norm_kw)

  def _compute_attention(self, query: jax.Array, key: jax.Array, value: jax.Array,
                         decoder_positions: Optional[jax.Array] = None,
                         decoder_segment_ids: Optional[jax.Array] = None) -> jax.Array:
    batch, target_length, num_query_heads, head_dim = query.shape
    num_kv_heads = key.shape[2]
    groups = num_query_heads // num_kv_heads
    query_g = jnp.reshape(query, (batch, target_length, num_kv_heads, groups, head_dim))

    def _compute_local_block(q_g, k, v, mask=None):
      weights = jnp.einsum("btkgd,bskd->bkgts", q_g, k)
      if mask is not None:
        weights = jnp.where(mask, weights, -1e10)
      if self.config.float32_logits:
        weights = weights.astype(jnp.float32)
      logits = jnp.reshape(weights, (batch, num_query_heads, target_length, -1))
      local_max = jnp.max(logits, axis=-1, keepdims=True)
      local_exp = jnp.exp(logits - local_max)
      local_sum = jnp.sum(local_exp, axis=-1, keepdims=True)
      local_exp = jnp.reshape(local_exp.astype(v.dtype), (batch, num_kv_heads, groups, target_length, -1))
      local_out = jnp.einsum("bkgts,bskd->btkgd", local_exp, v)
      local_out = jnp.reshape(local_out, (batch, target_length, num_query_heads, head_dim))
      local_max = jnp.transpose(local_max, (0, 2, 1, 3))
      local_sum = jnp.transpose(local_sum, (0, 2, 1, 3))
      return local_out, local_max, local_sum

    source_length = key.shape[1]
    q_pos = jnp.arange(target_length) if decoder_positions is None else decoder_positions
    k_pos = jnp.arange(source_length) if decoder_positions is None else decoder_positions
    causal_mask = k_pos[..., None, :] <= q_pos[..., :, None]
    if decoder_segment_ids is not None:
      seg_mask = decoder_segment_ids[..., :, None] == decoder_segment_ids[..., None, :]
      mask = causal_mask & seg_mask
    else:
      mask = causal_mask
    mask = mask[:, None, None, :, :]
    out, _, lsum = _compute_local_block(query_g, key, value, mask)
    return (out / lsum).astype(self.config.dtype)

  def __call__(self, inputs: jax.Array, decoder_positions: jax.Array,
               decoder_segment_ids: Optional[jax.Array] = None, deterministic: bool = False, **kwargs) -> jax.Array:
    query = self.query(inputs)
    key = self.key(inputs)
    value = self.value(inputs)
    if self.config.use_qk_norm:
      query = self.query_norm(query)
      key = self.key_norm(key)
    query = apply_rope(query, decoder_positions, max_timescale=self.config.rope_max_timescale)
    key = apply_rope(key, decoder_positions, max_timescale=self.config.rope_max_timescale)
    query_pre_attn_scalar = self.head_dim**-0.5
    query = query * query_pre_attn_scalar
    attn_out = self._compute_attention(query=query, key=key, value=value, decoder_positions=decoder_positions,
                                       decoder_segment_ids=decoder_segment_ids)
    return self.out(attn_out)


class Qwen3DecoderLayer(nnx.Module):
  """Qwen3 pre-norm decoder layer."""

  def __init__(self, config: Config, mesh: Optional[Mesh] = None, *, rngs: nnx.Rngs):
    self.config = config
    norm_kw = {"epsilon": config.normalization_layer_epsilon, "dtype": config.dtype,
               "param_dtype": config.weight_dtype, "rngs": rngs}
    self.pre_self_attention_layer_norm = nnx.RMSNorm(num_features=config.emb_dim, **norm_kw)
    self.self_attention = Qwen3Attention(config=config, rngs=rngs)
    self.post_self_attention_layer_norm = nnx.RMSNorm(num_features=config.emb_dim, **norm_kw)
    self.mlp = Qwen3MLP(config=config, rngs=rngs)

  def __call__(self, inputs: jax.Array, decoder_segment_ids: Optional[jax.Array] = None,
               decoder_positions: Optional[jax.Array] = None, deterministic: bool = False, **kwargs) -> jax.Array:
    normed_attn_in = self.pre_self_attention_layer_norm(inputs)
    attn_out = self.self_attention(normed_attn_in, decoder_positions=decoder_positions,
                                   decoder_segment_ids=decoder_segment_ids, deterministic=deterministic)
    x = (inputs + attn_out).astype(self.config.dtype)
    normed_mlp_in = self.post_self_attention_layer_norm(x)
    mlp_out = self.mlp(normed_mlp_in)
    output = (x + mlp_out).astype(self.config.dtype)
    return output


class Qwen3Decoder(nnx.Module):
  """Qwen3 decoder stack with sequential layers and tied output embeddings."""

  def __init__(self, config: Config, mesh: Optional[Mesh] = None, *, rngs: nnx.Rngs,
               layer_cls: type[nnx.Module] = Qwen3DecoderLayer):
    self.config = config
    self.layers = nnx.Sequential(*(layer_cls(config=config, rngs=rngs) for _ in range(config.num_decoder_layers)))
    self.decoder_norm = nnx.RMSNorm(num_features=config.emb_dim, epsilon=config.normalization_layer_epsilon,
                                    dtype=config.dtype, param_dtype=config.weight_dtype, rngs=rngs)

  def apply_output_head(self, shared_embedding: Any, y: jax.Array, deterministic: bool = False) -> jax.Array:
    y = self.decoder_norm(y)
    attend_dtype = jnp.float32 if self.config.logits_dot_in_fp32 else self.config.dtype
    logits = jnp.dot(y.astype(attend_dtype), shared_embedding.embedding[...].astype(attend_dtype).T)
    if self.config.normalize_embedding_logits:
      logits = logits / jnp.sqrt(y.shape[-1])
    if self.config.final_logits_soft_cap:
      logits = logits / self.config.final_logits_soft_cap
      logits = jnp.tanh(logits) * self.config.final_logits_soft_cap
    if self.config.cast_logits_to_fp32:
      logits = logits.astype(jnp.float32)
    return logits

  def __call__(self, shared_embedding: Any, decoder_input_tokens: jax.Array, decoder_positions: jax.Array,
               decoder_segment_ids: Optional[jax.Array] = None, deterministic: bool = False,
               **kwargs) -> tuple[jax.Array, jax.Array]:
    y = shared_embedding(decoder_input_tokens)
    y = self.layers(y, decoder_segment_ids=decoder_segment_ids, decoder_positions=decoder_positions,
                    deterministic=deterministic)
    logits = self.apply_output_head(shared_embedding, y, deterministic=deterministic)
    return logits, y


class Qwen3Model(nnx.Module):
  """Top-level Qwen3 autoregressive language model in m3 format."""

  def __init__(self, config: Config, mesh: Optional[Mesh] = None, *, rngs: nnx.Rngs, **kwargs):
    self.config = config
    self.token_embedder = nnx.Embed(num_embeddings=config.vocab_size, features=config.emb_dim,
                                    dtype=config.dtype, param_dtype=config.weight_dtype, rngs=rngs)
    self.decoder = Qwen3Decoder(config=config, rngs=rngs)

  def logits_from_hidden_states_for_vocab_tiling(self, hidden_states: jax.Array, deterministic: bool,
                                                  **kwargs) -> jax.Array:
    return self.decoder.apply_output_head(shared_embedding=self.token_embedder, y=hidden_states,
                                          deterministic=deterministic)

  def __call__(self, decoder_input_tokens: jnp.ndarray, decoder_positions: jnp.ndarray,
               decoder_segment_ids: Optional[jnp.ndarray] = None, cache: Any = None,
               enable_dropout: bool = False, true_length: Optional[int] = None, **kwargs) -> jax.Array:
    deterministic = kwargs.get("deterministic", not enable_dropout)
    logits, _ = self.decoder(shared_embedding=self.token_embedder, decoder_input_tokens=decoder_input_tokens,
                             decoder_positions=decoder_positions, decoder_segment_ids=decoder_segment_ids,
                             deterministic=deterministic)
    return logits


def create_qwen3_model(config: Config, mesh: Optional[Mesh] = None, *, rngs: Optional[nnx.Rngs] = None,
                       **kwargs) -> Qwen3Model:
  """Create a Qwen3Model instance."""
  if rngs is None:
    rngs = nnx.Rngs(0)
  return Qwen3Model(config, mesh, rngs=rngs, **kwargs)
