"""Multi-head latent attention for the DeepSeek FP8 batch-split schedule."""

import jax
from lilax._backend.layers import attention_op
from lilax._backend.layers.quantization.configure import configure_kv_quant
from lilax._backend.models.deepseek_batchsplit_fp8.attention.norm import rms_norm
from lilax._backend.models.deepseek_batchsplit_fp8.attention.projections import kv_projection, query_projection
from lilax._backend.models.deepseek_batchsplit_fp8.dot import dot
from lilax._backend.models.deepseek_batchsplit_fp8.sharding import with_data_parallel_constraint
from lilax._backend.models.deepseek_batchsplit_fp8.staggering import staggered_call


def attention_op_from_config(cfg, *, mesh, quant):
  """Builds the attention op used by MLA."""
  return attention_op.AttentionOp(
      config=cfg,
      mesh=mesh,
      attention_kernel=cfg.attention,
      max_target_length=cfg.max_target_length,
      max_prefill_predict_length=cfg.max_prefill_predict_length,
      quant=quant,
      kv_quant=configure_kv_quant(cfg),
      num_query_heads=cfg.num_query_heads,
      num_kv_heads=cfg.num_kv_heads,
      dropout_rate=cfg.dropout_rate,
      dtype=cfg.dtype,
      attention_type=cfg.attention_type,
  )


def mla_with_norms(
    inputs,
    weights,
    decoder_positions,
    decoder_segment_ids,
    *,
    mesh,
    model_mode,
    attn_op,
    normalization_layer_epsilon,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    rope_max_timescale,
    num_query_heads,
    max_position_embeddings,
    original_max_position_embeddings,
    beta_fast,
    beta_slow,
    rope_factor,
    mscale,
    dtype,
    quant,
):
  """Performs MLA with pre- and post-normalization."""
  (pre_attn_scale, post_attn_scale), attn_ws = weights

  def fn(args):
    x, dseg, dpos = args
    y = rms_norm(
        x,
        pre_attn_scale,
        epsilon=normalization_layer_epsilon,
        dtype=dtype,
    )
    out = x + with_data_parallel_constraint(
        mla(
            y,
            dpos,
            dseg,
            attn_ws,
            model_mode=model_mode,
            epsilon=normalization_layer_epsilon,
            kv_lora_rank=kv_lora_rank,
            kv_norm_epsilon=normalization_layer_epsilon,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            rope_theta=rope_max_timescale,
            num_query_heads=num_query_heads,
            max_position_embeddings=max_position_embeddings,
            original_max_position_embeddings=original_max_position_embeddings,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            rope_factor=rope_factor,
            dtype=dtype,
            mscale=mscale,
            attention_op_fn=attn_op,
            quant=quant,
        ),
        mesh,
    )
    return out, rms_norm(
        out,
        post_attn_scale,
        epsilon=normalization_layer_epsilon,
        dtype=dtype,
    )

  return staggered_call(fn, list(zip(inputs, decoder_segment_ids, decoder_positions)))


def mla(
    inputs,
    positions,
    segment_ids,
    weights,
    *,
    model_mode,
    epsilon,
    kv_lora_rank,
    kv_norm_epsilon,
    qk_nope_head_dim,
    qk_rope_head_dim,
    num_query_heads,
    rope_theta,
    max_position_embeddings,
    original_max_position_embeddings,
    beta_fast,
    beta_slow,
    rope_factor,
    mscale,
    attention_op_fn,
    dtype,
    quant,
):
  """Performs MLA."""
  (
      wq_a_weights,
      wq_b_weights,
      q_norm_scale_weights,
      wkv_a_weights,
      wkv_b_weights,
      kv_norm_scale_weights,
      out_weights,
  ) = weights
  query = query_projection(
      inputs,
      positions,
      wq_a_weights,
      wq_b_weights,
      q_norm_scale_weights,
      epsilon=epsilon,
      qk_rope_head_dim=qk_rope_head_dim,
      rope_theta=rope_theta,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      beta_fast=beta_fast,
      beta_slow=beta_slow,
      rope_factor=rope_factor,
      dtype=dtype,
      qk_nope_head_dim=qk_nope_head_dim,
      mscale=mscale,
      quant=quant,
  )
  query = jax.ad_checkpoint.checkpoint_name(query, "query_proj")
  key, value = kv_projection(
      inputs,
      positions,
      wkv_a_weights,
      wkv_b_weights,
      kv_norm_scale_weights,
      kv_lora_rank=kv_lora_rank,
      kv_norm_epsilon=kv_norm_epsilon,
      qk_rope_head_dim=qk_rope_head_dim,
      rope_theta=rope_theta,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      beta_fast=beta_fast,
      beta_slow=beta_slow,
      rope_factor=rope_factor,
      dtype=dtype,
      qk_nope_head_dim=qk_nope_head_dim,
      num_query_heads=num_query_heads,
      quant=quant,
  )
  key = jax.ad_checkpoint.checkpoint_name(key, "key_proj")
  value = jax.ad_checkpoint.checkpoint_name(value, "value_proj")
  out = attention_op_fn(
      query,
      key,
      value,
      segment_ids,
      model_mode,
      cached_values=[None, None],
  )
  out = jax.ad_checkpoint.checkpoint_name(out, "attention_out")
  out = dot(out, out_weights, quant=quant, axes=2)
  out = jax.ad_checkpoint.checkpoint_name(out, "out_proj")
  return out
