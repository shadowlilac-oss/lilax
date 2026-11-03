"""Alternative DeepSeek model definition with batch-split schedule."""

import functools

import jax
import jax.numpy as jnp
from lilax._backend.models.deepseek_batchsplit_fp8 import collectives  # pylint: disable=unused-import
from lilax._backend.models.deepseek_batchsplit_fp8.attention.mla import attention_op_from_config, mla_with_norms
from lilax._backend.models.deepseek_batchsplit_fp8.microbatch import merge, split
from lilax._backend.models.deepseek_batchsplit_fp8.moe.layer import moe
from lilax._backend.models.deepseek_batchsplit_fp8.sharding import with_data_parallel_constraint
from lilax._backend.models.deepseek_batchsplit_fp8.weights.fetch import fetch_weights
from lilax._backend.models.deepseek_batchsplit_fp8.weights.gather import gather_weights


def scan_batch_split_layers(
    inputs,
    params,
    positions,
    segment_ids,
    *,
    model_mode,
    mesh,
    quant,
    cfg,
    policy,
):
  """Scans the layers with batch-split schedule."""

  def batch_split_scan_fn(inputs, weights, dpos, dseg):
    weights = gather_weights(weights, mesh)
    xs = batch_split_schedule(
        inputs,
        weights,
        dpos,
        dseg,
        model_mode=model_mode,
        mesh=mesh,
        quant=quant,
        cfg=cfg,
    )
    return xs, None

  batch_split_scan_fn_checkpointed = jax.checkpoint(
      batch_split_scan_fn,
      # No need to prevent CSE inside scan.
      prevent_cse=False,
      policy=policy,
  )
  weights = fetch_weights(params, cfg.dtype)
  # `jax.lax.scan` expects the leading dimension of weights to be the scan
  # dimension, but the weights are initialized/loaded with the param scan
  # axis as the scan dimension, so swap the axes.
  weights = jax.tree.map(lambda x: jnp.swapaxes(x, 0, cfg.param_scan_axis), weights)

  activation_pspec = jax.sharding.PartitionSpec(
      ("data", "fsdp", "fsdp_transpose", "expert", "context"),
      None,
      None,
  )
  inputs = jax.shard_map(
      functools.partial(split, split_factor=cfg.batch_split_factor),
      mesh=mesh,
      in_specs=activation_pspec,
      out_specs=[activation_pspec] * cfg.batch_split_factor,
  )(inputs)
  dpos = split(positions, split_factor=cfg.batch_split_factor)
  dseg = split(segment_ids, split_factor=cfg.batch_split_factor)
  outputs, _ = jax.lax.scan(
      functools.partial(batch_split_scan_fn_checkpointed, dpos=dpos, dseg=dseg),
      inputs,
      weights,
  )
  outputs = jax.shard_map(
      functools.partial(merge, split_factor=cfg.batch_split_factor),
      mesh=mesh,
      in_specs=([activation_pspec] * cfg.batch_split_factor,),
      out_specs=activation_pspec,
  )(outputs)
  return outputs


def batch_split_schedule(
    inputs,
    weights,
    positions,
    segment_ids,
    *,
    model_mode,
    mesh,
    quant,
    cfg,
):
  """Applies the DeepSeek MoE layer with batch-split schedule."""
  xs = [with_data_parallel_constraint(x, mesh) for x in inputs]
  xs = jax.ad_checkpoint.checkpoint_name(xs, "decoder_layer_input")

  attn_op = attention_op_from_config(cfg, mesh=mesh, quant=quant)
  norm_mla_ws, moe_ws = weights
  xs = mla_with_norms(
      xs,
      norm_mla_ws,
      positions,
      segment_ids,
      mesh=mesh,
      model_mode=model_mode,
      attn_op=attn_op,
      normalization_layer_epsilon=cfg.normalization_layer_epsilon,
      kv_lora_rank=cfg.kv_lora_rank,
      qk_nope_head_dim=cfg.qk_nope_head_dim,
      qk_rope_head_dim=cfg.qk_rope_head_dim,
      rope_max_timescale=cfg.rope_max_timescale,
      num_query_heads=cfg.num_query_heads,
      max_position_embeddings=cfg.max_position_embeddings,
      original_max_position_embeddings=cfg.original_max_position_embeddings,
      beta_fast=cfg.beta_fast,
      beta_slow=cfg.beta_slow,
      rope_factor=cfg.rope_factor,
      mscale=cfg.mscale,
      dtype=cfg.dtype,
      quant=quant,
  )

  xs = moe(
      xs,
      moe_ws,
      mesh=mesh,
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      routed_scaling_factor=cfg.routed_scaling_factor,
      expert_axis_name="expert",
      use_gather_mosaic_kernel=False,
      config=cfg,
      quant=quant,
  )
  return xs
