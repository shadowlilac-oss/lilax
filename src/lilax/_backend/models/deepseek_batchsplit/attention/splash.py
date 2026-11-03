"""TPU Splash attention kernel for the DeepSeek BF16 batch-split schedule."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask

from lilax._backend.kernels import attention


def init_splash_kernel(config):
  """Initializes the Splash kernel."""
  sa_config = attention.splash_attention_kernel.BlockSizes(
      block_q=min(config.sa_block_q, config.max_target_length),
      block_kv=min(config.sa_block_kv, config.max_target_length),
      block_kv_compute=min(config.sa_block_kv_compute, config.max_target_length),
      block_q_dkv=min(config.sa_block_q_dkv, config.max_target_length),
      block_kv_dkv=min(config.sa_block_kv_dkv, config.max_target_length),
      block_kv_dkv_compute=min(config.sa_block_kv_dkv_compute, config.max_target_length),
      block_q_dq=None if config.sa_use_fused_bwd_kernel else min(config.sa_block_q_dq, config.max_target_length),
      block_kv_dq=None if config.sa_use_fused_bwd_kernel else min(config.sa_block_kv_dq, config.max_target_length),
      use_fused_bwd_kernel=config.sa_use_fused_bwd_kernel,
      q_layout=attention.splash_attention_kernel.QKVLayout[config.sa_q_layout],
      k_layout=attention.splash_attention_kernel.QKVLayout[config.sa_k_layout],
      v_layout=attention.splash_attention_kernel.QKVLayout[config.sa_v_layout],
  )
  mask = splash_attention_mask.CausalMask(shape=(config.max_target_length, config.max_target_length))
  multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * config.num_query_heads)
  return attention.splash_attention_kernel.make_splash_mha(
      mask=multi_head_mask,
      head_shards=1,  # the size of the axis if sharding over heads
      q_seq_shards=1,  # axis for sequence sharding
      block_sizes=sa_config,
      attn_logits_soft_cap=None,
      residual_checkpoint_name="context",
      save_residuals=True,
  )


def tpu_flash_attention(
    query,
    key,
    value,
    mesh,
    splash_kernel,
    activation_pspec,
):
  """TPU Flash Attention."""
  # Transpose to ('batch', 'heads', 'length', 'kv')
  query = jnp.transpose(query, axes=(0, 2, 1, 3))
  key = jnp.transpose(key, axes=(0, 2, 1, 3))
  value = jnp.transpose(value, axes=(0, 2, 1, 3))

  q_pspec = jax.sharding.PartitionSpec(*activation_pspec + (None,))
  kv_pspec = jax.sharding.PartitionSpec(*activation_pspec + (None,))
  lse_pspec = activation_pspec

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(
          q_pspec,
          kv_pspec,
          kv_pspec,
      ),
      out_specs=(
          q_pspec,
          lse_pspec,
      ),
      check_vma=False,
  )
  def wrap_flash_attention_manual(query, key, value):
    attention_output, logsumexp = jax.vmap(splash_kernel.manual_fwd, in_axes=(0, 0, 0, None, None), out_axes=(0, 0))(
        query,
        key,
        value,
        None,
        None,
    )
    return attention_output, logsumexp

  attention_output, logsumexp = wrap_flash_attention_manual(
      query,
      key,
      value,
  )
  return jnp.transpose(attention_output, axes=(0, 2, 1, 3)), logsumexp


def tpu_flash_attention_bwd(
    attention_out_grad,
    query,
    key,
    value,
    attention_output,
    logsumexp,
    mesh,
    splash_kernel,
    activation_pspec,
):
  """TPU Flash Attention backward."""
  # Transpose to ('batch', 'heads', 'length', 'kv')
  query = jnp.transpose(query, axes=(0, 2, 1, 3))
  key = jnp.transpose(key, axes=(0, 2, 1, 3))
  value = jnp.transpose(value, axes=(0, 2, 1, 3))
  attention_output = jnp.transpose(attention_output, axes=(0, 2, 1, 3))
  attention_out_grad = jnp.transpose(attention_out_grad, axes=(0, 2, 1, 3))

  q_pspec = jax.sharding.PartitionSpec(*activation_pspec + (None,))
  kv_pspec = jax.sharding.PartitionSpec(*activation_pspec + (None,))
  lse_pspec = activation_pspec

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(
          (
              lse_pspec,  # logsumexp.
              q_pspec,
              kv_pspec,
              kv_pspec,
              q_pspec,  # attention_output.
          ),
          (
              q_pspec,
              lse_pspec,
          ),
      ),
      out_specs=(
          q_pspec,
          kv_pspec,
          kv_pspec,
      ),
      check_vma=False,
  )
  def wrap_flash_attention_manual_bwd(res, grad):
    logsumexp, query, key, value, attention_output = res
    attention_output_grad, _ = grad
    dq, dk, dv = jax.vmap(splash_kernel.manual_bwd, in_axes=(0, 0, 0, 0, 0, 0, None, None), out_axes=(0, 0, 0))(
        query,
        key,
        value,
        attention_output,
        logsumexp,
        attention_output_grad,
        None,
        None,
    )
    return dq, dk, dv

  dq, dk, dv = wrap_flash_attention_manual_bwd(
      (logsumexp, query, key, value, attention_output), (attention_out_grad, None)
  )
  dq = jnp.transpose(dq, axes=(0, 2, 1, 3))
  dk = jnp.transpose(dk, axes=(0, 2, 1, 3))
  dv = jnp.transpose(dv, axes=(0, 2, 1, 3))
  return dq, dk, dv
