"""Pre-normalized residual MLA block across microbatches for the DeepSeek BF16 batch-split schedule."""

import functools

import jax

from lilax._backend.models.deepseek_batchsplit.attention.mla import mla, mla_bwd, mla_remat
from lilax._backend.models.deepseek_batchsplit.gradients import sum_grads
from lilax._backend.models.deepseek_batchsplit.norm import rms_norm
from lilax._backend.models.deepseek_batchsplit.staggering import staggered_call


def mla_with_norms(
    inputs,
    weights,
    yarn_freqs,
    *,
    mesh,
    config,
    splash_kernel,
    normalization_layer_epsilon,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    num_query_heads,
    max_position_embeddings,
    original_max_position_embeddings,
    rope_factor,
    mscale,
    pairwise_swap_and_negate_mask,
    dtype,
    activation_pspec,
):
  """Performs MLA with pre-normalization."""
  pre_attn_scale, attn_ws = weights

  def fn(args):
    x, yarn_freqs = args
    y = rms_norm(
        x,
        pre_attn_scale,
        epsilon=normalization_layer_epsilon,
        dtype=dtype,
        out_sharding=jax.sharding.NamedSharding(mesh, activation_pspec),
    )
    mla_out, mla_res = mla(
        y,
        yarn_freqs,
        attn_ws,
        epsilon=normalization_layer_epsilon,
        kv_lora_rank=kv_lora_rank,
        kv_norm_epsilon=normalization_layer_epsilon,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        num_query_heads=num_query_heads,
        max_position_embeddings=max_position_embeddings,
        original_max_position_embeddings=original_max_position_embeddings,
        rope_factor=rope_factor,
        pairwise_swap_and_negate_mask=pairwise_swap_and_negate_mask,
        dtype=dtype,
        mscale=mscale,
        config=config,
        splash_kernel=splash_kernel,
        mesh=mesh,
        activation_pspec=activation_pspec,
    )
    # Prevent fusion with MoE ops, especially the RMS norm.
    # Unfortunately, this seems to be needed to avoid slight numerical differences
    # between the fwd pass and remat.
    out = jax.lax.optimization_barrier(mla_out + x)
    return out, {"mla_out": out, **mla_res}

  return staggered_call(fn, list(zip(inputs, yarn_freqs)))


def mla_with_norms_remat(
    residuals,
    weights,
    yarn_freqs,
    *,
    mesh,
    config,
    splash_kernel,
    normalization_layer_epsilon,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    num_query_heads,
    max_position_embeddings,
    original_max_position_embeddings,
    rope_factor,
    mscale,
    pairwise_swap_and_negate_mask,
    dtype,
    activation_pspec,
):
  """Performs remat for the mla_with_norms function."""
  xs = residuals.pop("layer_inputs")
  pre_attn_scale, attn_ws = weights

  def remat_fn(args):
    x, yarn_freqs, attn_out, lse = args
    y, pre_attn_rms_norm_bwd = jax.vjp(
        functools.partial(
            rms_norm,
            epsilon=normalization_layer_epsilon,
            dtype=dtype,
            out_sharding=jax.sharding.NamedSharding(mesh, activation_pspec),
        ),
        x,
        pre_attn_scale,
    )
    mla_out, mla_bwds = mla_remat(
        (y, attn_out, lse),
        yarn_freqs,
        attn_ws,
        epsilon=normalization_layer_epsilon,
        kv_lora_rank=kv_lora_rank,
        kv_norm_epsilon=normalization_layer_epsilon,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        num_query_heads=num_query_heads,
        max_position_embeddings=max_position_embeddings,
        original_max_position_embeddings=original_max_position_embeddings,
        rope_factor=rope_factor,
        pairwise_swap_and_negate_mask=pairwise_swap_and_negate_mask,
        dtype=dtype,
        mscale=mscale,
        config=config,
        splash_kernel=splash_kernel,
        mesh=mesh,
        activation_pspec=activation_pspec,
    )
    # Prevent fusion with MoE ops, especially the RMS norm.
    # Unfortunately, this seems to be needed to avoid slight numerical differences
    # between the fwd pass and remat.
    out = jax.lax.optimization_barrier(x + mla_out)
    return out, (pre_attn_rms_norm_bwd, mla_bwds)

  bwds = [None] * len(xs)
  for i, x in enumerate(zip(xs, yarn_freqs, residuals.pop("attn_out"), residuals.pop("lse"))):
    xs[i], bwds[i] = remat_fn(x)
  return xs, bwds


def mla_with_norms_bwd(
    outputs_grad,
    bwds,
):
  """Performs the backward pass for the mla_with_norms function."""

  def bwd_fn(args):
    output_grad, (pre_attn_rms_norm_bwd, mla_bwds) = args
    x_grad, mla_out_grad = output_grad, output_grad
    y_grad, attn_ws_grad = mla_bwd(mla_out_grad, mla_bwds)
    x_grad_partial, pre_attn_scale_grad = pre_attn_rms_norm_bwd(y_grad)
    return x_grad + x_grad_partial, (pre_attn_scale_grad, attn_ws_grad)

  norm_mla_ws_grad = []
  for i, g in enumerate(outputs_grad):
    outputs_grad[i], ws_grad = bwd_fn((g, bwds[i]))
    norm_mla_ws_grad.append(ws_grad)
  (pre_attn_scale_grad, attn_ws_grad) = sum_grads(norm_mla_ws_grad)
  return outputs_grad, (pre_attn_scale_grad, attn_ws_grad)
