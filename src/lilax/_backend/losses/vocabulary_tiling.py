"""Vocabulary tiled cross-entropy for NNX models."""

import functools

from flax import nnx
import jax
import jax.numpy as jnp

from lilax._backend.common.common_types import ShardMode
from lilax._backend.losses import cross_entropy
from lilax._backend.distributed.sharding import (
    FSDP_MESH_AXES, create_sharding, get_logical_axis_rules, get_physical_spec_without_axes, maybe_shard_with_name,
)


_OUTPUT_HEAD_PATH_KEYS = ("token_embedder", "shared_embedding", "decoder_norm", "logits_dense")


def _is_output_head_param_path(path, _value):
  def _name(key):
    for attr in ("key", "attr", "name"):
      value = getattr(key, attr, None)
      if value is not None:
        return str(value)
    return str(key)

  keys = [_name(key) for key in path]
  return any(key in keys for key in _OUTPUT_HEAD_PATH_KEYS)


def vocab_tiling_nnx_loss(model, hidden_states, data, config, is_train):
  """Compute cross-entropy in tiles, retaining one tile of logits in backward."""
  labels = data["targets"]
  segmentation = data["targets_segmentation"]
  deterministic = not config.enable_dropout if is_train else True
  hidden_spec = create_sharding(model.mesh, ("activation_embed_and_logits_batch", "activation_length", "activation_embed"))
  label_spec = create_sharding(model.mesh, ("activation_embed_and_logits_batch", "activation_length"))
  reshaped_hidden_spec = create_sharding(
      model.mesh, ("num_tile", "activation_embed_and_logits_batch_sequence", "activation_embed")
  )
  reshaped_data_spec = create_sharding(model.mesh, ("num_tile", "activation_embed_and_logits_batch_sequence"))
  chunked_hidden_spec = create_sharding(model.mesh, ("activation_embed_and_logits_batch_sequence", "activation_embed"))
  chunked_data_spec = create_sharding(model.mesh, ("activation_embed_and_logits_batch_sequence",))
  chunked_logits_spec = create_sharding(model.mesh, ("activation_embed_and_logits_batch_sequence", "activation_vocab"))
  shard = functools.partial(maybe_shard_with_name, shard_mode=config.shard_mode, debug_sharding=config.debug_sharding)

  def reshape(inputs, shape, sharding):
    out_sharding = sharding if config.shard_mode == ShardMode.EXPLICIT else None
    return shard(jax.lax.reshape(inputs, shape, out_sharding=out_sharding), sharding)

  hidden_states = shard(hidden_states, hidden_spec)
  labels = shard(labels, label_spec)
  segmentation = shard(segmentation, label_spec)
  graphdef, head_params, other_params, rest = nnx.split(model, _is_output_head_param_path, nnx.Param, ...)
  if config.vocab_tiling_ag_once:
    head_spec = get_physical_spec_without_axes(
        nnx.get_partition_spec(head_params), model.mesh,
        FSDP_MESH_AXES + ("context", "context_usp_ulysses", "expert"), get_logical_axis_rules(),
    )
    head_params = maybe_shard_with_name(head_params, head_spec, shard_mode=config.shard_mode)

  def logits_for_chunk(head, other, state, hidden):
    local_model = nnx.merge(graphdef, head, other, state, copy=True)
    logits = local_model.logits_from_hidden_states_for_vocab_tiling(hidden, deterministic, "train")
    return shard(logits, chunked_logits_spec)

  @jax.custom_vjp
  def chunked_loss(head, other, state, hidden, targets, segments):
    result, _ = forward(head, other, state, hidden, targets, segments)
    return result

  def forward(head, other, state, hidden, targets, segments):
    batch_size, seq_len, emb_dim = hidden.shape
    tile_size = (batch_size * seq_len) // config.num_vocab_tiling
    hidden = reshape(hidden, (config.num_vocab_tiling, tile_size, emb_dim), reshaped_hidden_spec)
    targets = reshape(targets, (config.num_vocab_tiling, tile_size), reshaped_data_spec)
    segments = reshape(segments, (config.num_vocab_tiling, tile_size), reshaped_data_spec)

    def scan_body(acc, chunks):
      hidden_chunk, target_chunk, segment_chunk = chunks
      hidden_chunk = shard(hidden_chunk, chunked_hidden_spec)
      target_chunk = shard(target_chunk, chunked_data_spec)
      segment_chunk = shard(segment_chunk, chunked_data_spec)
      logits = logits_for_chunk(head, other, state, hidden_chunk)
      xent, z_loss = cross_entropy.cross_entropy_with_logits(
          logits, jax.nn.one_hot(target_chunk, config.vocab_size), z_loss=config.z_loss_multiplier
      )
      mask = segment_chunk != 0
      return (acc[0] + jnp.sum(xent * mask), acc[1] + jnp.sum(z_loss * mask)), None

    initial = (jnp.zeros((), dtype=jnp.float32), jnp.zeros((), dtype=jnp.float32))
    result, _ = jax.lax.scan(scan_body, initial, (hidden, targets, segments))
    return result, (head, other, state, hidden, targets, segments, batch_size, seq_len, emb_dim)

  def backward(residuals, cotangents):
    head, other, state, hidden, targets, segments, batch_size, seq_len, emb_dim = residuals
    loss_cotangent, _ = cotangents

    def single_loss(input_head, input_hidden, input_targets, input_segments):
      logits = logits_for_chunk(input_head, other, state, input_hidden)
      xent, _ = cross_entropy.cross_entropy_with_logits(
          logits, jax.nn.one_hot(input_targets, config.vocab_size), z_loss=config.z_loss_multiplier
      )
      return jnp.sum(xent * (input_segments != 0))

    def scan_body(grad_head, chunks):
      hidden_chunk, target_chunk, segment_chunk = chunks
      hidden_chunk = shard(hidden_chunk, chunked_hidden_spec)
      target_chunk = shard(target_chunk, chunked_data_spec)
      segment_chunk = shard(segment_chunk, chunked_data_spec)
      _, vjp = jax.vjp(lambda p, h: single_loss(p, h, target_chunk, segment_chunk), head, hidden_chunk)
      head_update, grad_hidden = vjp(1.0)
      grad_hidden = shard(grad_hidden, chunked_hidden_spec)
      grad_head = jax.tree_util.tree_map(lambda acc, update: acc + update, grad_head, head_update)
      return grad_head, grad_hidden

    grad_head, grad_hidden = jax.lax.scan(
        scan_body, jax.tree_util.tree_map(jnp.zeros_like, head), (hidden, targets, segments)
    )
    grad_hidden = shard(grad_hidden, reshaped_hidden_spec)
    grad_head = jax.tree_util.tree_map(lambda g: g * loss_cotangent, grad_head)
    grad_hidden *= loss_cotangent
    grad_head = jax.tree_util.tree_map(lambda x, y: y.astype(x.dtype), head, grad_head)
    grad_hidden = reshape(grad_hidden, (batch_size, seq_len, emb_dim), hidden_spec)
    return (
        grad_head, jax.tree_util.tree_map(jnp.zeros_like, other), jax.tree_util.tree_map(jnp.zeros_like, state),
        grad_hidden.astype(hidden.dtype), None, None,
    )

  chunked_loss.defvjp(forward, backward)
  return chunked_loss(head_params, other_params, rest, hidden_states, labels, segmentation)
