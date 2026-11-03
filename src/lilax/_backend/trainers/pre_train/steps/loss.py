"""Native pretraining loss and its auxiliary outputs."""

from typing import Any, TypedDict

from flax import nnx
import jax
import jax.numpy as jnp

from lilax._backend.diffusion.block_diffusion import target_alignment as block_diffusion_target_alignment
from lilax._backend.experimental.lineage import lineage_adapter
from lilax._backend.layers import moe
from lilax._backend.layers.attention_mla import indexer_losses
from lilax._backend.layers.multi_token_prediction import calculate_mtp_loss, mtp_acceptance, mtp_losses
from lilax._backend.trainers.pre_train.moe.bias import collect_moe_bias_updates
from lilax._backend.trainers.pre_train.moe.capacity_aux import update_moe_capacity_aux
from lilax._backend.utils import max_logging, maxtext_utils
from lilax._backend.distributed import sharding
from lilax._backend.utils.globals import EPS
from lilax._backend.losses import cross_entropy
from lilax._backend.utils.vocabulary_tiling import vocab_tiling_nnx_loss


class EncoderKwargs(TypedDict, total=False):
  encoder_images: Any
  encoder_image_masks: Any
  encoder_videos: Any
  encoder_video_masks: Any
  encoder_video_grid_thw: Any


def _one_hot_targets(targets, config, mesh):
  if not getattr(config, "lm_head_fsdp_as_vocab_tp", False):
    return jax.nn.one_hot(targets, config.vocab_size)
  return sharding.vocab_parallel_one_hot(
      targets, config.vocab_size, mesh, sharding.lm_head_logical_axes(vocab_parallel=True)[2], config.shard_mode
  )


def _validate_block_diffusion_masks(data):
  required_masks = {"corruption_mask", "targets_loss_mask"}
  missing_masks = required_masks - data.keys()
  if missing_masks:
    raise ValueError(f"Block-diffusion loss requires explicit batch masks; missing {sorted(missing_masks)}")
  target_shape = data["targets"].shape
  for mask_name in required_masks:
    if data[mask_name].shape != target_shape:
      raise ValueError(f"{mask_name} must match targets shape; got {data[mask_name].shape} and {target_shape}")


def _truncate_to_micro_batch(config, data, is_train):
  if is_train:
    for key, value in data.items():
      data[key] = value[: config.micro_batch_size_to_train_on, :]
  else:
    for key, value in data.items():
      data[key] = value[: config.micro_batch_size_to_eval_on, :]


def _encoder_kwargs(config, data) -> EncoderKwargs:
  encoder_images = data.get("images") if config.use_multimodal else None
  encoder_image_masks = data.get("image_masks") if config.use_multimodal else None
  if "video_grid_thw" in data:
    return {
        "encoder_videos": encoder_images,
        "encoder_video_masks": encoder_image_masks,
        "encoder_video_grid_thw": data.get("video_grid_thw"),
    }
  return {"encoder_images": encoder_images, "encoder_image_masks": encoder_image_masks}


def _pop_intermediate_outputs(model, config):
  mtp_losses_state, mtp_acceptance_state = None, None
  if config.mtp_num_layers > 0:
    mtp_losses_state = nnx.pop(model, mtp_losses)
    mtp_acceptance_state = nnx.pop(model, mtp_acceptance)
  indexer_losses_state = None
  if config.use_indexer:
    indexer_losses_state = nnx.pop(model, indexer_losses)
  intermediates = nnx.pop(model, nnx.Intermediate)
  intermediate_outputs = intermediates.to_pure_dict()
  intermediate_outputs = moe.finalize_deferred_intermediates(intermediate_outputs, config)
  if mtp_losses_state is not None and mtp_acceptance_state is not None:
    intermediate_outputs["mtp_losses"] = mtp_losses_state.to_pure_dict()
    intermediate_outputs["mtp_acceptance"] = mtp_acceptance_state.to_pure_dict()
  if indexer_losses_state is not None:
    intermediate_outputs["indexer_losses"] = indexer_losses_state.to_pure_dict()
  return intermediate_outputs


def _masked_cross_entropy_sums(model, config, data, logits, is_block_diffusion, targets_loss_mask):
  one_hot_targets = _one_hot_targets(data["targets"], config, model.mesh)
  xent, z_loss = cross_entropy.cross_entropy_with_logits(logits, one_hot_targets, z_loss=config.z_loss_multiplier)
  xent = sharding.maybe_shard_with_logical(
      xent, ("activation_embed_and_logits_batch", "activation_length"), model.mesh,
      config.shard_mode, debug_sharding=config.debug_sharding,
  )
  z_loss = sharding.maybe_shard_with_logical(
      z_loss, ("activation_embed_and_logits_batch", "activation_length"), model.mesh,
      config.shard_mode, debug_sharding=config.debug_sharding,
  )
  if is_block_diffusion:
    xent = xent * targets_loss_mask
    z_loss = z_loss * targets_loss_mask
  else:
    xent = xent * (data["targets_segmentation"] != 0)
    z_loss = z_loss * (data["targets_segmentation"] != 0)
  return jnp.sum(xent), jnp.sum(z_loss)


def _add_auxiliary_losses(config, intermediate_outputs, loss, total_weights, manual_gradient_accumulation, is_train):
  def _add_aux_loss(total, aux_loss):
    if manual_gradient_accumulation and is_train:
      return total + aux_loss * jnp.asarray(total_weights, jnp.float32)
    return total + aux_loss

  mtp_loss = 0.0
  if config.mtp_num_layers > 0 and is_train:
    mtp_loss = calculate_mtp_loss(intermediate_outputs, config)
    loss = _add_aux_loss(loss, mtp_loss)
  indexer_loss = 0.0
  if config.use_indexer and config.indexer_loss_scaling_factor > 0.0:
    indexer_losses_list = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "indexer_loss")
    if indexer_losses_list:
      indexer_loss = jnp.mean(jnp.concatenate([jnp.atleast_1d(value) for value in indexer_losses_list]))
      loss = _add_aux_loss(loss, indexer_loss)
    else:
      max_logging.debug("No Indexer loss found. Defaulting to 0.0.")
  moe_lb_loss = 0.0
  if config.num_experts > 1:
    moe_lb_losses = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "moe_lb_loss")
    if moe_lb_losses:
      lb_losses = jnp.concatenate(moe_lb_losses)
      moe_lb_loss = jnp.sum(lb_losses) if moe.uses_megatron_seq_aux_loss(config) else jnp.mean(lb_losses)
      loss = _add_aux_loss(loss, moe_lb_loss)
    else:
      max_logging.debug("\nNo MoE load balance loss found. Defaulting to 0.0.")
  return loss, mtp_loss, indexer_loss, moe_lb_loss


def loss_fn(model, config, data, dropout_rng, params, sparsity_state=None, is_train=True):
  del dropout_rng, params, sparsity_state
  is_block_diffusion = getattr(config, "training_objective", "causal_lm") == "block_diffusion"
  if getattr(config, "attention_type", "global") == "block_diffusion" and not is_block_diffusion:
    raise ValueError(
        "Block-diffusion attention requires target-aligned block-diffusion losses; "
        "causal next-token labels would leak within a bidirectional block."
    )
  if is_block_diffusion:
    _validate_block_diffusion_masks(data)

  _truncate_to_micro_batch(config, data, is_train)
  if lineage_adapter.is_native(config):
    return lineage_adapter.loss_and_aux(model, config, data, is_train=is_train)
  forced_routing_kwargs = (
      {"forced_routed_experts": data["forced_routed_experts"]} if "forced_routed_experts" in data else {}
  )
  if is_block_diffusion:
    targets_loss_mask = (data["targets_loss_mask"] != 0) & (data["targets_segmentation"] != 0)
    target_positions = data.get("targets_position", data["inputs_position"])
  else:
    targets_loss_mask = None
    target_positions = None
  encoder_kwargs = _encoder_kwargs(config, data)
  logits = model(
      decoder_input_tokens=data["inputs"],
      decoder_positions=data["inputs_position"],
      decoder_segment_ids=data["inputs_segmentation"],
      **encoder_kwargs,
      enable_dropout=config.enable_dropout if is_train else False,
      decoder_target_tokens=data["targets"],
      decoder_target_mask=data["targets_segmentation"],
      **forced_routing_kwargs,
  )
  intermediate_outputs = _pop_intermediate_outputs(model, config)

  is_dense_warmup = config.use_indexer and config.indexer_loss_scaling_factor > 0.0 and not config.indexer_sparse_training
  if is_dense_warmup and is_train:
    xent_sum = 0.0
    total_z_loss = 0.0
  elif config.num_vocab_tiling > 1:
    hidden_state_key = ("decoder", "hidden_states")
    hidden_states = maxtext_utils.get_nested_value(intermediate_outputs, hidden_state_key)[0]
    xent_sum, total_z_loss = vocab_tiling_nnx_loss(model, hidden_states, data, config, is_train)
  else:
    if is_block_diffusion:
      logits = block_diffusion_target_alignment.align_logits_to_targets(
          logits, config.block_diffusion_logit_alignment, target_positions, data["targets_segmentation"] != 0
      )
    xent_sum, total_z_loss = _masked_cross_entropy_sums(
        model, config, data, logits, is_block_diffusion, targets_loss_mask
    )

  if is_block_diffusion:
    assert targets_loss_mask is not None
    total_weights = jnp.sum(targets_loss_mask)
  else:
    total_weights = jnp.sum(data["targets_segmentation"] != 0)
  manual_gradient_accumulation = config.gradient_accumulation_steps > 1 and not config.use_tunix_gradient_accumulation
  if manual_gradient_accumulation:
    loss = xent_sum
  else:
    loss = xent_sum / (total_weights + EPS)
  total_z_loss = total_z_loss / (total_weights + EPS)
  loss, mtp_loss, indexer_loss, moe_lb_loss = _add_auxiliary_losses(
      config, intermediate_outputs, loss, total_weights, manual_gradient_accumulation, is_train
  )

  moe_bias_updates, mtp_moe_bias_updates = collect_moe_bias_updates(config, intermediate_outputs)
  if not is_train and config.mtp_eval_target_module > 0:
    intermediate_outputs["logits"] = logits
  aux = {
      "intermediate_outputs": intermediate_outputs,
      "xent_sum": xent_sum,
      "z_loss": total_z_loss,
      "total_weights": total_weights,
      "moe_lb_loss": moe_lb_loss,
      "indexer_loss": indexer_loss,
      "moe_bias_updates": moe_bias_updates,
      "mtp_moe_bias_updates": mtp_moe_bias_updates,
      "mtp_loss": mtp_loss,
      "batch_stats": (intermediate_outputs.get("batch_stats", None) if hasattr(intermediate_outputs, "get") else None),
  }
  update_moe_capacity_aux(aux, config, intermediate_outputs)
  return loss, aux
