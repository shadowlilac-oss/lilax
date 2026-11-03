"""Native pretraining loss and training loop."""

from typing import Any, TypedDict
import contextlib
import datetime
import functools
import gc
import os
import time

from flax import nnx, traverse_util
from flax.core.spmd import logical_axis_rules
from flax.nnx import variablelib
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
import numpy as np
from lilax._backend.common.checkpoints import checkpointing
from lilax._backend.common import metric_logger, profiler
from lilax._backend.common.checkpoints import snapshotting
from lilax._backend.common.goodput import GoodputEvent, RECORD_JOB_END_TIME, maybe_record_goodput, record_goodput
from lilax._backend.common.metric_logger import record_activation_metrics
from lilax._backend.diffusion.block_diffusion import target_alignment as block_diffusion_target_alignment
from lilax._backend.experimental.lineage import lineage_adapter
from lilax._backend.layers import moe
from lilax._backend.layers.attention_mla import indexer_losses
from lilax._backend.layers.multi_token_prediction import calculate_mtp_acceptance_rate, calculate_mtp_loss, mtp_acceptance, mtp_losses
from lilax._backend.utils import elastic_utils, exceptions, gcs_utils, max_logging, max_utils, maxtext_utils, maxtext_utils_nnx
from lilax._backend.common.telemetry import mllog_utils
from lilax._backend.utils import qk_clip_utils, sharding, train_utils
from lilax._backend.utils.diagnostics import jaxpr_dump
from lilax._backend.utils.gradient_accumulation import gradient_accumulation_loss_and_grad
from lilax._backend.utils.globals import EPS
from lilax._backend.utils.vocabulary_tiling import vocab_tiling_nnx_loss


class EncoderKwargs(TypedDict, total=False):
  encoder_images: Any
  encoder_image_masks: Any
  encoder_videos: Any
  encoder_video_masks: Any
  encoder_video_grid_thw: Any


def get_first_step(model, state):
  del model
  if hasattr(state, "inner_state"):
    step_val = state.step.get_value() if hasattr(state.step, "get_value") else state.step
    return int(step_val)
  return int(state.optimizer.step.get_value())


def _one_hot_targets(targets, config, mesh):
  if not getattr(config, "lm_head_fsdp_as_vocab_tp", False):
    return jax.nn.one_hot(targets, config.vocab_size)
  return sharding.vocab_parallel_one_hot(
      targets, config.vocab_size, mesh, sharding.lm_head_logical_axes(vocab_parallel=True)[2], config.shard_mode
  )


def loss_fn(model, config, data, dropout_rng, params, sparsity_state=None, is_train=True):
  del dropout_rng, params, sparsity_state
  is_block_diffusion = getattr(config, "training_objective", "causal_lm") == "block_diffusion"
  if getattr(config, "attention_type", "global") == "block_diffusion" and not is_block_diffusion:
    raise ValueError(
        "Block-diffusion attention requires target-aligned block-diffusion losses; "
        "causal next-token labels would leak within a bidirectional block."
    )
  if is_block_diffusion:
    required_masks = {"corruption_mask", "targets_loss_mask"}
    missing_masks = required_masks - data.keys()
    if missing_masks:
      raise ValueError(f"Block-diffusion loss requires explicit batch masks; missing {sorted(missing_masks)}")
    target_shape = data["targets"].shape
    for mask_name in required_masks:
      if data[mask_name].shape != target_shape:
        raise ValueError(f"{mask_name} must match targets shape; got {data[mask_name].shape} and {target_shape}")

  if is_train:
    for key, value in data.items():
      data[key] = value[: config.micro_batch_size_to_train_on, :]
  else:
    for key, value in data.items():
      data[key] = value[: config.micro_batch_size_to_eval_on, :]
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
  encoder_images = data.get("images") if config.use_multimodal else None
  encoder_image_masks = data.get("image_masks") if config.use_multimodal else None
  is_video = "video_grid_thw" in data
  encoder_kwargs: EncoderKwargs
  if is_video:
    encoder_kwargs = {
        "encoder_videos": encoder_images,
        "encoder_video_masks": encoder_image_masks,
        "encoder_video_grid_thw": data.get("video_grid_thw"),
    }
  else:
    encoder_kwargs = {"encoder_images": encoder_images, "encoder_image_masks": encoder_image_masks}
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
    one_hot_targets = _one_hot_targets(data["targets"], config, model.mesh)
    xent, z_loss = max_utils.cross_entropy_with_logits(logits, one_hot_targets, z_loss=config.z_loss_multiplier)
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
    xent_sum = jnp.sum(xent)
    total_z_loss = jnp.sum(z_loss)

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

  moe_bias_updates = None
  mtp_moe_bias_updates = None
  if config.routed_bias and config.routed_bias_update_rate > 0.0:
    for path, value in jax.tree_util.tree_leaves_with_path(intermediate_outputs):
      keys = tuple(key.key for key in path if hasattr(key, "key"))
      if not keys or keys[-1] != "moe_bias_updates":
        continue
      if "decoder" in keys:
        moe_bias_updates = (value,)
      elif "mtp_block" in keys:
        if mtp_moe_bias_updates is None:
          mtp_moe_bias_updates = []
        mtp_moe_bias_updates.append(value)
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
  te_moe_block = getattr(config, "te_moe_block", False)
  if te_moe_block:
    overflow_values = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "te_moe_capacity_overflow")
    total_recv_values = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "te_moe_total_recv_tokens")
    capacity_values = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "te_moe_recv_capacity_per_rank")
    if not overflow_values or not total_recv_values or not capacity_values:
      raise ValueError("te_moe_block=True did not produce TE MoE receive-capacity intermediates.")
    aux.update({
        "te_moe_capacity_overflow": jnp.any(jnp.concatenate(overflow_values)),
        "te_moe_max_total_recv_tokens": jnp.max(jnp.concatenate(total_recv_values)),
        "te_moe_recv_capacity_per_rank": jnp.min(jnp.concatenate(capacity_values)),
    })
  has_moe_overflow = jnp.bool_(False)
  if config.moe_dropless_fallback == "step":
    moe_overflow_flags = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "moe_has_overflow", ravel=False)
    if moe_overflow_flags:
      has_moe_overflow = jnp.any(jnp.stack([jnp.any(value) for value in moe_overflow_flags]))
  aux["has_moe_overflow"] = has_moe_overflow
  if config.moe_dropless_fallback == "layer":
    fallback_flags = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "moe_dropless_fallback")
    aux["moe_dropless_fallback_layers"] = sum(
        (jnp.sum(jnp.asarray(value).astype(jnp.int32)) for value in fallback_flags), jnp.int32(0)
    )
  if getattr(config, "log_required_ragged_buffer_factor", False):
    required_rbf_values = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "moe_required_rbf")
    if required_rbf_values:
      aux["moe_required_rbf"] = jnp.concatenate([value.astype(jnp.float32) for value in required_rbf_values])
  if getattr(config, "moe_log_max_load_ratio", False):
    load_ratios = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "moe_max_load_ratio")
    if load_ratios:
      load_ratios = jnp.concatenate(load_ratios)
      aux["moe_max_load_ratio"] = jnp.max(load_ratios)
      aux["moe_max_load_ratio_mean"] = jnp.mean(load_ratios)
  return loss, aux


def _rollback_on_overflow(new_state, old_leaves, has_moe_overflow):
  if not isinstance(has_moe_overflow, jax.core.Tracer) and not bool(has_moe_overflow):
    return new_state

  def _select(new, old):
    dtype = getattr(new, "dtype", None)
    if dtype is not None and jnp.issubdtype(dtype, jax.dtypes.extended):
      return new
    return jnp.where(has_moe_overflow, old, new)

  new_leaves, treedef = jax.tree_util.tree_flatten(new_state)
  if len(new_leaves) != len(old_leaves):
    raise ValueError(
        "moe_dropless_fallback='step': cannot roll back the step -- the updated state has "
        f"{len(new_leaves)} leaves but the input state had {len(old_leaves)}."
    )
  return jax.tree_util.tree_unflatten(treedef, [_select(new, old) for new, old in zip(new_leaves, old_leaves)])


def step_diagnostics(raw_grads, has_moe_overflow, bias_values, bias_updates):
  leaves = [
      grad for grad in jax.tree_util.tree_leaves(raw_grads)
      if hasattr(grad, "dtype") and jnp.issubdtype(grad.dtype, jnp.inexact)
  ]
  max_abs = jnp.max(jnp.stack([jnp.max(jnp.abs(grad)).astype(jnp.float32) for grad in leaves])) if leaves else jnp.float32(0)
  checksum = jnp.float32(0)
  for bias in bias_values:
    checksum = checksum + jnp.sum(jnp.asarray(bias, jnp.float32))
  nonzero = jnp.float32(0)
  for update in bias_updates:
    nonzero = nonzero + jnp.sum(jnp.asarray(update) != 0).astype(jnp.float32)
  overflow = jnp.asarray(False if has_moe_overflow is None else has_moe_overflow).astype(jnp.float32)
  return {
      "diag/max_abs_grad": max_abs,
      "diag/moe_bias_checksum": checksum,
      "diag/moe_bias_update_nonzero": nonzero,
      "diag/moe_overflow": overflow,
  }


def _find_gate_bias(module: nnx.Module | None) -> nnx.Variable | None:
  if module is None:
    return None
  for _, node in nnx.iter_graph(module):
    if type(node).__name__ == "GateLogit" and hasattr(node, "bias") and node.bias is not None:
      return node.bias
  return None


def _decode_moe_bias_update(update, config):
  update = jnp.asarray(update[0] if isinstance(update, (tuple, list)) else update)
  if not getattr(config, "using_pipeline_parallelism", False):
    return update
  counts = update.astype(jnp.float32)
  return jnp.sign(jnp.mean(counts, axis=-1, keepdims=True) - counts) * config.routed_bias_update_rate


def _apply_moe_bias_updates_by_path(root: nnx.Module, intermediates: dict, config) -> None:
  for path, update in traverse_util.flatten_dict(intermediates).items():
    if path[-1] != "moe_bias_updates":
      continue
    target = root
    for key in path[:-1]:
      target = getattr(target, key)
    bias = _find_gate_bias(target)
    if bias is None:
      continue
    update = _decode_moe_bias_update(update, config)
    if update.shape != bias.value.shape:
      raise ValueError(
          f"moe_bias_updates at {'/'.join(map(str, path))} has shape {update.shape}, "
          f"but the gate bias it updates has shape {bias.value.shape}."
      )
    bias.value = bias.value + update.astype(bias.value.dtype)


def train_step(model, config, state_mesh_shardings, params_shardings, state, data, dropout_rng=None):
  del dropout_rng
  input_state_leaves = jax.tree_util.tree_leaves(state) if config.moe_dropless_fallback == "step" else None
  state = nnx.merge(model, state)
  loss_model, loss_params, loss_rng = state.model, None, None
  if config.gradient_accumulation_steps > 1:
    loss, aux, raw_grads = gradient_accumulation_loss_and_grad(
        loss_fn, config, loss_model, loss_params, params_shardings, data, loss_rng
    )
    if lineage_adapter is not None and lineage_adapter.is_native(config):
      aux = lineage_adapter.finalize_accumulated_step(loss_model, aux)
  else:
    owg_type = variablelib.variable_type_from_name("_overwrite_with_gradient", allow_register=True)
    custom_param_filter = nnx.Any(owg_type)
    train_param_type = (
        getattr(nnx, "LoRAParam", nnx.Param)
        if getattr(getattr(config, "lora", None), "enable_lora", False)
        else nnx.Param
    )
    nnx.pop(state.model, nnx.Intermediate)
    model_graphdef, curr_params, custom_params, rest = nnx.split(state.model, train_param_type, custom_param_filter, ...)
    if config.parameter_memory_host_offload:
      device_param_shardings = jax.tree_util.tree_map_with_path(
          maxtext_utils_nnx.move_memory_to_device, params_shardings, is_leaf=lambda value: isinstance(value, NamedSharding)
      )
      curr_params = jax.device_put(curr_params, device_param_shardings)
      nnx.update(state.model, curr_params)
    if config.shard_optimizer_over_data:
      param_sharding_lookup = {}
      for path, value in jax.tree_util.tree_leaves_with_path(
          params_shardings, is_leaf=lambda leaf: isinstance(leaf, (nnx.Variable, NamedSharding, jax.sharding.Sharding))
      ):
        param_sharding_lookup[path] = value.get_value() if isinstance(value, nnx.Variable) else value

      def _maybe_shard_param(path, var):
        if path in param_sharding_lookup:
          return sharding.maybe_shard_with_name(var, param_sharding_lookup[path], shard_mode=config.shard_mode)
        return var

      curr_params = jax.tree_util.tree_map_with_path(
          _maybe_shard_param, curr_params, is_leaf=lambda value: isinstance(value, nnx.Variable)
      )
      nnx.update(state.model, curr_params)

    def diff_wrapper(curr_params, custom_params, rest, config, data):
      local_model = nnx.merge(model_graphdef, curr_params, custom_params, rest, copy=True)
      loss, aux = loss_fn(local_model, config, data, None, None, is_train=True)
      non_param_rest = nnx.state(local_model, nnx.Not(nnx.Any(nnx.Param, custom_param_filter, nnx.Intermediate)))
      return loss, (aux, non_param_rest)

    grad_func = jax.value_and_grad(diff_wrapper, argnums=(0, 1), has_aux=True)
    with max_utils.model_loop_unroll_context(config):
      (loss, (aux, non_param_rest)), (raw_grads, custom_grads) = grad_func(curr_params, custom_params, rest, config, data)
    nnx.update(state.model, nnx.State.merge(custom_grads, non_param_rest))

  raw_grads = jax.tree_util.tree_map(
      lambda grad: grad.astype(config.grad_dtype) if grad.dtype == jnp.float32 else grad, raw_grads
  )
  if config.parameter_memory_host_offload:
    raw_grads = jax.device_put(raw_grads, max_utils.with_memory_kind(params_shardings, "device"))
  intermediate_outputs = aux["intermediate_outputs"]
  xent_sum = aux["xent_sum"]
  total_weights = aux["total_weights"]
  moe_lb_loss = aux["moe_lb_loss"]
  has_moe_overflow = aux.get("has_moe_overflow")
  indexer_loss = aux.get("indexer_loss", 0.0)
  z_loss = aux.get("z_loss", 0.0)
  moe_bias_updates = aux.get("moe_bias_updates")
  mtp_moe_bias_updates = aux.get("mtp_moe_bias_updates")
  mtp_loss = aux.get("mtp_loss", 0.0)
  new_opt_state = None
  bias_metrics = {}
  diag_bias_values, diag_bias_updates = [], []
  if config.gradient_clipping_threshold > 0:
    grads = maxtext_utils.apply_gradient_clipping(raw_grads, None, config.gradient_clipping_threshold)
  else:
    grads = raw_grads
  if config.optimizer_memory_host_offload:
    device_opt_shardings = jax.tree_util.tree_map_with_path(
        maxtext_utils_nnx.move_memory_to_device, state_mesh_shardings.optimizer,
        is_leaf=lambda value: isinstance(value, NamedSharding),
    )
    opt_state = nnx.state(state.optimizer)
    new_opt_state = jax.device_put(opt_state, device_opt_shardings)
    nnx.update(state.optimizer, new_opt_state)
  if config.skip_step_on_spikes:
    grad_norm = max_utils.l2norm_pytree(grads)
    state.apply_gradients(grads, loss=loss, grad_norm=grad_norm)
  else:
    state.apply_gradients(grads)
  new_state = state
  if lineage_adapter.is_native(config):
    diag_bias_values = list(aux.get("diag_bias_values", ()))
    diag_bias_updates = list(aux.get("diag_bias_updates", ()))
  elif config.routed_bias and config.routed_bias_update_rate > 0.0:
    if getattr(config, "model_name", "").startswith("deepseek4"):
      max_logging.log("DeepSeek V4: Applying auxiliary-loss-free routing bias via pure NNX MoEBiasVar.")
      flat_intermediates = traverse_util.flatten_dict(aux.get("intermediate_outputs", {}))
      for path, update in flat_intermediates.items():
        if path[-1] != "moe_bias_updates":
          continue
        target = new_state.model
        prefix = path[1:-1] if path[0] == "intermediates" else path[:-1]
        for key in prefix:
          if hasattr(target, key):
            target = getattr(target, key)
          elif isinstance(target, dict) and key in target:
            target = target[key]
          else:
            target = None
            break
        if target is None:
          continue
        for _, node in nnx.iter_graph(target):
          if type(node).__name__ == "GateLogit" and hasattr(node, "bias") and node.bias is not None:
            update_val = _decode_moe_bias_update(update, config)
            name_prefix = "-".join(map(str, prefix))
            if getattr(config, "log_moe_bias_norms", False):
              bias_metrics[f"learning/moe_bias_before_norm_{name_prefix}"] = jnp.linalg.norm(node.bias.value)
            node.bias.value = node.bias.value + jnp.array(update_val)
            diag_bias_values.append(node.bias.value)
            diag_bias_updates.append(update_val)
            if getattr(config, "log_moe_bias_norms", False):
              bias_metrics[f"learning/moe_bias_update_norm_{name_prefix}"] = jnp.linalg.norm(jnp.array(update_val))
    else:
      if getattr(config, "using_pipeline_parallelism", False):
        _apply_moe_bias_updates_by_path(new_state.model.decoder, intermediate_outputs.get("decoder", {}), config)
      else:
        decoder_layer = getattr(new_state.model.decoder, "moe_layers", new_state.model.decoder)
        decoder_bias = _find_gate_bias(decoder_layer)
        if decoder_bias is not None and moe_bias_updates is not None:
          decoder_update = jnp.array(moe_bias_updates[0])
          decoder_bias.value = decoder_bias.value + decoder_update
          diag_bias_values.append(decoder_bias.value)
          diag_bias_updates.append(decoder_update)
      if mtp_moe_bias_updates is not None and hasattr(new_state.model, "mtp_block"):
        for index, update in enumerate(mtp_moe_bias_updates):
          mtp_layer = getattr(new_state.model.mtp_block, f"mtp_layer_{index + 1}", None)
          mtp_bias = _find_gate_bias(mtp_layer)
          if mtp_bias is not None:
            mtp_update = _decode_moe_bias_update(update, config)
            mtp_bias.value = mtp_bias.value + mtp_update
            diag_bias_values.append(mtp_bias.value)
            diag_bias_updates.append(mtp_update)
  lm_loss = xent_sum / (total_weights + EPS)
  scalar_metrics = {
      "learning/loss": loss, "learning/lm_loss": lm_loss, "learning/perplexity": jnp.exp(lm_loss),
      "learning/z_loss": z_loss, "learning/moe_lb_loss": moe_lb_loss, "learning/indexer_loss": indexer_loss,
      "learning/mtp_loss": mtp_loss, "learning/total_weights": total_weights,
  }
  if getattr(config, "te_moe_block", False):
    scalar_metrics.update({
        "learning/te_moe_capacity_overflow": aux["te_moe_capacity_overflow"].astype(jnp.int32),
        "learning/te_moe_max_total_recv_tokens": aux["te_moe_max_total_recv_tokens"],
        "learning/te_moe_recv_capacity_per_rank": aux["te_moe_recv_capacity_per_rank"],
    })
  if config.moe_dropless_fallback == "layer":
    scalar_metrics["learning/moe_dropless_fallback_layers"] = aux["moe_dropless_fallback_layers"]
  if "moe_max_load_ratio" in aux:
    scalar_metrics["learning/moe_max_load_ratio"] = aux["moe_max_load_ratio"]
    scalar_metrics["learning/moe_max_load_ratio_mean"] = aux["moe_max_load_ratio_mean"]
  scalar_metrics.update(bias_metrics)
  if config.use_qk_clip:
    new_state = qk_clip_utils.apply_qk_clip_nnx(new_state, intermediate_outputs, config)
    global_max_logit = qk_clip_utils.calculate_max_logit_metric(intermediate_outputs)
    if global_max_logit is not None:
      scalar_metrics["learning/max_logits"] = global_max_logit
  if not config.optimizer_memory_host_offload:
    scalar_metrics["learning/grad_norm"] = max_utils.l2norm_pytree(grads)
    scalar_metrics["learning/raw_grad_norm"] = max_utils.l2norm_pytree(raw_grads)
    model_params = nnx.state(new_state.model, nnx.Param)
    scalar_metrics["learning/param_norm"] = max_utils.l2norm_pytree(model_params)
  if getattr(config, "log_step_diagnostics", False):
    scalar_metrics.update(step_diagnostics(raw_grads, has_moe_overflow, diag_bias_values, diag_bias_updates))
    if "learning/raw_grad_norm" not in scalar_metrics:
      scalar_metrics["learning/raw_grad_norm"] = max_utils.l2norm_pytree(raw_grads)
      scalar_metrics["learning/grad_norm"] = max_utils.l2norm_pytree(grads)
      scalar_metrics["learning/param_norm"] = max_utils.l2norm_pytree(nnx.state(new_state.model, nnx.Param))
  if config.skip_step_on_spikes:
    opt_state = nnx.to_pure_dict(nnx.state(new_state.optimizer)).get("opt_state", {})
    is_skipped = opt_state.get("is_skipped") if isinstance(opt_state, dict) else None
    if is_skipped is not None:
      scalar_metrics["optim/step_skipped"] = is_skipped.astype(jnp.float32)
  metrics = {"scalar": scalar_metrics, "scalars": {}}
  if config.moe_dropless_fallback == "step":
    metrics["has_moe_overflow"] = has_moe_overflow if has_moe_overflow is not None else jnp.bool_(False)
  if getattr(config, "log_required_ragged_buffer_factor", False) and "moe_required_rbf" in aux:
    metrics["moe_required_rbf"] = aux["moe_required_rbf"]
  if getattr(config, "record_internal_nn_metrics", False):
    record_activation_metrics(metrics, intermediate_outputs, config)
  out_state = nnx.state(new_state, nnx.Not(nnx.Intermediate))
  if input_state_leaves is not None and has_moe_overflow is not None:
    out_state = _rollback_on_overflow(out_state, input_state_leaves, has_moe_overflow)
  return out_state, metrics


def eval_step(model, config, state, data, dropout_rng=None):
  del dropout_rng
  state = nnx.merge(model, state)
  loss, aux = loss_fn(state.model, config, data, None, None, is_train=False)
  mtp_acceptance_rate = 0.0
  if config.mtp_eval_target_module > 0:
    mtp_acceptance_rate = calculate_mtp_acceptance_rate(aux["intermediate_outputs"], config)
  metrics = {"scalar": {
      "evaluation/loss": loss,
      "evaluation/z_loss": aux.get("z_loss", 0.0),
      "evaluation/total_loss": aux["xent_sum"],
      "evaluation/total_weights": aux["total_weights"],
      "evaluation/moe_lb_loss": aux["moe_lb_loss"],
      "evaluation/indexer_loss": aux.get("indexer_loss", 0.0),
      "evaluation/mtp_loss": aux.get("mtp_loss", 0.0),
      "evaluation/mtp_acceptance_rate_percent": mtp_acceptance_rate,
  }}
  if config.moe_dropless_fallback == "step":
    metrics["has_moe_overflow"] = aux.get("has_moe_overflow", False)
  if config.moe_dropless_fallback == "layer":
    metrics["scalar"]["evaluation/moe_dropless_fallback_layers"] = aux["moe_dropless_fallback_layers"]
  return metrics


def build_dropless_graphdef(config, jit_model, state):
  reconstructed = nnx.merge(jit_model, state)
  for _, module in nnx.iter_graph(reconstructed):
    if type(module).__name__ == "RoutedMoE":
      module.force_dropless = True
      module.num_moe_token_chunks = getattr(config, "retry_num_moe_token_chunks", 2)
      module.moe_chunk_barrier = True
    elif hasattr(module, "get_remat_policy"):
      module.remat_policy_override = "full"
  jit_model_dropless, _ = nnx.split(reconstructed)
  del reconstructed, _
  gc.collect()
  return jit_model_dropless


def build_eval_graphdef(config, jit_model, state):
  if config.eval_ragged_buffer_factor <= 0:
    return None
  reconstructed = nnx.merge(jit_model, state)
  n_moe = 0
  for _, module in nnx.iter_graph(reconstructed):
    if type(module).__name__ == "RoutedMoE":
      module.ragged_buffer_factor_override = float(config.eval_ragged_buffer_factor)
      n_moe += 1
  max_logging.log(f"eval graphdef: ragged_buffer_factor_override={config.eval_ragged_buffer_factor} on {n_moe} RoutedMoE")
  jit_model_eval, _ = nnx.split(reconstructed)
  del reconstructed, _
  gc.collect()
  return jit_model_eval


def build_first_phase_graphdef(config, jit_model, state):
  if getattr(config, "first_phase_ragged_buffer_factor", 0.0) <= 0:
    return None
  reconstructed = nnx.merge(jit_model, state)
  n_moe = 0
  for _, module in nnx.iter_graph(reconstructed):
    if type(module).__name__ == "RoutedMoE":
      module.ragged_buffer_factor_override = float(config.first_phase_ragged_buffer_factor)
      n_moe += 1
  max_logging.log(
      f"first-phase graphdef: ragged_buffer_factor_override={config.first_phase_ragged_buffer_factor} on {n_moe} RoutedMoE"
  )
  jit_model_first_phase, _ = nnx.split(reconstructed)
  del reconstructed, _
  gc.collect()
  return jit_model_first_phase


def aot_compile_step(p_step, lower_args, compiler_options, prefix):
  compiled = p_step.lower(*lower_args).compile(compiler_options=compiler_options)
  max_utils.print_compiled_memory_stats(compiled.memory_analysis(), prefix=prefix)
  return compiled


def make_synthetic_batch(config, shaped_batch, seed=0):
  names = sorted(shaped_batch)
  seq_len = int(getattr(config, "max_target_length", 0))

  def _seq_axis(shape):
    axes = [index for index, size in enumerate(shape) if size == seq_len and index > 0]
    return axes[0] if axes else len(shape) - 1

  def _gen(key):
    out = {}
    keys = jax.random.split(key, len(names))
    for sub_key, name in zip(keys, names):
      shape, dtype = shaped_batch[name].shape, shaped_batch[name].dtype
      if name in ("inputs", "targets"):
        out[name] = jax.random.randint(sub_key, shape, 0, max(int(config.vocab_size), 1), dtype=dtype)
      elif name.endswith("_position"):
        out[name] = jax.lax.broadcasted_iota(dtype, shape, _seq_axis(shape))
      elif jnp.issubdtype(dtype, jnp.floating):
        out[name] = jnp.zeros(shape, dtype)
      else:
        out[name] = jnp.ones(shape, dtype)
    return out

  out_shardings = {name: getattr(shaped_batch[name], "sharding", None) for name in names}
  return jax.jit(_gen, out_shardings=out_shardings)(jax.random.PRNGKey(seed))


def _array_leaf_ids(tree):
  return {id(leaf) for leaf in jax.tree_util.tree_leaves(tree) if isinstance(leaf, jax.Array)}


def _delete_outputs(outputs, keep_ids):
  for leaf in jax.tree_util.tree_leaves(outputs):
    if isinstance(leaf, jax.Array) and id(leaf) not in keep_ids and not leaf.is_deleted():
      leaf.delete()


def warmup_programs(
    programs, state, mesh, train_rules, eval_rules, train_batch, eval_batch, rng_args=(), copy_train_state=False
):
  keep_ids = _array_leaf_ids(state)
  timings = {}
  for name, kind, p_step in programs:
    if p_step is None:
      continue
    rules, batch = (train_rules, train_batch) if kind == "train" else (eval_rules, eval_batch)
    if batch is None:
      max_logging.log(f"warmup_programs_in_init: skipping {name} (no synthetic {kind} batch)")
      continue
    start = time.perf_counter()
    with jax.set_mesh(mesh), logical_axis_rules(rules):
      arg_state = state
      if kind == "train" and copy_train_state:
        arg_state = jax.tree_util.tree_map(lambda value: value.copy() if isinstance(value, jax.Array) else value, state)
      outputs = p_step(arg_state, batch, *rng_args)
      jax.block_until_ready(outputs)
    timings[name] = time.perf_counter() - start
    _delete_outputs(outputs, keep_ids)
    if arg_state is not state:
      _delete_outputs(arg_state, keep_ids)
    del outputs, arg_state
    max_logging.log(f"warmup_programs_in_init: {name} ({kind}) {timings[name]:.3f} s")
  gc.collect()
  return timings


def start_run_clock(config, start_step, warmup_fn=None, barrier_fn=None, python_vars=None):
  mllog_utils.init_print(config)
  if warmup_fn is not None:
    start = time.perf_counter()
    warmup_fn()
    warmed = time.perf_counter()
    if barrier_fn is None:
      barrier_fn = lambda: multihost_utils.sync_global_devices("warmup_programs_in_init")
    barrier_fn()
    done = time.perf_counter()
    max_logging.log(
        f"warmup_programs_in_init: warmup {warmed - start:.3f} s, barrier wait {done - warmed:.3f} s "
        f"(process {jax.process_index()})"
    )
    if python_vars is not None:
      python_vars["last_step_completion"] = datetime.datetime.now()
  mllog_utils.init_stop()
  mllog_utils.run_start()
  mllog_utils.block_start(config, start_step)


def _maybe_log_required_rbf(config, step, metrics, program):
  if not getattr(config, "log_required_ragged_buffer_factor", False) or "moe_required_rbf" not in metrics:
    return
  per_layer = np.asarray(jax.device_get(metrics["moe_required_rbf"]), dtype=np.float32).ravel()
  if per_layer.size == 0:
    return
  per_layer_str = ",".join(f"{value:.4f}" for value in per_layer)
  max_logging.log(f"REQUIRED_RBF step={step} max={per_layer.max():.4f} per_layer=[{per_layer_str}] program={program}")


def training_loop_iteration(
    jax_device_state: dict[str, Any], python_vars: dict[str, Any], immutable_data: dict[str, Any],
    elastic_retry_cache: elastic_utils.RetryCache | None = None,
):
  state = jax_device_state["state"]
  init_rng = jax_device_state["init_rng"]
  mesh = jax_device_state["mesh"]
  p_train_step = jax_device_state["p_train_step"]
  p_train_step_dropless = jax_device_state.get("p_train_step_dropless", None)
  p_train_step_first_phase = jax_device_state.get("p_train_step_first_phase", None)
  p_eval_step = jax_device_state["p_eval_step"]
  p_eval_step_dropless = jax_device_state.get("p_eval_step_dropless", None)
  step = python_vars["step"]
  last_step_completion = python_vars["last_step_completion"]
  data_loader = python_vars["data_loader"]
  rampup_manager = python_vars["rampup_manager"]
  recorder = python_vars["recorder"]
  checkpoint_manager = python_vars["checkpoint_manager"]
  data_iterator = python_vars["data_iterator"]
  eval_data_iterator = python_vars["eval_data_iterator"]
  metric_logger_instance = python_vars["metric_logger_instance"]
  prof = python_vars["prof"]
  config = immutable_data["config"]
  snapshotter = None if elastic_retry_cache is None else elastic_retry_cache.snapshotter
  logical_axis_rules_for_train = immutable_data["logical_axis_rules_for_train"]
  logical_axis_rules_for_eval = immutable_data["logical_axis_rules_for_eval"]
  eval_interval = immutable_data["eval_interval"]
  eval_steps = immutable_data["eval_steps"]
  start_step = immutable_data["start_step"]
  eval_start_step = immutable_data["eval_start_step"]
  dump_hlo = immutable_data["dump_hlo"]
  dump_step = immutable_data["dump_step"]
  dump_hlo_local_dir = immutable_data["dump_hlo_local_dir"]
  dump_hlo_gcs_dir = immutable_data["dump_hlo_gcs_dir"]
  dump_hlo_module_name = immutable_data["dump_hlo_module_name"]
  dump_hlo_delete_local_after = immutable_data["dump_hlo_delete_local_after"]
  dump_hlo_upload_all = immutable_data["dump_hlo_upload_all"]
  prof.maybe_activate_profiler(step, state)
  with jax.profiler.StepTraceAnnotation("train", step_num=step):
    example_batch = data_loader.load_next_batch(rampup_manager=rampup_manager)
    if config.enable_diloco:
      step_rng_args = (jax.jit(jax.random.fold_in)(init_rng, step),)
    else:
      step_rng_args = ()
    with maybe_record_goodput(recorder, GoodputEvent.STEP, step):
      with jax.set_mesh(mesh), logical_axis_rules(logical_axis_rules_for_train):
        in_first_phase = config.retry_dropless_first_steps > 0 and step - start_step < config.retry_dropless_first_steps
        if in_first_phase and p_train_step_dropless is not None and p_train_step_first_phase is None:
          max_logging.log(f"Step {step}: dropless first-phase program")
          state, metrics = p_train_step_dropless(state, example_batch, *step_rng_args)
          _maybe_log_required_rbf(config, step, metrics, "dropless")
        else:
          if in_first_phase and p_train_step_first_phase is not None:
            max_logging.log(
                f"Step {step}: first-phase program (ragged_buffer_factor={config.first_phase_ragged_buffer_factor})"
            )
            p_attempt, attempt_name = p_train_step_first_phase, "first_phase"
          else:
            p_attempt, attempt_name = p_train_step, "normal"
          state, metrics = p_attempt(state, example_batch, *step_rng_args)
          _maybe_log_required_rbf(config, step, metrics, attempt_name)
          if config.moe_dropless_fallback == "step" and p_train_step_dropless is not None and bool(metrics.get("has_moe_overflow")):
            max_logging.log(f"Step {step}: MoE ragged buffer overflow detected! Replaying step with dropless buffer...")
            state, metrics = p_train_step_dropless(state, example_batch, *step_rng_args)
            _maybe_log_required_rbf(config, step, metrics, "dropless")

  step_time_delta = datetime.datetime.now() - last_step_completion
  last_step_completion = datetime.datetime.now()
  completed_step = step + 1
  checkpointing.maybe_save_checkpoint(
      checkpoint_manager, state, config, data_iterator, step, training_graph=immutable_data["training_graph"]
  )
  snapshotting.maybe_save_snapshot(snapshotter, config, state, step, checkpoint_manager, data_iterator)
  if dump_hlo and step == (dump_step if dump_step >= 0 else start_step):
    jax.block_until_ready(state)
    gcs_utils.upload_dump(
        dump_hlo_local_dir, dump_hlo_gcs_dir, module_name=dump_hlo_module_name,
        delete_local_after=dump_hlo_delete_local_after, all_host_upload=dump_hlo_upload_all,
    )
  ran_eval = (
      eval_interval > 0 and step >= start_step and step >= eval_start_step
      and (step - eval_start_step) % eval_interval == 0
  )
  if ran_eval:
    assert eval_data_iterator
    prefill_thread = python_vars.get("eval_prefill_thread")
    if prefill_thread is not None and prefill_thread.is_alive():
      wait_t = time.perf_counter()
      prefill_thread.join(timeout=600)
      max_logging.log(f"eval cache prefill: waited {time.perf_counter() - wait_t:.3f} s before eval")
    if hasattr(eval_data_iterator, "reset"):
      eval_data_iterator.reset()
    metric_logger_instance.reset_eval_metrics()
    mllog_utils.eval_start(config, completed_step, start_step=start_step)
    max_logging.log(f"Starting eval after train step {step}")
    eval_step_count = 0
    last_eval_step_completion = datetime.datetime.now()
    metric_logger_instance.mark_eval_loop_start()
    for eval_batch in eval_data_iterator:
      eval_batch = jax.device_put(
          eval_batch, sharding.get_input_data_sharding(config, mesh, rules=config.logical_axis_rules_for_eval)
      )
      with jax.set_mesh(mesh), logical_axis_rules(logical_axis_rules_for_eval):
        eval_metrics = p_eval_step(state, eval_batch, *step_rng_args)
        if config.moe_dropless_fallback == "step" and p_eval_step_dropless is not None and bool(eval_metrics.get("has_moe_overflow")):
          max_logging.log(
              f"Eval step {eval_step_count}: MoE ragged buffer overflow detected! "
              "Replaying eval step with dropless buffer..."
          )
          eval_metrics = p_eval_step_dropless(state, eval_batch, *step_rng_args)
      eval_step_time_delta = datetime.datetime.now() - last_eval_step_completion
      last_eval_step_completion = datetime.datetime.now()
      metric_logger_instance.buffer_and_write_metrics(
          eval_metrics, eval_step_count, step_time_delta=eval_step_time_delta, is_training=False
      )
      eval_step_count += 1
      if 0 < eval_steps <= eval_step_count:
        break
  prof.maybe_deactivate_profiler(step, state)
  if step == start_step:
    max_utils.print_mem_stats("After params initialized")
  metric_logger_instance.buffer_and_write_metrics(metrics, step, step_time_delta)
  if not ran_eval:
    mllog_utils.step_end(config, completed_step)
  if config.elastic_enabled and completed_step < immutable_data["steps"]:
    snapshotting.maybe_back_up_and_scale_up(snapshotter, config, state, step, checkpoint_manager, data_iterator)
  jax_device_state["state"] = state
  python_vars["last_step_completion"] = last_step_completion
  return metrics


def train_loop(config, recorder, state=None, elastic_retry_cache=None, *, setup=None, reporter_factory=None,
               training_graph=None):
  if setup is None:
    setup = train_utils.setup_train_loop(config, recorder, elastic_retry_cache=elastic_retry_cache)
  (
      init_rng, checkpoint_manager, state_mesh_shardings, model, mesh, learning_rate_schedule,
      data_iterator, data_loader, rampup_manager, eval_data_iterator, state,
  ) = setup
  train_utils.maybe_apply_dcn_throttling(config)
  start_step = get_first_step(model, state)
  train_utils.validate_completed_steps(start_step, config.steps)
  jit_model_dropless = None
  jit_model_eval = None
  jit_model_first_phase = None
  if config.enable_diloco:
    jit_model = model
  else:
    jit_model, state = nnx.split(state)
    if config.moe_dropless_fallback == "step":
      jit_model_dropless = build_dropless_graphdef(config, jit_model, state)
      jit_model_first_phase = build_first_phase_graphdef(config, jit_model, state)
    jit_model_eval = build_eval_graphdef(config, jit_model, state)
  if config.enable_diloco:
    params_shardings = state_mesh_shardings.params
  else:
    params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)
  p_train_step, p_eval_step = elastic_utils.get_or_build(
      elastic_retry_cache,
      "train_and_eval_step",
      functools.partial(
          train_utils.jit_train_and_eval_step, config, jit_model, mesh, state, state_mesh_shardings,
          train_step, eval_step, eval_data_iterator, params_shardings,
      ),
  )
  if jit_model_eval is not None and p_eval_step is not None:
    p_eval_step = train_utils.jit_eval_step(
        config, jit_model_eval, state_mesh_shardings,
        sharding.get_input_data_sharding(config, mesh, rules=config.logical_axis_rules_for_eval), eval_step,
    )
  p_train_step_dropless = None
  p_eval_step_dropless = None
  if jit_model_dropless is not None:
    p_train_step_dropless, p_eval_step_dropless = train_utils.jit_train_and_eval_step(
        config, jit_model_dropless, mesh, state, state_mesh_shardings, train_step,
        eval_step=eval_step, eval_data_iterator=eval_data_iterator, params_shardings=params_shardings,
    )
  p_train_step_first_phase = None
  if jit_model_first_phase is not None:
    p_train_step_first_phase, _ = train_utils.jit_train_and_eval_step(
        config, jit_model_first_phase, mesh, state, state_mesh_shardings, train_step, params_shardings=params_shardings
    )
  precompiled_train = False
  precompiled_eval = False
  shaped_eval_batch = None
  with jax.set_mesh(mesh), logical_axis_rules(config.logical_axis_rules):
    data_sharding = sharding.get_input_data_sharding(config, mesh)
    shaped_batch = maxtext_utils.get_shaped_batch(config, batch_sharding=data_sharding)
    if config.shard_optimizer_over_data:
      state = sharding.reshard_train_state(state, state_mesh_shardings, config.enable_single_controller)
    if config.enable_diloco:
      lower_args = (state, shaped_batch, init_rng)
    else:
      lower_args = (state, shaped_batch)
    jaxpr_dump.maybe_dump_jaxpr(config, p_train_step, lower_args)
    if config.compiled_trainstep_file == "" and not jax.config.jax_enable_pgle:
      compiler_options = max_utils.parse_libtpu_flags_to_dict(config.compile_xla_flags)
      compiled = p_train_step.lower(*lower_args).compile(compiler_options=compiler_options)
      compiled_stats = compiled.memory_analysis()
      max_utils.print_compiled_memory_stats(compiled_stats, prefix="train")
      precompiled_train = True
      if p_train_step_dropless is not None:
        aot_compile_step(p_train_step_dropless, lower_args, compiler_options, prefix="train_dropless")
      if p_train_step_first_phase is not None:
        aot_compile_step(p_train_step_first_phase, lower_args, compiler_options, prefix="train_first_phase")
  if p_eval_step is not None:
    with jax.set_mesh(mesh), logical_axis_rules(config.logical_axis_rules_for_eval):
      data_sharding_eval = sharding.get_input_data_sharding(config, mesh, rules=config.logical_axis_rules_for_eval)
      shaped_eval_batch = maxtext_utils.get_shaped_batch(config, batch_sharding=data_sharding_eval, is_eval=True)
  if p_eval_step is not None and config.compiled_trainstep_file == "" and not jax.config.jax_enable_pgle:
    with jax.set_mesh(mesh), logical_axis_rules(config.logical_axis_rules_for_eval):
      compiler_options = max_utils.parse_libtpu_flags_to_dict(config.compile_xla_flags)
      if config.enable_diloco:
        eval_lower_args = (state, shaped_eval_batch, init_rng)
      else:
        eval_lower_args = (state, shaped_eval_batch)
      compiled_eval = p_eval_step.lower(*eval_lower_args).compile(compiler_options=compiler_options)
      compiled_eval_stats = compiled_eval.memory_analysis()
      max_utils.print_compiled_memory_stats(compiled_eval_stats, prefix="eval")
      precompiled_eval = True
      if p_eval_step_dropless is not None:
        aot_compile_step(p_eval_step_dropless, eval_lower_args, compiler_options, prefix="eval_dropless")
  metric_logger_instance = metric_logger.MetricLogger(
      config=config, learning_rate_schedule=learning_rate_schedule, start_step=start_step
  )
  def on_capture_complete(**capture):
    if metric_logger_instance.reporter is not None:
      metric_logger_instance.reporter.log_capture(**capture)

  prof = profiler.Profiler(config, offset_step=start_step, on_capture_complete=on_capture_complete)
  if config.enable_diloco:
    setup_params = state.params
  else:
    _, setup_params, _ = nnx.split(state.model, nnx.Param, ...)
  metric_logger_instance.write_setup_info_to_tensorboard(setup_params)
  elastic_utils.record_elastic_reinit_end()
  jax_device_state = {
      "state": state, "init_rng": init_rng, "mesh": mesh, "state_mesh_shardings": state_mesh_shardings,
      "p_train_step": p_train_step, "p_train_step_dropless": p_train_step_dropless,
      "p_train_step_first_phase": p_train_step_first_phase, "p_eval_step": p_eval_step,
      "p_eval_step_dropless": p_eval_step_dropless, "model": model,
  }
  python_vars = {
      "step": start_step, "last_step_completion": datetime.datetime.now(), "data_loader": data_loader,
      "rampup_manager": rampup_manager, "recorder": recorder, "checkpoint_manager": checkpoint_manager,
      "data_iterator": data_iterator, "eval_data_iterator": eval_data_iterator,
      "metric_logger_instance": metric_logger_instance, "prof": prof,
  }
  immutable_data = {
      "config": config, "training_graph": training_graph, "logical_axis_rules_for_train": config.logical_axis_rules,
      "logical_axis_rules_for_eval": config.logical_axis_rules_for_eval,
      "shard_optimizer_over_data": config.shard_optimizer_over_data, "shard_mode": config.shard_mode,
      "steps": config.steps, "eval_interval": config.eval_interval, "eval_steps": config.eval_steps,
      "eval_start_step": config.eval_start_step, "save_checkpoint_on_completion": config.save_checkpoint_on_completion,
      "start_step": start_step, "dump_hlo": config.dump_hlo, "dump_step": config.dump_step,
      "dump_hlo_local_dir": config.dump_hlo_local_dir, "dump_hlo_gcs_dir": config.dump_hlo_gcs_dir,
      "dump_hlo_module_name": config.dump_hlo_module_name, "dump_hlo_delete_local_after": config.dump_hlo_delete_local_after,
      "dump_hlo_upload_all": config.dump_hlo_upload_all,
  }
  warmup_fn = None
  if getattr(config, "warmup_programs_in_init", False):
    if not (precompiled_train or precompiled_eval):
      max_logging.log("warmup_programs_in_init: no precompiled program (compiled_trainstep_file or AutoPGLE); skipped")
    else:
      warmup_list = []
      if precompiled_train:
        warmup_list += [("train", "train", p_train_step), ("train_first_phase", "train", p_train_step_first_phase)]
        if config.warmup_skip_train_dropless:
          if p_train_step_dropless is not None:
            max_logging.log("warmup_programs_in_init: skipping train_dropless (warmup_skip_train_dropless=True)")
        else:
          warmup_list.append(("train_dropless", "train", p_train_step_dropless))
      if precompiled_eval:
        warmup_list += [("eval", "eval", p_eval_step), ("eval_dropless", "eval", p_eval_step_dropless)]
      train_batch_sharding = getattr(data_loader, "input_data_shardings", None) or data_sharding
      train_shaped = {
          key: jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=train_batch_sharding)
          for key, value in shaped_batch.items()
      }
      warmup_train_batch = make_synthetic_batch(config, train_shaped, seed=0) if precompiled_train else None
      warmup_eval_batch = make_synthetic_batch(config, shaped_eval_batch, seed=1) if precompiled_eval else None
      copy_train_state = True
      warmup_rng_args = (jax.jit(jax.random.fold_in)(init_rng, start_step),) if config.enable_diloco else ()

      def _run_warmup():
        warmup_programs(
            warmup_list, state, mesh, config.logical_axis_rules, config.logical_axis_rules_for_eval,
            warmup_train_batch, warmup_eval_batch, rng_args=warmup_rng_args, copy_train_state=copy_train_state,
        )

      warmup_fn = _run_warmup
  job_completed_gracefully = False
  te_moe_overflow_window = []
  try:
    if reporter_factory is not None:
      metric_logger_instance.reporter = reporter_factory(config)
    train_utils.prepare_before_run_start(
        config, state, learning_rate_schedule, start_step, mesh, data_loader, shaped_batch, shaped_eval_batch=shaped_eval_batch
    )
    python_vars["last_step_completion"] = datetime.datetime.now()
    start_run_clock(config, start_step, warmup_fn=warmup_fn, python_vars=python_vars)
    python_vars["eval_prefill_thread"] = None
    if config.eval_cache_prefill_in_background and eval_data_iterator is not None:
      python_vars["eval_prefill_thread"] = train_utils.start_eval_cache_prefill(eval_data_iterator)
    while python_vars["step"] < immutable_data["steps"]:
      step = python_vars["step"]
      metrics = training_loop_iteration(jax_device_state, python_vars, immutable_data, elastic_retry_cache=elastic_retry_cache)
      python_vars["step"] += 1
      if getattr(config, "te_moe_block", False):
        te_moe_overflow_window.append((
            int(step), metrics["scalar"]["learning/te_moe_capacity_overflow"],
            metrics["scalar"]["learning/te_moe_max_total_recv_tokens"],
            metrics["scalar"]["learning/te_moe_recv_capacity_per_rank"],
        ))
        check_overflow = len(te_moe_overflow_window) == config.te_ep_overflow_check_every_n_steps or step == config.steps - 1
        if check_overflow:
          checked_window = jax.device_get(tuple(te_moe_overflow_window))
          overflowing_steps = [entry for entry in checked_window if bool(np.asarray(entry[1]))]
          te_moe_overflow_window.clear()
        else:
          overflowing_steps = []
        if overflowing_steps:
          overflow_step = overflowing_steps[0][0]
          observed = max(int(np.asarray(entry[2])) for entry in overflowing_steps)
          capacity = min(int(np.asarray(entry[3])) for entry in overflowing_steps)
          prof.deactivate()
          message = (
              "TE MoE receive capacity overflow at training step "
              f"{overflow_step} (detected at the {config.te_ep_overflow_check_every_n_steps}-step "
              f"check ending at step {step}): observed padded receive demand {observed} exceeds "
              f"recv_capacity_per_rank {capacity} (ragged_buffer_factor={config.ragged_buffer_factor}). "
              "The optimizer update was skipped; increase ragged_buffer_factor, "
              "or set it to -1 to reserve worst-case dropless capacity."
          )
          max_logging.error(message)
          raise RuntimeError(message)
    state = jax_device_state["state"]
    if immutable_data["save_checkpoint_on_completion"]:
      checkpointing.maybe_save_checkpoint(
          checkpoint_manager, state, config, data_iterator, training_graph=training_graph
      )
    if checkpoint_manager is not None:
      checkpointing.wait_until_finished(checkpoint_manager)
    job_completed_gracefully = True
  except exceptions.StopTraining as error:
    prof.deactivate()
    max_logging.log(f"Training stopped: {str(error)}")
    job_completed_gracefully = True
  finally:
    metric_logger_instance.flush_metrics_and_cleanup()
    if job_completed_gracefully:
      record_goodput(recorder, RECORD_JOB_END_TIME)
      samples_count = (python_vars["step"] - immutable_data["start_step"]) * config.global_batch_size_to_train_on
      mllog_utils.run_stop(status="aborted", current_epoch_num=samples_count, step=python_vars["step"])
    train_utils.maybe_cleanup_dcn_throttling(config)
  return state


def run(config, recorder, elastic_retry_cache=None, *, setup=None, setup_fn=None, reporter_factory=None,
        launch_xprof=False, xprof_port=8791, training_graph=None):
  with max_utils.maybe_get_transformer_engine_context(config), profiler.nnx_named_scope_context():
    if setup_fn is not None:
      setup = setup_fn(config, recorder, elastic_retry_cache)
    server = (
        profiler.xprof_server(os.path.join(config.tensorboard_dir, ""), xprof_port, jax.process_index())
        if launch_xprof else contextlib.nullcontext()
    )
    with server:
      return train_loop(
          config, recorder, elastic_retry_cache=elastic_retry_cache, setup=setup, reporter_factory=reporter_factory,
          training_graph=training_graph,
      )


def run_elastic(config, recorder, *, setup=None, setup_fn=None, reinitialize=None, reinitialize_first=False,
                reporter_factory=None, launch_xprof=False, xprof_port=8791, training_graph=None):
  if not config.elastic_enabled:
    return run(config, recorder, setup=setup, setup_fn=setup_fn, reporter_factory=reporter_factory,
               launch_xprof=launch_xprof, xprof_port=xprof_port, training_graph=training_graph)

  retry_cache = elastic_utils.RetryCache()
  retry_cache.snapshotter = snapshotting.create_snapshotter(config)

  def on_elastic_event(scale_up):
    elastic_utils.record_elastic_event_start(recorder, scale_up=scale_up)
    if retry_cache.snapshotter is not None and not scale_up:
      retry_cache.snapshotter = snapshotting.reset_snapshotter(retry_cache.snapshotter)

  def on_slices_ready():
    elastic_utils.record_elastic_wait_end_and_reinit_start(recorder)

  initialized = False

  def initialize_attempt():
    nonlocal initialized
    if (initialized or reinitialize_first) and reinitialize is not None:
      initialized = True
      return reinitialize()
    initialized = True
    return config, recorder

  def attempt():
    attempt_config, attempt_recorder = elastic_utils.get_or_build(retry_cache, "initialize", initialize_attempt)
    return run(attempt_config, attempt_recorder, elastic_retry_cache=retry_cache, setup=setup, setup_fn=setup_fn,
               reporter_factory=reporter_factory, launch_xprof=launch_xprof, xprof_port=xprof_port,
               training_graph=training_graph)

  return elastic_utils.elastic_retry(
      config, callback_fn=on_elastic_event, pre_callback_fn=on_slices_ready, retry_cache=retry_cache,
  )(attempt)()
