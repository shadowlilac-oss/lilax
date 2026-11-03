"""Functions for gradient accumulation."""

import functools

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from flax import nnx

from lilax._backend.common.common_types import ShardMode
from lilax._backend.layers.moe import load_balance_updates_from_counts
from lilax._backend.utils.xla_options import loop_unrolling
from lilax._backend.utils.sharding import maybe_shard_with_name


def gradient_accumulation_loss_and_grad(_loss_fn, config, model, params, params_shardings, data, dropout_rng):
  def _maybe_shard_with_name(inputs, sharding_names):
    return maybe_shard_with_name(inputs, sharding_names, config.shard_mode, debug_sharding=config.debug_sharding)

  is_nnx = isinstance(model, nnx.Module)
  param_mesh = jax.tree.leaves(params_shardings)[0].mesh
  data_parallel_active = config.shard_mode == ShardMode.EXPLICIT and param_mesh.shape.get("data", 1) > 1
  if data_parallel_active:
    ga_params_shardings = params_shardings
    grad_shardings = params_shardings
  else:
    ga_params_shardings = grad_shardings = params_shardings

  if is_nnx:
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

  if config.shard_optimizer_over_data:
    def convert_to_bf16(param):
      if param.dtype == jnp.float32:
        return param.astype(jnp.bfloat16)
      return param

    ga_params = jax.tree_util.tree_map(convert_to_bf16, params)
  else:
    ga_params = params

  ga_params = jax.tree.map(_maybe_shard_with_name, ga_params, ga_params_shardings)
  use_tunix_ga = getattr(config, "use_tunix_gradient_accumulation", False)
  token_capacity = getattr(config, "global_batch_size_to_train_on", 0) * getattr(config, "max_target_length", 0)
  loss_scale = 1.0 / float(token_capacity) if (token_capacity > 0 and not use_tunix_ga) else 1.0

  @functools.wraps(_loss_fn)
  def _scaled_loss_fn(*args, **kwargs):
    loss, aux = _loss_fn(*args, **kwargs)
    return loss * loss_scale, aux

  if is_nnx:
    grad_func = nnx.value_and_grad(_scaled_loss_fn, argnums=0, has_aux=True)
  else:
    grad_func = jax.value_and_grad(_scaled_loss_fn, argnums=4, has_aux=True)

  def accumulate_gradient(acc_grad_and_loss, data):
    ga_params = acc_grad_and_loss["ga_params"]
    loop_unroll_ctx = loop_unrolling.model_loop_unroll_context(config)
    if is_nnx:
      local_model = nnx.merge(graphdef, ga_params, acc_grad_and_loss["rest_state"], copy=True)
      with loop_unroll_ctx:
        (_, aux), cur_batch_gradient = grad_func(local_model, config, data, None, None, is_train=True)
      _, _, next_rest_state = nnx.split(local_model, nnx.Param, ...)
      acc_grad_and_loss["rest_state"] = next_rest_state
    else:
      rng = (
          jax.random.fold_in(dropout_rng, acc_grad_and_loss["total_weights"].astype(jnp.int32))
          if dropout_rng is not None
          else None
      )
      with loop_unroll_ctx:
        (_, aux), cur_batch_gradient = grad_func(model, config, data, rng, ga_params, is_train=True)
    acc_grad_and_loss["loss"] += aux["xent_sum"] + aux.get("dpo_loss", 0.0)
    acc_grad_and_loss["moe_lb_loss"] += aux["moe_lb_loss"]
    acc_grad_and_loss["indexer_loss"] += aux["indexer_loss"]
    acc_grad_and_loss["mtp_loss"] += aux["mtp_loss"]
    acc_grad_and_loss["grad"] = jax.tree_util.tree_map(lambda x, y: x + y, cur_batch_gradient, acc_grad_and_loss["grad"])
    acc_grad_and_loss["total_weights"] += aux["total_weights"]
    return acc_grad_and_loss, aux

  def reshape_to_microbatch_accumulations(batch_arr):
    num_microbatches = config.gradient_accumulation_steps
    microbatch_shape = (batch_arr.shape[0] // num_microbatches, num_microbatches) + batch_arr.shape[1:]
    reshaped_batch_arr = jnp.reshape(batch_arr, microbatch_shape)
    return jnp.swapaxes(reshaped_batch_arr, 0, 1)

  data = jax.tree_util.tree_map(reshape_to_microbatch_accumulations, data)
  init_grad = jax.tree_util.tree_map(jnp.zeros_like, ga_params)
  init_grad = jax.tree.map(_maybe_shard_with_name, init_grad, grad_shardings)
  init_grad_and_loss = {
      "loss": 0.0,
      "grad": init_grad,
      "total_weights": 0,
      "moe_lb_loss": 0.0,
      "indexer_loss": 0.0,
      "mtp_loss": 0.0,
      "ga_params": ga_params,
  }
  if is_nnx:
    init_grad_and_loss["rest_state"] = rest

  grad_and_loss, aux = jax.lax.scan(
      accumulate_gradient, init_grad_and_loss, data, length=config.gradient_accumulation_steps
  )
  has_weights = grad_and_loss["total_weights"] > 0
  denominator = jnp.maximum(grad_and_loss["total_weights"], 1)
  loss = (
      grad_and_loss["loss"] / denominator
      + grad_and_loss["moe_lb_loss"] / config.gradient_accumulation_steps
      + grad_and_loss["indexer_loss"] / config.gradient_accumulation_steps
      + grad_and_loss["mtp_loss"] / config.gradient_accumulation_steps
  )
  loss = jnp.where(has_weights, loss, 0.0)
  raw_grads = grad_and_loss["grad"]
  if data_parallel_active:
    unreduced_shardings = jax.tree.map(update_sharding_for_unreduced, params_shardings)
    raw_grads = jax.tree.map(_maybe_shard_with_name, raw_grads, unreduced_shardings)
  raw_grads = jax.tree.map(_maybe_shard_with_name, raw_grads, params_shardings)
  divisor = config.gradient_accumulation_steps if use_tunix_ga else denominator * loss_scale
  raw_grads = jax.tree_util.tree_map(
      lambda arr: jnp.where(has_weights, (arr / divisor).astype(arr.dtype), jnp.zeros_like(arr)),
      raw_grads,
  )
  scanned_aux = aux
  aux = jax.tree.map(lambda x: jnp.sum(x, axis=0), aux)
  for key in ("moe_lb_loss", "indexer_loss", "mtp_loss"):
    if key in aux:
      aux[key] = aux[key] / config.gradient_accumulation_steps
  if getattr(config, "routed_bias", False) and getattr(config, "routed_bias_update_rate", 0.0) > 0.0:
    def finalize_bias(counts):
      if isinstance(counts, jax.Array) and jnp.issubdtype(counts.dtype, jnp.integer):
        return load_balance_updates_from_counts(counts, config.num_experts, config.routed_bias_update_rate)
      return counts

    for key in ("moe_bias_updates", "mtp_moe_bias_updates"):
      if aux.get(key) is not None:
        aux[key] = jax.tree.map(finalize_bias, aux[key])
    if "intermediate_outputs" in aux:
      aux["intermediate_outputs"] = jax.tree_util.tree_map_with_path(
          lambda path, val: (
              finalize_bias(val) if path and getattr(path[-1], "key", None) == "moe_bias_updates" else val
          ),
          aux["intermediate_outputs"],
      )
  if "te_moe_capacity_overflow" in scanned_aux:
    aux["te_moe_capacity_overflow"] = jnp.any(scanned_aux["te_moe_capacity_overflow"], axis=0)
    aux["te_moe_max_total_recv_tokens"] = jnp.max(scanned_aux["te_moe_max_total_recv_tokens"], axis=0)
    aux["te_moe_recv_capacity_per_rank"] = jnp.min(scanned_aux["te_moe_recv_capacity_per_rank"], axis=0)
  if "moe_required_rbf" in scanned_aux:
    aux["moe_required_rbf"] = jnp.max(scanned_aux["moe_required_rbf"], axis=0)
  if "moe_max_load_ratio" in scanned_aux:
    aux["moe_max_load_ratio"] = jnp.max(scanned_aux["moe_max_load_ratio"], axis=0)
    aux["moe_max_load_ratio_mean"] = jnp.mean(scanned_aux["moe_max_load_ratio_mean"], axis=0)

  if is_nnx:
    nnx.update(model, grad_and_loss["rest_state"])
  return loss, aux, raw_grads


def update_sharding_for_reduced(sharding: NamedSharding) -> NamedSharding:
  return sharding.update(spec=sharding.spec.update(reduced={"data"}))


def update_sharding_for_unreduced(sharding: NamedSharding) -> NamedSharding:
  return sharding.update(spec=sharding.spec.update(unreduced={"data"}))
