"""Shard-mapped dropless MoE forward and backward for the DeepSeek BF16 batch-split schedule."""

import functools

import jax

from lilax._backend.models.deepseek_batchsplit.moe.stages import route_compute_unroute, route_compute_unroute_bwd


def moe(
    xs,
    weights,
    *,
    mesh,
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    n_routing_groups,
    topk_routing_group,
    top_k_in_group,
    expert_axis_name,
    num_expert_shards,
    capacity_factor,
    use_gather_mosaic_kernel,
    config,
    normalization_layer_epsilon,
    dtype,
    activation_pspec,
):
  """Performs dropless MoE with tensor/expert parallelism."""
  return jax.shard_map(
      functools.partial(
          route_compute_unroute,
          num_experts=num_experts,
          num_experts_per_tok=num_experts_per_tok,
          routed_scaling_factor=routed_scaling_factor,
          n_routing_groups=n_routing_groups,
          topk_routing_group=topk_routing_group,
          top_k_in_group=top_k_in_group,
          expert_axis_name=expert_axis_name,
          num_expert_shards=num_expert_shards,
          capacity_factor=capacity_factor,
          use_gather_mosaic_kernel=use_gather_mosaic_kernel,
          normalization_layer_epsilon=normalization_layer_epsilon,
          dtype=dtype,
          config=config,
      ),
      mesh=mesh,
      in_specs=(
          [activation_pspec] * config.batch_split_factor,
          (
              jax.sharding.PartitionSpec(None),
              (
                  jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                  jax.sharding.PartitionSpec(None),
              ),
              (
                  jax.sharding.PartitionSpec(expert_axis_name, None, None, reduced={"data", "fsdp"}),
                  jax.sharding.PartitionSpec(expert_axis_name, None, None, reduced={"data", "fsdp"}),
                  jax.sharding.PartitionSpec(expert_axis_name, None, None, reduced={"data", "fsdp"}),
              ),
              (
                  jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                  jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                  jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
              ),
          ),
      ),
      out_specs=(
          [
              activation_pspec,
              (
                  activation_pspec,
                  jax.sharding.PartitionSpec(*activation_pspec[:-1]),
                  jax.sharding.PartitionSpec(*activation_pspec[:-1]),
                  jax.sharding.PartitionSpec(activation_pspec[0], None),
              ),
          ],
          {
              "mlpwi_0": [jax.sharding.PartitionSpec(*activation_pspec[:-1])] * config.batch_split_factor,
              "mlpwi_1": [jax.sharding.PartitionSpec(*activation_pspec[:-1])] * config.batch_split_factor,
          },
      ),
      check_vma=True,
  )([x.astype(config.dtype) for x in xs], weights)


def moe_bwd(
    residuals,
    outputs_grad,
    weights,
    *,
    mesh,
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    n_routing_groups,
    topk_routing_group,
    top_k_in_group,
    expert_axis_name,
    num_expert_shards,
    use_gather_mosaic_kernel,
    config,
    normalization_layer_epsilon,
    dtype,
    activation_pspec,
):
  """Performs the backward pass for the moe function."""
  return jax.shard_map(
      functools.partial(
          route_compute_unroute_bwd,
          num_experts=num_experts,
          num_experts_per_tok=num_experts_per_tok,
          routed_scaling_factor=routed_scaling_factor,
          n_routing_groups=n_routing_groups,
          topk_routing_group=topk_routing_group,
          top_k_in_group=top_k_in_group,
          expert_axis_name=expert_axis_name,
          num_expert_shards=num_expert_shards,
          use_gather_mosaic_kernel=use_gather_mosaic_kernel,
          normalization_layer_epsilon=normalization_layer_epsilon,
          dtype=dtype,
          config=config,
      ),
      mesh=mesh,
      in_specs=(
          {
              "mla_out": [activation_pspec] * config.batch_split_factor,
              "mlpwi_0": [jax.sharding.PartitionSpec(*activation_pspec[:-1])] * config.batch_split_factor,
              "mlpwi_1": [jax.sharding.PartitionSpec(*activation_pspec[:-1])] * config.batch_split_factor,
          },
          [
              activation_pspec,
              (
                  activation_pspec,
                  jax.sharding.PartitionSpec(*activation_pspec[:-1]),
                  jax.sharding.PartitionSpec(*activation_pspec[:-1]),
                  jax.sharding.PartitionSpec(activation_pspec[0], None),
              ),
          ],
          (
              jax.sharding.PartitionSpec(None),
              (
                  jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                  jax.sharding.PartitionSpec(None),
              ),
              (
                  jax.sharding.PartitionSpec(expert_axis_name, None, None, reduced={"data", "fsdp"}),
                  jax.sharding.PartitionSpec(expert_axis_name, None, None, reduced={"data", "fsdp"}),
                  jax.sharding.PartitionSpec(expert_axis_name, None, None, reduced={"data", "fsdp"}),
              ),
              (
                  jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                  jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                  jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
              ),
          ),
      ),
      out_specs=(
          [activation_pspec] * config.batch_split_factor,
          (
              jax.sharding.PartitionSpec(None),
              (
                  jax.sharding.PartitionSpec(None, None, unreduced={"data", "fsdp", "expert"}),
                  jax.sharding.PartitionSpec(None),
              ),
              (
                  jax.sharding.PartitionSpec(expert_axis_name, None, None, unreduced={"data", "fsdp"}),
                  jax.sharding.PartitionSpec(expert_axis_name, None, None, unreduced={"data", "fsdp"}),
                  jax.sharding.PartitionSpec(expert_axis_name, None, None, unreduced={"data", "fsdp"}),
              ),
              (
                  jax.sharding.PartitionSpec(None, None, unreduced={"data", "fsdp", "expert"}),
                  jax.sharding.PartitionSpec(None, None, unreduced={"data", "fsdp", "expert"}),
                  jax.sharding.PartitionSpec(None, None, unreduced={"data", "fsdp", "expert"}),
              ),
          ),
      ),
      check_vma=True,
  )(residuals, outputs_grad, weights)
