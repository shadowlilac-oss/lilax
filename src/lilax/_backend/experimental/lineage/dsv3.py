"""High level DeepSeekV3 model running dense layers then sparse layers."""

from collections.abc import Mapping
from typing import Any

import jax
import jaxtyping as jt
import typeguard

from lilax._backend.experimental.lineage import dsv3_dense_layers
from lilax._backend.experimental.lineage import dsv3_expert_shuffle
from lilax._backend.experimental.lineage import dsv3_sparse_layers
from lilax._backend.experimental.lineage import dsv3_types
from lilax._backend.experimental.lineage import quantization
from lilax._backend.experimental.lineage.xla_transforms import max_async_overlap_transform as max_async_overlap_transform_module


@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3(
    x: jt.Num[jax.Array, "B T D"],
    w: dsv3_types.DSv3WeightsPytree,
    yarn_freqs: tuple[jt.Num[jax.Array, "B T 1 R"], jt.Num[jax.Array, "B T 1 R"]],
    splash_kernel: Any,
    *,
    num_experts: int,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    n_routing_groups: int,
    topk_routing_group: int,
    topk_in_group: int,
    qk_head_dim: int,
    rope_head_dim: int,
    num_query_heads: int,
    mscale: float,
    kv_lora_rank: int,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    rope_factor: int,
    mesh: jax.sharding.Mesh,
    norm_fn: dsv3_dense_layers.NormFn,
    rope_fn: dsv3_dense_layers.RopeFn,
    gmm_fn: dsv3_sparse_layers.GmmFn,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    expert_axis_name: str = "expert",
    capacity_factor: float = 2.0,
    router_dtype: jax.typing.DTypeLike | None = None,
    segment_ids: jt.Num[jax.Array, "B T"] | None = None,
    max_async_overlap_transform: bool = False,
    ragged_buffer_factor: float,
    quant: quantization.QuantConfig | None = None,
    expert_permutations: jt.Int[jax.Array, "L E"] | None = None,
) -> tuple[jt.Num[jax.Array, "B T D"], dsv3_types.DSv3RouterAux[jax.Array]]:
  """Executes dense and sparse layers, returning output tokens and local router statistics."""
  if max_async_overlap_transform:
    max_async_overlap_transform_module.register_transform()

  if expert_permutations is not None:
    num_sparse_layers = jax.tree.leaves(w.sparse)[0].shape[0]
    dsv3_expert_shuffle.validate_expert_permutations(
        expert_permutations, num_experts=num_experts, num_layers=num_sparse_layers
    )
    expert_permutations = expert_permutations[:num_sparse_layers]

  x = dsv3_dense_layers.dsv3_dense_layers(
      x, w.dense, yarn_freqs, splash_kernel,
      qk_head_dim=qk_head_dim, rope_head_dim=rope_head_dim, num_query_heads=num_query_heads,
      mscale=mscale, kv_lora_rank=kv_lora_rank, max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings, rope_factor=rope_factor,
      mesh=mesh, norm_fn=norm_fn, rope_fn=rope_fn, axis_mapping=axis_mapping, segment_ids=segment_ids, quant=quant,
  )
  x, aux = dsv3_sparse_layers.dsv3_sparse_layers(
      x, w.sparse, yarn_freqs, splash_kernel,
      num_experts=num_experts, num_experts_per_tok=num_experts_per_tok, routed_scaling_factor=routed_scaling_factor,
      n_routing_groups=n_routing_groups, topk_routing_group=topk_routing_group, topk_in_group=topk_in_group,
      qk_head_dim=qk_head_dim, rope_head_dim=rope_head_dim, num_query_heads=num_query_heads,
      mscale=mscale, kv_lora_rank=kv_lora_rank, max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings, rope_factor=rope_factor,
      mesh=mesh, norm_fn=norm_fn, rope_fn=rope_fn, gmm_fn=gmm_fn, axis_mapping=axis_mapping,
      expert_axis_name=expert_axis_name, capacity_factor=capacity_factor, router_dtype=router_dtype,
      segment_ids=segment_ids, ragged_buffer_factor=ragged_buffer_factor, quant=quant,
      expert_permutations=expert_permutations,
  )
  return x, aux
