"""Routing state shared by token dispatch, expert computation and combine."""

from typing import Any, Optional

from flax import struct
import jax


@struct.dataclass
class RouteMetadata:
  """EP communication state needed to undo the forward all-to-all after expert computation."""

  # Index of this device's EP shard.
  expert_shard_id: int
  # Permutation of [tokens received by this expert shard], sorted by local expert ID.
  local_sorted_indices: Optional[jax.Array]
  # Shape [num_ep].  Aggregates group_sizes per EP shard; tracks how many local tokens are routed to each EP shard.
  reshaped_group_sizes: Optional[jax.Array]
  # Shape [num_ep, num_ep]. all_gather of reshaped_group_sizes across EP shards.
  # [i, j] = number of tokens from batch shard i sent to expert shard j.
  all_shards_group_sizes: Optional[jax.Array]


@struct.dataclass
class RouteOutput:
  """Holds state of routing output"""

  # Shape [num experts], tracks number of local tokens routed to every expert.
  group_sizes: jax.Array
  # Indices of experts chosen for each token.
  selected_experts: jax.Array
  # Tokens sorted by experts they are routed to.
  sorted_selected_experts: jax.Array
  # Weights for each of the selected experts.
  weights: jax.Array
  # Auxiliary loss for token distribution among experts.
  lb_loss: Optional[jax.Array]
  # Dynamic bias updates for loss-free load balancing, used only for Deepseek models
  bias_updates: Optional[jax.Array]
  # Shape [local experts], tracks number of local tokens routed to every local expert.
  local_group_sizes: Optional[jax.Array] = None
  has_overflow: bool | jax.Array = False
  # log_required_ragged_buffer_factor probe: minimum ragged_buffer_factor that would have avoided drops, max over
  # the mesh (float32 scalar). None when the probe is off.
  required_rbf: Optional[jax.Array] = None
  # Scalar max over expert shards (and, after the mesh reduction, over EP groups) of the
  # tokens routed to a shard divided by the perfectly balanced load. Only set when
  # moe_log_max_load_ratio=True on the ring-of-experts ragged path.
  max_load_ratio: Optional[jax.Array] = None
  # Inverse of sorted_selected_experts on the ring-of-experts ragged path, so the unsort backward does not re-sort to
  # get it. Only set when moe_routing_maps is not "remat".
  topk_argsort_indices: Optional[jax.Array] = None
  # moe_tc_ragged_sort: TcRouting shared by the TensorCore ragged sort and unsort of this chunk.
  tc_routing: Optional[Any] = None
