"""Expert-parallel token exchange: all-to-all parameters and local permutation of received tokens."""

import enum

import jax
import jax.numpy as jnp

from lilax._backend.kernels.ragged.ragged_sort import a2a_ragged_sort
from lilax._backend.layers.moe_ops.dispatch import sorting


def _truncate_matrix(all_shards_group_sizes: jax.Array, buffer_size: int) -> jax.Array:
  """Truncates the traffic matrix to fit in buffer_size on receiver side.

  When ragged_buffer_factor > 0, the receiver buffer has a fixed capacity
  (buffer_size). Due to routing imbalance, some shards might receive more tokens
  than this capacity. We use a prefix sum to deterministically truncate the
  received tokens on all shards, ensuring we don't write out of bounds.
  """
  cumsum = jnp.cumsum(all_shards_group_sizes, axis=0)
  clamped_cumsum = jnp.minimum(cumsum, buffer_size)
  clamped_cumsum_extended = jnp.concatenate(
      [
          jnp.zeros((1, all_shards_group_sizes.shape[1]), dtype=clamped_cumsum.dtype),
          clamped_cumsum,
      ],
      axis=0,
  )
  return jnp.diff(clamped_cumsum_extended, axis=0)


class ExpertDispatch:
  @staticmethod
  def _maybe_truncate_local_group_size(
      all_shard_local_sizes: jax.Array,
      buffer_size: int,
      ragged_buffer_factor: float,
  ) -> jax.Array:
    if ragged_buffer_factor > 0.0:
      flat_sizes = all_shard_local_sizes.reshape(-1)
      cumsum = jnp.cumsum(flat_sizes)
      clamped_cumsum = jnp.minimum(cumsum, buffer_size)
      clamped_cumsum_extended = jnp.concatenate([jnp.zeros((1,), dtype=clamped_cumsum.dtype), clamped_cumsum])
      truncated_flat_sizes = jnp.diff(clamped_cumsum_extended)
      truncated_all_shard_local_sizes = truncated_flat_sizes.reshape(all_shard_local_sizes.shape)
      return jnp.sum(truncated_all_shard_local_sizes, axis=0)
    else:
      return jnp.sum(all_shard_local_sizes, axis=0)

  @staticmethod
  def local_permute(
      inputs,
      global_group_sizes,
      local_expert_size,
      shard_index,
      is_offset=False,
      global_sorted_experts=None,
      use_custom_sort_vjp=True,
      use_ragged_sort=False,
      ragged_buffer_factor=-1.0,
      use_single_sparsecore=False,
      *,
      truncate_local_group_size,
  ):
    all_shard_local_sizes = jax.lax.dynamic_slice_in_dim(
        global_group_sizes, shard_index * local_expert_size, local_expert_size, axis=1
    )
    local_sizes = all_shard_local_sizes.reshape(-1)
    local_group_size = truncate_local_group_size(
        all_shard_local_sizes, inputs.shape[0], ragged_buffer_factor
    )
    if is_offset:
      divided_assignments = jnp.floor_divide(global_sorted_experts, local_expert_size)
      expert_indices = jnp.where(
          divided_assignments == shard_index, jnp.mod(global_sorted_experts, local_expert_size), local_expert_size
      )
    else:
      base_indices = jnp.mod(jnp.arange(local_sizes.shape[0]), local_expert_size)
      expert_indices = jnp.repeat(base_indices, local_sizes, total_repeat_length=inputs.shape[0])

    sorted_indices = jnp.argsort(expert_indices)
    if use_ragged_sort:
      valid_end = jnp.sum(local_group_size).astype(jnp.int32)
      sorted_inputs = a2a_ragged_sort(inputs, sorted_indices, valid_end, use_single_sparsecore=use_single_sparsecore)
    else:
      sorted_inputs = sorting._sort_activations(inputs, sorted_indices, use_custom_sort_vjp)
    sorted_experts_ids = expert_indices[sorted_indices]
    return sorted_inputs, sorted_indices, local_group_size, sorted_experts_ids

  @staticmethod
  def get_all_to_all_params(
      all_shards_group_sizes,
      shard_id,
      num_expert_parallelism,
      is_batch_sharded=True,
      ragged_buffer_factor=-1.0,
      buffer_size=None,
      is_dispatch=True,
  ):
    class TransformStrategy(enum.Enum):
      INPUT_OFFSET = enum.auto()
      SEND_SIZE = enum.auto()
      OUTPUT_OFFSET = enum.auto()
      RECV_SIZE = enum.auto()

    def transform_array(input_array, shard_id, strategy, is_batch_sharded):
      if is_batch_sharded:
        if strategy == TransformStrategy.INPUT_OFFSET:
          local_array = input_array[shard_id]
          return jnp.concatenate((jnp.array([0]), jnp.cumsum(local_array)[:-1]))
        elif strategy == TransformStrategy.SEND_SIZE:
          return input_array[shard_id]
        elif strategy == TransformStrategy.OUTPUT_OFFSET:
          zero_row = jnp.zeros((1,) + input_array.shape[1:], dtype=input_array.dtype)
          array_with_zeros = jnp.concatenate((zero_row, input_array), axis=0)
          cumulated_array = jnp.cumsum(array_with_zeros, axis=0, dtype=input_array.dtype)
          return cumulated_array[shard_id]
        elif strategy == TransformStrategy.RECV_SIZE:
          return input_array[:, shard_id]
        else:
          raise ValueError(f"Unknown transform array strategy: {strategy}")
      else:
        if strategy == TransformStrategy.INPUT_OFFSET:
          return jnp.zeros(num_expert_parallelism, dtype=input_array.dtype)
        elif strategy == TransformStrategy.SEND_SIZE:
          return jnp.repeat(input_array[shard_id], num_expert_parallelism)
        elif strategy == TransformStrategy.OUTPUT_OFFSET:
          output_offset = jnp.concatenate((jnp.array([0]), jnp.cumsum(input_array[:-1])))[shard_id]
          return jnp.repeat(output_offset, num_expert_parallelism)
        elif strategy == TransformStrategy.RECV_SIZE:
          return input_array
        else:
          raise ValueError(f"Unknown transform array strategy: {strategy}")

    if ragged_buffer_factor > 0.0:
      assert buffer_size is not None
      truncated_all_shards_group_sizes = _truncate_matrix(all_shards_group_sizes, buffer_size)
      if is_dispatch:
        input_offsets = transform_array(all_shards_group_sizes, shard_id, TransformStrategy.INPUT_OFFSET, is_batch_sharded)
        send_sizes = transform_array(truncated_all_shards_group_sizes, shard_id, TransformStrategy.SEND_SIZE, is_batch_sharded)
        output_offsets = transform_array(
            truncated_all_shards_group_sizes, shard_id, TransformStrategy.OUTPUT_OFFSET, is_batch_sharded
        )
        recv_sizes = transform_array(truncated_all_shards_group_sizes, shard_id, TransformStrategy.RECV_SIZE, is_batch_sharded)
      else:
        transposed_all_shards = jnp.transpose(all_shards_group_sizes)
        transposed_truncated = jnp.transpose(truncated_all_shards_group_sizes)
        input_offsets = transform_array(transposed_truncated, shard_id, TransformStrategy.INPUT_OFFSET, is_batch_sharded)
        send_sizes = transform_array(transposed_truncated, shard_id, TransformStrategy.SEND_SIZE, is_batch_sharded)
        output_offsets = transform_array(transposed_all_shards, shard_id, TransformStrategy.OUTPUT_OFFSET, is_batch_sharded)
        recv_sizes = transform_array(transposed_truncated, shard_id, TransformStrategy.RECV_SIZE, is_batch_sharded)
    else:
      matrix = all_shards_group_sizes if is_dispatch else jnp.transpose(all_shards_group_sizes)
      input_offsets = transform_array(matrix, shard_id, TransformStrategy.INPUT_OFFSET, is_batch_sharded)
      send_sizes = transform_array(matrix, shard_id, TransformStrategy.SEND_SIZE, is_batch_sharded)
      output_offsets = transform_array(matrix, shard_id, TransformStrategy.OUTPUT_OFFSET, is_batch_sharded)
      recv_sizes = transform_array(matrix, shard_id, TransformStrategy.RECV_SIZE, is_batch_sharded)
    return input_offsets, send_sizes, output_offsets, recv_sizes
