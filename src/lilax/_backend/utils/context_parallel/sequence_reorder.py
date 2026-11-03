"""Context parallel causal load balancing reorder of sequence batches."""

from functools import partial

import jax
import jax.numpy as jnp

from lilax._backend.common.common_types import ReorderStrategy


@partial(jax.jit, static_argnames=("cp_size", "seq_dim", "to_contiguous"))
def reorder_sequence(tensor, cp_size: int, seq_dim: int = 1, to_contiguous: bool = False):
  """Reorder a sequence for causal load balancing or restore contiguous order."""
  if tensor is None:
    return tensor
  seq_len = tensor.shape[seq_dim]
  group_size = seq_len // (2 * cp_size)
  if cp_size % 2 != 0:
    raise ValueError(f"{cp_size=} must be a multiple of 2.")
  if seq_len % (cp_size * 2) != 0:
    raise ValueError(f"{tensor.shape=} is not a multiple of {cp_size*2=}")
  seq_dim = seq_dim % tensor.ndim
  ori_tensor_shape = tensor.shape
  reshaped = tensor.reshape(*ori_tensor_shape[:seq_dim], 2 * cp_size, group_size, *ori_tensor_shape[seq_dim + 1 :])
  swapped = jnp.swapaxes(reshaped, 0, seq_dim)
  if not to_contiguous:
    first_half, second_half = jnp.split(swapped, 2, axis=0)
    second_half_reversed = second_half[::-1, ...]
    stacked = jnp.stack([first_half, second_half_reversed], axis=1)
    permuted = stacked.reshape(2 * cp_size, *swapped.shape[1:])
  else:
    reshaped_swapped = swapped.reshape(cp_size, 2, *swapped.shape[1:])
    first_half = reshaped_swapped[:, 0, ...]
    second_half_reversed = reshaped_swapped[:, 1, ...]
    second_half = second_half_reversed[::-1, ...]
    permuted = jnp.concatenate([first_half, second_half], axis=0)
  reordered = jnp.swapaxes(permuted, 0, seq_dim)
  return reordered.reshape(ori_tensor_shape)


@partial(jax.jit, static_argnums=(1, 2, 3))
def reorder_causal_load_balanced(batch, cp_size, reorder_strategy, hardware="tpu"):
  """Reorder batch sequences using the appropriate hardware backend."""
  reorder_keys = {
      "inputs", "targets", "inputs_position", "targets_position", "inputs_segmentation",
      "targets_segmentation", "corruption_mask", "targets_loss_mask",
  }
  if hardware in ("gpu", "gpu_multiprocess"):
    from transformer_engine.jax.attention import ReorderStrategy as TE_ReorderStrategy
    from transformer_engine.jax.attention import reorder_causal_load_balancing

    reorder_strategy_map = {
        ReorderStrategy.DUAL_CHUNK_SWAP: TE_ReorderStrategy.DualChunkSwap,
        ReorderStrategy.STRIPED: TE_ReorderStrategy.Striped,
    }
    return {
        key: reorder_causal_load_balancing(value, reorder_strategy_map[reorder_strategy], cp_size=cp_size, seq_dim=1)
        if key in reorder_keys else value
        for key, value in batch.items()
    }
  if reorder_strategy == ReorderStrategy.STRIPED:
    raise ValueError(
        f"STRIPED reorder strategy requires Transformer Engine and is only supported on GPU, got hardware={hardware!r}."
    )
  return {key: reorder_sequence(value, cp_size=cp_size) if key in reorder_keys else value for key, value in batch.items()}


def reordered_cp_size(config, mesh) -> int:
  """The cp_size the input pipeline reordered the batch with; 1 if it did not."""
  if mesh is None or not getattr(config, "context_parallel_load_balance", False):
    return 1
  return mesh.shape.get(config.context_sharding, 1)
