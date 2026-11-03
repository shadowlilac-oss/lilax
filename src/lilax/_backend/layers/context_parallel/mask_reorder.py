"""Context parallel causal load balancing reorder of NumPy attention masks."""

import numpy as np


def reorder_mask_load_balancing(tensor, cp_size: int, seq_dim: int):
  """Reorder a NumPy mask for causal load balancing."""
  seq_len = tensor.shape[seq_dim]
  group_size = seq_len // (2 * cp_size)
  if cp_size % 2 != 0:
    raise ValueError(f"{cp_size=} must be a multiple of 2.")
  if seq_len % (cp_size * 2) != 0:
    raise ValueError(f"{tensor.shape=} is not a multiple of {cp_size*2=}")
  ori_tensor_shape = tensor.shape
  reshaped = tensor.reshape(*ori_tensor_shape[:seq_dim], 2 * cp_size, group_size, *ori_tensor_shape[seq_dim + 1 :])
  first_half = np.arange(cp_size)
  second_half = np.arange(2 * cp_size - 1, cp_size - 1, -1)
  src_indices = np.stack([first_half, second_half], axis=1).reshape(-1)
  reordered = np.take(reshaped, src_indices, axis=seq_dim)
  return reordered.reshape(ori_tensor_shape)
