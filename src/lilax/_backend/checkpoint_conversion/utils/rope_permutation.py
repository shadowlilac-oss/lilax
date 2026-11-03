"""RoPE weight permutations between Hugging Face and MaxText layouts."""

import jax
import numpy as np


def permute_to_match_maxtext_rope(arr):
  """Permutes the Huggingface Rope to match the MaxText logic."""
  assert arr.shape[-1] % 2 == 0, "The last dimension for rope has to be even."
  evens, odds = np.split(arr, 2, axis=arr.ndim - 1)  # pylint: disable=W0632
  x = np.empty_like(arr)
  x[..., ::2] = evens
  x[..., 1::2] = odds
  return x


def unpermute_from_match_maxtext_rope(arr, model_size):
  """
  Function to get the RoPE values in correct ordering
  """
  if model_size[:8] != "llama3.1":
    return arr
  evens = arr[..., ::2]
  odds = arr[..., 1::2]
  return jax.numpy.concatenate((evens, odds), axis=arr.ndim - 1)
