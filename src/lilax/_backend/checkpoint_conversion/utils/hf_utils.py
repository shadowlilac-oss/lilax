"""Hugging Face checkpoint comparison utilities."""

import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate


def convert_jax_weight_to_torch(weight: "jax.Array", dtype: None | str = None) -> torch.Tensor:
  expected_dtype = str(weight.dtype) if dtype is None else dtype
  expected_shape = weight.shape
  weight = multihost_utils.process_allgather(weight)
  weight = np.array(weight, dtype="float32")
  return torch.from_numpy(weight).to(getattr(torch, expected_dtype)).reshape(expected_shape)


def check_arrays_match(arrayA, arrayB, atol=0.01, rtol=1e-5):
  is_A_torch = isinstance(arrayA, torch.Tensor)
  is_B_torch = isinstance(arrayB, torch.Tensor)
  if is_A_torch and not is_B_torch:
    arrayB = convert_jax_weight_to_torch(arrayB)
  elif is_B_torch and not is_A_torch:
    arrayA = convert_jax_weight_to_torch(arrayA)
  if isinstance(arrayA, torch.Tensor):
    if not torch.allclose(arrayA, arrayB, rtol=rtol, atol=atol):
      abs_diff = torch.abs(arrayA - arrayB)
      mismatch_indices = abs_diff > (atol + rtol * torch.abs(arrayB))
      num_mismatched_elements = mismatch_indices.sum().item()
      max_abs_diff_val = torch.max(abs_diff).item()
      print(f"Arrays do not match within rtol={rtol}, atol={atol}.")
      print(f"Maximum absolute difference: {max_abs_diff_val:.6f}")
      print(f"Number of mismatched elements in {arrayB.shape}: {num_mismatched_elements}")
      if num_mismatched_elements > 0:
        print("Examples of mismatched elements (ArrayA vs ArrayB, limited to first 5):")
        actual_limit = min(num_mismatched_elements, 5)
        mismatched_A_samples = arrayA[mismatch_indices].flatten()[:actual_limit]
        mismatched_B_samples = arrayB[mismatch_indices].flatten()[:actual_limit]
        for sample_a, sample_b in zip(mismatched_A_samples, mismatched_B_samples):
          print(f"  A: {sample_a.item():.6f}, B: {sample_b.item():.6f}, Diff: {(sample_a - sample_b).item():.6f}")
      return False
  elif not jnp.allclose(arrayA, arrayB, rtol=rtol, atol=atol):
    abs_diff = jnp.abs(arrayA - arrayB)
    mismatch_indices = abs_diff > (atol + rtol * jnp.abs(arrayB))
    num_mismatched_elements = jnp.sum(mismatch_indices).item()
    max_abs_diff_val = jnp.max(abs_diff).item()
    print(f"JAX arrays do not match within rtol={rtol}, atol={atol}.")
    print(f"Maximum absolute difference: {max_abs_diff_val:.6f}")
    print(f"Number of mismatched elements in {arrayB.shape}: {num_mismatched_elements}")
    return False
  return True


def check_predicted_tokens_match(logits_a, logits_b, tolerance=0.1):
  if not 0.0 <= tolerance <= 1.0:
    raise ValueError("Tolerance must be between 0.0 and 1.0")
  disagreement_rate = get_logits_comparison_metrics(logits_a, logits_b)["disagreement_top1"]
  if disagreement_rate > tolerance:
    raise AssertionError(
        f"Token prediction mismatch: {disagreement_rate:.1%} of tokens disagree "
        f"(exceeds tolerance of {tolerance:.1%})"
    )


def get_logits_comparison_metrics(logitsA, logitsB):
  if isinstance(logitsA, jax.Array):
    logitsA = convert_jax_weight_to_torch(logitsA)
  if isinstance(logitsA, np.ndarray):
    logitsA = torch.tensor(logitsA)
  if isinstance(logitsB, jax.Array):
    logitsB = convert_jax_weight_to_torch(logitsB)
  if isinstance(logitsB, np.ndarray):
    logitsB = torch.tensor(logitsB)
  probs_A = F.softmax(logitsA, dim=-1)
  probs_B = F.softmax(logitsB, dim=-1)
  kl_div = F.kl_div(torch.log(probs_B), probs_A, reduction="none", log_target=False)
  max_kl_div = torch.max(kl_div.sum(dim=-1))
  max_abs_diff = torch.abs(probs_A - probs_B).max()
  sorted_logits_A = torch.argsort(logitsA, dim=1)
  sorted_logits_B = torch.argsort(logitsB, dim=1)
  ranking_A_top5, ranking_B_top5 = sorted_logits_A[:, -5:], sorted_logits_B[:, -5:]
  disagreement_top5 = torch.mean(((torch.abs(ranking_B_top5 - ranking_A_top5) > 0).sum(dim=1) > 0).float())
  ranking_A_top1, ranking_B_top1 = sorted_logits_A[:, -1:], sorted_logits_B[:, -1:]
  disagreement_top1 = torch.mean(((torch.abs(ranking_B_top1 - ranking_A_top1) > 0).sum(dim=1) > 0).float())
  metrics = {
      "max_kl_div": float(max_kl_div),
      "abs_diff": float(max_abs_diff),
      "disagreement_top5": float(disagreement_top5),
      "disagreement_top1": float(disagreement_top1),
  }
  print(tabulate([[key, value] for key, value in metrics.items()], headers=["Metric", "Value"], tablefmt="orgtbl"))
  return metrics
