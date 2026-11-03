"""Evaluate generated checkpoint conversion code."""

import bz2
import lzma
from pathlib import Path
import zlib

from lilax._backend.experimental.agent.base import BaseAgent


def compressed_size(data: bytes, method: str = "gzip") -> int:
  if method == "gzip":
    return len(zlib.compress(data))
  if method == "bz2":
    return len(bz2.compress(data))
  if method == "lzma":
    return len(lzma.compress(data))
  raise ValueError(f"Unknown compression method: {method}")


def estimate_kolmogorov(filepath: str) -> dict:
  with open(filepath, "rb") as file:
    data = file.read()
  results = {
      "original_size": len(data),
      "gzip_size": compressed_size(data, "gzip"),
      "bz2_size": compressed_size(data, "bz2"),
      "lzma_size": compressed_size(data, "lzma"),
  }
  results["approx_k_complexity"] = min(results["gzip_size"], results["bz2_size"], results["lzma_size"])
  return results


def evaluate_conversion(ground_truth_path: str, generated_path: str, prompt_path: str | None = None) -> str:
  agent = BaseAgent()
  with open(ground_truth_path, "rt", encoding="utf8") as file:
    ground_truth = file.read()
  with open(generated_path, "rt", encoding="utf8") as file:
    generated = file.read()
  if prompt_path is None:
    prompt_path = Path(__file__).parent / "prompts" / "rate_outputs.txt"
  with open(prompt_path, "rt", encoding="utf8") as file:
    prompt = file.read().format(ground_truth=ground_truth, dsl_chain=generated)
  return agent.generate_text(prompt)
