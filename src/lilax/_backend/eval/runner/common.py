"""Shared helpers for evaluation runners."""

import os

from lilax._backend.eval.reporting.gcs_reporter import upload_results
from lilax._backend.eval.runner.server_manager import VllmServerManager


def resolve_token(cfg: dict, hf_token: str | None) -> str | None:
  return hf_token or os.environ.get("HF_TOKEN") or None


def build_server_manager(cfg: dict, token: str | None, *, enable_chat_api: bool = False) -> VllmServerManager:
  hf_path = cfg["hf_path"]
  model_name = cfg["model_name"]
  checkpoint_path = cfg.get("checkpoint_path")
  hf_mode = cfg.get("hf_mode", False)
  use_maxtext_adapter = bool(checkpoint_path) and not hf_mode
  tensor_parallel_size = int(cfg.get("tensor_parallel_size", 4))
  max_model_len = int(cfg["max_model_len"])
  server_host = cfg.get("server_host", "localhost")
  server_port = int(cfg.get("server_port", 8000))
  max_num_batched_tokens = cfg.get("max_num_batched_tokens")
  if max_num_batched_tokens is not None:
    max_num_batched_tokens = int(max_num_batched_tokens)
  max_num_seqs = cfg.get("max_num_seqs")
  if max_num_seqs is not None:
    max_num_seqs = int(max_num_seqs)
  expert_parallel_size = int(cfg.get("expert_parallel_size") or 1)
  data_parallel_size = int(cfg.get("data_parallel_size") or 1)
  hbm_memory_utilization = float(cfg.get("hbm_memory_utilization") or 0.3)
  concurrency = cfg.get("concurrency")
  if concurrency is not None:
    concurrency = int(concurrency)
  server_env = {"HF_TOKEN": token} if token else None
  return VllmServerManager(
      model_path=hf_path,
      checkpoint_path=checkpoint_path if use_maxtext_adapter else None,
      maxtext_model_name=model_name if use_maxtext_adapter else None,
      host=server_host,
      port=server_port,
      tensor_parallel_size=tensor_parallel_size,
      expert_parallel_size=expert_parallel_size,
      data_parallel_size=data_parallel_size,
      max_model_len=max_model_len,
      max_num_batched_tokens=max_num_batched_tokens,
      max_num_seqs=max_num_seqs,
      hbm_memory_utilization=hbm_memory_utilization,
      enable_chat_api=enable_chat_api,
      concurrency=concurrency,
      env=server_env,
  )


def maybe_upload_to_gcs(output: dict, gcs_results_path: str | None) -> None:
  if gcs_results_path:
    upload_results(output["local_path"], gcs_results_path)
