"""Custom dataset evaluation runner."""

import logging
import time

import jax
from jax.experimental import multihost_utils
from transformers import AutoTokenizer
import yaml

from lilax._backend.eval.datasets.registry import get_dataset
from lilax._backend.eval.reporting.json_reporter import write_results
from lilax._backend.eval.runner.async_client import generate_batch
from lilax._backend.eval.runner.common import build_server_manager, maybe_upload_to_gcs, resolve_token
from lilax._backend.eval.runner.warmup import warmup_server
from lilax._backend.eval.scoring.registry import get_scorer

logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
  with open(config_path, encoding="utf-8") as file:
    return yaml.safe_load(file) or {}


def _merge_config(base: dict, overrides: dict) -> dict:
  merged = dict(base)
  for key, value in overrides.items():
    if value is not None:
      merged[key] = value
  return merged


def _derive_from_maxtext_config(maxtext_config_path: str) -> dict:
  raw = _load_config(maxtext_config_path)
  prefill_len = raw.get("max_prefill_predict_length")
  target_len = raw.get("max_target_length")
  derived: dict = {}
  if target_len is not None:
    derived["max_model_len"] = int(target_len)
    logger.info("Derived max_model_len=%d from MaxText config max_target_length.", derived["max_model_len"])
  if prefill_len is not None and target_len is not None:
    derived["max_tokens_default"] = int(target_len) - int(prefill_len)
    logger.info("Derived max_tokens_default=%d from max_target_length - max_prefill_predict_length.",
                derived["max_tokens_default"])
  for key in ("base_output_directory", "run_name"):
    if raw.get(key):
      derived[key] = raw[key]
  return derived


def _build_results_path(cfg: dict) -> str:
  base_output_directory = cfg.get("base_output_directory", "").rstrip("/")
  run_name = cfg.get("run_name", "")
  if not base_output_directory or not run_name:
    raise ValueError("Cannot build eval results_path.")
  return f"{base_output_directory}/{run_name}/eval_results"


def run_eval(cfg: dict, hf_token: str | None = None) -> dict:
  benchmark = cfg["benchmark"]
  model_name = cfg["model_name"]
  hf_path = cfg["hf_path"]
  results_path = cfg["results_path"]
  num_samples = cfg.get("num_samples")
  max_tokens = int(cfg.get("max_tokens", 1024))
  temperature = float(cfg.get("temperature", 0.0))
  concurrency = int(cfg.get("concurrency", 64))
  request_timeout = int(cfg.get("request_timeout", 600))
  max_retries = int(cfg.get("max_retries", 3))
  if "max_model_len" not in cfg:
    raise ValueError("Error: max_model_len is required.")
  gcs_results_path = cfg.get("gcs_results_path")
  token = resolve_token(cfg, hf_token)
  logger.info("Loading tokenizer from %s.", hf_path)
  tokenizer = AutoTokenizer.from_pretrained(hf_path, token=token)
  logger.info("Loading benchmark dataset: %s", benchmark)
  dataset = get_dataset(benchmark)
  requests = dataset.sample_requests(num_samples=num_samples, tokenizer=tokenizer)
  logger.info("Loaded %d samples.", len(requests))
  prompts = [request.prompt for request in requests]
  references = [request.reference for request in requests]
  with build_server_manager(cfg, token) as server:
    is_rank0 = jax.process_index() == 0
    if is_rank0:
      base_url = server.base_url
      if not cfg.get("skip_warmup"):
        warmup_server(base_url=base_url, model=model_name, sample_requests=requests)
      logger.info("Generating responses for %d prompts.", len(prompts))
      start = time.time()
      results = generate_batch(
          prompts=prompts, base_url=base_url, model=model_name, max_tokens=max_tokens,
          temperature=temperature, concurrency=concurrency, request_timeout=request_timeout, max_retries=max_retries,
      )
      elapsed = time.time() - start
      logger.info("Generation completed in %.1fs (%.1f samples/s).", elapsed, len(prompts) / elapsed)
    multihost_utils.sync_global_devices("eval_runner_complete")
  if not is_rank0:
    return {}
  responses = [result.text for result in results]
  errors = [result for result in results if result.error]
  if errors:
    logger.warning("%d generation errors (out of %d).", len(errors), len(results))
  scorer = get_scorer(benchmark)
  scores = scorer(responses, references)
  logger.info("Scores: %s", scores)
  generation_stats = {
      "total_samples": len(prompts),
      "num_errors": len(errors),
      "elapsed_s": round(elapsed, 2),
      "samples_per_second": round(len(prompts) / elapsed, 2),
      "total_prompt_tokens": sum(result.prompt_tokens for result in results),
      "total_completion_tokens": sum(result.completion_tokens for result in results),
  }
  output = write_results(benchmark=benchmark, model_name=model_name, scores=scores,
                         generation_stats=generation_stats, config=cfg, results_path=results_path)
  maybe_upload_to_gcs(output, gcs_results_path)
  return output
