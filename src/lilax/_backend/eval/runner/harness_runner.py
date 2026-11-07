"""lm-evaluation-harness and evalchemy evaluation runner."""

import logging
from contextlib import nullcontext

import jax
from jax.experimental import multihost_utils
import lm_eval
from lm_eval.tasks import TaskManager

from lilax._backend.eval.reporting.json_reporter import write_results
from lilax._backend.eval.runner.common import build_server_manager, maybe_upload_to_gcs, resolve_token
from lilax._backend.eval.runner.warmup import warmup_server

logger = logging.getLogger(__name__)


def _map_results(raw_results: dict, tasks: list[str]) -> dict:
  scores: dict[str, float] = {}
  results_section = raw_results.get("results", {})
  for task in tasks:
    task_results = results_section.get(task, {})
    accuracy = None
    for key in ("acc,none", "exact_match,strict-match", "exact_match,flexible-extract", "exact_match,none", "acc", "score"):
      if task_results.get(key) is not None:
        accuracy = task_results[key]
        break
    accuracy_norm = None
    for key in ("acc_norm,none", "acc_norm"):
      if task_results.get(key) is not None:
        accuracy_norm = task_results[key]
        break
    if accuracy is not None:
      scores[f"{task}_accuracy"] = round(float(accuracy) * 100, 2)
    if accuracy_norm is not None:
      scores[f"{task}_accuracy_norm"] = round(float(accuracy_norm) * 100, 2)
    has_custom_keys = False
    for suffix in (
        "prompt_level_strict_acc,none", "inst_level_strict_acc,none", "prompt_level_loose_acc,none",
        "inst_level_loose_acc,none", "prompt_level_strict_acc", "inst_level_strict_acc",
        "prompt_level_loose_acc", "inst_level_loose_acc",
    ):
      if task_results.get(suffix) is not None:
        scores[f"{task}_{suffix.replace(',none', '')}"] = round(float(task_results[suffix]) * 100, 2)
        has_custom_keys = True
    if accuracy is None and not has_custom_keys and task_results:
      logger.warning("No known accuracy keys found for task '%s'. Available: %s", task, list(task_results.keys()))
  return scores


def run_harness(cfg: dict, hf_token: str | None = None, *, endpoint=None, eval_dataset=None, compute_metrics=None) -> dict:
  backend = cfg.get("backend", "lm_eval")
  model_name = cfg["model_name"]
  hf_path = cfg["hf_path"]
  tasks = cfg["tasks"]
  results_path = cfg["results_path"]
  num_fewshot = cfg.get("num_fewshot", 0)
  num_samples = cfg.get("num_samples")
  gcs_results_path = cfg.get("gcs_results_path")
  apply_chat_template = cfg.get("apply_chat_template", False)
  fewshot_as_multiturn = cfg.get("fewshot_as_multiturn", False)
  gen_kwargs = cfg.get("gen_kwargs")
  token = resolve_token(cfg, hf_token)
  lm_model_type = "local-chat-completions" if backend == "evalchemy" else "local-completions"
  raw_results: dict = {}
  with (build_server_manager(cfg, token, enable_chat_api=backend == "evalchemy")
        if endpoint is None else nullcontext(endpoint)) as server:
    is_rank0 = jax.process_index() == 0
    if is_rank0:
      if not cfg.get("skip_warmup"):
        warmup_server(base_url=server.base_url, model=model_name)
      completions_path = "/v1/chat/completions" if backend == "evalchemy" else "/v1/completions"
      model_args_parts = [
          f"model={model_name}",
          f"base_url={server.base_url}{completions_path}",
          "tokenizer_backend=huggingface",
          f"tokenizer={hf_path}",
      ]
      if token:
        model_args_parts.append(f"token={token}")
      model_args = ",".join(model_args_parts)
      logger.info("Running %s tasks %s via %s at %s", backend, tasks, lm_model_type, server.base_url)
      simple_eval_kwargs: dict = {
          "model": lm_model_type,
          "model_args": model_args,
          "tasks": tasks,
          "num_fewshot": num_fewshot,
          "limit": num_samples,
          "log_samples": compute_metrics is not None,
      }
      if eval_dataset is not None:
        manager = TaskManager()
        loaded = manager.load(list(tasks))
        for name, task in loaded["tasks"].items():
          source = eval_dataset[name] if isinstance(eval_dataset, dict) else eval_dataset
          split = task.config.test_split if task.has_test_docs() else task.config.validation_split
          task.dataset[split] = source
          task.task_docs = task.eval_docs
        simple_eval_kwargs["tasks"] = list(loaded["tasks"].values())
        simple_eval_kwargs["task_manager"] = manager
      if cfg.get("max_num_seqs") is not None:
        simple_eval_kwargs["batch_size"] = cfg["max_num_seqs"]
      if apply_chat_template:
        simple_eval_kwargs["apply_chat_template"] = True
      if fewshot_as_multiturn:
        simple_eval_kwargs["fewshot_as_multiturn"] = True
      if gen_kwargs:
        simple_eval_kwargs["gen_kwargs"] = gen_kwargs
      raw_results = lm_eval.simple_evaluate(**simple_eval_kwargs)
    multihost_utils.sync_global_devices(f"harness_{backend}_complete")
  if not is_rank0:
    return {}
  scores = _map_results(raw_results, tasks)
  if compute_metrics is not None:
    for task_name, samples in raw_results.get("samples", {}).items():
      responses = [sample["filtered_resps"][0] for sample in samples]
      references = [sample["target"] for sample in samples]
      if any(not isinstance(value, str) for value in (*responses, *references)):
        raise ValueError("compute_metrics requires a generation task with text predictions and text references.")
      scores.update({f"{task_name}_{name}": value for name, value in compute_metrics(responses, references).items()})
  logger.info("%s scores: %s", backend, scores)
  output = write_results(
      benchmark="+".join(tasks), model_name=model_name, scores=scores,
      generation_stats={f"{backend}_config": raw_results.get("config", {}),
                        f"{backend}_results": raw_results.get("results", {})},
      config=cfg, results_path=results_path,
  )
  maybe_upload_to_gcs(output, gcs_results_path)
  return output
