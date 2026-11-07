"""Run OpenAI simple-evals tasks against a local vLLM chat server."""

from __future__ import annotations

import logging
import random
import threading
import time
from contextlib import nullcontext
from typing import Any

import jax
from jax.experimental import multihost_utils
import openai
from openai import OpenAI

from lilax._backend.eval.reporting.json_reporter import write_results
from lilax._backend.eval.reporting.simple_evals_debug_reporter import (
    SimpleEvalsDebugCollector, build_debug_report, write_debug_report,
)
from lilax._backend.eval.runner.common import build_server_manager, maybe_upload_to_gcs, resolve_token
from lilax._backend.eval.runner.warmup import warmup_chat_server
from lilax._backend.eval.simple_evals_template import common as simple_evals_common
from lilax._backend.eval.simple_evals_template.aime_eval import AIMEEval
from lilax._backend.eval.simple_evals_template.drop_eval import DropEval
from lilax._backend.eval.simple_evals_template.gpqa_eval import GPQAEval
from lilax._backend.eval.simple_evals_template.gsm8k_eval import GSM8KEval
from lilax._backend.eval.simple_evals_template.mgsm_eval import MGSMEval
from lilax._backend.eval.simple_evals_template.mmlu_eval import MMLUEval
from lilax._backend.eval.simple_evals_template.types import SamplerBase, SamplerResponse

DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."
_MAX_ATTEMPTS = 3
_REQUEST_TIMEOUT_S = 600.0
SUPPORTED_TASKS = ("mmlu", "gpqa", "gsm8k", "drop", "mgsm", "mgsm_en", "aime2024", "aime2025")
logger = logging.getLogger(__name__)


def _describe_api_error(exc: Exception | None) -> str:
  if exc is None:
    return "n/a"
  parts = [f"{type(exc).__name__}: {exc}"]
  request = getattr(exc, "request", None)
  if request is not None:
    parts.append(f"request={getattr(request, 'method', '?')} {getattr(request, 'url', '?')}")
  response = getattr(exc, "response", None)
  if response is not None:
    body = getattr(exc, "body", None)
    text = str(body) if body is not None else getattr(response, "text", "")
    parts.append(f"status={getattr(response, 'status_code', '?')} body={text[:500]}")
  if exc.__cause__ is not None:
    parts.append(f"cause={type(exc.__cause__).__name__}: {exc.__cause__}")
  return " | ".join(parts)


def _build_vllm_sampler(
    base_url: str, model_name: str, max_tokens: int, temperature: float,
    system_message: str | None = DEFAULT_SYSTEM_MESSAGE, reasoning_effort: str | None = None,
    debug_collector=None, concurrency: int = 1, continue_on_request_error: bool = False,
):
  class _VllmChatSampler(SamplerBase):
    def __init__(self) -> None:
      self.client = OpenAI(base_url=f"{base_url}/v1", api_key="EMPTY", max_retries=0, timeout=_REQUEST_TIMEOUT_S)
      self.model = model_name
      self.max_tokens = max_tokens
      self.temperature = temperature
      self.system_message = system_message
      self.reasoning_effort = reasoning_effort
      self.request_semaphore = threading.BoundedSemaphore(concurrency)
      self.continue_on_request_error = continue_on_request_error

    def _pack_message(self, role: str, content: Any) -> dict:
      return {"role": str(role), "content": content}

    def __call__(self, message_list):
      with self.request_semaphore:
        return self._sample(message_list)

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
      status = getattr(getattr(exc, "response", None), "status_code", None)
      return isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)) or (
          status in (408, 409, 429) or (status is not None and status >= 500)
      )

    @staticmethod
    def _record_attempt_error(debug_record, exc: Exception, attempt_start: float) -> None:
      if debug_record is None:
        return
      debug_record["attempt_errors"].append({
          "type": type(exc).__name__, "status": getattr(getattr(exc, "response", None), "status_code", None),
          "detail": _describe_api_error(exc), "latency_s": time.monotonic() - attempt_start,
      })

    def _failure_response(self, message_list, debug_record, exc: Exception, *, status: str, error_message: str):
      response_metadata = {"usage": None, "status": status}
      if debug_record is not None:
        debug_record.update(
            status=status, response_text="", finish_reason=None, prompt_tokens=None, completion_tokens=None,
            total_tokens=None, error_type=type(exc).__name__,
            error_status=getattr(getattr(exc, "response", None), "status_code", None),
            error_detail=_describe_api_error(exc),
        )
        debug_collector.finish(debug_record)
        response_metadata.update(debug_record)
      if self.continue_on_request_error:
        return SamplerResponse(response_text="", response_metadata=response_metadata, actual_queried_message_list=message_list)
      raise RuntimeError(error_message) from exc

    @staticmethod
    def _success_response(message_list, response, debug_record, attempt_start: float):
      choice = response.choices[0]
      message = choice.message
      content = message.content or ""
      usage = response.usage
      response_metadata = {"usage": usage}
      if debug_record is not None:
        response_extra = getattr(response, "model_extra", None) or {}
        diagnostics = response_extra.get("diagnostics") or getattr(response, "diagnostics", None) or {}
        debug_record.update(
            status="success", response_text=content, reasoning_text=getattr(message, "reasoning", None),
            raw_response_text=diagnostics.get("raw_output"),
            successful_attempt_latency_s=time.monotonic() - attempt_start,
            finish_reason=getattr(choice, "finish_reason", None), prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None), total_tokens=getattr(usage, "total_tokens", None),
        )
        debug_collector.finish(debug_record)
        response_metadata.update(debug_record)
      return SamplerResponse(response_text=content, response_metadata=response_metadata, actual_queried_message_list=message_list)

    def _sample(self, message_list):
      if self.system_message:
        message_list = [self._pack_message("system", self.system_message)] + message_list
      debug_record = debug_collector.begin(message_list) if debug_collector else None
      call_start = time.monotonic()
      last_exc: Exception | None = None
      for trial in range(_MAX_ATTEMPTS):
        attempt_start = time.monotonic()
        if debug_record is not None:
          debug_record["attempt_count"] += 1
        try:
          extra_body = {"include_reasoning": debug_record is not None, "include_raw_output": debug_record is not None}
          if self.reasoning_effort:
            extra_body["reasoning_effort"] = self.reasoning_effort
          response = self.client.chat.completions.create(
              model=self.model, messages=message_list, temperature=self.temperature,
              max_tokens=self.max_tokens, extra_body=extra_body,
          )
          return self._success_response(message_list, response, debug_record, attempt_start)
        except openai.BadRequestError as exc:
          logger.warning("vLLM rejected an invalid request: %s", _describe_api_error(exc))
          self._record_attempt_error(debug_record, exc, attempt_start)
          return self._failure_response(
              message_list, debug_record, exc, status="bad_request",
              error_message="vLLM rejected the evaluation request as invalid.",
          )
        except Exception as exc:
          last_exc = exc
          self._record_attempt_error(debug_record, exc, attempt_start)
          if not self._is_retryable(exc):
            logger.error("Non-retryable vLLM response failure: %s", _describe_api_error(exc))
            break
          if trial + 1 < _MAX_ATTEMPTS:
            backoff = random.uniform(0, min(30.0, float(2**trial)))
            logger.warning(
                "vLLM request failed (trial %d/%d) [%s] cause=%s attempt_elapsed=%.1fs "
                "total_elapsed=%.1fs thread=%s; retrying in %.1fs: %s",
                trial + 1, _MAX_ATTEMPTS, type(exc).__name__,
                type(getattr(exc, "__cause__", None)).__name__ if getattr(exc, "__cause__", None) else "n/a",
                time.monotonic() - attempt_start, time.monotonic() - call_start,
                threading.current_thread().name, backoff, _describe_api_error(exc),
            )
            time.sleep(backoff)
      assert last_exc is not None
      attempts_made = trial + 1
      if debug_record is not None:
        attempts_made = debug_record.get("attempt_count", attempts_made)
      logger.error(
          "vLLM request failed after %d attempt(s) and %.1fs total (thread=%s): %s",
          attempts_made, time.monotonic() - call_start, threading.current_thread().name,
          _describe_api_error(last_exc), exc_info=last_exc,
      )
      return self._failure_response(
          message_list, debug_record, last_exc, status="error",
          error_message=f"vLLM request failed after {attempts_made} attempt(s).",
      )

  return _VllmChatSampler()


def _effective_repeats(task: str, num_examples: int | None, n_repeats: int | None) -> int:
  if task not in ("gpqa", "aime2024", "aime2025") or num_examples is not None:
    return 1
  if n_repeats is not None:
    return n_repeats
  return 4 if task == "gpqa" else 1


def _build_eval(task: str, num_examples: int | None, n_repeats: int | None, *, dataset=None):
  if task == "mmlu":
    return MMLUEval(num_examples=num_examples, dataset=dataset)
  if task == "gpqa":
    return GPQAEval(num_examples=num_examples, n_repeats=_effective_repeats(task, num_examples, n_repeats), dataset=dataset)
  if task == "drop":
    return DropEval(num_examples=num_examples, dataset=dataset)
  if task == "mgsm":
    return MGSMEval(num_examples_per_lang=num_examples or 250, dataset=dataset)
  if task == "mgsm_en":
    return MGSMEval(num_examples_per_lang=num_examples or 250, languages=["en"], dataset=dataset)
  if task == "gsm8k":
    return GSM8KEval(num_examples=num_examples, dataset=dataset)
  if task in ("aime2024", "aime2025"):
    return AIMEEval(year=2024 if task == "aime2024" else 2025, num_examples=num_examples,
                    n_repeats=_effective_repeats(task, num_examples, n_repeats), dataset=dataset)
  raise ValueError(f"Unsupported simple_evals task: {task}. Supported: {list(SUPPORTED_TASKS)}.")


def _map_scores(task: str, eval_result) -> dict:
  scores: dict[str, float] = {}
  if eval_result.score is not None:
    scores[f"{task}_accuracy"] = round(float(eval_result.score) * 100, 2)
  for name, value in (eval_result.metrics or {}).items():
    if isinstance(value, (int, float)):
      scores[f"{task}_{name}"] = round(float(value), 4)
  return scores


def _validate_tasks(tasks: list[str]) -> None:
  unsupported = [task for task in tasks if task not in SUPPORTED_TASKS]
  if unsupported:
    raise ValueError(
        f"Unsupported simple_evals task(s): {unsupported}. "
        f"Phase 1 supports grader-free evals only: {list(SUPPORTED_TASKS)}."
    )


def run_simple_evals(cfg: dict, hf_token: str | None = None, *, endpoint=None, eval_dataset=None, compute_metrics=None) -> dict:
  """Run selected simple-evals tasks and write their aggregate results."""
  model_name = cfg["model_name"]
  tasks = cfg["tasks"]
  results_path = cfg["results_path"]
  num_samples = cfg.get("num_samples")
  n_repeats = cfg.get("n_repeats")
  if n_repeats is not None:
    n_repeats = int(n_repeats)
  max_tokens = int(cfg.get("max_tokens") or 2048)
  temperature_value = cfg.get("temperature")
  temperature = float(0.5 if temperature_value is None else temperature_value)
  system_message = cfg.get("system_message", DEFAULT_SYSTEM_MESSAGE)
  reasoning_effort = cfg.get("reasoning_effort")
  log_debug_info = bool(cfg.get("log_debug_info", False))
  continue_on_request_error = bool(cfg.get("continue_on_request_error", False))
  gcs_results_path = cfg.get("gcs_results_path")
  token = resolve_token(cfg, hf_token)
  _validate_tasks(tasks)
  scores: dict[str, float] = {}
  eval_results: dict[str, Any] = {}
  task_durations: dict[str, float] = {}
  debug_collector = SimpleEvalsDebugCollector() if log_debug_info else None
  is_rank0 = False
  with (build_server_manager(cfg, token, enable_chat_api=True) if endpoint is None else nullcontext(endpoint)) as server:
    is_rank0 = jax.process_index() == 0
    if is_rank0:
      simple_evals_common.set_default_num_threads(server.concurrency)
      if not cfg.get("skip_warmup"):
        warmup_chat_server(
            base_url=server.base_url, model=model_name, concurrency=server.concurrency,
            max_model_len=int(cfg["max_model_len"]), system_message=system_message,
            reasoning_effort=reasoning_effort, max_tokens=min(16, max_tokens),
        )
      sampler = _build_vllm_sampler(
          base_url=server.base_url, model_name=model_name, max_tokens=max_tokens, temperature=temperature,
          system_message=system_message, reasoning_effort=reasoning_effort, debug_collector=debug_collector,
          concurrency=server.concurrency, continue_on_request_error=continue_on_request_error,
      )
      for task in tasks:
        logger.info("Running simple_evals task '%s' at %s", task, server.base_url)
        dataset = (eval_dataset[task] if isinstance(eval_dataset, dict) and task in eval_dataset else eval_dataset)
        eval_obj = _build_eval(task, num_samples, n_repeats, dataset=dataset)
        if debug_collector is not None:
          debug_collector.set_task(task)
        task_start = time.monotonic()
        eval_result = eval_obj(sampler)
        task_durations[task] = time.monotonic() - task_start
        eval_results[task] = eval_result
        task_scores = _map_scores(task, eval_result)
        if compute_metrics is not None:
          responses = [conversation[-1]["content"] for conversation in eval_result.convos]
          references = [item["correct_answer"] for item in eval_result.metadata["example_level_metadata"]]
          if any(not isinstance(value, str) for value in (*responses, *references)):
            raise ValueError("compute_metrics requires text predictions and single text references.")
          task_scores.update({f"{task}_{name}": value for name, value in compute_metrics(responses, references).items()})
        logger.info("simple_evals task '%s' scores: %s", task, task_scores)
        scores.update(task_scores)
    multihost_utils.sync_global_devices("simple_evals_complete")
  if not is_rank0:
    return {}
  output = write_results(
      benchmark="+".join(tasks), model_name=model_name, scores=scores,
      generation_stats={
          "num_samples": num_samples, "n_repeats": n_repeats,
          "effective_repeats": {task: _effective_repeats(task, num_samples, n_repeats) for task in tasks},
          "task_variants": {
              task: ("all_11_languages" if task == "mgsm" else "english_only" if task == "mgsm_en" else "default")
              for task in tasks
          },
          "effective_concurrency": server.concurrency,
          "protocol_version": "openai-simple-evals-652c89d-maxtext-v2",
      },
      config=cfg, results_path=results_path,
  )
  maybe_upload_to_gcs(output, gcs_results_path)
  if debug_collector is not None:
    report = build_debug_report(
        collector=debug_collector, eval_results=eval_results, task_durations=task_durations,
        model_name=model_name, max_tokens=max_tokens, max_model_len=int(cfg["max_model_len"]),
    )
    debug_local_path = write_debug_report(report, results_path, output["local_path"])
    output["debug_local_path"] = debug_local_path
    if gcs_results_path:
      secondary_debug_path = (
          gcs_results_path if gcs_results_path.endswith("/") else f"{gcs_results_path.rsplit('.', 1)[0]}.debug.txt"
      )
      maybe_upload_to_gcs({"local_path": debug_local_path}, secondary_debug_path)
  return output
