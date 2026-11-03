import atexit
import itertools
import multiprocessing
import os
import threading
import time
from typing import Any, Callable, Optional

_POOL = None
_POOL_NUM_PROCS = None
_DEFAULT_MAX_PROCS = 8
_MAX_TASKS_PER_CHILD = 100
_TERMINATE_GRACE_SECONDS = 2.0


def silent_worker_init() -> None:
  os.environ["JAX_PLATFORMS"] = "cpu"
  os.environ["TPU_VISIBLE_DEVICES"] = ""
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
  os.environ["TPU_MIN_LOG_LEVEL"] = "3"
  os.environ["TPU_STDERR_LOG_LEVEL"] = "3"
  os.environ["GRPC_VERBOSITY"] = "ERROR"
  import math_verify
  from math_verify import parse, verify
  from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
  from sympy import Basic, MatrixBase
  from sympy.parsing import sympy_parser


def are_equal_under_sympy(gold: Any, prediction: Any) -> bool:
  from sympy.parsing import sympy_parser

  try:
    gold_expr = sympy_parser.parse_expr(str(gold), evaluate=False)
    pred_expr = sympy_parser.parse_expr(str(prediction), evaluate=False)
    if gold_expr == pred_expr:
      return True
  except Exception:
    pass
  return False


def verify_math_worker(golds: list[str], predictions: list[str]) -> float:
  try:
    from math_verify import parse, verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    from sympy import Basic, MatrixBase

    gold_targets = (ExprExtractionConfig(), LatexExtractionConfig())
    pred_targets = (ExprExtractionConfig(), LatexExtractionConfig())
    extracted_predictions = list(
        itertools.chain.from_iterable(parse(pred, pred_targets, parsing_timeout=None) for pred in predictions)
    )
    extracted_golds = list(itertools.chain.from_iterable(parse(gold, gold_targets, parsing_timeout=None) for gold in golds))
    if not extracted_predictions or not extracted_golds:
      return 0.0
    for gold in extracted_golds:
      for pred in extracted_predictions:
        if isinstance(gold, (Basic, MatrixBase)) and isinstance(pred, (Basic, MatrixBase)):
          if are_equal_under_sympy(gold, pred):
            return 1.0
          if "**" not in str(gold) and "**" not in str(pred):
            if verify(gold, pred, timeout_seconds=None):
              return 1.0
        elif verify(gold, pred, timeout_seconds=None):
          return 1.0
    return 0.0
  except Exception:
    return 0.0


def _shutdown_pool() -> None:
  global _POOL, _POOL_NUM_PROCS
  pool = _POOL
  _POOL = None
  _POOL_NUM_PROCS = None
  if pool is None:
    return
  try:
    workers = list(getattr(pool, "_pool", []))
    for worker in workers:
      if worker.is_alive():
        worker.terminate()
    deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
    for worker in workers:
      remaining = max(0.0, deadline - time.monotonic())
      worker.join(timeout=remaining)
    for worker in workers:
      if worker.is_alive():
        worker.kill()
        worker.join(timeout=1.0)
    thread = threading.Thread(target=pool.terminate, daemon=True)
    thread.start()
    thread.join(timeout=_TERMINATE_GRACE_SECONDS)
  except Exception:
    pass


def _get_pool(num_procs: int) -> multiprocessing.pool.Pool:
  global _POOL, _POOL_NUM_PROCS
  if _POOL is None or _POOL_NUM_PROCS != num_procs:
    _shutdown_pool()
    ctx = multiprocessing.get_context("spawn")
    _POOL = ctx.Pool(processes=num_procs, initializer=silent_worker_init, maxtasksperchild=_MAX_TASKS_PER_CHILD)
    _POOL_NUM_PROCS = num_procs
  return _POOL


atexit.register(_shutdown_pool)


def math_verify_pool(
    trainer_config: Any, items: list[tuple[int, list[str], list[str]]], scores: list[float], timeout: float = 300,
    num_procs: Optional[int] = None, log_fn: Optional[Callable[[str], None]] = None,
) -> list[float]:
  if not items:
    return scores
  cpu_count = multiprocessing.cpu_count()
  if num_procs is None:
    num_procs = min(_DEFAULT_MAX_PROCS, cpu_count)
  else:
    num_procs = max(1, min(num_procs, cpu_count))
  count = 0
  pool = _get_pool(num_procs)
  active_jobs = [(idx, pool.apply_async(verify_math_worker, (golds, predictions))) for idx, golds, predictions in items]
  start_time = time.time()
  while active_jobs and (time.time() - start_time < timeout):
    for index in range(len(active_jobs) - 1, -1, -1):
      idx, job = active_jobs[index]
      if job.ready():
        try:
          score = job.get(0)
          if score > 0.0:
            scores[idx] = max(scores[idx], trainer_config.reward_exact_answer)
          count += 1
        except Exception as error:
          if log_fn:
            log_fn(f"math_verify_pool failed ({error}) for idx: {idx}")
        active_jobs.pop(index)
    time.sleep(0.1)
  if log_fn:
    log_fn(f"math_verify_pool: Processed {count}/{len(items)} items ({len(active_jobs)} timed out).")
  if len(active_jobs) > 0:
    _shutdown_pool()
  return scores
