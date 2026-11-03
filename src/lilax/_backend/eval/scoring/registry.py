"""Registry mapping benchmark names to scorers."""

from typing import Callable

from lilax._backend.eval.scoring import rouge_scorer

SCORER_REGISTRY: dict[str, Callable[..., dict]] = {
    "mlperf_openorca": rouge_scorer.score_batch,
    "openorca": rouge_scorer.score_batch,
}


def get_scorer(benchmark_name: str) -> Callable[..., dict]:
  key = benchmark_name.lower()
  if key not in SCORER_REGISTRY:
    raise KeyError(f"No scorer registered for benchmark '{benchmark_name}'. Available: {sorted(SCORER_REGISTRY)}. ")
  return SCORER_REGISTRY[key]


def register_scorer(benchmark_name: str, scorer_fn: Callable[..., dict]) -> None:
  SCORER_REGISTRY[benchmark_name.lower()] = scorer_fn
