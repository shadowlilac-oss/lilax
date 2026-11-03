"""Native vLLM next-token and candidate scoring."""

from lilax._backend.configs.runtime import resolve_runtime_config
from lilax._backend.inference.vllm_score import VllmScorer, score as _score, setup_scoring_environment


def score(config, **options):
  """Score the configured prompt or a JSONL prompt batch using vLLM."""
  setup_scoring_environment()
  return _score(resolve_runtime_config(config), **options)


__all__ = ["VllmScorer", "score"]
