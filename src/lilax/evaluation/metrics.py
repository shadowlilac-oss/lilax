"""Text metrics supported by the existing benchmark scorers."""

from lilax._backend.eval.scoring.rouge_scorer import score_batch as score_rouge

__all__ = ["score_rouge"]
