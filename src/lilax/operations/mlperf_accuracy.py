"""MLPerf accuracy evaluation operations."""

from lilax._backend.inference.mlperf.accuracy import evaluate_accuracy
from lilax._backend.inference.mlperf.accuracy_fast import evaluate_accuracy_fast

__all__ = ["evaluate_accuracy", "evaluate_accuracy_fast"]
