"""Read completed Tunix metric buffers with their source-selected reducers."""

import jax
import numpy as np
from tunix.sft import utils


def trainer_scalar_metrics(trainer, metrics):
  loss = float(metrics.loss)
  values = {"loss": loss, "perplexity": float(np.exp(loss))}
  learning_rate = trainer._try_get_learning_rate()
  if learning_rate is not None:
    values["learning_rate"] = float(learning_rate)
  for name, (items, reducer) in metrics.additional_metrics.items():
    weighted = [utils.is_weighted_metric(item) for item in items]
    if any(weighted) and not all(weighted):
      raise TypeError("metrics must not mix weighted and scalar values")
    if all(weighted) and getattr(reducer, "__name__", "") in (
        "weighted_metric_mean", "global_weighted_mean", "mean_of_means",
    ):
      result = reducer(items)
    else:
      items = [item.compute() for item in items] if all(weighted) else items
      result = reducer([np.asarray(item, dtype=np.float32) if isinstance(item, jax.Array) else item for item in items])
    values[name] = float(result)
  return values
