"""ROUGE scorer for MLPerf OpenOrca."""

import evaluate
import nltk
import numpy as np

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def score_batch(responses: list[str], references: list[str], use_stemmer: bool = True) -> dict:
  if len(responses) != len(references):
    raise ValueError(f"Length mismatch: {len(responses)} responses vs {len(references)} references.")
  metric = evaluate.load("rouge")
  preds = []
  targets = []
  for response, reference in zip(responses, references):
    preds.append("\n".join(nltk.sent_tokenize(response.strip())))
    targets.append("\n".join(nltk.sent_tokenize(reference.strip())))
  result = metric.compute(predictions=preds, references=targets)
  result = {key: float(round(np.mean(value) * 100, 4)) for key, value in result.items()}
  result["gen_num"] = len(preds)
  return result
