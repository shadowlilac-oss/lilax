"""GEPA prompt optimization for the NuminaMath example."""

import random
import re

import gepa
from datasets import load_dataset
from gepa.adapters.default_adapter.default_adapter import DefaultAdapter
from gepa.core.adapter import EvaluationBatch


def load_math_dataset():
  """Stream and split the notebook's numeric AMC/AIME problems."""
  dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
  examples = []
  for item in dataset:
    if item["source"] == "amc_aime":
      solution = item["solution"]
      match = re.search(r"\\boxed\{([^}]+)\}", solution)
      if match:
        answer = match.group(1).strip()
        if answer.replace(".", "", 1).replace("-", "", 1).isdigit():
          examples.append({"input": item["problem"], "additional_context": {"solution": solution}, "answer": answer})
    if len(examples) >= 600:
      break

  random.Random(42).shuffle(examples)
  return examples[:200], examples[200:500], examples[500:600]


def extract_math_answer(prediction):
  """Extract the first boxed answer, including nested braces."""
  match = re.search(r"\\boxed\{", prediction)
  if not match:
    return ""
  start = match.end()
  depth = 1
  for index, char in enumerate(prediction[start:]):
    if char == "{":
      depth += 1
    elif char == "}":
      depth -= 1
    if depth == 0:
      return prediction[start:start + index].strip()
  return ""


class MathAdapter(DefaultAdapter):
  """Score boxed answers using the notebook's numeric equality metric."""

  def evaluate(self, batch, candidate, capture_traces=False):
    eval_batch = super().evaluate(batch, candidate, capture_traces)
    scores = []
    trajectories = list(eval_batch.trajectories) if eval_batch.trajectories else [{} for _ in batch]
    for index, (data, output) in enumerate(zip(batch, eval_batch.outputs)):
      answer = extract_math_answer(output.get("full_assistant_response", ""))
      try:
        score = 1.0 if float(answer) == float(str(data["answer"]).strip()) else 0.0
      except ValueError:
        score = 0.0
      scores.append(score)
      if capture_traces:
        trajectories[index]["extracted_answer"] = answer
    return EvaluationBatch(outputs=eval_batch.outputs, scores=scores, trajectories=trajectories)


def optimize_math_prompt(model, reflection_lm, trainset, valset, seed_candidate):
  """Optimize a system prompt against the notebook's math metric.

  ``model`` may be a LiteLLM model name or a chat completion callable accepted
  by GEPA's DefaultAdapter. A running OpenAI-compatible server can be selected
  through LiteLLM's usual configuration.
  """
  adapter = MathAdapter(
      model=model,
      max_litellm_workers=2,
      litellm_batch_completion_kwargs={"max_tokens": 8192, "temperature": 0.0},
  )
  return gepa.optimize(
      seed_candidate=seed_candidate,
      trainset=trainset,
      valset=valset,
      adapter=adapter,
      max_metric_calls=1000,
      reflection_lm=reflection_lm,
      display_progress_bar=False,
      reflection_minibatch_size=8,
      skip_perfect_score=True,
      use_merge=True,
      max_merge_invocations=5,
      merge_val_overlap_floor=5,
  )
