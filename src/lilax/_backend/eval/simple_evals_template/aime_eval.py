"""AIME 2024 and 2025 evaluations."""

from __future__ import annotations

import random
import re

from huggingface_hub import hf_hub_download
import pandas

from . import common
from .common import HTML_JINJA
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult

_AIME_2024_URL = (
    "https://huggingface.co/datasets/Maxwell-Jia/AIME_2024/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
_AIME_2025_URL = (
    "https://huggingface.co/datasets/math-ai/aime25/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"
)
_DATASET_BY_YEAR = {
    2024: (_AIME_2024_URL, "Problem", "Answer"),
    2025: (_AIME_2025_URL, "problem", "answer"),
}
QUERY_TEMPLATE = (
    "Solve this competition math problem. The final answer is always an integer between 0 and 999. "
    "Think step by step, then write the final answer by itself on the last line in the format "
    '"Answer: $N" (without quotes), where $N is the integer answer.\n\n{problem}'
)
_ANSWER_LINE_RE = re.compile(r"(?i)(?:^|\n)\s*Answer\s*:\s*\$?([0-9]{1,3})\$?\s*\Z")


def extract_answer(response_text: str) -> str | None:
  match = _ANSWER_LINE_RE.search(response_text)
  return match.group(1) if match else None


class AIMEEval(Eval):

  def __init__(self, year: int, num_examples: int | None = None, n_repeats: int = 1, *, dataset=None):
    if year not in _DATASET_BY_YEAR:
      raise ValueError(f"Unsupported AIME year: {year}. Supported: {sorted(_DATASET_BY_YEAR)}.")
    url, problem_col, answer_col = _DATASET_BY_YEAR[year]
    if dataset is None:
      try:
        if "AIME_2024" in url:
          parquet_path = hf_hub_download(
              repo_id="Maxwell-Jia/AIME_2024", repo_type="dataset", filename="default/train/0000.parquet"
          )
        else:
          parquet_path = hf_hub_download(
              repo_id="math-ai/aime25", repo_type="dataset", filename="default/test/0000.parquet"
          )
      except Exception as exc:
        raise RuntimeError(f"Could not download AIME dataset: {exc}") from exc
      df = pandas.read_parquet(parquet_path)
      dataset = [row.to_dict() for _, row in df.iterrows()]
    examples = [{"problem": row[problem_col], "answer": str(int(row[answer_col]))} for row in dataset]
    rng = random.Random(0)
    if num_examples:
      assert n_repeats == 1, "n_repeats only supported for num_examples = None"
      examples = rng.sample(examples, num_examples)
    self.examples = examples * n_repeats
    self.year = year

  def __call__(self, sampler: SamplerBase) -> EvalResult:
    def fn(row: dict):
      prompt_messages = [sampler._pack_message(content=QUERY_TEMPLATE.format(problem=row["problem"]), role="user")]
      sampler_response = sampler(prompt_messages)
      response_text = sampler_response.response_text
      actual_queried_prompt_messages = sampler_response.actual_queried_message_list
      extracted_answer = extract_answer(response_text) if common.request_succeeded(sampler_response) else None
      score = 1.0 if extracted_answer is not None and int(extracted_answer) == int(row["answer"]) else 0.0
      html = common.jinja_env.from_string(HTML_JINJA).render(
          prompt_messages=actual_queried_prompt_messages,
          next_message={"content": response_text, "role": "assistant"},
          score=score,
          correct_answer=row["answer"],
          extracted_answer=extracted_answer,
      )
      convo = actual_queried_prompt_messages + [{"content": response_text, "role": "assistant"}]
      return SingleEvalResult(
          html=html,
          score=score,
          convo=convo,
          metrics={"chars": len(response_text)},
          example_level_metadata={
              "request_id": sampler_response.response_metadata.get("request_id"),
              "request_status": sampler_response.response_metadata.get("status", "success"),
              "score": score,
              "correct_answer": row["answer"],
              "extracted_answer": extracted_answer,
          },
      )

    results = common.map_with_progress(fn, self.examples)
    return common.aggregate_results(results)
