"""GSM8K evaluation."""

from __future__ import annotations

import random
import re

from huggingface_hub import hf_hub_download
import pandas

from . import common
from .common import HTML_JINJA
from .mgsm_eval import score_mgsm
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult

_DATASET_REPO_ID = "openai/gsm8k"
_TEST_PARQUET_FILENAME = "main/test-00000-of-00001.parquet"
QUERY_TEMPLATE = (
    "Solve this math problem. Give the reasoning steps before giving the final answer on the last line "
    'by itself in the format of "Answer:". Do not add anything other than the integer answer after '
    '"Answer:".\n\n{question}'
)
_ANSWER_LINE_RE = re.compile(r"(?i)(?:^|\n)\s*Answer\s*:\s*(-?[\d,]+(?:\.\d+)?)\s*\Z")


def extract_answer(response_text: str) -> str | None:
  match = _ANSWER_LINE_RE.search(response_text)
  return match.group(1).replace(",", "") if match else None


def _load_gsm8k_examples() -> list[dict]:
  try:
    parquet_path = hf_hub_download(
        repo_id=_DATASET_REPO_ID, repo_type="dataset", filename=_TEST_PARQUET_FILENAME
    )
  except Exception as exc:
    raise RuntimeError(
        "Could not download the canonical GSM8K main/test split from Hugging Face. "
        "Check outbound Hub access and HF_TOKEN for any proxy policy."
    ) from exc
  df = pandas.read_parquet(parquet_path)
  examples = [row.to_dict() for _, row in df.iterrows()]
  if len(examples) != 1319:
    raise ValueError(f"Expected 1319 GSM8K main/test examples, found {len(examples)}.")
  return examples


class GSM8KEval(Eval):

  def __init__(self, num_examples: int | None = None):
    examples = _load_gsm8k_examples()
    for example in examples:
      example["target"] = example["answer"].split("####")[-1].strip().replace(",", "")
    if num_examples:
      examples = random.Random(0).sample(examples, num_examples)
    self.examples = examples

  def __call__(self, sampler: SamplerBase) -> EvalResult:
    def fn(row: dict):
      prompt_messages = [sampler._pack_message(content=QUERY_TEMPLATE.format(question=row["question"]), role="user")]
      sampler_response = sampler(prompt_messages)
      response_text = sampler_response.response_text
      actual_queried_prompt_messages = sampler_response.actual_queried_message_list
      extracted_answer = extract_answer(response_text) if common.request_succeeded(sampler_response) else None
      score = score_mgsm(row["target"], extracted_answer or "")
      html = common.jinja_env.from_string(HTML_JINJA).render(
          prompt_messages=actual_queried_prompt_messages,
          next_message={"content": response_text, "role": "assistant"},
          score=score,
          correct_answer=row["target"],
          extracted_answer=extracted_answer or None,
      )
      convo = actual_queried_prompt_messages + [{"content": response_text, "role": "assistant"}]
      return SingleEvalResult(
          html=html,
          score=score,
          convo=convo,
          example_level_metadata={
              "request_id": sampler_response.response_metadata.get("request_id"),
              "request_status": sampler_response.response_metadata.get("status", "success"),
              "score": score,
              "correct_answer": row["target"],
              "extracted_answer": extracted_answer or None,
          },
      )

    results = common.map_with_progress(fn, self.examples)
    return common.aggregate_results(results)
