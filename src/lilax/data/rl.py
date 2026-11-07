"""Prepare Hugging Face records for the existing RL prompt and reward format."""

import json
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from lilax._backend.trainers.post_train.rl import utils_rl
from lilax._backend.utils.globals import MAXTEXT_PKG_DIR
from lilax.rollout.rewards import RewardSpecialTokens


class RLDataOptions(BaseModel):
  model_config = ConfigDict(extra="forbid")
  train_fraction: float = Field(1.0, gt=0.0, le=1.0)
  test_batch_start_index: int = 0
  processor: Callable[..., dict] | None = None


def prepare_rl_dataset(dataset, *, processing_class, options=None):
  options = options or RLDataOptions()
  tokenizer = getattr(processing_class, "tokenizer", processing_class)
  with (Path(MAXTEXT_PKG_DIR) / "examples/chat_templates/gsm8k_rl.json").open(encoding="utf-8") as stream:
    template = json.load(stream)
  dataset_name = getattr(dataset.info, "dataset_name", "") or ""
  dataset_name = "openai/gsm8k" if dataset_name == "gsm8k" else dataset_name

  def process(record):
    if options.processor is not None:
      return options.processor(record)
    if all(name in record for name in ("prompts", "question", "answer")):
      return record
    return utils_rl.process_data(dataset_name, tokenizer, template, RewardSpecialTokens(), record)

  prepared = dataset.map(process)
  prepared._lilax_rl_options = options
  return prepared


def _batch_rl_dataset(dataset, config, processing_class, *, evaluation=False, agentic=False):
  """Apply the existing prompt filter and training duration without an evaluation cap."""
  tokenizer = getattr(processing_class, "tokenizer", processing_class)
  prepared = dataset if hasattr(dataset, "_lilax_rl_options") else prepare_rl_dataset(
      dataset, processing_class=processing_class,
  )
  options = prepared._lilax_rl_options
  prepared = prepared.filter(lambda row: len(tokenizer.tokenize(row["prompts"])) <= config.max_prefill_predict_length)
  if hasattr(prepared, "to_iterable_dataset"):
    prepared = prepared.to_iterable_dataset()
  if evaluation:
    prepared = prepared.skip(options.test_batch_start_index)
    batch_size = config.batch_size if config.eval_batch_size <= 0 else config.eval_batch_size
    return prepared.batch(batch_size, drop_last_batch=False)
  if agentic:
    prepared = prepared.map(lambda row: {"prompts": row["question"]})
  size = int(config.num_batches * config.batch_size * options.train_fraction)
  return prepared.take(size).repeat(config.num_epoch).batch(config.batch_size, drop_last_batch=True)
