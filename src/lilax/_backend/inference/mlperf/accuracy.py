# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MLPerf accuracy evaluation."""

import json
import os

import evaluate
import nltk
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def get_groundtruth(processed_dataset_file):
  """Load the ground truth labels from the processed dataset file."""
  ext = os.path.splitext(processed_dataset_file)[1].lower()
  if ext == ".parquet":
    data = pd.read_parquet(processed_dataset_file)
  elif ext == ".csv":
    data = pd.read_csv(processed_dataset_file)
  elif ext in (".json", ".jsonl"):
    data = pd.read_json(processed_dataset_file)
  else:
    raise ValueError(
        f"Unsupported dataset file format: {processed_dataset_file}. "
        "Please use safe formats like Parquet (.parquet), CSV (.csv), or JSON/JSONL (.json/.jsonl)."
    )
  return data["output"]


def postprocess_text(preds, targets):
  preds = [pred.strip() for pred in preds]
  targets = [target.strip() for target in targets]
  preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
  targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]
  return preds, targets


def evaluate_accuracy(checkpoint_path, mlperf_accuracy_file, dataset_file, *, verbose=False, dtype="int64"):
  """Score an MLPerf accuracy log and return its ROUGE and generation statistics."""
  metric = evaluate.load("rouge")
  nltk.download("punkt")
  print(f"Loading checkpoint from {checkpoint_path}")
  tokenizer = AutoTokenizer.from_pretrained(
      checkpoint_path, model_max_length=2048, padding_side="left", use_fast=False
  )
  targets = get_groundtruth(dataset_file)
  target_required = []
  preds_token_ids = []
  eval_dtype = {"int32": np.int32, "int64": np.int64, "float": np.float32}[dtype]
  with open(mlperf_accuracy_file, "rt", encoding="utf8") as f:
    results = json.load(f)

  seen = set()
  gen_tok_len = 0
  for pred in results:
    qsl_idx = pred["qsl_idx"]
    if qsl_idx in seen:
      continue
    seen.add(qsl_idx)
    target_required.append(targets[qsl_idx])
    pred = np.frombuffer(bytes.fromhex(pred["data"]), eval_dtype)
    if pred[0] > 32000 or pred[0] < 0:
      pred = [1, *pred[1:]]
    gen_tok_len += len(pred)
    preds_token_ids.append(pred)

  preds_decoded_text = tokenizer.batch_decode(preds_token_ids, skip_special_tokens=True)
  preds, targets = postprocess_text(preds_decoded_text, target_required)
  result = metric.compute(predictions=preds, references=targets, use_stemmer=True, use_aggregator=False)
  result = {k: round(np.mean(v) * 100, 4) for k, v in result.items()}
  prediction_lens = [len(pred) for pred in preds]
  gen_num = len(preds)
  result = {
      **result,
      "gen_len": np.sum(prediction_lens),
      "gen_num": gen_num,
      "gen_tok_len": gen_tok_len,
      "tokens_per_sample": round(gen_tok_len / gen_num, 1),
  }
  print("\nResults\n")
  print(result)
  return result
