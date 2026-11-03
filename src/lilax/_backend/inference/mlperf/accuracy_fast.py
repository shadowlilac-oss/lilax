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
"""Parallel MLPerf accuracy evaluation."""

from functools import partial
import json
from multiprocessing import Pool, cpu_count

import evaluate
import nltk
import numpy as np
import tqdm
from transformers import AutoTokenizer, LlamaTokenizer

from lilax._backend.inference.mlperf.accuracy import get_groundtruth


def split_data(preds: list[str], refs: list[str], num_chunks: int) -> list[tuple[list[str], list[str]]]:
  chunk_size = len(preds) // num_chunks + (1 if len(preds) % num_chunks else 0)
  chunks = []
  for i in range(0, len(preds), chunk_size):
    chunks.append((preds[i : i + chunk_size], refs[i : i + chunk_size]))
  return chunks


def compute_rouge_chunk(chunk: tuple[list[str], list[str]], metric) -> dict:
  preds, refs = chunk
  return metric.compute(predictions=preds, references=refs, use_stemmer=True, use_aggregator=False)


def aggregate_rouge_scores(chunk_results: list[dict]) -> dict:
  all_scores = {}
  for scores in chunk_results:
    for metric, values in scores.items():
      if metric not in all_scores:
        all_scores[metric] = []
      all_scores[metric].extend(values)
  return {metric: round(np.mean(values) * 100, 4) for metric, values in all_scores.items()}


def process_batch(batch, tokenizer, eval_dtype):
  preds_token_ids = []
  seen = set()
  gen_tok_len = 0
  target_indices = []
  for pred in batch:
    qsl_idx = pred["qsl_idx"]
    if qsl_idx in seen:
      continue
    seen.add(qsl_idx)
    target_indices.append(qsl_idx)
    pred_data = np.frombuffer(bytes.fromhex(pred["data"]), eval_dtype)
    if pred_data[0] > 32000 or pred_data[0] < 0:
      pred_data = np.concatenate([[1], pred_data[1:]])
    gen_tok_len += len(pred_data)
    preds_token_ids.append(pred_data)
  preds_decoded = tokenizer.batch_decode(preds_token_ids, skip_special_tokens=True)
  return preds_decoded, target_indices, gen_tok_len


def postprocess_text(pred, target):
  pred = pred.strip()
  target = target.strip()
  pred = "\n".join(nltk.sent_tokenize(pred))
  target = "\n".join(nltk.sent_tokenize(target))
  return pred, target


def chunk_list(lst, n):
  chunk_size = len(lst) // n + (1 if len(lst) % n else 0)
  return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def evaluate_accuracy_fast(
    mlperf_accuracy_file, dataset_file, *, checkpoint_path=None, tokenizer_path=None,
    verbose=False, dtype="int64", num_workers=None
):
  """Score an MLPerf accuracy log in parallel and return its statistics."""
  num_workers = num_workers or cpu_count()
  print(f"Using {num_workers} worker processes")
  if checkpoint_path:
    print(f"Loading checkpoint from {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path, model_max_length=2048, padding_side="left", use_fast=False
    )
  elif tokenizer_path:
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = LlamaTokenizer(tokenizer_path)
  else:
    raise ValueError("Either --checkpoint-path or --tokenizer-path must be provided")

  metric = evaluate.load("rouge")
  nltk.download("punkt", quiet=True)
  print(f"Getting groundtruth from {dataset_file}")
  targets = get_groundtruth(dataset_file)
  eval_dtype = {"int32": np.int32, "int64": np.int64, "float": np.float32}[dtype]
  print(f"Loading accuracy log from {mlperf_accuracy_file}")
  with open(mlperf_accuracy_file, "rt", encoding="utf8") as f:
    results = json.load(f)

  result_chunks = chunk_list(results, num_workers)
  process_func = partial(process_batch, tokenizer=tokenizer, eval_dtype=eval_dtype)
  total_gen_tok_len = 0
  all_preds = []
  all_target_indices = []
  print("Processing predictions...")
  with Pool(num_workers) as pool:
    for preds, target_indices, gen_tok_len in tqdm.tqdm(pool.imap(process_func, result_chunks), total=len(result_chunks)):
      all_preds.extend(preds)
      all_target_indices.extend(target_indices)
      total_gen_tok_len += gen_tok_len

  target_required = [targets[idx] for idx in all_target_indices]
  print("Post-processing texts...")
  with Pool(num_workers) as pool:
    processed_pairs = list(tqdm.tqdm(pool.starmap(postprocess_text, zip(all_preds, target_required)), total=len(all_preds)))
  preds, refs = zip(*processed_pairs)
  print("Computing ROUGE scores...")
  data_chunks = split_data(preds, refs, num_workers)
  with Pool(num_workers) as pool:
    chunk_results = list(
        tqdm.tqdm(pool.imap(partial(compute_rouge_chunk, metric=metric), data_chunks), total=len(data_chunks))
    )
  rouge_scores = aggregate_rouge_scores(chunk_results)
  prediction_lens = [len(pred) for pred in preds]
  gen_num = len(preds)
  result = {
      **rouge_scores,
      "gen_len": np.sum(prediction_lens),
      "gen_num": gen_num,
      "gen_tok_len": total_gen_tok_len,
      "tokens_per_sample": round(total_gen_tok_len / gen_num, 1),
  }
  print("\nResults\n")
  print(result)
  return result
