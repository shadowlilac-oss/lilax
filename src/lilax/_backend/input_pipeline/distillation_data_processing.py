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

"""Preprocess conversational datasets for knowledge distillation."""

from dataclasses import dataclass, field

import datasets

import grain.python as grain
import numpy as np
from grain._src.python.dataset.sources import tfrecord_dataset
from grain.experimental import TFRecordIterDataset
from lilax._backend.diffusion.block_diffusion import corruption as block_diffusion_corruption
from lilax._backend.input_pipeline import encoding_dsv4
from lilax._backend.input_pipeline import tokenizer as input_tokenizer
from lilax._backend.input_pipeline.protos import example_pb2
from lilax._backend.multimodal import processor as mm_processor
from lilax._backend.input_pipeline.conversation import conversational_columns
from lilax._backend.input_pipeline.transforms import hf_tokenization
from lilax._backend.utils import max_logging


@dataclass
class InputRequest:
  prompt: str = ""
  prompt_token_ids: list[int] = field(default_factory=list)
  actual_completion: str = ""
  max_output_tokens: int = 0


def map_to_prompt_completion(example):
  """Separate user and assistant messages into prompt and completion columns."""
  messages = example["messages"]
  example["prompt"] = [message for message in messages if message["role"] == "user"]
  example["completion"] = [message for message in messages if message["role"] == "assistant"]
  return example


def extract_content(example, data_column_names):
  """Extract the content of each message in the selected columns."""
  for column in data_column_names:
    example[column] = [data["content"] for data in example[column]]
  return example


def process_dataset(dataset, data_columns):
  """Preprocess a conversational dataset for distillation data generation."""
  dataset = dataset.select_columns(data_columns)

  supported_columns = [["prompt", "completion"], ["messages"]]
  assert any(
      set(data_columns) == set(supported) for supported in supported_columns
  ), f"Dataset column names mismatch. Expected columns to match one of {supported_columns}, but got {data_columns}"
  assert conversational_columns.is_conversational(dataset.features, data_columns), "Dataset is not in conversational format."

  if "messages" in data_columns:
    dataset = dataset.map(map_to_prompt_completion, remove_columns=data_columns)

  data_column_names = ["prompt", "completion"]
  return dataset.map(extract_content, fn_kwargs={"data_column_names": data_column_names})


def load_dataset(dataset_path, data_split, hf_access_token, dataset_type="huggingface"):
  """Load a dataset from Hugging Face."""
  assert dataset_type == "huggingface", "Only dataset from Hugging Face is supported."
  return datasets.load_dataset(dataset_path, split=data_split, token=hf_access_token)


def filter_dataset(dataset, tokenizer, max_target_length, max_prefill_length, use_chat_template=False):
  """Filter long prompts and create requests for the remaining samples."""
  filtered_dataset = []
  for data in dataset:
    prompt = data["prompt"][0]
    actual_completion = data["completion"][0]

    max_output_length = max_target_length - max_prefill_length
    max_output_tokens = min(max_output_length, len(tokenizer.encode(actual_completion)))
    if use_chat_template:
      message = [{"role": "user", "content": prompt}]
      prompt_token_ids = hf_tokenization.extract_token_ids(
          tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=True)
      )
    else:
      prompt_token_ids = tokenizer.encode(prompt)

    if len(prompt_token_ids) > max_prefill_length:
      continue

    request = InputRequest(prompt, prompt_token_ids, actual_completion, max_output_tokens)
    filtered_dataset.append(request)
  if len(filtered_dataset) < len(dataset):
    max_logging.log("Some prompts are longer than `max-prefill-length` and will be filtered out.")
    max_logging.log(f"Filtering reduced dataset batch from {len(dataset)} to {len(filtered_dataset)} samples.")
  return filtered_dataset
