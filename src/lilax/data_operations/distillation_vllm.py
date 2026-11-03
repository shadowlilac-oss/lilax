# Copyright 2023–2026 Google LLC
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

"""Generate response-based distillation data using vLLM."""

import os

from datasets import Dataset, load_dataset
import transformers
from vllm import LLM, SamplingParams


def generate_distillation_data(
    *,
    hf_access_token,
    dataset_path="HuggingFaceH4/ultrachat_200k",
    data_split="train_sft",
    data_columns=None,
    use_chat_template=False,
    max_prefill_length=256,
    max_target_length=2048,
    num_generations=1,
    num_prompts=5120,
    output_file=None,
    teacher_model=None,
    tp_size=4,
    max_model_len=4096,
    max_new_tokens=512,
    max_num_batched_tokens=2048,
    max_num_seqs=256,
    gpu_memory_utilization=0.98,
):
  """Generate assistant responses and save them as a Parquet dataset.

  As in the source script, hf_access_token, max_prefill_length, and
  max_target_length are accepted but not used.
  """
  prompt_column = data_columns[0] if data_columns else "messages"
  if output_file is None:
    output_file = os.path.join(os.environ.get("BASE_DIRECTORY", "."), "datasets", "distillation_data.parquet")
  if teacher_model is None:
    teacher_model = os.path.join(os.environ.get("BASE_DIRECTORY", "."), "qwen3-32b")

  def apply_chat_template(example):
    messages = example[prompt_column]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"formatted_prompt": prompt}

  print(f"Loading dataset {dataset_path} ({data_split})...")
  dataset = load_dataset(dataset_path, split=data_split)
  dataset = dataset.select(range(min(len(dataset), num_prompts)))

  print(f"Loading tokenizer {teacher_model}...")
  tokenizer = transformers.AutoTokenizer.from_pretrained(teacher_model)

  if use_chat_template:
    print("Formatting prompts...")
    dataset = dataset.map(apply_chat_template, desc="Applying chat template")
    prompts = dataset["formatted_prompt"]
  else:
    prompts = dataset[prompt_column]

  print(f"Initializing vLLM with model {teacher_model}...")
  llm = LLM(
      model=teacher_model,
      max_model_len=max_model_len,
      tensor_parallel_size=tp_size,
      max_num_batched_tokens=max_num_batched_tokens,
      max_num_seqs=max_num_seqs,
      gpu_memory_utilization=gpu_memory_utilization,
      enforce_eager=False,
  )
  sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=max_new_tokens, n=num_generations)

  print("Running inference...")
  outputs = llm.generate(prompts, sampling_params)

  results = []
  for output, original_item in zip(outputs, dataset):
    for completion in output.outputs:
      msgs = list(original_item[prompt_column])
      msgs.append({"role": "assistant", "content": completion.text})
      results.append({"messages": msgs})

  os.makedirs(os.path.dirname(output_file), exist_ok=True)
  print(f"Saving results to {output_file} (Parquet)")
  ds = Dataset.from_list(results)
  ds.to_parquet(output_file)
