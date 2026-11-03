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

"""Dequantize GPT-OSS MXFP4 checkpoints."""

import json
import os
from glob import glob

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

MODEL_INDEX_FILE = "model.safetensors.index.json"


def e2m1_to_fp(x: torch.Tensor):
  x = torch.as_tensor(x, dtype=torch.int8)
  sign = 1 - 2 * ((x >> 3) & 0x01)
  exp = 2.0 ** (((x >> 1) & 0x3) - 1)
  is_subnormal = (x & 0b111) == 0b001
  m = torch.where(is_subnormal, 1.0, (1 / 2 * (x & 0x1) + 1).to(torch.float32))
  is_zero = (x & 0b111) == 0
  return torch.where(is_zero, 0.0, m * exp * sign)


def dequantize_mxfp4(blocks_2x_e2m1: torch.Tensor, scales_e8m0: torch.Tensor, data_type: torch.dtype):
  scales_e8m0 = torch.as_tensor(scales_e8m0, dtype=torch.float32)
  scales = (2.0 ** (scales_e8m0 - 127))[..., None, None]
  x = torch.stack([e2m1_to_fp(blocks_2x_e2m1), e2m1_to_fp(blocks_2x_e2m1 >> 4)], -1) * scales
  return x.reshape((x.shape[:-3] + (-1,))).to(data_type)


def main(input_path: str, output_path: str, target_dtype: torch.dtype, cache_size: int):
  torch.set_default_dtype(target_dtype)
  os.makedirs(input_path, exist_ok=True)
  with open(os.path.join(input_path, MODEL_INDEX_FILE), "r", encoding="utf-8") as f:
    model_index = json.load(f)
  weight_map = model_index["weight_map"]
  loaded_files = {}
  fp4_weight_names = []
  block_suffix = "_blocks"
  scale_suffix = "_scales"

  def get_tensor(tensor_name):
    file_name = weight_map[tensor_name]
    if file_name not in loaded_files:
      loaded_files[file_name] = load_file(os.path.join(input_path, file_name), device="cuda")
    return loaded_files[file_name][tensor_name]

  for safetensor_file in tqdm(sorted(glob(os.path.join(input_path, "*.safetensors")))):
    file_name = os.path.basename(safetensor_file)
    current_state_dict = load_file(safetensor_file, device="cuda")
    loaded_files[file_name] = current_state_dict
    new_state_dict = {}
    for weight_name, weight in current_state_dict.items():
      fp4_weight_names.append(weight_name)
      if weight_name.endswith(block_suffix):
        weight_prefix = weight_name.removesuffix(block_suffix)
        scale = get_tensor(weight_prefix + scale_suffix)
        dequantized_weight = dequantize_mxfp4(weight, scale, target_dtype)
        new_state_dict[weight_prefix] = dequantized_weight.permute(0, 2, 1).contiguous()
        print(f"{weight_name}: Dequantized weight type is {dequantized_weight.dtype}")
      elif weight_name.endswith(scale_suffix):
        print(f"{weight_name}: Skip scale conversion as expected")
        continue
      else:
        if weight.dtype != target_dtype:
          weight = weight.to(target_dtype)
          print(f"{weight_name}: Original weight type is {weight.dtype}, and cast to {target_dtype}")
        else:
          print(f"{weight_name}: Original weight type is {weight.dtype}, and no conversion needed")
        new_state_dict[weight_name] = weight
    save_file(new_state_dict, os.path.join(output_path, file_name))
    if len(loaded_files) > cache_size:
      del loaded_files[next(iter(loaded_files))]
      torch.cuda.empty_cache()

  for weight_name in fp4_weight_names:
    if weight_name.endswith(block_suffix):
      file_name = weight_map.pop(weight_name)
      weight_map[weight_name.removesuffix(block_suffix)] = file_name
    elif weight_name.endswith(scale_suffix):
      weight_map.pop(weight_name)
  print(f"weight_map: {weight_map}")
  with open(os.path.join(output_path, MODEL_INDEX_FILE), "w", encoding="utf-8") as f:
    json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)
