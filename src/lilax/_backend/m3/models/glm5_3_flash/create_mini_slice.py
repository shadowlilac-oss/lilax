"""Helper to extract mini-slice safetensors keys from index weight map."""

from collections.abc import Mapping
import json
import os
import shutil
from lilax._backend.m3.configs import glm5_3_flash_config as glm5_config
import safetensors.torch
import torch

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
)


def _normalize_key(key: str) -> str:
  prefix = "model.language_model."
  if key.startswith(prefix):
    return key[len(prefix) :]
  if key == "lm_head.weight" or key.startswith("lm_head."):
    return key
  raise ValueError(f"Unknown HF key prefix: {key!r}")


def extract_mini_slice_keys(
    weight_map: Mapping[str, str],
    num_hidden_layers: int,
    *,
    include_mtp_layer: int | None,
) -> dict[str, str]:
  """Filter the weight map for mini-slice layers and required model weights."""
  if num_hidden_layers <= 0:
    raise ValueError(f"num_hidden_layers must be positive, got {num_hidden_layers}.")
  if include_mtp_layer is not None and include_mtp_layer != glm5_config.MTP_LAYER_INDEX:
    raise ValueError(f"include_mtp_layer must be {glm5_config.MTP_LAYER_INDEX}, got" f" {include_mtp_layer}.")
  target_layers = set(range(num_hidden_layers))
  if include_mtp_layer is not None:
    target_layers.add(include_mtp_layer)
  counts = {layer: 0 for layer in target_layers}
  selected: dict[str, str] = {}
  for key, shard in weight_map.items():
    if key.startswith("model.visual.") or key.startswith("visual."):
      continue
    clean_key = _normalize_key(key)
    if any(clean_key.startswith(prefix) for prefix in ("embed_tokens.", "norm.", "lm_head.")):
      selected[key] = shard
    elif clean_key.startswith("layers."):
      layer = int(clean_key.split(".")[1])
      if layer in counts:
        selected[key] = shard
        counts[layer] += 1
    else:
      raise ValueError(f"Unrecognized HF key in weight_map: {key!r}")
  for required in ("embed_tokens.", "norm.", "lm_head."):
    if not any(_normalize_key(key).startswith(required) for key in selected):
      raise ValueError(f"No {required[:-1]} keys found in weight_map.")
  for layer, count in counts.items():
    if count == 0:
      raise ValueError(f"Required layer {layer} has zero matching keys in weight_map.")
  return selected


def write_mini_slice_checkpoint(
    src_hf_dir: str,
    dst_hf_dir: str,
    *,
    num_hidden_layers: int,
    include_mtp_layer: int | None,
) -> dict[str, str]:
  """Extract mini-slice layers from a multi-shard HF checkpoint."""
  index_path = os.path.join(src_hf_dir, "model.safetensors.index.json")
  if not os.path.exists(index_path):
    raise FileNotFoundError(f"Missing index file in {src_hf_dir}: {index_path}")
  with open(index_path, "r", encoding="utf-8") as file:
    weight_map = json.load(file)["weight_map"]

  selected = extract_mini_slice_keys(weight_map, num_hidden_layers, include_mtp_layer=include_mtp_layer)
  os.makedirs(dst_hf_dir, exist_ok=True)
  tensors: dict[str, torch.Tensor] = {}
  for shard in sorted(set(selected.values())):
    with safetensors.safe_open(os.path.join(src_hf_dir, shard), framework="pt") as file:
      for key in file.keys():
        if key in selected:
          tensors[key] = file.get_tensor(key)
  if missing := set(selected) - set(tensors):
    raise ValueError(f"Selected keys missing from safetensors shards: {missing}")
  safetensors.torch.save_file(tensors, os.path.join(dst_hf_dir, "model.safetensors"))
  sliced_map = {key: "model.safetensors" for key in sorted(tensors)}
  metadata = {"total_size": sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())}
  with open(os.path.join(dst_hf_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as file:
    json.dump({"metadata": metadata, "weight_map": sliced_map}, file, indent=2)
  for filename in TOKENIZER_FILES:
    src_file = os.path.join(src_hf_dir, filename)
    if os.path.exists(src_file):
      shutil.copy2(src_file, os.path.join(dst_hf_dir, filename))

  if os.path.exists(config_path := os.path.join(src_hf_dir, "config.json")):
    with open(config_path, "r", encoding="utf-8") as file:
      config_data = json.load(file)
    num_nextn = 1 if include_mtp_layer is not None else 0
    config_data["num_hidden_layers"] = num_hidden_layers
    config_data["num_nextn_predict_layers"] = num_nextn
    if "text_config" in config_data and isinstance(config_data["text_config"], dict):
      config_data["text_config"]["num_hidden_layers"] = num_hidden_layers
      config_data["text_config"]["num_nextn_predict_layers"] = num_nextn
      for attr in ("layer_types", "mlp_layer_types", "indexer_types"):
        if attr in config_data["text_config"] and isinstance(config_data["text_config"][attr], list):
          config_data["text_config"][attr] = config_data["text_config"][attr][:num_hidden_layers]
      for attr in ("kda_layers", "full_attn_layers"):
        if attr in config_data["text_config"] and isinstance(config_data["text_config"][attr], list):
          config_data["text_config"][attr] = [item for item in config_data["text_config"][attr] if item < num_hidden_layers]
      if "linear_attn_config" in config_data["text_config"] and isinstance(config_data["text_config"]["linear_attn_config"], dict):
        linear_config = config_data["text_config"]["linear_attn_config"]
        if "kda_layers" in linear_config and isinstance(linear_config["kda_layers"], list):
          linear_config["kda_layers"] = [index for index in linear_config["kda_layers"] if index < num_hidden_layers]
        if "full_attn_layers" in linear_config and isinstance(linear_config["full_attn_layers"], list):
          linear_config["full_attn_layers"] = [index for index in linear_config["full_attn_layers"] if index < num_hidden_layers]
    for attr in ("layer_types", "mlp_layer_types", "indexer_types"):
      if attr in config_data and isinstance(config_data[attr], list):
        config_data[attr] = config_data[attr][:num_hidden_layers]
    for attr in ("kda_layers", "full_attn_layers"):
      if attr in config_data and isinstance(config_data[attr], list):
        config_data[attr] = [item for item in config_data[attr] if item < num_hidden_layers]
    if "linear_attn_config" in config_data and isinstance(config_data["linear_attn_config"], dict):
      linear_config = config_data["linear_attn_config"]
      if "kda_layers" in linear_config and isinstance(linear_config["kda_layers"], list):
        linear_config["kda_layers"] = [index for index in linear_config["kda_layers"] if index < num_hidden_layers]
      if "full_attn_layers" in linear_config and isinstance(linear_config["full_attn_layers"], list):
        linear_config["full_attn_layers"] = [index for index in linear_config["full_attn_layers"] if index < num_hidden_layers]
    with open(os.path.join(dst_hf_dir, "config.json"), "w", encoding="utf-8") as file:
      json.dump(config_data, file, indent=2)
  return sliced_map
