"""Checkpoint conversion utility functions."""

import contextlib
import gc
import io
import json
import logging
import os
import pathlib
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import psutil
from etils import epath
from flax.training import train_state
from google.cloud import storage
from google.cloud.storage import transfer_manager
from huggingface_hub import HfApi, repo_exists, snapshot_download
from jax import tree
from jax.experimental import multihost_utils
from jaxtyping import Array
from safetensors import safe_open
from safetensors.flax import save as save_flax_to_bytes
from safetensors.numpy import save as numpy_save
from safetensors.numpy import save_file as numpy_save_file
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, Qwen3OmniMoeForConditionalGeneration
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from lilax._backend.checkpoint_conversion.utils._memory_monitor import MemoryMonitorTqdm, print_peak_memory, print_ram_usage
from lilax._backend.checkpoint_conversion.utils.tensor_handling import nesting_depth, stacked_axes
from lilax._backend.common.checkpoints import checkpointing
from lilax._backend.observability import max_logging

SAFE_TENSORS_CONFIG_FILE = "config.json"
SAFE_TENSORS_WEIGHTS_FILE = "model.safetensors"
SAFE_TENSORS_INDEX_FILE = "model.safetensors.index.json"
DEFAULT_MAX_SHARD_SIZE = 1024 * 1024 * 1024 * 3


def _get_local_directory(output_dir: str) -> str:
  if output_dir.startswith("gs://") or output_dir.startswith("hf://"):
    local_dir = os.path.join(os.path.expanduser("~"), ".cache", "maxtext_hf_conversion_temp", "temp_files")
  else:
    local_dir = output_dir
  os.makedirs(local_dir, exist_ok=True)
  return local_dir


def validate_and_filter_param_map_keys(param_map_keys, maxtext_state_keys):
  flattened_map_keys = set()
  for key in param_map_keys:
    if isinstance(key, tuple):
      flattened_map_keys.update(key)
    else:
      flattened_map_keys.add(key)
  missing_keys = maxtext_state_keys - flattened_map_keys
  if missing_keys:
    hint = ""
    ckpt_has_scanned = any("scanned_blocks" in key for key in missing_keys)
    map_has_scanned = any("scanned_blocks" in key for key in flattened_map_keys)
    if ckpt_has_scanned and not map_has_scanned:
      hint = "\nHint: checkpoint keys contain 'scanned_blocks' but param_map does not — try scan_layers=True."
    elif map_has_scanned and not ckpt_has_scanned:
      hint = "\nHint: param_map contains 'scanned_blocks' keys but checkpoint does not — try scan_layers=False."
    raise ValueError(
        "maxtext_state_dict must be a subset of flattened param_map"
        + f"\nparam map\n{param_map_keys}"
        + f"\nmaxtext:\n{maxtext_state_keys}"
        + f"\nmissing keys:\n{missing_keys}"
        + hint
    )
  extra_keys = flattened_map_keys - maxtext_state_keys
  if extra_keys:
    max_logging.log(f"Warning: extra keys in param_map are skipped: {extra_keys}")
  filtered_map_keys = []
  for key in param_map_keys:
    if (isinstance(key, str) and key in maxtext_state_keys) or (
        isinstance(key, tuple) and all(part in maxtext_state_keys for part in key)
    ):
      filtered_map_keys.append(key)
  return filtered_map_keys


def apply_hook_fns(weight, target_shape, hook_fns):
  if hook_fns is None:
    return weight
  if not isinstance(hook_fns, list):
    hook_fns = [hook_fns]
  for hook_fn in hook_fns:
    weight = hook_fn(weight, target_shape)
  return weight


def convert_jax_weight_to_numpy(weight: "jax.Array", dtype_str: None | str = None) -> np.ndarray:
  final_dtype_str = str(weight.dtype) if dtype_str is None else dtype_str
  expected_shape = weight.shape
  if str(weight.dtype) != final_dtype_str:
    weight = weight.astype(final_dtype_str)
  weight = multihost_utils.process_allgather(weight)
  return np.asarray(weight).reshape(expected_shape)


def _process(hf_path, processed_slice, output_weights, current_hook_fns, hf_shape_map, save_dtype):
  if isinstance(hf_path, tuple):
    if hf_path not in hf_shape_map:
      raise ValueError(f"HF path tuple '{hf_path}' not found in hf_shape_map.")
    target_hf_shapes = hf_shape_map[hf_path]
    if current_hook_fns:
      processed_slice = apply_hook_fns(processed_slice, target_hf_shapes, current_hook_fns)
    for path, single_slice, shape in zip(hf_path, processed_slice, target_hf_shapes):
      numpy_slice = convert_jax_weight_to_numpy(single_slice, save_dtype).reshape(shape)
      output_weights.append((path, numpy_slice))
  else:
    if hf_path not in hf_shape_map:
      raise ValueError(f"HF path '{hf_path}' not found in hf_shape_map.")
    target_hf_shape = hf_shape_map[hf_path]
    if current_hook_fns:
      processed_slice = apply_hook_fns(processed_slice, target_hf_shape, current_hook_fns)
    numpy_slice = convert_jax_weight_to_numpy(processed_slice, save_dtype).reshape(target_hf_shape)
    output_weights.append((hf_path, numpy_slice))


def process_maxtext_param(
    maxtext_param_key: str | tuple[str, ...],
    maxtext_param_weight: jax.Array | list[jax.Array],
    param_map: dict[str, Any],
    hook_fn_map: dict[str, Any],
    hf_shape_map: dict[str, Any],
    maxtext_config: Any,
) -> list[tuple[str, np.ndarray]]:
  max_logging.log(f"maxtext param: {maxtext_param_key}")
  if maxtext_param_key not in param_map:
    raise ValueError(f"MaxText param key '{maxtext_param_key}' not found in param_map.")
  hf_target_paths = param_map[maxtext_param_key]
  if hf_target_paths is None:
    max_logging.log(f"\tskipping parameter mapped to None: {maxtext_param_key}")
    return []
  if not hf_target_paths:
    raise ValueError(f"No HF target paths found for MaxText key '{maxtext_param_key}'")
  current_hook_fns = hook_fn_map.get(maxtext_param_key)
  output_weights = []
  if not isinstance(hf_target_paths, list):
    max_logging.log("\tunscan")
    _process(hf_target_paths, maxtext_param_weight, output_weights, current_hook_fns, hf_shape_map,
             save_dtype=maxtext_config.weight_dtype)
    return output_weights
  if not isinstance(hf_target_paths[0], list):
    if maxtext_config.scan_layers:
      max_logging.log("\tscan")
      key_str = maxtext_param_key[0] if isinstance(maxtext_param_key, tuple) else maxtext_param_key
      is_var_collection = "MoEBiasVar" in key_str or "Tid2EidVar" in key_str
      is_unscanned_moe = "MoeBlock" in key_str and not any(
          part in key_str for part in ("scanned_blocks", "moe_layers-", "decoder-layers-")
      ) and not key_str.startswith("params-decoder-layers_remainder-local_layers-")
      weight_sample = maxtext_param_weight[0] if isinstance(maxtext_param_weight, list) else maxtext_param_weight
      if is_var_collection or is_unscanned_moe:
        axis_to_slice = 0
      else:
        axis_to_slice = (
            maxtext_config.param_scan_axis if getattr(weight_sample, "ndim", 0) > maxtext_config.param_scan_axis else 0
        )
    else:
      max_logging.log("\tunscan moe")
      axis_to_slice = 0
    for index, hf_path in enumerate(hf_target_paths):
      if isinstance(maxtext_param_weight, list):
        weight_slice = [jax.lax.index_in_dim(value, index, axis=axis_to_slice, keepdims=False)
                        for value in maxtext_param_weight]
      else:
        weight_slice = jax.lax.index_in_dim(maxtext_param_weight, index, axis=axis_to_slice, keepdims=False)
      _process(hf_path, weight_slice, output_weights, current_hook_fns, hf_shape_map,
               save_dtype=maxtext_config.weight_dtype)
    return output_weights
  key_str = maxtext_param_key[0] if isinstance(maxtext_param_key, tuple) else maxtext_param_key
  depth = nesting_depth(hf_target_paths)
  axes = stacked_axes(key_str, maxtext_config, depth)
  max_logging.log("\tscan nested local" if axes != tuple(range(depth)) else "\tscan moe")

  def _emit_slices(weight, hf_paths, level):
    if level == depth:
      _process(hf_paths, weight, output_weights, current_hook_fns, hf_shape_map, save_dtype=maxtext_config.weight_dtype)
      return
    axis_to_slice = axes[level] - level
    for index, sub_paths in enumerate(hf_paths):
      if isinstance(weight, list):
        weight_slice = [jax.lax.index_in_dim(value, index, axis=axis_to_slice, keepdims=False) for value in weight]
      else:
        weight_slice = jax.lax.index_in_dim(weight, index, axis=axis_to_slice, keepdims=False)
      _emit_slices(weight_slice, sub_paths, level + 1)

  _emit_slices(maxtext_param_weight, hf_target_paths, 0)
  return output_weights


def create_huggingface_hub_repo_if_not_exist(repo_id, repo_type):
  if not repo_exists(repo_id, repo_type=repo_type):
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, private=True)
    max_logging.log(f"\n Created new HuggingFace Hub {repo_type} repo: {repo_id}.")


def save_config_file(config, local_path_to_save_to: str, output_dir_final: str, file_name: str,
                     remove_local_copy_after_upload: bool = False):
  if jax.process_index() == 0:
    if config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
      config.architectures = [MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]]
    if output_dir_final.startswith("hf://"):
      max_logging.log(f"  Serializing {file_name} to memory for Hugging Face Hub upload...")
      json_bytes = config.to_json_string().encode("utf-8")
      repo_id = output_dir_final.lstrip("hf://")
      api = HfApi()
      with io.BytesIO(json_bytes) as file_handle:
        api.upload_file(path_or_fileobj=file_handle, path_in_repo=file_name, repo_id=repo_id, repo_type="model")
      max_logging.log(f"  Successfully uploaded {file_name} to HF repo: {repo_id}")
    else:
      actual_local_file_path = os.path.join(local_path_to_save_to, file_name)
      config.to_json_file(actual_local_file_path)
      max_logging.log(f"   Saved {file_name} to {actual_local_file_path}")
      if output_dir_final.startswith("gs://"):
        upload_file_to_gcs(actual_local_file_path, os.path.join(output_dir_final, file_name),
                           remove_local_file_after_upload=remove_local_copy_after_upload)


def shard_checkpoint(weights_dict: dict[str, Array], max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
                     weights_name: str = "model.safetensors") -> tuple[dict[str, dict[str, Array]], None | dict]:
  current_shard: dict[str, Array] = {}
  shards: list[dict[str, Array]] = [current_shard]
  current_size = 0
  total_size = 0
  for key, tensor in sorted(weights_dict.items()):
    weight_size = tensor.size * tensor.itemsize
    if (current_size + weight_size > max_shard_size) and len(current_shard.items()):
      current_shard = {}
      shards.append(current_shard)
      current_size = 0
    current_shard[key] = tensor
    current_size += weight_size
    total_size += weight_size
  if len(shards) == 1:
    return {weights_name: shards[0]}, None
  shard_dict = {}
  weight_map = {}
  for index, shard in enumerate(shards, 1):
    shard_name = weights_name.replace(".safetensors", f"-{index:05d}-of-{len(shards):05d}.safetensors")
    shard_dict[shard_name] = shard
    for key in shard:
      weight_map[key] = shard_name
  return shard_dict, {"metadata": {"total_size": total_size}, "weight_map": weight_map}


def save_safetensor_file(state_dict, local_dir_to_save_to: str, output_dir_final: str, file_name: str):
  if jax.process_index() == 0:
    state_dict = {key: value for key, value in state_dict.items() if value is not None}
    if "model.safetensors" in state_dict and isinstance(state_dict["model.safetensors"], dict):
      state_dict = state_dict["model.safetensors"]
    for key in list(state_dict.keys()):
      value = state_dict[key]
      if isinstance(value, np.ndarray) and not value.flags["C_CONTIGUOUS"]:
        state_dict[key] = np.ascontiguousarray(value)
    if output_dir_final.startswith("gs://"):
      local_path = os.path.join(local_dir_to_save_to, file_name)
      numpy_save_file(state_dict, local_path, metadata={"format": "pt"})
      state_dict.clear()
      gc.collect()
      cloud_path = os.path.join(output_dir_final, file_name)
      upload_file_to_gcs(local_file=local_path, gs_bucket_path=cloud_path, remove_local_file_after_upload=True)
    elif output_dir_final.startswith("hf://"):
      max_logging.log(f"  Serializing {file_name} to memory for Hugging Face Hub upload...")
      serialized_content = save_flax_to_bytes(state_dict, metadata={"format": "pt"})
      repo_id = output_dir_final.lstrip("hf://")
      api = HfApi()
      with io.BytesIO(serialized_content) as file_handle:
        api.upload_file(path_or_fileobj=file_handle, path_in_repo=file_name, repo_id=repo_id, repo_type="model")
      max_logging.log(f"  Successfully uploaded {file_name} to HF repo: {repo_id}")
    else:
      local_path = os.path.join(local_dir_to_save_to, file_name)
      numpy_save_file(state_dict, local_path, metadata={"format": "pt"})
      max_logging.log(f"   Saved {file_name} to {local_path}")


def save_index_file(index: dict, local_dir_to_save_to: str, output_dir_final: str, file_name: str,
                    remove_local_copy_after_upload: bool = False):
  if jax.process_index() == 0:
    local_path = os.path.join(local_dir_to_save_to, file_name)
    if output_dir_final.startswith("hf://"):
      max_logging.log(f"   Serialized {file_name} to memory for Hugging Face Hub upload.")
      json_bytes = json.dumps(index, indent=2).encode("utf-8")
      repo_id = output_dir_final.lstrip("hf://")
      api = HfApi()
      with io.BytesIO(json_bytes) as file_handle:
        api.upload_file(path_or_fileobj=file_handle, path_in_repo=file_name, repo_id=repo_id, repo_type="model")
      max_logging.log(f"   Successfully uploaded {file_name} to HF repo: {repo_id}")
    else:
      with open(local_path, "wt", encoding="utf8") as file_handle:
        json.dump(index, file_handle, indent=2)
      max_logging.log(f"   Saved {file_name} to {local_path}")
      if output_dir_final.startswith("gs://"):
        upload_file_to_gcs(local_path, os.path.join(output_dir_final, file_name),
                           remove_local_file_after_upload=remove_local_copy_after_upload)
        max_logging.log(f"   Successfully uploaded {file_name} to GCS: {output_dir_final}")


def save_weight_files(shards, index, local_dir_to_save_to: str, output_dir_final: str, parallel_threads=4,
                      remove_local_copy_after_upload: bool = False):
  if index is None:
    save_safetensor_file(shards, local_dir_to_save_to, output_dir_final, SAFE_TENSORS_WEIGHTS_FILE)
  else:
    if output_dir_final.startswith("gs://"):
      def _save_and_upload(shard_name, shard_dict):
        local_path = os.path.join(local_dir_to_save_to, shard_name)
        numpy_save_file(shard_dict, local_path, metadata={"format": "pt"})
        shard_dict.clear()
        del shard_dict
        gc.collect()
        cloud_path = os.path.join(output_dir_final, shard_name)
        upload_file_to_gcs(local_file=local_path, gs_bucket_path=cloud_path, remove_local_file_after_upload=True)

      with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
        futures = []
        for shard_name in list(shards.keys()):
          shard_dict = shards.pop(shard_name)
          if "model.safetensors" in shard_dict and isinstance(shard_dict["model.safetensors"], dict):
            shard_dict = shard_dict["model.safetensors"]
          shard_dict = {key: value for key, value in shard_dict.items() if value is not None}
          futures.append(executor.submit(_save_and_upload, shard_name, shard_dict))
        for future in futures:
          future.result()
    else:
      for shard_name in list(shards.keys()):
        shard_dict = shards.pop(shard_name)
        save_safetensor_file(shard_dict, local_dir_to_save_to, output_dir_final, shard_name)
        shard_dict.clear()
        gc.collect()
    save_index_file(index, local_dir_to_save_to, output_dir_final, SAFE_TENSORS_INDEX_FILE,
                    remove_local_copy_after_upload)


@contextlib.contextmanager
def get_local_save_path_manager(output_dir: str):
  if output_dir.startswith("gs://") or output_dir.startswith("hf://"):
    with tempfile.TemporaryDirectory(prefix="maxtext_hf_save_") as temp_dir:
      max_logging.log(f"   Using temporary local staging directory: {temp_dir}")
      yield temp_dir, True
  else:
    os.makedirs(output_dir, exist_ok=True)
    max_logging.log(f"   Using local directory: {output_dir}")
    yield output_dir, False


def save_model_files(weight_arrays: dict, config, tokenizer: None | Any, processor, output_dir: str, parallel_threads=4):
  if output_dir.startswith("hf://"):
    create_huggingface_hub_repo_if_not_exist(repo_id=output_dir.lstrip("hf://"), repo_type="model")
    repo_id = output_dir.lstrip("hf://")
  else:
    repo_id = None
  max_logging.log(f"\n-> Saving model and tokenizer (if provided) to {output_dir}...")
  with get_local_save_path_manager(output_dir) as (current_save_path, is_temp_path):
    remove_local_copy = is_temp_path
    if jax.process_index() == 0:
      files_to_upload = []
      if processor is not None:
        max_logging.log(f"    Saving image processor files to {current_save_path}...")
        saved_image_processor_files = processor.save_pretrained(current_save_path)
        max_logging.log(f"    Processor files saved locally: {saved_image_processor_files}")
      elif tokenizer is not None:
        max_logging.log(f"    Saving tokenizer files to {current_save_path}...")
        saved_tokenizer_files = tokenizer.save_pretrained(current_save_path)
        max_logging.log(f"    Tokenizer files saved locally: {saved_tokenizer_files}")
      files_to_upload = [os.path.join(current_save_path, name) for name in os.listdir(current_save_path)]
      if output_dir.startswith("gs://"):
        for local_file_path in files_to_upload:
          if not os.path.exists(local_file_path):
            max_logging.log(f"   Warning: Tokenizer file {local_file_path} not found locally. Skipping upload to GCS.")
            continue
          file_name = os.path.basename(local_file_path)
          upload_file_to_gcs(local_file_path, os.path.join(output_dir, file_name),
                             remove_local_file_after_upload=remove_local_copy)
      elif output_dir.startswith("hf://") and repo_id:
        api = HfApi()
        for local_file_path in files_to_upload:
          if not os.path.exists(local_file_path):
            max_logging.log(f"   Warning: Tokenizer file {local_file_path} not found locally. Skipping upload to HF Hub.")
            continue
          file_name = os.path.basename(local_file_path)
          api.upload_file(path_or_fileobj=local_file_path, path_in_repo=file_name, repo_id=repo_id, repo_type="model")
          if remove_local_copy:
            os.remove(local_file_path)
            max_logging.log(f"   Removed local copy: {local_file_path}")
      save_config_file(config, current_save_path, output_dir, SAFE_TENSORS_CONFIG_FILE, remove_local_copy)
    shards, index = shard_checkpoint(weight_arrays)
    weight_arrays.clear()
    gc.collect()
    save_weight_files(shards, index, current_save_path, output_dir, parallel_threads, remove_local_copy)
  if jax.process_index() == 0:
    max_logging.log(f"✅ Model and tokenizer (if provided) successfully processed for {output_dir}")


def upload_state_dict_to_gcs(state_dict: dict, gs_bucket_path: str):
  gs_bucket_path = gs_bucket_path.removeprefix("gs://")
  bucket_name, *blob_path_parts = gs_bucket_path.split("/")
  blob_name = "/".join(blob_path_parts)
  data = numpy_save(state_dict, metadata={"format": "pt"})
  buffer = io.BytesIO(data)
  buffer.seek(0)
  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(blob_name)
  print(f"-> Uploading in-memory state_dict to {gs_bucket_path}...")
  blob.upload_from_file(buffer, content_type="application/octet-stream", timeout=600)
  print(f"✅ Uploaded to {bucket.name}/{blob_name}")


def upload_file_to_gcs(local_file: str, gs_bucket_path: str, remove_local_file_after_upload=False):
  gs_bucket_path = gs_bucket_path.removeprefix("gs://")
  bucket_name = gs_bucket_path.split("/")[0]
  blob_name = gs_bucket_path[len(bucket_name):].lstrip("/")
  max_logging.log(f"-> Uploading {local_file} to {gs_bucket_path}...")
  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(blob_name)
  blob.upload_from_filename(local_file, timeout=600)
  max_logging.log(f"✅ Uploaded {local_file} to {bucket.name}/{blob_name}")
  if remove_local_file_after_upload:
    os.remove(local_file)
    max_logging.log(f"✅ Deleted {local_file}")


def upload_folder_to_gcs(local_folder: str, gs_bucket_path: str, num_workers: int = 4):
  start_time = time.time()
  gs_bucket_path = gs_bucket_path.removeprefix("gs://")
  bucket_name = gs_bucket_path.split("/")[0]
  destination_dir = gs_bucket_path[len(bucket_name):]
  if destination_dir.startswith("/"):
    destination_dir = destination_dir[1:]
  if destination_dir != "" and not destination_dir.endswith("/"):
    destination_dir += "/"
  files_in_local_folder = os.listdir(local_folder)
  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)
  results = transfer_manager.upload_many_from_filenames(
      bucket, files_in_local_folder, source_directory=local_folder, max_workers=num_workers,
      blob_name_prefix=destination_dir, timeout=600, deadline=None,
  )
  for name, result in zip(files_in_local_folder, results):
    if isinstance(result, Exception):
      max_logging.log(f"Failed to upload {name}: {result}")
    else:
      max_logging.log(f"✅ Uploaded {name} to {bucket.name}/{destination_dir}{name}")
  max_logging.log(f"Upload completed in {time.time() - start_time}s")


def _recursive_update(d: dict, u: dict) -> dict:
  for key, value in u.items():
    if isinstance(value, dict) and isinstance(d.get(key), dict):
      _recursive_update(d[key], value)
    else:
      d[key] = value
  return d


def load_orbax_checkpoint(config) -> dict:
  ckptr = ocp.Checkpointer(
      ocp.PyTreeCheckpointHandler(
          restore_concurrent_gb=config.checkpoint_storage_concurrent_gb,
          use_ocdbt=config.checkpoint_storage_use_ocdbt,
          use_zarr3=config.checkpoint_storage_use_zarr3,
      )
  )
  devices = np.array(jax.devices()).reshape((-1,))
  single_device_mesh = jax.sharding.Mesh(devices, ("x",))

  def create_restore_args(tree_metadata):
    if hasattr(tree_metadata, "shape"):
      restore_dtype = None
      if hasattr(tree_metadata, "dtype") and tree_metadata.dtype is not None:
        try:
          orig_dtype = jnp.dtype(tree_metadata.dtype)
          if jnp.issubdtype(orig_dtype, jnp.floating) and getattr(config, "weight_dtype", None) is not None:
            restore_dtype = jnp.dtype(config.weight_dtype)
          else:
            restore_dtype = orig_dtype
        except (TypeError, ValueError):
          restore_dtype = None
      return ocp.ArrayRestoreArgs(
          sharding=jax.sharding.NamedSharding(single_device_mesh, jax.sharding.PartitionSpec()), dtype=restore_dtype,
      )
    elif isinstance(tree_metadata, dict):
      return {key: create_restore_args(value) for key, value in tree_metadata.items()}
    else:
      return None

  lora_path = config.lora.lora_restore_path
  paths = [path for path in [config.load_parameters_path, lora_path] if path]
  merged_dict = {}
  for index, path in enumerate(paths):
    checkpoint_path = epath.Path(path)
    metadata = ckptr.metadata(checkpoint_path)
    checkpoint_tree = metadata.item_metadata.tree
    if isinstance(checkpoint_tree, dict):
      if "params" in checkpoint_tree:
        checkpoint_tree = {"params": checkpoint_tree["params"]}
        max_logging.log(f"Filtering checkpoint to only load 'params' from {path}")
      else:
        filtered_tree = {key: value for key, value in checkpoint_tree.items() if key not in ("opt_state", "optimizer")}
        if len(filtered_tree) < len(checkpoint_tree):
          checkpoint_tree = filtered_tree
          max_logging.log(f"Filtering checkpoint to exclude optimizer keys from {path}")
    restore_args = jax.tree_util.tree_map(
        lambda value: create_restore_args(value) if hasattr(value, "shape") else None,
        checkpoint_tree, is_leaf=lambda value: hasattr(value, "shape"),
    )
    restored = ckptr.restore(
        checkpoint_path, args=ocp.args.PyTreeRestore(item=checkpoint_tree, restore_args=restore_args, partial_restore=True),
    )
    if index == 0:
      merged_dict = restored
    else:
      _recursive_update(merged_dict, restored)
  return merged_dict


def save_adapter_files(output_dir, weights, config, found_modules, model_id):
  """Save HF LoRA adapter weights and configuration."""
  os.makedirs(output_dir, exist_ok=True)
  adapter_file = os.path.join(output_dir, "adapter_model.safetensors")
  numpy_save_file(weights, adapter_file)
  adapter_config = {
      "base_model_name_or_path": model_id,
      "peft_type": "LORA",
      "task_type": "CAUSAL_LM",
      "r": config.lora.lora_rank,
      "lora_alpha": config.lora.lora_alpha,
      "target_modules": list(found_modules),
      "lora_dropout": 0.0,
      "bias": "none",
      "inference_mode": True,
  }
  config_file = os.path.join(output_dir, "adapter_config.json")
  with open(config_file, "w", encoding="utf-8") as config_file_handle:
    json.dump(adapter_config, config_file_handle, indent=4)


def param_key_parts_from_path(path_tuple) -> list[str]:
  parts: list[str] = []
  for entry in path_tuple:
    key = getattr(entry, "key", getattr(entry, "idx", getattr(entry, "name", entry)))
    if (isinstance(key, int) or (isinstance(key, str) and key.isdigit())) and parts:
      parts[-1] = f"{parts[-1]}_{key}"
    else:
      parts.append(str(key))
  if parts and parts[-1] == "value":
    parts.pop()
  return parts


def extract_nnx_weights(weights_dict: dict) -> dict[str, np.ndarray]:
  result = {}
  leaves_with_paths = jax.tree_util.tree_leaves_with_path(weights_dict)
  for path_tuple, leaf_value in leaves_with_paths:
    path_keys = param_key_parts_from_path(path_tuple)
    if "to_nnx__rngs" in path_keys or any(key == "rngs" or key.endswith("_rngs") for key in path_keys):
      continue
    maxtext_param_key = "params-" + "-".join(path_keys)
    if not isinstance(leaf_value, (jax.Array, np.ndarray)):
      raise ValueError(f"Leaf value for {maxtext_param_key} is not an array. Type: {type(leaf_value)}.")
    result[maxtext_param_key] = leaf_value
  return result


def extract_linen_weights(weights_dict: dict, prefix: str = "params-") -> dict[str, np.ndarray]:
  result = {}
  leaves_with_paths = jax.tree_util.tree_leaves_with_path(weights_dict)
  for path_tuple, leaf_value in leaves_with_paths:
    path_keys = param_key_parts_from_path(path_tuple)
    maxtext_param_key = prefix + "-".join(path_keys)
    if not isinstance(leaf_value, (jax.Array, np.ndarray)):
      raise ValueError(f"Leaf value for {maxtext_param_key} is not an array. Type: {type(leaf_value)}.")
    result[maxtext_param_key] = leaf_value
  return result


def detect_and_extract_checkpoint(checkpoint_dict: dict) -> dict[str, np.ndarray]:
  actual_weights_dict = checkpoint_dict.get("params")
  if actual_weights_dict is None:
    if "base" in checkpoint_dict and isinstance(checkpoint_dict["base"], dict):
      max_logging.log("Detected NNX-RL checkpoint structure (with 'base' wrapper)")
      return extract_nnx_weights(checkpoint_dict["base"])
    else:
      max_logging.log("Detected NNX-SFT checkpoint structure")
      return extract_nnx_weights(checkpoint_dict)
  else:
    if isinstance(actual_weights_dict, dict) and "params" in actual_weights_dict:
      extra_collections = [key for key in actual_weights_dict if key != "params"]
      if extra_collections:
        max_logging.log(
            f"Detected multi-collection Linen checkpoint structure with collections: {list(actual_weights_dict.keys())}"
        )
        result = {}
        for col_name, col_dict in actual_weights_dict.items():
          result.update(extract_linen_weights(col_dict, prefix=f"{col_name}-"))
        return result
      else:
        actual_weights_dict = actual_weights_dict["params"]
        max_logging.log("Detected Linen checkpoint structure")
    else:
      max_logging.log("Detected Linen checkpoint structure (single params layer)")
    return extract_linen_weights(actual_weights_dict)


def load_hf_dict_from_transformers(model_id: str, token: str, revision: str | None = None, dtype: str = "auto"):
  if model_id in ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]:
    model_classes = [Qwen3OmniMoeForConditionalGeneration]
  else:
    model_classes = [AutoModelForCausalLM, AutoModel]
  last_exception = None
  for model_class in model_classes:
    try:
      hf_model = model_class.from_pretrained(model_id, token=token, revision=revision, dtype=dtype)
      break
    except (ValueError, OSError, RuntimeError, ImportError) as error:
      max_logging.log(f"Failed to load using {model_class.__name__}: {error!r}")
      last_exception = error
  else:
    error_message = (
        f"Could not load HuggingFace model {model_id} with any of the attempted classes. " f"Last error: {last_exception}"
    )
    raise ValueError(error_message) from last_exception
  return hf_model.state_dict()


def load_hf_dict_from_safetensors(model_id_or_path, token, revision, framework="pt"):
  if os.path.isdir(model_id_or_path):
    local_path = model_id_or_path
  else:
    local_path = snapshot_download(repo_id=model_id_or_path, token=token, revision=revision)
  ckpt_paths = sorted(pathlib.Path(local_path).rglob("[!.]*.safetensors"))
  hf_state_dict = {}
  max_logging.log(f"Loading {len(ckpt_paths)} checkpoints")
  for ckpt_path in tqdm(ckpt_paths, total=len(ckpt_paths)):
    with safe_open(ckpt_path, framework=framework, device="cpu") as file_handle:
      for key in file_handle.keys():
        hf_state_dict[key] = file_handle.get_tensor(key)
  return hf_state_dict


def shard_jax_weights(jax_weights, device_count, mem_info):
  if len(jax.devices()) != device_count:
    max_logging.log(
        "WARNING: hardware/simulated device mismatch. "
        f"Actual JAX devices: {len(jax.devices())}, Requested count: {device_count}."
    )
  max_logging.log(f"Shard weights across {len(jax.devices())} devices")
  max_logging.debug("Note: Axis 0 sharding is the default and will not be logged individually.")
  mesh = jax.sharding.Mesh(jax.devices(), "checkpoint_sharding_axis")
  s0 = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  s1 = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("checkpoint_sharding_axis"))
  s2 = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "checkpoint_sharding_axis"))
  s3 = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None))

  def checkpoint_device_put(arr):
    if not isinstance(arr, (np.ndarray, jax.Array)):
      arr = np.array(arr)
    if len(arr.shape) == 0:
      max_logging.debug("0D scalar detected, replicating")
      return jax.device_put(arr, device=s0)
    elif arr.shape[0] % device_count == 0:
      return jax.device_put(arr, device=s1)
    elif len(arr.shape) > 1 and arr.shape[1] % device_count == 0:
      max_logging.debug(f"Sharding axis 1. Tensor shape {arr.shape}")
      return jax.device_put(arr, device=s2)
    else:
      max_logging.debug(f"Not sharding. Tensor shape {arr.shape}")
      return jax.device_put(arr, device=s3)

  start = time.time()
  jax_weights_flat, jax_weights_struct = tree.flatten(jax_weights)
  del jax_weights
  gc.collect()
  jax_weights_new = []
  jax_weights_flat.reverse()
  num_weights = len(jax_weights_flat)
  for _ in tqdm(range(num_weights)):
    jax_weight = jax_weights_flat.pop()
    jax_weights_new.append(checkpoint_device_put(jax_weight))
    del jax_weight
    gc.collect()
    logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  jax_weights = tree.unflatten(jax_weights_struct, jax_weights_new)
  max_logging.log(f"Elapse for checkpoint sharding: {(time.time() - start) / 60:.2f} min")
  return jax_weights


def save_weights_to_checkpoint(maxtext_model_path: str, jax_weights: dict, device_count: int, use_ocdbt: bool,
                               use_zarr3: bool, config=None):
  mem_info = psutil.Process()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  gc.collect()
  if device_count > 1:
    jax_weights = shard_jax_weights(jax_weights, device_count, mem_info)
  else:
    max_logging.log("Single device: Skip sharding")
  start = time.time()
  step_number_to_save_new_ckpt = 0
  enable_checkpointing = True
  async_checkpointing = False
  save_interval_steps = 1
  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      maxtext_model_path, enable_checkpointing, async_checkpointing, save_interval_steps,
      use_ocdbt=use_ocdbt, use_zarr3=use_zarr3, enable_continuous_checkpointing=True,
  )
  if checkpoint_manager is None:
    raise RuntimeError("Failed to create Orbax checkpoint manager.")
  params = jax_weights if (isinstance(jax_weights, dict) and "params" in jax_weights) else {"params": jax_weights}
  state_new = train_state.TrainState(
      step=step_number_to_save_new_ckpt, apply_fn=None, params=params, tx=None, opt_state={}
  )
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  if checkpointing.save_checkpoint(checkpoint_manager, step_number_to_save_new_ckpt, state_new, config=config):
    max_logging.log(f"saved a checkpoint at step {step_number_to_save_new_ckpt}")
  checkpointing.wait_until_finished(checkpoint_manager)
  max_logging.log(f"Elapse for checkpoint save: {(time.time() - start) / 60:.2f} min")
