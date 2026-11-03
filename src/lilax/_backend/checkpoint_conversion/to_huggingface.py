"""Convert MaxText checkpoints to Hugging Face models or PEFT adapters."""

import gc
import time

import jax
import jax.numpy as jnp
from transformers import AutoProcessor, AutoTokenizer

from lilax._backend.checkpoint_conversion.model_families.param_mapping_registry import HOOK_FNS, PARAM_MAPPING
from lilax._backend.checkpoint_conversion.model_families.hf_shape_registry import HF_SHAPE
from lilax._backend.checkpoint_conversion.model_families.hf_model_config_registry import HF_MODEL_CONFIGS
from lilax._backend.checkpoint_conversion.utils.utils import (
    validate_and_filter_param_map_keys,
    process_maxtext_param,
    save_model_files,
    load_orbax_checkpoint,
    detect_and_extract_checkpoint,
    MemoryMonitorTqdm,
    print_peak_memory,
    save_adapter_files,
)
from lilax._backend.utils import max_logging, max_utils
from lilax._backend.utils.globals import HF_IDS
from lilax._backend.utils.lora_utils import sync_lora_metadata
from lilax._backend.utils.model_creation_utils import verify_and_sync_scan_layers


def _get_lora_delta(key, lora_state_dict, lora_scaling):
  """Calculate the LoRA delta for a parameter key."""
  a_key, b_key = key + "_lora_a", key + "_lora_b"
  if a_key not in lora_state_dict and key.startswith("params-"):
    a_key, b_key = key[7:] + "_lora_a", key[7:] + "_lora_b"

  if a_key in lora_state_dict and b_key in lora_state_dict:
    data_a = jnp.asarray(lora_state_dict[a_key], dtype=jnp.float32)
    data_b = jnp.asarray(lora_state_dict[b_key], dtype=jnp.float32)
    is_attention = "attention" in key.lower() or "attn" in key.lower()
    if is_attention and data_a.ndim > 2:
      if data_a.ndim == 4:
        return jnp.einsum("lipr,lrpo->lipo", data_a, data_b) * lora_scaling
      return jnp.einsum("ipr,rpo->ipo", data_a, data_b) * lora_scaling
    else:
      if data_a.ndim == 3:
        rank = data_a.shape[2]
        if rank == data_b.shape[1] and rank != data_b.shape[0]:
          return jnp.einsum("lir,lro->lio", data_a, data_b) * lora_scaling
        elif rank == data_b.shape[0] and rank != data_b.shape[1]:
          return jnp.einsum("ilr,rlo->ilo", data_a, data_b) * lora_scaling
        else:
          if any(term in key for term in ["wo", "out-kernel", "out_proj"]):
            return jnp.einsum("ilr,rlo->ilo", data_a, data_b) * lora_scaling
          else:
            return jnp.einsum("lir,lro->lio", data_a, data_b) * lora_scaling
      return jnp.matmul(data_a, data_b) * lora_scaling
  return None


def _get_model_mappings(model_name: str, scan_layers: bool, hf_config_dict: dict, maxtext_config):
  """Retrieve parameter, shape, and hook mappings for the model."""
  if model_name not in PARAM_MAPPING or model_name not in HF_SHAPE or model_name not in HOOK_FNS:
    raise ValueError(f"Mappings not found for model: {model_name}. Available PARAM_MAPPING keys: {PARAM_MAPPING.keys()}")

  param_mapping = PARAM_MAPPING[model_name](hf_config_dict, maxtext_config, scan_layers)
  hook_fn_mapping = HOOK_FNS[model_name](hf_config_dict, maxtext_config, scan_layers, saving_to_hf=True)
  for hook_key in list(hook_fn_mapping.keys()):
    if isinstance(hook_key, tuple):
      hf_path = param_mapping.get(hook_key[0])
      if hf_path is not None:
        param_mapping[hook_key] = hf_path
        for key in hook_key:
          param_mapping.pop(key, None)

  return {
      "param_mapping": param_mapping,
      "shape_mapping": HF_SHAPE[model_name](hf_config_dict),
      "hook_fn_mapping": hook_fn_mapping,
  }


def _validate_or_update_architecture(hf_config, max_config, override: bool):
  """Validate or update Hugging Face architecture attributes."""
  attributes_to_check = [
      ("hidden_size", "emb_dim"),
      ("intermediate_size", "mlp_dim"),
      ("kv_lora_rank", "kv_lora_rank"),
      ("moe_intermediate_size", "moe_mlp_dim"),
      ("n_routed_experts", "num_experts"),
      ("n_shared_experts", "shared_experts"),
      ("num_attention_heads", "num_query_heads"),
      ("num_experts", "num_experts"),
      ("num_experts_per_tok", "num_experts_per_tok"),
      ("num_hidden_layers", "num_decoder_layers"),
      ("num_key_value_heads", "num_kv_heads"),
      ("num_local_experts", "num_experts"),
      ("q_lora_rank", "q_lora_rank"),
      ("qk_nope_head_dim", "qk_nope_head_dim"),
      ("qk_rope_head_dim", "qk_rope_head_dim"),
      ("v_head_dim", "v_head_dim"),
      ("vocab_size", "vocab_size"),
      ("global_head_dim", "global_head_dim"),
      ("num_global_key_value_heads", "global_num_kv_heads"),
      ("hc_mult", "mhc_expansion_rate"),
      ("num_hash_layers", "first_num_hash_layers"),
      ("index_n_heads", "indexer_n_heads"),
      ("index_head_dim", "indexer_head_dim"),
      ("o_lora_rank", "o_lora_rank"),
      ("o_groups", "o_groups"),
  ]
  if max_config.attention_type == "mla":
    attributes_to_check.extend(
        [
            ("qk_nope_head_dim", "qk_nope_head_dim"),
            ("qk_rope_head_dim", "qk_rope_head_dim"),
            ("v_head_dim", "v_head_dim"),
            ("kv_lora_rank", "kv_lora_rank"),
            ("q_lora_rank", "q_lora_rank"),
        ]
    )
  else:
    attributes_to_check.append(("head_dim", "head_dim"))

  mismatches = []
  target_cfg = getattr(hf_config, "text_config", hf_config) or hf_config
  for hf_attr, mt_attr in attributes_to_check:
    if not hasattr(max_config, mt_attr):
      continue
    try:
      hf_value = getattr(target_cfg, hf_attr)
    except (AttributeError, ValueError, RuntimeError):
      continue
    mt_value = getattr(max_config, mt_attr)
    if hf_value is None or mt_value is None:
      continue
    if max_config.model_name.startswith("gemma2") and hf_attr == "num_hidden_layers":
      if isinstance(mt_value, int):
        mt_value = mt_value * 2
    if "qwen3" in max_config.model_name and getattr(max_config, "num_experts", 0) > 1 and hf_attr == "intermediate_size":
      mt_value = mt_value * getattr(max_config, "num_experts_per_tok", 1)
    if hf_attr == "vocab_size" and isinstance(mt_value, int) and isinstance(hf_value, int):
      if mt_value >= hf_value and (mt_value - hf_value) < 256:
        mt_value = hf_value
    is_match = False
    if isinstance(hf_value, float) or isinstance(mt_value, float):
      try:
        is_match = abs(float(hf_value) - float(mt_value)) < 1e-6
      except (ValueError, TypeError):
        is_match = hf_value == mt_value
    else:
      is_match = hf_value == mt_value
    if not is_match:
      if override:
        max_logging.log(f"⚠️ Overwriting HF Config '{hf_attr}': {hf_value} -> {mt_value} (from MaxText '{mt_attr}')")
        setattr(target_cfg, hf_attr, mt_value)
      else:
        mismatches.append(f"{hf_attr} (HF={hf_value} vs MaxText={mt_value})")

  if mismatches:
    error_msg = (
        "Architecture mismatches detected between standard Hugging Face config and provided MaxText config:\n  - "
        + "\n  - ".join(mismatches)
        + "\n\nAction Required: Set override_model_architecture to force the conversion using MaxText values."
    )
    raise ValueError(error_msg)


def _apply_yarn_rope_config(hf_config, max_config) -> None:
  """Apply MaxText YaRN RoPE values without dropping model-specific fields."""
  rope_scaling = getattr(hf_config, "rope_scaling", None)
  if isinstance(rope_scaling, dict):
    rope_scaling = dict(rope_scaling)
  else:
    rope_scaling = {}
  rope_type_key = "type" if "type" in rope_scaling and "rope_type" not in rope_scaling else "rope_type"
  rope_scaling.update(
      {
          "beta_fast": float(getattr(max_config, "beta_fast", 32.0)),
          "beta_slow": float(getattr(max_config, "beta_slow", 1.0)),
          "factor": float(getattr(max_config, "rope_factor", 32.0)),
          "original_max_position_embeddings": getattr(max_config, "original_max_position_embeddings", 4096),
          "rope_theta": getattr(max_config, "rope_max_timescale"),
          rope_type_key: "yarn",
          "truncate": getattr(max_config, "rope_truncate", False),
      }
  )
  hf_config.rope_theta = getattr(max_config, "rope_max_timescale")
  hf_config.rope_scaling = rope_scaling
  hf_config.rope_parameters = rope_scaling


def _transform_weights_to_adapter(param_map, state_dict):
  """Extract standalone PEFT weights from a MaxText state dict."""
  processed_params_list = []
  found_hf_modules = set()
  for mt_key, hf_paths in param_map.items():
    a_key, b_key = mt_key + "_lora_a", mt_key + "_lora_b"
    if a_key not in state_dict and mt_key.startswith("params-"):
      a_key, b_key = mt_key[7:] + "_lora_a", mt_key[7:] + "_lora_b"
    if a_key in state_dict and b_key in state_dict:
      data_a, data_b = state_dict[a_key], state_dict[b_key]
      hf_paths = [hf_paths] if not isinstance(hf_paths, list) else hf_paths
      for index, hf_path in enumerate(hf_paths):
        found_hf_modules.add(hf_path.split(".")[-2])
        name = hf_path.replace(".weight", "")
        if data_a.ndim > 2:
          if data_a.shape[0] == len(hf_paths):
            layer_a = data_a[index, ...]
            layer_b = data_b[index, ...]
          else:
            layer_a = data_a[:, index, ...]
            layer_b = data_b[:, index, ...]
        else:
          layer_a = data_a
          layer_b = data_b
        if layer_a.ndim > 2:
          layer_a = layer_a[:, 0, :]
        if layer_b.ndim > 2:
          layer_b = layer_b[:, 0, :]
        processed_params_list.append((f"base_model.model.{name}.lora_A.weight", jax.numpy.asarray(layer_a.T)))
        processed_params_list.append((f"base_model.model.{name}.lora_B.weight", jax.numpy.asarray(layer_b.T)))
  return dict(processed_params_list), found_hf_modules


def _transform_weights_to_full_model(config, filtered_map_keys, state_dict, param_map, hook_fn_map, shape_map):
  """Transform MaxText weights to HF full model format, optionally merging LoRA."""
  processed_params_list = []
  lora_scaling = config.lora.lora_alpha / config.lora.lora_rank if config.lora.lora_rank > 0 else 1.0
  for key in MemoryMonitorTqdm(filtered_map_keys, leave=True):
    if isinstance(key, tuple):
      weight = [state_dict.pop(subkey, None) for subkey in key]
      delta = None
    else:
      delta = _get_lora_delta(key, state_dict, lora_scaling)
      weight = state_dict.pop(key, None)
      if weight is not None and delta is not None:
        if delta.shape != weight.shape and delta.size == weight.size:
          delta = delta.reshape(weight.shape)
        weight = (jnp.asarray(weight, dtype=jnp.float32) + delta).astype(weight.dtype)
    if weight is not None:
      processed_params_list.extend(process_maxtext_param(key, weight, param_map, hook_fn_map, shape_map, config))
      del weight
      del delta
      gc.collect()
  return dict(processed_params_list)


def _transform_and_save_weights(
    config, lora_restore_path, load_parameters_path, param_map, maxtext_state_dict, filtered_map_keys,
    hook_fn_map, shape_map, output_directory, hf_config_obj, tokenizer, processor,
):
  """Transform and save weights for the selected conversion mode."""
  start = time.time()
  if lora_restore_path and not load_parameters_path:
    transformed_hf_weights, found_hf_modules = _transform_weights_to_adapter(param_map, maxtext_state_dict)
    save_adapter_files(output_directory, transformed_hf_weights, config, found_hf_modules, HF_IDS.get(config.model_name))
    max_logging.log(f"✅ LoRA adapter successfully saved at {output_directory}")
  else:
    transformed_hf_weights = _transform_weights_to_full_model(
        config, filtered_map_keys, maxtext_state_dict, param_map, hook_fn_map, shape_map
    )
    if not transformed_hf_weights:
      raise ValueError("Error: No weights were transformed. Check mappings and parameter paths.")
    max_logging.log("\nSaving HuggingFace model...")
    save_model_files(transformed_hf_weights, hf_config_obj, tokenizer, processor, output_directory)
    max_logging.log(f"✅ MaxText model successfully saved in HuggingFace format at {output_directory}")
  max_logging.log(f"Elapse for transform and save: {(time.time() - start) / 60:.2f} min")


def convert_to_huggingface(config, *, override_model_architecture=False, hf_model_path=None):
  """Convert a checkpoint using a resolved native configuration and return its output path."""
  assert (
      config.load_full_state_path == ""
  ), "This script expects parameters, not a full state. Use generate_param_only_checkpoint first if needed."
  if not config.load_parameters_path and config.lora.lora_restore_path:
    temp_config = config.model_copy(update={"load_parameters_path": config.lora.lora_restore_path})
    temp_config = verify_and_sync_scan_layers(temp_config)
    config = config.model_copy(update={"scan_layers": temp_config.scan_layers})
  else:
    config = verify_and_sync_scan_layers(config)
  max_utils.print_system_information()
  overall_start = time.time()

  lora_restore_path = config.lora.lora_restore_path
  load_parameters_path = config.load_parameters_path
  if not load_parameters_path and not lora_restore_path:
    raise ValueError("Either load_parameters_path or lora_restore_path must be specified.")
  if lora_restore_path:
    sync_lora_metadata(config)
  max_logging.log("\nLoading Orbax checkpoint(s)...")
  start = time.time()
  checkpoint_dict = load_orbax_checkpoint(config)
  max_logging.log(f"Elapse for checkpoint load: {(time.time() - start) / 60:.2f} min")

  if not config.base_output_directory:
    output_directory = f"tmp/{config.run_name}"
  else:
    output_directory = config.base_output_directory
  model_key = config.model_name
  if model_key not in HF_IDS:
    raise ValueError(f"Unsupported model name: {config.model_name}. Supported models are: {list(HF_IDS.keys())}")
  hf_config_obj = HF_MODEL_CONFIGS[model_key]
  _validate_or_update_architecture(hf_config_obj, config, override=override_model_architecture)
  if getattr(config, "rope_type", None) == "yarn":
    _apply_yarn_rope_config(hf_config_obj, config)

  if model_key not in HF_IDS:
    raise ValueError(f"HF Tokenizer ID not found for model key: {model_key}")
  hf_token = config.hf_access_token
  hf_tokenizer_id = hf_model_path or HF_IDS[model_key]
  tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_id, token=hf_token)
  processor = AutoProcessor.from_pretrained(hf_tokenizer_id, token=hf_token) if config.use_multimodal else None
  mappings = _get_model_mappings(model_key, config.scan_layers, hf_config_obj.to_dict(), config)
  param_map = mappings["param_mapping"]
  shape_map = mappings["shape_mapping"]
  hook_fn_map = mappings["hook_fn_mapping"]
  maxtext_state_dict = detect_and_extract_checkpoint(checkpoint_dict)
  state_keys = {key.replace("_lora_a", "").replace("_lora_b", "") for key in maxtext_state_dict}
  filtered_map_keys = validate_and_filter_param_map_keys(param_map, state_keys)
  if not config.use_multimodal:
    filtered_map_keys = [
        key
        for key in filtered_map_keys
        if not (isinstance(key, str) and "vision_encoder" in key)
        and not (isinstance(key, tuple) and any("vision_encoder" in sub for sub in key))
    ]
  max_logging.log("\nProcessing weights...")
  _transform_and_save_weights(
      config, lora_restore_path, load_parameters_path, param_map, maxtext_state_dict, filtered_map_keys,
      hook_fn_map, shape_map, output_directory, hf_config_obj, tokenizer, processor,
  )
  max_logging.log(f"Overall Elapse: {(time.time() - overall_start) / 60:.2f} min")
  print_peak_memory()
  return output_directory
