"""Offline conversion of Lilax weight-only projection layouts."""

import argparse
import json
from pathlib import Path

import orbax.checkpoint as ocp

from lilax._backend.checkpoint_conversion.projection_layout import (
    is_hybrid_qwen_adapter, projection_containers, projection_layout_adapter, supports_mlp_layout,
    validate_mlp_only_layout,
)
from lilax._backend.checkpoint_conversion.model_families.gpt3 import fused as fused_gpt3
from lilax._backend.checkpoint_conversion.model_families.qwen3_5 import fused as fused_qwen3_5
from lilax._backend.checkpoint_conversion.model_families.qwen3_vl import fused as fused_qwen3_vl
from lilax._backend.checkpoint_conversion.model_families.qwen3_omni import fused as fused_qwen3_omni
from lilax._backend.configs.runtime import resolve_runtime_config
from lilax.configuration import MaxTextConfig
from lilax.configuration.values import load_config_values


def convert_checkpoint(source, destination, *, fused_qkv=None, fused_mlp=None):
  """Convert a Lilax save_pretrained checkpoint without optimizer or training state."""
  source = Path(source).expanduser().resolve()
  destination = Path(destination).expanduser().resolve()
  metadata_path = source / "maxtext_easy_config_metadata.json"
  if not (source / "config.json").is_file() or not metadata_path.is_file():
    raise ValueError("Source must be a Lilax save_pretrained weight-only checkpoint.")
  if destination.exists():
    raise ValueError(f"Destination already exists: {destination}")
  if source in destination.parents:
    raise ValueError("Destination must not be inside the source checkpoint.")
  saved_values = json.loads((source / "config.json").read_text(encoding="utf-8"))
  metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
  if not isinstance(saved_values, dict) or not isinstance(saved_values.get("model_name"), str):
    raise ValueError("Source checkpoint config must contain a model_name.")
  if not isinstance(metadata, dict) or set(metadata) != {"explicit_keys"}:
    raise ValueError("Source checkpoint configuration metadata must contain only explicit_keys.")
  explicit_keys = metadata["explicit_keys"]
  if not isinstance(explicit_keys, list) or any(
      not isinstance(key, str) or key not in saved_values for key in explicit_keys
  ):
    raise ValueError("Source checkpoint explicit_keys must be strings present in config.json.")
  values = load_config_values(saved_values["model_name"], _saved_values=saved_values, _apply_environment=False)
  config = MaxTextConfig._from_resolved(values, explicit_keys)
  validate_mlp_only_layout(config, fused_qkv=fused_qkv, fused_mlp=fused_mlp)
  layout_adapter = projection_layout_adapter(config)
  if layout_adapter is None:
    raise ValueError(f"Projection layout conversion is not supported for {config.model_name!r}.")
  if config.qwix_qat_config is not None:
    raise NotImplementedError("Projection layout conversion does not support Qwix QAT.")
  native_config = resolve_runtime_config(config)
  target_qkv = config.fused_qkv if fused_qkv is None else fused_qkv
  target_mlp = config.fused_mlp if fused_mlp is None else fused_mlp
  if target_mlp != config.fused_mlp and not supports_mlp_layout(layout_adapter):
    raise ValueError("This model's routed MLP layout cannot be converted.")
  if target_qkv and (config.share_kv_projections or config.num_kv_shared_layers):
    raise ValueError("Shared KV projections or layers are incompatible with fused QKV.")
  checkpointer = ocp.PyTreeCheckpointer(
      use_ocdbt=config.checkpoint_storage_use_ocdbt, use_zarr3=config.checkpoint_storage_use_zarr3,
  )
  state = checkpointer.restore(source)
  if any(key in state for key in ("step", "opt_state", "optimizer", "iter")):
    raise ValueError("Full training checkpoints cannot be converted as model weights.")

  def unwrap(node):
    if isinstance(node, dict):
      if len(node) == 1 and "value" in node and not isinstance(node["value"], dict):
        return node["value"]
      return {key: unwrap(value) for key, value in node.items()}
    return node

  def wrap(node):
    if isinstance(node, dict):
      return {key: wrap(value) for key, value in node.items()}
    return {"value": node}

  state = unwrap(state)

  def reject_prefused_moe(node):
    if isinstance(node, dict):
      if isinstance(node.get("GptOssMlp"), dict) and "wi" in node["GptOssMlp"]:
        raise NotImplementedError("Cannot split stored GPT-OSS prefused weights: source shard-count provenance is missing.")
      for value in node.values():
        reject_prefused_moe(value)

  reject_prefused_moe(state)

  def projection_modules(node, names, *, hybrid_attention=False, parent=None, path=()):
    if not isinstance(node, dict):
      return []
    wrappers = {"moe_block", "shared_expert", "shared_experts", "DeepSeekMoeBlock_0"}
    modules = [node[name] for name in names if isinstance(node.get(name), dict)
               and not wrappers.intersection(node[name]) and (not hybrid_attention or parent == "attention")
               and (layout_adapter not in (fused_qwen3_vl, fused_qwen3_omni) and not is_hybrid_qwen_adapter(layout_adapter)
                    or "decoder" in path or names == ("AudioMLP",) and "audio_encoder" in path)]
    for name, value in node.items():
      modules.extend(projection_modules(value, names, hybrid_attention=hybrid_attention, parent=name, path=path + (name,)))
    return modules

  if target_qkv != config.fused_qkv:
    attention = projection_modules(state, projection_containers(layout_adapter, mlp=False),
                                   hybrid_attention=is_hybrid_qwen_adapter(layout_adapter))
    required = {"qkv_proj"} if config.fused_qkv else {"query", "key", "value"}
    if not attention or any(not required.issubset(module) for module in attention):
      raise ValueError("Checkpoint does not contain the expected attention projection layout.")
  if target_mlp != config.fused_mlp:
    mlps = projection_modules(state, projection_containers(layout_adapter, mlp=True))
    audio_mlps = projection_modules(state, ("AudioMLP",)) if layout_adapter is fused_qwen3_omni and config.use_audio else []
    required = {"wi"} if config.fused_mlp or (layout_adapter is fused_gpt3 and len(config.mlp_activations) == 1) else {"wi_0", "wi_1"}
    if (not mlps and not audio_mlps or any(not required.issubset(module) for module in mlps)
        or any(not isinstance(module.get("wi"), dict) or "kernel" not in module["wi"]
               or module["wi"]["kernel"].ndim != (3 if config.fused_mlp else 2) for module in audio_mlps)):
      raise ValueError("Checkpoint does not contain the expected MLP projection layout.")
  if not is_hybrid_qwen_adapter(layout_adapter) or layout_adapter is fused_qwen3_5:
    state = layout_adapter.convert_native_state(state, native_config, fused_qkv=target_qkv, fused_mlp=target_mlp)
  else:
    converted = layout_adapter.convert_native_state({"decoder": state["decoder"]}, native_config,
                                                     fused_qkv=target_qkv, fused_mlp=target_mlp)
    state["decoder"] = converted["decoder"]
  checkpointer.save(destination, wrap(state))
  saved_values.pop("hf_access_token", None)
  saved_values.update({"fused_qkv": target_qkv, "fused_mlp": target_mlp})
  (destination / "config.json").write_text(json.dumps(saved_values, indent=2), encoding="utf-8")
  metadata["explicit_keys"] = sorted(set(metadata["explicit_keys"]) | {"fused_qkv", "fused_mlp"})
  (destination / "maxtext_easy_config_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
  return destination


def main():
  parser = argparse.ArgumentParser(prog="lilax convert")
  parser.add_argument("source", help="Lilax weight-only checkpoint")
  parser.add_argument("destination", help="new checkpoint directory")
  parser.add_argument("--fused-qkv", action=argparse.BooleanOptionalAction, default=None)
  parser.add_argument("--fused-mlp", action=argparse.BooleanOptionalAction, default=None)
  args = parser.parse_args()
  print(convert_checkpoint(args.source, args.destination, fused_qkv=args.fused_qkv, fused_mlp=args.fused_mlp))


if __name__ == "__main__":
  main()
