"""Resolve model checkpoint sources and conversion paths."""

import copy
import hashlib
import json

from pathlib import Path

from etils import epath
from huggingface_hub.constants import HF_HUB_CACHE
import jax

from lilax._backend.checkpoint_conversion.cpu_bootstrap import conversion_provenance, convert_on_cpu
from lilax._backend.common.checkpoints.model_path import model_checkpoint_format, model_checkpoint_path
from lilax.configuration import AutoConfig
from lilax.configuration.model_architecture import ArchitectureConfig
from lilax.configuration.model_mapping import project_model_config
from lilax.modeling.hub import get_model_capabilities, resolve_model_name, resolve_model_snapshot
from lilax.runtime import Runtime
from lilax.modeling import persistence as model_persistence


def resolve_pretrained_source(
    pretrained_model_name_or_path, *, config=None, runtime=None, revision=None, token=None, cache_dir=None,
    local_files_only=False, model_mode="train", rng_key=None, wrap_with_tunix_adapter=False,
    tokenizer_pad_id=None, quant_mode_str="train", lazy_load_tensors=True, eager_load_method="safetensors",
    save_dtype="bfloat16", simulated_cpu_devices_count=16, conversion_output_dir=None,
    fuse_qkv=None, fuse_mlp=None, precision_config=None, quantization_config=None,
):
  auto_config = config is None and runtime is None
  path = str(pretrained_model_name_or_path)
  local_path = Path(path).expanduser()
  is_hf = not path.startswith("gs://") and (
      not local_path.is_dir() or (local_path / "model.safetensors.index.json").is_file()
      or (local_path / "model.safetensors").is_file()
      or (local_path / "pytorch_model.bin.index.json").is_file()
      or (local_path / "pytorch_model.bin").is_file()
  )
  if (is_hf and local_path.is_dir() and model_checkpoint_path(local_path) != epath.Path(local_path)
      and model_checkpoint_format(local_path) != "huggingface"):
    is_hf = False
  if is_hf:
    supplied_config = config if config is not None else runtime.public_config if runtime is not None else None
    snapshot = resolve_model_snapshot(
        path, revision=revision, token=token, cache_dir=cache_dir, local_files_only=local_files_only
    )
    model_name = resolve_model_name(
        path, model_name=supplied_config.model_name if supplied_config else None, snapshot=snapshot
    )
    if not get_model_capabilities(model_name).hf_import:
      raise ValueError(f"Hugging Face import is not supported for {model_name!r}")
    if config is None and runtime is None:
      config = AutoConfig.from_pretrained(snapshot, model_name=model_name)
  if path.startswith("gs://") and config is None and runtime is None:
    raise ValueError("Loading a GCS checkpoint requires an explicit config or initialized runtime.")
  if config is None:
    if runtime is not None:
      config = runtime.public_config
    else:
      config = AutoConfig.from_pretrained(
          path, revision=revision, token=token, cache_dir=cache_dir, local_files_only=local_files_only
      )
  if not is_hf and local_path.is_dir():
    resolve_model_name(local_path, model_name=config.model_name)
  if runtime is None:
    config = copy.deepcopy(config)
    if quantization_config is not None:
      config.quantization_config = quantization_config
    if precision_config is not None:
      config.precision_config = precision_config
    if fuse_qkv is not None or fuse_mlp is not None:
      architecture = config.architecture_config or ArchitectureConfig()
      updates = {name: value for name, value in (("fused_qkv", fuse_qkv), ("fused_mlp", fuse_mlp)) if value is not None}
      config.architecture_config = architecture.model_copy(update=updates)
    runtime = Runtime(project_model_config(config))
  elif config is not runtime.public_config and config is not runtime.config:
    raise ValueError("The configuration must be the configuration used to initialize the runtime.")
  source_qat_config = None
  if not is_hf:
    source_qat_config = model_persistence.checkpoint_qat_config(str(local_path) if local_path.is_dir() else path)
  checkpoint = {
      "path": str(local_path) if local_path.is_dir() else path, "is_hf": is_hf,
      "snapshot": snapshot if is_hf else None, "conversion_output_dir": conversion_output_dir,
      "model_mode": model_mode, "rng_key": rng_key, "wrap_with_tunix_adapter": wrap_with_tunix_adapter,
      "tokenizer_pad_id": tokenizer_pad_id, "quant_mode_str": quant_mode_str,
      "lazy_load_tensors": lazy_load_tensors, "eager_load_method": eager_load_method, "revision": revision,
      "save_dtype": save_dtype, "simulated_cpu_devices_count": simulated_cpu_devices_count,
      "qat_config": source_qat_config,
      "auto_config": auto_config and not is_hf,
      "public_config": config,
      "cache_dir": cache_dir,
  }
  return runtime, checkpoint


def resolve_checkpoint_path(checkpoint, runtime):
  checkpoint_path = checkpoint["path"]
  if checkpoint["is_hf"]:
    native_path = runtime.config.load_parameters_path
    if native_path and epath.Path(native_path).exists():
      checkpoint_path = native_path
    else:
      conversion_config = runtime.public_config.model_copy(update={"scan_layers": runtime.config.scan_layers})
      if checkpoint["conversion_output_dir"] is None:
        shared_output = runtime.config.base_output_directory
        if jax.process_count() > 1 and not shared_output:
          raise ValueError(
              "Multihost Hub conversion needs a shared Trainer output_dir or BASE_OUTPUT_DIRECTORY for staging."
          )
        root = shared_output or str(Path(checkpoint["cache_dir"] or HF_HUB_CACHE).expanduser().resolve())
        provenance = conversion_provenance(
            conversion_config, checkpoint["snapshot"], checkpoint["revision"], checkpoint["save_dtype"],
            checkpoint["simulated_cpu_devices_count"], None, "explicit",
        )
        digest = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode("utf-8")).hexdigest()
        checkpoint["conversion_output_dir"] = str(epath.Path(root) / "lilax-models" / digest)
      checkpoint_path = convert_on_cpu(
          conversion_config, checkpoint["snapshot"], checkpoint["conversion_output_dir"],
          lazy_load_tensors=checkpoint["lazy_load_tensors"], eager_load_method=checkpoint["eager_load_method"],
          revision=checkpoint["revision"], save_dtype=checkpoint["save_dtype"],
          simulated_cpu_devices_count=checkpoint["simulated_cpu_devices_count"], conversion_mode="explicit",
      )
  return checkpoint_path
