"""Native MaxText LoRA adapter APIs."""

import copy
from dataclasses import dataclass
import json
import os

from flax import nnx
from flax.traverse_util import flatten_dict
from huggingface_hub import hf_hub_download
import numpy as np
import torch
from tunix.sft import checkpoint_manager

from lilax._backend.checkpoint_conversion.to_huggingface import _get_model_mappings, _transform_weights_to_adapter
from lilax._backend.checkpoint_conversion.to_maxtext import (
    convert_hf_lora_key_to_maxtext, convert_lora_to_maxtext_adapter,
)
from lilax._backend.checkpoint_conversion.model_families.hf_model_config_registry import HF_MODEL_CONFIGS
from lilax._backend.checkpoint_conversion.model_families.param_mapping_registry import PARAM_MAPPING
from lilax._backend.checkpoint_conversion.utils.utils import (
    extract_nnx_weights, load_hf_dict_from_safetensors, save_adapter_files,
)
from lilax._backend.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from lilax._backend.utils import lora_utils
from lilax._backend.utils.globals import HF_IDS
from lilax.modeling.models import MaxTextModel


@dataclass(kw_only=True)
class LoraConfig:
  r: int
  lora_alpha: float
  target_modules: str = ""
  lora_weight_qtype: str | None = None
  lora_tile_size: int | None = None


class PeftModel(MaxTextModel):
  """A native model with a Qwix LoRA adapter and its original runtime."""

  def __init__(self, native_model, runtime, adapter_config=None, *, base_model=None, peft_config=None, adapter_source=None):
    super().__init__(native_model, runtime, construction_config=adapter_config,
                     config=base_model.config if base_model is not None else None)
    if base_model is not None:
      self.generation_config = base_model.generation_config
    self._adapter_config = self.construction_config if base_model is None else None
    self._base_model = base_model
    self._peft_config = peft_config
    self._adapter_source = adapter_source
    self._pending_native_model = None

  @property
  def existing_graph_config(self):
    """Return the config of the adapted or base graph, without materializing."""
    if self.is_materialized:
      return self.construction_config
    if self._pending_native_model is not None:
      return self._adapter_config
    return self._base_model.existing_graph_config if self._base_model is not None else None

  @property
  def pending_qat_config(self):
    if self.existing_graph_config is not None or self._base_model is None:
      return super().pending_qat_config
    return self._base_model.pending_qat_config

  def prepare_for_full_state_restore(self, qat_config, *, runtime):
    if self.existing_graph_config is not None or self._base_model is None:
      super().prepare_for_full_state_restore(qat_config, runtime=runtime)
    if self._base_model is not None:
      self._base_model.prepare_for_full_state_restore(qat_config, runtime=runtime)
    return self

  def materialize(self, *, runtime):
    if runtime is not self.runtime:
      raise ValueError("A model must be materialized with its original runtime.")
    if self.is_materialized:
      return self
    model = self._base_model
    model.materialize(runtime=runtime)
    config = self._adapter_config if self._pending_native_model is not None else copy.deepcopy(model.construction_config)
    if model.existing_graph_qat_config is not None and not config.lora_input_adapters_path:
      raise ValueError(
          "PEFT LoRA cannot be applied to a Qwix QAT model or QAT checkpoint. "
          "Load a base model without Qwix QAT to apply LoRA, or use QAT without PEFT LoRA."
      )
    source = self._adapter_source
    if source is not None and source[0] == "huggingface":
      _, model_id, revision, token = source
      token = config.hf_access_token if token is None else token
      adapter_path = str(model_id)
      config_path = os.path.join(adapter_path, "adapter_config.json") if os.path.isdir(adapter_path) else hf_hub_download(
          adapter_path, "adapter_config.json", token=token, revision=revision,
      )
      with open(config_path, "r", encoding="utf-8") as config_file:
        metadata = json.load(config_file)
      if metadata["peft_type"] != "LORA":
        raise ValueError(f"Unsupported PEFT adapter type: {metadata['peft_type']}")
      if metadata["r"] != self._peft_config.r or metadata["lora_alpha"] != self._peft_config.lora_alpha:
        raise ValueError("Adapter rank and alpha must match the supplied LoraConfig.")
      weights = load_hf_dict_from_safetensors(adapter_path, token, revision)
      _validate_hf_adapter_weights(config, weights)
      adapter = convert_lora_to_maxtext_adapter(config, weights, save_dtype="float32")
    if self._pending_native_model is None:
      _set_lora_config(config, self._peft_config)
      self._pending_native_model = lora_utils.apply_lora_to_model(model.native_model, runtime.mesh, config)
      self._adapter_config = config
    native_model = self._pending_native_model
    if source is not None and source[0] == "huggingface":
      adapter_model = native_model.base if isinstance(native_model, TunixMaxTextAdapter) else native_model
      expected_paths = {tuple(str(part) for part in path) for path, _ in nnx.to_flat_state(nnx.state(adapter_model, nnx.LoRAParam))}
      adapter_paths = {tuple(str(part) for part in path[:-1]) for path in flatten_dict(adapter)}
      missing_paths = sorted(expected_paths - adapter_paths)
      if missing_paths:
        raise ValueError(f"Hugging Face adapter is missing model LoRA parameters: {missing_paths}")
      lora_utils.apply_converted_lora_adapter(adapter_model, adapter)
    elif source is not None:
      manager = checkpoint_manager.CheckpointManager(root_directory=str(source[1]))
      try:
        if manager.latest_step() is not None:
          manager.maybe_restore(native_model, restore_only_lora_params=True)
        else:
          restore_config = copy.deepcopy(config)
          restore_config.lora.lora_restore_path = str(source[1])
          native_model = lora_utils.restore_lora_from_path(native_model, restore_config)
      finally:
        manager.close()
    self._native_model = native_model
    self.construction_config = config
    self._adapter_config = config
    self._base_model = None
    self._peft_config = None
    self._adapter_source = None
    self._pending_native_model = None
    return self

  def save_pretrained(self, save_directory, *, format="huggingface"):
    """Save adapter parameters in Hugging Face or native Orbax format."""
    self.materialize(runtime=self.runtime)
    if format == "huggingface":
      config = self._adapter_config
      hf_config = HF_MODEL_CONFIGS[config.model_name]
      param_map = _get_model_mappings(config.model_name, config.scan_layers, hf_config.to_dict(), config)["param_mapping"]
      native_model = self.native_model
      adapter_model = native_model.base if isinstance(native_model, TunixMaxTextAdapter) else native_model
      state = nnx.to_pure_dict(nnx.state(adapter_model, nnx.LoRAParam))
      weights, found_modules = _transform_weights_to_adapter(param_map, extract_nnx_weights(state))
      weights = {name: np.asarray(weight) for name, weight in weights.items()}
      save_adapter_files(str(save_directory), weights, config, found_modules, HF_IDS.get(config.model_name))
      return
    if format != "maxtext":
      raise ValueError(f"Unsupported adapter format: {format}")
    manager = checkpoint_manager.CheckpointManager(root_directory=str(save_directory))
    try:
      manager.save(
          0, self.native_model, save_only_lora_params=True, force=True,
          custom_metadata={"lora": self._adapter_config.lora.model_dump()},
      )
    finally:
      manager.close()

  @classmethod
  def from_pretrained(cls, model, model_id, peft_config: LoraConfig, *, format="maxtext", revision=None, token=None):
    """Apply an adapter and restore native Orbax or Hugging Face weights."""
    if format not in ("huggingface", "maxtext"):
      raise ValueError(f"Unsupported adapter format: {format}")
    adapted_model = cls(
        None, model.runtime, base_model=model, peft_config=peft_config,
        adapter_source=(format, model_id, revision, token),
    )
    if model.is_materialized:
      adapted_model.materialize(runtime=model.runtime)
    return adapted_model


def _validate_hf_adapter_weights(config, weights):
  model_key = config.model_name.replace("-Instruct", "")
  hf_config = HF_MODEL_CONFIGS[model_key].to_dict()
  param_mapping = PARAM_MAPPING[model_key](hf_config, config, config.scan_layers)
  pairs = {}
  if not weights:
    raise ValueError("The Hugging Face adapter contains no weights.")
  for name, weight in weights.items():
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
      raise ValueError(f"Unsupported Hugging Face adapter weight dtype for {name}: {weight.dtype}")
    if name.endswith((".lora_A.weight", ".lora_A")):
      factor = "A"
    elif name.endswith((".lora_B.weight", ".lora_B")):
      factor = "B"
    else:
      raise ValueError(f"Unsupported Hugging Face adapter weight: {name}")
    target, index = convert_hf_lora_key_to_maxtext(name, param_mapping)
    if not target:
      raise ValueError(f"Hugging Face adapter weight has no MaxText mapping: {name}")
    if isinstance(target, tuple):
      target = target[0]
    key = (target, 0 if index is None else index)
    factors = pairs.setdefault(key, set())
    if factor in factors:
      raise ValueError(f"Multiple Hugging Face adapter weights map to the same MaxText parameter: {name}")
    factors.add(factor)
  for (target, index), factors in pairs.items():
    if factors != {"A", "B"}:
      raise ValueError(f"Hugging Face adapter is missing a LoRA factor for {target} at index {index}")


def _set_lora_config(config, peft_config):
  config.lora.enable_lora = True
  config.lora.lora_rank = peft_config.r
  config.lora.lora_alpha = peft_config.lora_alpha
  config.lora.lora_module_path = peft_config.target_modules
  config.lora.lora_weight_qtype = peft_config.lora_weight_qtype
  config.lora.lora_tile_size = peft_config.lora_tile_size


def get_peft_model(model, peft_config: LoraConfig) -> PeftModel:
  """Apply a native LoRA adapter to the existing NNX model."""
  adapted_model = PeftModel(
      None, model.runtime, base_model=model, peft_config=peft_config,
  )
  if model.is_materialized:
    adapted_model.materialize(runtime=model.runtime)
  return adapted_model


__all__ = ["LoraConfig", "PeftModel", "get_peft_model"]
