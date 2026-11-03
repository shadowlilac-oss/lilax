"""Weaver parameter mappings and transformation hooks for checkpoint conversion."""

from lilax._backend.checkpoint_conversion.model_families.qwen3_vl.param_mapping import (
    QWEN3_VL_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    QWEN3_VL_MAXTEXT_TO_HF_PARAM_MAPPING,
)


def WEAVER_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  """Returns mapping from MaxText to HuggingFace Weaver weight paths."""
  qwen3_vl_mapping = QWEN3_VL_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers)
  mapping = {}

  def translate_hf_key(val):
    if isinstance(val, list):
      return [translate_hf_key(v) for v in val]
    elif isinstance(val, str):
      if val.startswith("model.language_model."):
        val = val[len("model.language_model.") :]
      elif val.startswith("model.visual."):
        val = val[len("model.visual.") :]
      elif val.startswith("model."):
        val = val[len("model.") :]
      val = val.replace("self_attn.q_proj", "self_attn.to_q")
      val = val.replace("self_attn.k_proj", "self_attn.to_k")
      val = val.replace("self_attn.v_proj", "self_attn.to_v")
      val = val.replace("self_attn.o_proj", "self_attn.to_out")
      val = val.replace("self_attn.q_norm", "self_attn.norm_q")
      val = val.replace("self_attn.k_norm", "self_attn.norm_k")
      return val
    return val

  for key, value in qwen3_vl_mapping.items():
    mapping[key] = translate_hf_key(value)
  return mapping


def WEAVER_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  """Creates parameter transformation functions for Weaver."""
  return QWEN3_VL_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers, saving_to_hf)
