import re

from lilax._backend.checkpoint_conversion.model_families.param_mapping_registry import PARAM_MAPPING, VLLM_HOOK_FNS
from lilax._backend.integration.tunix import weight_mapping


STANDALONE_VLLM_WEIGHT_MAPPING = weight_mapping.StandaloneVllmWeightMapping()

_SHARDING_KNOWLEDGE_MAP = {
    "base.token_embedder.embedding": ("model", None),
    "base.decoder.decoder_norm.scale": (None,),
    "base.decoder.logits_dense.kernel": (None, "model"),
    "base.decoder.layers.pre_self_attention_layer_norm.scale": (None, "layer"),
    "base.decoder.layers.self_attention.query.kernel": (None, "layer", "model", None),
    "base.decoder.layers.self_attention.key.kernel": (None, "layer", "model", None),
    "base.decoder.layers.self_attention.value.kernel": (None, "layer", "model", None),
    "base.decoder.layers.self_attention.query_norm.scale": (None, "layer"),
    "base.decoder.layers.self_attention.key_norm.scale": (None, "layer"),
    "base.decoder.layers.self_attention.out.kernel": ("model", "layer", None, None),
    "base.decoder.layers.post_self_attention_layer_norm.scale": (None, "layer"),
    "base.decoder.layers.mlp.wi_0.kernel": (None, "layer", "model"),
    "base.decoder.layers.mlp.wi_1.kernel": (None, "layer", "model"),
    "base.decoder.layers.mlp.wo.kernel": ("model", "layer", None),
    "base.decoder.layers.moe_block.gate.kernel": (None, "layer", "model"),
    "base.decoder.layers.moe_block.wi_0": ("expert", "layer", None, "model"),
    "base.decoder.layers.moe_block.wi_1": ("expert", "layer", None, "model"),
    "base.decoder.layers.moe_block.wo": ("expert", "layer", "model", None),
    "base.decoder.layers.self_attention.wq_a.kernel": (None, "layer", "model", None),
    "base.decoder.layers.self_attention.wq_b.kernel": (None, "layer", "model", None),
    "base.decoder.layers.self_attention.q_norm.scale": (None, "layer"),
    "base.decoder.layers.self_attention.wkv_a.kernel": (None, "layer", "model", None),
    "base.decoder.layers.self_attention.wkv_b.kernel": (None, "layer", "model", None),
    "base.decoder.layers.self_attention.kv_norm.scale": (None, "layer"),
    "base.decoder.layers.moe_block.shared_experts.wi_0.kernel": (None, "layer", "model"),
    "base.decoder.layers.moe_block.shared_experts.wi_1.kernel": (None, "layer", "model"),
    "base.decoder.layers.moe_block.shared_experts.wo.kernel": ("model", "layer", None),
    "base.decoder.layers.moe_block.gate.bias": (None, "layer", "model"),
}


class VllmWeightMapping:
  def __init__(self, model_name, config=None, use_standalone_mappings=False):
    self.model_name = model_name
    self.config = config
    self.use_standalone_mappings = use_standalone_mappings
    self._sharding_knowledge_map = _SHARDING_KNOWLEDGE_MAP

  def to_hf_mapping(self):
    if self.use_standalone_mappings:
      return STANDALONE_VLLM_WEIGHT_MAPPING[self.model_name].to_hf_mapping()
    return self.convert_hf_map_to_sharding_map(
        PARAM_MAPPING[self.model_name](self.config, maxtext_config=None, scan_layers=True)
    )

  def to_hf_transpose_keys(self):
    if self.use_standalone_mappings:
      return STANDALONE_VLLM_WEIGHT_MAPPING[self.model_name].to_hf_transpose_keys()
    return {}

  def to_hf_hook_fns(self):
    if self.use_standalone_mappings:
      return STANDALONE_VLLM_WEIGHT_MAPPING[self.model_name].to_hf_hook_fns()
    model_family = self.model_name.split("-")[0]
    if model_family in VLLM_HOOK_FNS:
      return VLLM_HOOK_FNS[model_family]()
    return {}

  def lora_to_hf_mappings(self):
    base_mappings = self.to_hf_mapping()
    if not base_mappings:
      return None
    lora_mapping = {}
    for maxtext_key, (hf_key, sharding_spec) in base_mappings.items():
      segments = set(maxtext_key.split("."))
      is_input_proj = any(part in segments for part in ["wi_0", "wi_1", "query", "key", "value", "wq_a", "wq_b", "wkv_a", "wkv_b"])
      is_output_proj = any(part in segments for part in ["wo", "out"])
      if not (is_input_proj or is_output_proj):
        continue
      maxtext_lora_a = maxtext_key + "_lora_a"
      maxtext_lora_b = maxtext_key + "_lora_b"
      if hf_key.endswith(".kernel"):
        hf_lora_a = hf_key.replace(".kernel", ".kernel_lora_a")
        hf_lora_b = hf_key.replace(".kernel", ".kernel_lora_b")
      elif hf_key.endswith(".weight"):
        hf_lora_a = hf_key.replace(".weight", ".weight_lora_a")
        hf_lora_b = hf_key.replace(".weight", ".weight_lora_b")
      else:
        hf_lora_a = hf_key + "_lora_a"
        hf_lora_b = hf_key + "_lora_b"
      if is_input_proj:
        sharding_a, sharding_b = (None, "layer", None), sharding_spec
      else:
        sharding_a, sharding_b = sharding_spec, (None, "layer", None)
      lora_mapping[maxtext_lora_a] = (hf_lora_a, sharding_a)
      lora_mapping[maxtext_lora_b] = (hf_lora_b, sharding_b)
    return lora_mapping

  def _generalize_maxtext_key(self, maxtext_key):
    generic_key = maxtext_key.replace("params-", "base.").replace("-", ".")
    generic_key = re.sub(r"\.dense_layers\.", ".layers.", generic_key)
    generic_key = re.sub(r"\.moe_layers\.", ".layers.", generic_key)
    generic_key = re.sub(r"DeepSeekMoeBlock_0\.MoeBlock_0\.", "moe_block.", generic_key)
    generic_key = re.sub(r"DeepSeekMoeBlock_0\.shared_experts\.", "moe_block.shared_experts.", generic_key)
    return re.sub(r"layers_(\d+)\.", "layers.", generic_key)

  def _generalize_hf_value(self, hf_value):
    if isinstance(hf_value, str):
      first_name = hf_value
    elif isinstance(hf_value, list):
      if not hf_value:
        return None
      first_name = hf_value[0][0] if isinstance(hf_value[0], list) else hf_value[0]
    else:
      raise TypeError(f"Unknown value type in map: {type(hf_value)}")
    wildcard_name = re.sub(r"layers\.(\d+)\.", "layers.*.", first_name)
    return re.sub(r"experts\.(\d+)\.", "experts.*.", wildcard_name)

  def _correct_hf_wildcard_name(self, wildcard_name):
    corrected_name = wildcard_name
    if "layernorm.weight" in corrected_name or "_norm.weight" in corrected_name:
      corrected_name = corrected_name.replace(".weight", ".scale")
    elif corrected_name == "model.embed_tokens.weight":
      corrected_name = "model.embed.embedding"
    elif corrected_name == "lm_head.weight":
      corrected_name = "model.lm_head"
    elif corrected_name == "model.norm.weight":
      corrected_name = "model.norm.scale"
    elif corrected_name.endswith(".weight"):
      corrected_name = corrected_name.replace(".weight", ".kernel")
    return corrected_name

  def convert_hf_map_to_sharding_map(self, hf_mapping):
    sharding_map = {}
    for maxtext_key, hf_value in hf_mapping.items():
      generic_key = self._generalize_maxtext_key(maxtext_key)
      wildcard_name = self._generalize_hf_value(hf_value)
      if wildcard_name is None:
        continue
      corrected_name = self._correct_hf_wildcard_name(wildcard_name)
      sharding_tuple = self._sharding_knowledge_map.get(generic_key)
      if sharding_tuple is None:
        if "layers." not in generic_key:
          print(f"Warning: No sharding rule found for key: {generic_key}")
        continue
      sharding_map[generic_key] = (corrected_name, sharding_tuple)
    return sharding_map
