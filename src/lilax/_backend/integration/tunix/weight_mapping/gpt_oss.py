from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class GPT_OSS_VLLM_MAPPING:
  @staticmethod
  def lora_to_hf_mappings():
    return None

  @staticmethod
  def to_hf_hook_fns():
    return {}

  @staticmethod
  def to_hf_transpose_keys():
    return {}

  @staticmethod
  def to_hf_mapping(
      layer_cycle_interval: int = 2, total_num_layers: int = 36, interleave_style: str = "modulo"
  ) -> Dict[str, Tuple[str, Tuple[Optional[str], ...]]]:
    mapping = {
        "base.token_embedder.embedding": ("embedder.input_embedding_table_VD", ("model", None)),
        "base.decoder.decoder_norm.scale": ("final_norm.scale", (None,)),
        "base.decoder.logits_dense.kernel": ("lm_head.input_embedding_table_DV", (None, "model")),
    }
    layers_per_block = total_num_layers // layer_cycle_interval
    for block_idx in range(layer_cycle_interval):
      src_block = f"base.decoder.layers.layers_{block_idx}"
      if interleave_style == "modulo":
        target_indices = range(block_idx, total_num_layers, layer_cycle_interval)
      else:
        start = block_idx * layers_per_block
        target_indices = range(start, start + layers_per_block)
      regex_indices = "|".join(map(str, target_indices))
      layer_regex = rf"layers\.({regex_indices})"
      mapping.update({
          f"{src_block}.pre_self_attention_layer_norm.scale": (f"{layer_regex}.pre_attention_norm.scale", (None, "layer")),
          f"{src_block}.post_self_attention_layer_norm.scale": (f"{layer_regex}.pre_mlp_norm.scale", (None, "layer")),
          f"{src_block}.GptOssAttention.query.kernel": (f"{layer_regex}.attn.kernel_q_DNH", (None, "layer", "model", None)),
          f"{src_block}.GptOssAttention.key.kernel": (f"{layer_regex}.attn.kernel_k_DKH", (None, "layer", "model", None)),
          f"{src_block}.GptOssAttention.value.kernel": (f"{layer_regex}.attn.kernel_v_DKH", (None, "layer", "model", None)),
          f"{src_block}.GptOssAttention.out.kernel": (f"{layer_regex}.attn.kernel_o_proj_NHD", ("model", "layer", None, None)),
          f"{src_block}.GptOssAttention.query.bias": (f"{layer_regex}.attn.bias_q_NH", (None, "layer", None)),
          f"{src_block}.GptOssAttention.key.bias": (f"{layer_regex}.attn.bias_k_KH", (None, "layer", None)),
          f"{src_block}.GptOssAttention.value.bias": (f"{layer_regex}.attn.bias_v_KH", (None, "layer", None)),
          f"{src_block}.GptOssAttention.out.bias": (f"{layer_regex}.attn.bias_o_D", (None, "layer")),
          f"{src_block}.GptOssAttention.sinks": (f"{layer_regex}.attn.sinks_N", (None, "layer")),
          f"{src_block}.GptOssMlp.gate.kernel": (f"{layer_regex}.custom_module.router.kernel_DE", (None, "layer", "model")),
          f"{src_block}.GptOssMlp.gate.bias": (f"{layer_regex}.custom_module.router.bias_E", ("model", "layer")),
          f"{src_block}.GptOssMlp.wi_0": (f"{layer_regex}.custom_module.gate_proj_kernel", ("model", "layer", None)),
          f"{src_block}.GptOssMlp.wi_0_bias": (f"{layer_regex}.custom_module.gate_proj_bias", ("model", "layer")),
          f"{src_block}.GptOssMlp.wi_1": (f"{layer_regex}.custom_module.up_proj_kernel", ("model", "layer", None)),
          f"{src_block}.GptOssMlp.wi_1_bias": (f"{layer_regex}.custom_module.up_proj_bias", ("model", "layer")),
          f"{src_block}.GptOssMlp.wo": (f"{layer_regex}.custom_module.mlp2_weight_EFD", ("model", "layer", None)),
          f"{src_block}.GptOssMlp.wo_bias": (f"{layer_regex}.custom_module.mlp2_bias_ED", ("model", "layer")),
      })
    return mapping
