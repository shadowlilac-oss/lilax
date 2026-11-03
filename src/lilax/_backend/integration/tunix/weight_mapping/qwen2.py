from dataclasses import dataclass


@dataclass
class QWEN2_VLLM_MAPPING:
  @staticmethod
  def to_hf_hook_fns():
    return {}

  @staticmethod
  def to_hf_transpose_keys():
    return {}

  @staticmethod
  def lora_to_hf_mappings():
    return None

  @staticmethod
  def to_hf_mapping():
    return {
        "base.token_embedder.embedding": ("model.embed_tokens.weight", ("model", None)),
        "base.decoder.decoder_norm.scale": ("model.norm.weight", (None,)),
        "base.decoder.logits_dense.kernel": ("model.lm_head", (None, "model")),
        "base.decoder.layers.mlp.wi_0.kernel": ("model.layers.*.mlp.gate_proj.weight", (None, "layer", "model")),
        "base.decoder.layers.mlp.wi_1.kernel": ("model.layers.*.mlp.up_proj.weight", (None, "layer", "model")),
        "base.decoder.layers.mlp.wo.kernel": ("model.layers.*.mlp.down_proj.weight", ("model", "layer", None)),
        "base.decoder.layers.pre_self_attention_layer_norm.scale": ("model.layers.*.input_layernorm.weight", (None, "layer")),
        "base.decoder.layers.post_self_attention_layer_norm.scale": ("model.layers.*.post_attention_layernorm.weight", (None, "layer")),
        "base.decoder.layers.self_attention.query.kernel": ("model.layers.*.self_attn.q_proj.weight", (None, "layer", "model", None)),
        "base.decoder.layers.self_attention.key.kernel": ("model.layers.*.self_attn.k_proj.weight", (None, "layer", "model", None)),
        "base.decoder.layers.self_attention.value.kernel": ("model.layers.*.self_attn.v_proj.weight", (None, "layer", "model", None)),
        "base.decoder.layers.self_attention.out.kernel": ("model.layers.*.self_attn.o_proj.weight", ("model", "layer", None, None)),
        "base.decoder.layers.self_attention.query.bias": ("model.layers.*.self_attn.q_proj.bias", (None, "layer", "model", None)),
        "base.decoder.layers.self_attention.key.bias": ("model.layers.*.self_attn.k_proj.bias", (None, "layer", "model", None)),
        "base.decoder.layers.self_attention.value.bias": ("model.layers.*.self_attn.v_proj.bias", (None, "layer", "model", None)),
    }
