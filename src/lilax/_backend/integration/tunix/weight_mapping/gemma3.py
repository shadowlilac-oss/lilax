import dataclasses

import numpy as np


@dataclasses.dataclass
class GEMMA3_VLLM_MAPPING:
  @staticmethod
  def to_hf_hook_fns():
    def scale_embedding(arr):
      hidden_size = arr.shape[1]
      normalizer = np.dtype(arr.dtype).type(hidden_size**0.5)
      return arr / normalizer
    return {"base.token_embedder.embedding": scale_embedding}

  @staticmethod
  def to_hf_transpose_keys():
    return {}

  @staticmethod
  def lora_to_hf_mappings():
    return None

  @staticmethod
  def to_hf_mapping():
    return {
        "base.token_embedder.embedding": ("model.language_model.embed_tokens.kernel", ("model", None)),
        "base.decoder.decoder_norm.scale": ("model.language_model.norm.scale", (None,)),
        "base.decoder.layers.pre_self_attention_norm.scale": ("model.language_model.layers.*.input_layernorm.scale", (None, "layer")),
        "base.decoder.layers.post_self_attention_norm.scale": ("model.language_model.layers.*.post_attention_layernorm.scale", (None, "layer")),
        "base.decoder.layers.self_attention.query_norm.scale": ("model.language_model.layers.*.self_attn.q_norm.scale", (None, "layer")),
        "base.decoder.layers.self_attention.key_norm.scale": ("model.language_model.layers.*.self_attn.k_norm.scale", (None, "layer")),
        "base.decoder.layers.pre_ffw_norm.scale": ("model.language_model.layers.*.pre_feedforward_layernorm.scale", (None, "layer")),
        "base.decoder.layers.post_ffw_norm.scale": ("model.language_model.layers.*.post_feedforward_layernorm.scale", (None, "layer")),
        "base.decoder.layers.mlp.wi_0.kernel": ("model.language_model.layers.*.mlp.gate_proj.kernel", (None, "layer", "model")),
        "base.decoder.layers.mlp.wi_1.kernel": ("model.language_model.layers.*.mlp.up_proj.kernel", (None, "layer", "model")),
        "base.decoder.layers.mlp.wo.kernel": ("model.language_model.layers.*.mlp.down_proj.kernel", ("model", "layer", None)),
        "base.decoder.layers.self_attention.query.kernel": ("model.language_model.layers.*.self_attn.q_proj.kernel", (None, "layer", "model", None)),
        "base.decoder.layers.self_attention.key.kernel": ("model.language_model.layers.*.self_attn.k_proj.kernel", (None, "layer", "model", None)),
        "base.decoder.layers.self_attention.value.kernel": ("model.language_model.layers.*.self_attn.v_proj.kernel", (None, "layer", "model", None)),
        "base.decoder.layers.self_attention.out.kernel": ("model.language_model.layers.*.self_attn.o_proj.kernel", ("model", "layer", None, None)),
    }
