from dataclasses import dataclass

import jax
import numpy as np


@dataclass
class LLAMA3_VLLM_MAPPING:
  @staticmethod
  def to_hf_hook_fns():
    def reorder_rope(arr):
      return jax.numpy.concatenate((arr[..., ::2], arr[..., 1::2]), axis=arr.ndim - 1)

    def transform_query_kernel(arr):
      depth_scale = np.dtype("float32").type(np.sqrt(arr.shape[-1]))
      return reorder_rope(arr * depth_scale)

    return {
        "base.decoder.layers.self_attention.query.kernel": transform_query_kernel,
        "base.decoder.layers.self_attention.key.kernel": reorder_rope,
    }

  @staticmethod
  def to_hf_transpose_keys():
    return {}

  @staticmethod
  def lora_to_hf_mappings():
    return None

  @staticmethod
  def to_hf_mapping():
    return {
        "base.token_embedder.embedding": ("model.embed.embedding", ("model", None)),
        "base.decoder.decoder_norm.scale": ("model.norm.scale", (None,)),
        "base.decoder.logits_dense.kernel": ("model.lm_head", (None, "model")),
        "base.decoder.layers.mlp.wi_0.kernel": ("model.layers.*.mlp.gate_proj.kernel", (None, "layer", "model")),
        "base.decoder.layers.mlp.wi_1.kernel": ("model.layers.*.mlp.up_proj.kernel", (None, "layer", "model")),
        "base.decoder.layers.mlp.wo.kernel": ("model.layers.*.mlp.down_proj.kernel", ("model", "layer", None)),
        "base.decoder.layers.pre_self_attention_layer_norm.scale": ("model.layers.*.input_layernorm.scale", (None, "layer")),
        "base.decoder.layers.post_self_attention_layer_norm.scale": ("model.layers.*.post_attention_layernorm.scale", (None, "layer")),
        "base.decoder.layers.self_attention.query.kernel": ("model.layers.*.self_attn.q_proj.kernel", (None, "layer", "model", None)),
        "base.decoder.layers.self_attention.key.kernel": ("model.layers.*.self_attn.k_proj.kernel", (None, "layer", "model", None)),
        "base.decoder.layers.self_attention.value.kernel": ("model.layers.*.self_attn.v_proj.kernel", (None, "layer", "model", None)),
        "base.decoder.layers.self_attention.out.kernel": ("model.layers.*.self_attn.o_proj.kernel", ("model", "layer", None, None)),
    }
