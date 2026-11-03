"""Gemma4 MaxText to vLLM weight converter."""

import gc
import logging

import jax
import jax.numpy as jnp
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.process_weights.moe_weights import FusedMoEWeights
from tpu_inference.layers.common.process_weights.moe_weights import process_moe_weights

from lilax._backend.integration.vllm.torchax_converter.base import BaseMaxTextToVLLMConverter
from lilax._backend.integration.vllm.torchax_converter.base import timer
from lilax._backend.integration.vllm.torchax_converter.base import GREEN
from lilax._backend.integration.vllm.torchax_converter.base import RESET


class Gemma4MaxTextToVLLMConverter(BaseMaxTextToVLLMConverter):
  """Converts MaxText Gemma4 weights to the layout expected by a vLLM Gemma4 model."""

  NUM_SLOTS = 6

  def __init__(self, config, mesh):
    super().__init__(config, mesh)
    assert self.num_layers % self.NUM_SLOTS == 0, f"num_layers {self.num_layers} must be divisible by {self.NUM_SLOTS}"
    self.num_reps = self.num_layers // self.NUM_SLOTS
    self.is_moe = config.model_name == "gemma4-26b"
    self.d_model = config.base_emb_dim

  def convert(self, model_state: dict, **kwargs):
    """Convert a MaxText Gemma4 model state into vLLM weight tensors."""
    logging.info(
        "\n%sStarting Gemma4 Conversion (is_moe=%s, num_layers=%d, num_reps=%d)...%s",
        GREEN,
        self.is_moe,
        self.num_layers,
        self.num_reps,
        RESET,
    )
    self.vllm_state = {}
    blocks = model_state["base"]["decoder"]["scanned_blocks"]
    prefix = "vllm_model.language_model.model.layers"

    with timer("Convert Global Weights"):
      self._convert_global(model_state)
    with timer("Convert Layer Norms"):
      self._convert_norms(blocks, prefix)
    with timer("Convert Attention Weights"):
      self._convert_attn_weights(blocks, prefix)
    if self.is_moe:
      with timer("Convert MoE Weights"):
        self._convert_moe_weights(blocks, prefix)
    else:
      with timer("Convert Dense MLP Weights"):
        self._convert_dense_mlp_weights(blocks, prefix)

    return self.vllm_state

  def _convert_global(self, params):
    """Convert non-layered weights (embed_tokens, lm_head, final norm)."""
    logging.info("_convert_global: embed_tokens (de-normalize) + lm_head (tied) + final_norm...")
    normalizer = self.d_model**0.5

    @jax.jit
    def _denorm_embed(x):
      return (x / normalizer).astype(x.dtype)

    raw_embedding = _denorm_embed(params["base"]["token_embedder"]["embedding"])
    self.vllm_state["vllm_model.language_model.model.embed_tokens.weight"] = raw_embedding
    self.vllm_state["vllm_model.language_model.lm_head.weight"] = raw_embedding
    self.vllm_state["vllm_model.language_model.model.norm.weight"] = params["base"]["decoder"]["decoder_norm"]["scale"]
    logging.info("_convert_global: done")

  def _convert_attn(self, params):
    """Satisfy abstract interface; Gemma4 uses _convert_attn_weights instead."""
    blocks = params["base"]["decoder"]["scanned_blocks"]
    prefix = "vllm_model.language_model.model.layers"
    self._convert_attn_weights(blocks, prefix)

  def _convert_moe(self, params):
    """Satisfy abstract interface; Gemma4 uses _convert_moe_weights/_convert_dense_mlp_weights."""
    blocks = params["base"]["decoder"]["scanned_blocks"]
    prefix = "vllm_model.language_model.model.layers"
    if self.is_moe:
      self._convert_moe_weights(blocks, prefix)
    else:
      self._convert_dense_mlp_weights(blocks, prefix)

  @staticmethod
  @jax.jit
  def _pack_attn(q, k, v, o, qnorm, knorm):
    """Prepare separate attention projections and norms for all layers in a slot."""
    q = jnp.transpose(q, (1, 2, 3, 0)).reshape(q.shape[1], -1, q.shape[0])
    k = jnp.transpose(k, (1, 2, 3, 0)).reshape(k.shape[1], -1, k.shape[0])
    v = jnp.transpose(v, (1, 2, 3, 0)).reshape(v.shape[1], -1, v.shape[0])
    o = jnp.transpose(o, (1, 3, 0, 2)).reshape(o.shape[1], o.shape[3], -1)
    qnorm = jnp.transpose(qnorm, (1, 0))
    knorm = jnp.transpose(knorm, (1, 0))
    return (
        jnp.unstack(q),
        jnp.unstack(k),
        jnp.unstack(v),
        jnp.unstack(o),
        jnp.unstack(qnorm),
        jnp.unstack(knorm),
    )

  def _convert_norms(self, blocks, prefix):
    """Converts all 4 per-layer norm vectors across all layers."""

    @jax.jit
    def _unstack_norm(x):
      return jnp.unstack(x, axis=1)

    for slot in range(self.NUM_SLOTS):
      slot_data = blocks[f"layers_{slot}"]
      pre_attn = _unstack_norm(slot_data["pre_self_attention_norm"]["scale"])
      post_attn = _unstack_norm(slot_data["post_self_attention_norm"]["scale"])
      pre_ffw = _unstack_norm(slot_data["pre_ffw_norm"]["scale"])
      post_ffw = _unstack_norm(slot_data["post_ffw_norm"]["scale"])
      for rep in range(self.num_reps):
        i = rep * self.NUM_SLOTS + slot
        self.vllm_state[f"{prefix}.{i}.input_layernorm.weight"] = pre_attn[rep]
        self.vllm_state[f"{prefix}.{i}.post_attention_layernorm.weight"] = post_attn[rep]
        self.vllm_state[f"{prefix}.{i}.pre_feedforward_layernorm.weight"] = pre_ffw[rep]
        self.vllm_state[f"{prefix}.{i}.post_feedforward_layernorm.weight"] = post_ffw[rep]
      del pre_attn, post_attn, pre_ffw, post_ffw
    gc.collect()

  def _convert_attn_weights(self, blocks, prefix):
    """Converts separate q/k/v proj, o proj, q-norm, k-norm for all layers."""

    @jax.jit
    def _pack_local(attn):
      q = attn["query"]["kernel"]
      k = attn["key"]["kernel"]
      v = attn["value"]["kernel"]
      return Gemma4MaxTextToVLLMConverter._pack_attn(
          q,
          k,
          v,
          attn["out"]["kernel"],
          attn["query_norm"]["scale"],
          attn["key_norm"]["scale"],
      )

    @jax.jit
    def _pack_global(attn):
      q = attn["query"]["kernel"]
      k = attn["key"]["kernel"]
      return Gemma4MaxTextToVLLMConverter._pack_attn(
          q,
          k,
          k,
          attn["out"]["kernel"],
          attn["query_norm"]["scale"],
          attn["key_norm"]["scale"],
      )

    for slot in range(self.NUM_SLOTS):
      is_global = slot == self.NUM_SLOTS - 1
      attn = blocks[f"layers_{slot}"]["self_attention"]
      pack_fn = _pack_global if is_global else _pack_local
      q_layers, k_layers, v_layers, o_layers, qnorm_layers, knorm_layers = pack_fn(attn)
      num_kv_heads = self.config.global_num_kv_heads if is_global else self.config.base_num_kv_heads
      tp = min(self.vllm_tp, num_kv_heads)
      for rep in range(self.num_reps):
        i = rep * self.NUM_SLOTS + slot
        q, k, v = q_layers[rep], k_layers[rep], v_layers[rep]
        q_per_tp = q.shape[0] // tp
        kv_per_tp = k.shape[0] // tp
        qkv = jnp.concatenate(
            [
                q.reshape(tp, q_per_tp, q.shape[1]),
                k.reshape(tp, kv_per_tp, k.shape[1]),
                v.reshape(tp, kv_per_tp, v.shape[1]),
            ],
            axis=1,
        ).reshape(-1, q.shape[1])
        self.vllm_state[f"{prefix}.{i}.self_attn.qkv_proj.weight"] = qkv
        self.vllm_state[f"{prefix}.{i}.self_attn.o_proj.weight"] = o_layers[rep]
        self.vllm_state[f"{prefix}.{i}.self_attn.q_norm.weight"] = qnorm_layers[rep]
        self.vllm_state[f"{prefix}.{i}.self_attn.k_norm.weight"] = knorm_layers[rep]
      del q_layers, k_layers, v_layers, o_layers, qnorm_layers, knorm_layers
    gc.collect()

  def _convert_moe_weights(self, blocks, prefix):
    """Converts router, routed experts, shared expert, and MoE norms (26B)."""

    def _pack_moe(routed, shared, extra):
      router_proj = jnp.unstack(jnp.transpose(routed["gate"]["kernel"], (1, 2, 0)), axis=0)
      router_scale = jnp.unstack(extra["pre_forward_scale_2"], axis=1)
      per_expert_scale = jnp.unstack(routed["per_expert_scale"], axis=1)
      w0 = jnp.transpose(routed["wi_0"], (1, 0, 3, 2))
      w1 = jnp.transpose(routed["wi_1"], (1, 0, 3, 2))
      gate_up = jnp.concatenate([w0, w1], axis=2)
      gate_up_proj = jnp.unstack(gate_up, axis=0)
      down_proj = jnp.unstack(jnp.transpose(routed["wo"], (1, 0, 3, 2)), axis=0)
      sh_gate = jnp.unstack(jnp.transpose(shared["wi_0"]["kernel"], (1, 2, 0)), axis=0)
      sh_up = jnp.unstack(jnp.transpose(shared["wi_1"]["kernel"], (1, 2, 0)), axis=0)
      sh_down = jnp.unstack(jnp.transpose(shared["wo"]["kernel"], (1, 2, 0)), axis=0)
      pre_ln_2 = jnp.unstack(extra["pre_feedforward_layernorm_2"]["scale"], axis=1)
      post_ln_1 = jnp.unstack(extra["post_feedforward_layernorm_1"]["scale"], axis=1)
      post_ln_2 = jnp.unstack(extra["post_feedforward_layernorm_2"]["scale"], axis=1)

      return (
          router_proj,
          router_scale,
          per_expert_scale,
          gate_up_proj,
          down_proj,
          sh_gate,
          sh_up,
          sh_down,
          pre_ln_2,
          post_ln_1,
          post_ln_2,
      )

    for slot in range(self.NUM_SLOTS):
      moe_block = blocks[f"layers_{slot}"]["mlp"]["moe_block"]
      routed = moe_block["MoeBlock_0"]
      shared = moe_block["shared_experts"]
      extra = blocks[f"layers_{slot}"]["mlp"]
      (
          router_proj,
          router_scale,
          per_expert_scale,
          gate_up_proj,
          down_proj,
          sh_gate,
          sh_up,
          sh_down,
          pre_ln_2,
          post_ln_1,
          post_ln_2,
      ) = _pack_moe(routed, shared, extra)

      for rep in range(self.num_reps):
        i = rep * self.NUM_SLOTS + slot
        p = f"{prefix}.{i}"
        self.vllm_state[f"{p}.router.proj.weight"] = router_proj[rep]
        self.vllm_state[f"{p}.router.scale"] = router_scale[rep]
        self.vllm_state[f"{p}.moe.per_expert_scale"] = per_expert_scale[rep]
        processed = process_moe_weights(
            FusedMoEWeights(
                w13_weight=gate_up_proj[rep],
                w13_weight_scale=None,
                w13_bias=None,
                w2_weight=down_proj[rep],
                w2_weight_scale=None,
                w2_bias=None,
            ),
            moe_backend=MoEBackend.GMM_TP,
            w13_reorder_size=self.vllm_tp,
            w13_interleave=False,
        )
        self.vllm_state[f"{p}.moe.experts.w13_weight"] = processed.w13_weight
        self.vllm_state[f"{p}.moe.experts.w2_weight"] = processed.w2_weight
        sh_g, sh_u = sh_gate[rep], sh_up[rep]
        sh_per_tp = sh_g.shape[0] // self.vllm_tp
        shared_gate_up = jnp.concatenate(
            [
                sh_g.reshape(self.vllm_tp, sh_per_tp, sh_g.shape[1]),
                sh_u.reshape(self.vllm_tp, sh_per_tp, sh_u.shape[1]),
            ],
            axis=1,
        ).reshape(-1, sh_g.shape[1])
        self.vllm_state[f"{p}.mlp.gate_up_proj.weight"] = shared_gate_up
        self.vllm_state[f"{p}.mlp.down_proj.weight"] = sh_down[rep]
        self.vllm_state[f"{p}.pre_feedforward_layernorm_2.weight"] = pre_ln_2[rep]
        self.vllm_state[f"{p}.post_feedforward_layernorm_1.weight"] = post_ln_1[rep]
        self.vllm_state[f"{p}.post_feedforward_layernorm_2.weight"] = post_ln_2[rep]

      del router_proj, router_scale, per_expert_scale, gate_up_proj, down_proj
      del sh_gate, sh_up, sh_down, pre_ln_2, post_ln_1, post_ln_2
      gc.collect()

  def _convert_dense_mlp_weights(self, blocks, prefix):
    """Converts gate/up/down projections for all layers (31B only)."""

    @jax.jit
    def _pack_mlp(mlp):
      gate = jnp.unstack(jnp.transpose(mlp["wi_0"]["kernel"], (1, 2, 0)), axis=0)
      up = jnp.unstack(jnp.transpose(mlp["wi_1"]["kernel"], (1, 2, 0)), axis=0)
      down = jnp.unstack(jnp.transpose(mlp["wo"]["kernel"], (1, 2, 0)), axis=0)
      return gate, up, down

    for slot in range(self.NUM_SLOTS):
      mlp = blocks[f"layers_{slot}"]["mlp"]
      gate_layers, up_layers, down_layers = _pack_mlp(mlp)
      for rep in range(self.num_reps):
        i = rep * self.NUM_SLOTS + slot
        p = f"{prefix}.{i}"
        self.vllm_state[f"{p}.mlp.gate_proj.weight"] = gate_layers[rep]
        self.vllm_state[f"{p}.mlp.up_proj.weight"] = up_layers[rep]
        self.vllm_state[f"{p}.mlp.down_proj.weight"] = down_layers[rep]
      del gate_layers, up_layers, down_layers
      gc.collect()
