"""Native logical axis rules, data sharding, and weight sharding configuration fields."""

import copy
from typing import Any

from pydantic.fields import Field
from pydantic.main import BaseModel


DEFAULT_DATA_SHARDING: list[list[str]] = [
    [
        "data",
        "stage",
        "fsdp",
        "fsdp_transpose",
        "context",
        "context_usp_ulysses",
        "context_autoregressive",
        "tensor",
        "tensor_sequence",
        "expert",
        "autoregressive",
    ]
]

DEFAULT_LOGICAL_AXIS_RULES: list[list] = [
    ["circular_repeats", []],
    # ==========================================
    # Vocabulary Embedding
    # ==========================================
    # Vocab Activations
    ["activation_embed_and_logits_batch", ["data", "stage", "fsdp", "fsdp_transpose", "expert"]],
    [
        "activation_embed_and_logits_batch_sequence",
        ["data", "stage", "fsdp", "fsdp_transpose", "context", "context_usp_ulysses", "expert"],
    ],
    ["activation_vocab", ["tensor", "tensor_sequence"]],
    ["activation_vocab", ["tensor"]],
    ["activation_vocab", "tensor_sequence"],
    # Vocab Weights
    ["vocab", ["tensor", "tensor_sequence", "autoregressive"]],
    ["embed_vocab", ["fsdp", "fsdp_transpose", "context", "context_usp_ulysses", "expert"]],
    # The four rules below are only used when lm_head_fsdp_as_vocab_tp is on; they rotate the
    # untied LM head off its embed dimension and onto its vocab dimension, so the FSDP axes
    # gather the hidden state (batch x length x embed) instead of the kernel (embed x vocab).
    ["activation_embed_and_logits_batch_no_fsdp", ["data", "stage", "expert"]],
    ["activation_vocab_fsdp", ["tensor", "tensor_sequence", "fsdp", "fsdp_transpose"]],
    ["vocab_fsdp", ["tensor", "tensor_sequence", "autoregressive", "fsdp", "fsdp_transpose"]],
    ["embed_vocab_replicated", []],
    # ==========================================
    # Attention
    # ==========================================
    # Attention Activations
    ["activation_batch_attn", ["data", "fsdp", "fsdp_transpose", "expert"]],
    ["activation_input_length_attn", ["tensor_sequence", "context"]],
    ["activation_heads", ["tensor", "tensor_sequence", "autoregressive"]],
    ["activation_kv_heads", ["tensor", "tensor_sequence"]],
    ["activation_length_attn", ["context", "context_usp_ulysses"]],
    ["activation_q_length", ["context", "context_usp_ulysses"]],
    ["activation_kv_length", []],
    ["activation_embed_attn", ["tensor"]],
    ["activation_kv", ["tensor", "tensor_sequence"]],
    ["activation_kv_batch", ["data", "fsdp", "fsdp_transpose", "expert"]],
    ["activation_kv_head_dim", ["tensor", "tensor_sequence"]],
    # Attention Weights
    ["heads", ["tensor", "tensor_sequence", "autoregressive"]],
    ["q_heads", ["tensor", "tensor_sequence", "autoregressive"]],
    ["kv_heads", ["tensor", "tensor_sequence", "autoregressive"]],
    ["qkv", []],
    ["kv", []],
    ["kv_head_dim", []],
    ["q_lora", ["fsdp", "fsdp_transpose", "context", "context_usp_ulysses", "expert"]],
    ["q_lora", ["fsdp", "context", "context_usp_ulysses", "expert"]],
    ["q_lora_up_proj", []],
    ["kv_lora", ["fsdp", "fsdp_transpose", "context", "context_usp_ulysses", "expert"]],
    ["kv_lora", ["fsdp", "context", "context_usp_ulysses", "expert"]],
    ["kv_lora_up_proj", []],
    ["embed_attn", ["fsdp", "fsdp_transpose", "context", "context_usp_ulysses", "expert"]],
    # ==========================================
    # Mixture of Experts (MoE)
    # ==========================================
    # MoE Activations
    ["activation_batch_moe", ["data", "fsdp", "fsdp_transpose", "expert"]],
    ["activation_length_moe", ["context", "context_usp_ulysses"]],
    ["activation_norm_length_moe", ["tensor_sequence", "context", "context_usp_ulysses"]],
    ["activation_embed_moe", ["tensor"]],
    ["activation_mlp_moe", ["tensor", "tensor_sequence"]],
    ["activation_exp", ["expert"]],
    # MoE Weights
    ["exp", "expert"],
    ["mlp_moe", ["fsdp_transpose", "tensor", "tensor_sequence", "autoregressive"]],
    ["embed_moe", ["fsdp", "fsdp_transpose", "context", "context_usp_ulysses"]],
    ["embed_moe", ["fsdp", "context", "context_usp_ulysses"]],
    ["embed_router", ["fsdp", "fsdp_transpose", "context", "context_usp_ulysses"]],
    ["embed_router", ["fsdp", "context", "context_usp_ulysses"]],
    # ==========================================
    # Standard MLP / Dense Layers / Model Structure
    # ==========================================
    # Dense Activations
    ["segment_ids_batch", ["data", "fsdp", "fsdp_transpose", "expert"]],
    ["activation_mlp", ["tensor", "tensor_sequence"]],
    # Note activation batch and length also get used in vocab
    ["activation_batch", ["data", "fsdp", "fsdp_transpose", "expert"]],
    ["activation_length", ["context", "context_usp_ulysses"]],
    ["activation_norm_length", ["tensor_sequence", "context", "context_usp_ulysses"]],
    ["activation_embed", ["tensor"]],
    ["activation_stage", "stage"],
    # General Weights
    ["mlp", ["fsdp_transpose", "tensor", "tensor_sequence", "autoregressive"]],
    ["gdn_head", ["fsdp_transpose", "tensor", "tensor_sequence", "autoregressive"]],
    ["embed", ["fsdp", "fsdp_transpose", "context", "context_usp_ulysses", "expert"]],
    ["embed", ["fsdp", "context", "context_usp_ulysses", "expert"]],
    ["norm", ["tensor"]],
    ["layers", "stage"],
    ["diloco", "diloco"],
    ["engram_dim", ["tensor"]],
    ["dense_layers", []],
    ["moe_layers", []],
    ["local_layers", []],
    ["mhc", []],
    # ==========================================
    # Inference (Prefill, Decode, Cache)
    # ==========================================
    ["prefill_activation_length", ["context", "context_usp_ulysses"]],
    ["prefill_activation_norm_length", ["tensor_sequence", "context", "context_usp_ulysses"]],
    ["activation_prefill_kv_batch", ["data", "fsdp", "fsdp_transpose", "expert"]],
    ["decode_batch", ["data", "fsdp", "fsdp_transpose", "expert"]],
    ["decode_length", []],
    ["cache_heads", ["autoregressive", "tensor", "tensor_sequence"]],
    ["paged_kv_heads", ["tensor"]],
    ["cache_batch_prefill", []],
    ["cache_batch", []],
    ["cache_heads_none", []],
    ["cache_kv", []],
    ["cache_sequence", []],
    ["num_pages", []],
    ["tokens_per_page", []],
    ["paged_kv_head_dim_size", []],
    # ==========================================
    # Deprecated / Scheduled for Removal
    # ==========================================
    ["mlp_no_fsdp", ["tensor", "tensor_sequence", "autoregressive"]],
    ["exp_with_fsdp", "fsdp"],
]


class LayoutAndSharding(BaseModel):
  """Configuration for data and model sharding rules."""

  logical_axis_rules: Any = Field(
      default_factory=lambda: copy.deepcopy(DEFAULT_LOGICAL_AXIS_RULES),
      description="Rules for mapping logical axes to physical mesh axes.",
  )
  logical_axis_rules_for_eval: Any = Field(
      default_factory=list,
      description="Rules for mapping logical axes to physical mesh axes during evaluation.",
  )
  data_sharding: Any = Field(
      default_factory=lambda: copy.deepcopy(DEFAULT_DATA_SHARDING),
      description="Sharding for input data.",
  )
  context_sharding: str = Field("context", description="Physical axis name for context parallelism.")
  ulysses_context_sharding: str = Field(
      "context_usp_ulysses",
      description=(
          "Physical axis name for the Ulysses head exchange under context_parallel_strategy='usp'. "
          "context_sharding names the ring dimension; this names the all-to-all dimension."
      ),
  )
  input_data_sharding_logical_axes: list[str] = Field(
      ["activation_embed_and_logits_batch", "activation_norm_length"],
      description="Logical axes for sharding input data.",
  )
  sharding_tolerance: float = Field(
      0.02,
      ge=0.0,
      le=1.0,
      description="Allowed percentage of non-sharded parameters.",
  )
  check_vma: bool = Field(
      False,
      description="Enabled check_vma flag in shard_map calls. Recommended for improved performance but only supported "
      "with auto sharding, megablox kernel, and EP / FSDP parallelisms.",
  )
  shard_optimizer_over_data: bool = Field(False, description="Enable ZeRO-1 optimizer sharding over the data axis.")
  dense_fsdp_use_two_stage_all_gather: bool = Field(
      False,
      description="Use two separate All-Gather calls for dense MLP weights sharded on both FSDP and FSDP-transpose.",
  )
  dense_weight_grad_in_kernel_order: bool | None = Field(
      None,
      description=(
          "Trace the per-layer attention and MLP projection dots inside a jax.sharding.auto_axes region so that "
          "XLA folds the transpose back into the gradient dot and each weight gradient comes out in its kernel's "
          "stored axis order. This is the in-loop counterpart of lm_head_weight_grad_in_kernel_order and, like it, "
          "None means on wherever it can act: shard_mode=explicit, where the Sharding custom-call on every dot "
          "output blocks that fold. Without it explicit is 0.33%-1.48% slower than auto at every measured layer "
          "count that is not a multiple of 8; with it explicit lands within 0.2% of auto everywhere "
          "(docs/guides/optimization/shard_mode_performance.md section 4.8). Stored kernels, their initialization "
          "and the arithmetic are untouched, so gradients are bit-identical to the default rule. No effect under "
          "shard_mode=auto, where XLA folds the transpose itself, or on quantized layers."
      ),
  )
  save_fsdp_gathered_weights: bool = Field(
      False,
      description=(
          "Gather each FSDP-sharded layer kernel once at the top of the scan body, in config.dtype, and keep the "
          "gathered value as a remat residual so the backward pass reads it instead of re-gathering. Mode-neutral: "
          "auto and explicit get the same answer to within half a point. Whether it is a win depends on how much of "
          "the backward pass is FSDP all-gather rather than activation traffic, so it has a sign condition: measured "
          "on v5p with the production SparseCore libtpu flags it is worth -3% to -8% at up to 2,048 tokens per "
          "device-step (mixtral-8x7b -8.3%, llama2-7b -5.2%, qwen3-8b -4.7%, and -6.3% at base_emb_dim 4096), but it "
          "costs +0.5% to +1.0% above ~4,096 tokens per device-step and regresses the gemma2/gemma3 family at any "
          "size, whose 256K-vocab logits dominate the step. See "
          "docs/guides/optimization/shard_mode_performance.md section 4.11 for the full table and the rule. Off by "
          "default for that reason and because the residuals grow HLO temporaries by 25%-75% (deepseek2-16b runs "
          "without it and OOMs with it); measure both step time and peak HBM before enabling. Numerics are "
          "unchanged -- only kernels are touched, and every consumer already casts them to config.dtype."
      ),
  )
  internal_compile: bool = Field(
      False,
      description="Use internal_compile to bypass open-source topology mappings.",
  )
  internal_compile_num_devices: int = Field(-1, description="Number of devices when using internal_compile.")
  compile_xla_flags: str = Field("", description="Compiler options for compilation only.")
  prefetch_fsdp_weights: bool = Field(
      False,
      description=(
          "Prefetch (all-gather) the next scanned layer's FSDP-sharded weights while the current layer runs, in"
          " both the forward and backward pass. Requires scan_layers=True. remat_policy applies to each layer;"
          " gathered weights are never saved and are re-gathered for the backward pass. Has no effect when no"
          " FSDP mesh axis spans more than one device. See docs/guides/optimization/fsdp_prefetch.md."
      ),
  )
  prefetch_fsdp_defer_grad_reduce_scatter: bool = Field(
      False,
      description=(
          "With prefetch_fsdp_weights and shard_mode=explicit, issue each layer's weight-gradient"
          " reduce-scatters at the start of the next layer's backward step, tagged for XLA's FSDP scheduler,"
          " which runs them on the SparseCore dedicated to FSDP collectives"
          " (--xla_tpu_explicit_fsdp_dedicated_sparse_core_id) and overlaps them with that layer's backward."
          " Keeps one extra layer of unreduced weight gradients live (more HBM). When False, they run untagged"
          " on the other SparseCore and complete within their own layer's backward step. For inhomogeneous"
          " blocks prefetched per sub-layer, the reduce-scatters are deferred to the next sub-layer's step."
      ),
  )
