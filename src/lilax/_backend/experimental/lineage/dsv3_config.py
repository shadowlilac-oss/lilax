"""Configuration dataclasses for the Lineage DSv3 model."""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from lilax._backend.experimental.lineage import dsv3_expert_shuffle
from lilax._backend.experimental.lineage import dsv3_sharding
from lilax._backend.experimental.lineage import quantization

QUANT_CONFIGS: dict[str, quantization.QuantConfig | None] = {
    "none": None,
    "fp8_full": quantization.FP8_FULL,
    "fp8_full_splash_qk": quantization.FP8_FULL_SPLASH_QK,
    "fp8_full_splash_qk_e3": quantization.FP8_FULL_SPLASH_QK_E3,
    "fp8_full_combine": quantization.FP8_FULL_COMBINE,
    "fp8_full_combine_per_hop": quantization.FP8_FULL_COMBINE_PER_HOP,
    "fp8_full_splash_qk_e3_combine_per_hop": quantization.FP8_FULL_SPLASH_QK_E3_COMBINE_PER_HOP,
}


def quant_config_from_name(name: str | None) -> quantization.QuantConfig | None:
  """Returns the quantization config selected by name, or None for bf16."""
  name = "none" if name is None else name
  if name not in QUANT_CONFIGS:
    raise ValueError(f"quantization must be one of {sorted(QUANT_CONFIGS)}, got: {name!r}")
  return QUANT_CONFIGS[name]


@dataclasses.dataclass(frozen=True)
class DSv3ModelConfig:
  """Model dimensions and dtypes."""

  vocab_size: int
  emb_dim: int
  num_dense_layers: int
  num_sparse_layers: int
  num_mtp_layers: int
  num_query_heads: int
  num_kv_heads: int
  q_lora_rank: int
  kv_lora_rank: int
  qk_head_dim: int
  rope_head_dim: int
  v_head_dim: int
  mscale: float
  mlp_dim: int
  moe_mlp_dim: int
  num_experts: int
  num_experts_per_tok: int
  n_routing_groups: int
  topk_routing_group: int
  routed_scaling_factor: float
  rope_theta: int
  max_position_embeddings: int
  original_max_position_embeddings: int
  rope_factor: int
  beta_fast: int
  beta_slow: int
  norm_epsilon: float
  tied_head: bool = False
  iota_embed: bool = False
  dtype: jax.typing.DTypeLike = jnp.bfloat16
  weight_dtype: jax.typing.DTypeLike = jnp.float32
  router_dtype: jax.typing.DTypeLike | None = None
  router_bias_dtype: jax.typing.DTypeLike = jnp.bfloat16
  head_dot_in_fp32: bool = False
  cast_logits_to_fp32: bool = True

  @property
  def topk_in_group(self) -> int:
    return self.num_experts_per_tok // self.topk_routing_group

  @property
  def logits_dtype(self) -> jax.typing.DTypeLike:
    if self.cast_logits_to_fp32 or self.head_dot_in_fp32:
      return jnp.float32
    return self.dtype


@dataclasses.dataclass(frozen=True)
class DSv3KernelConfig:
  """Kernel and runtime knobs of the Lineage layers."""

  max_target_length: int
  sa_block_q: int = 2048
  sa_block_kv: int = 2048
  sa_block_kv_compute: int = 2048
  sa_block_q_dkv: int = 2048
  sa_block_kv_dkv: int = 2048
  sa_block_kv_dkv_compute: int = 2048
  sa_q_layout: Any = "SEQ_MINOR"
  sa_k_layout: Any = "SEQ_MINOR"
  sa_v_layout: Any = "HEAD_DIM_MINOR"
  qk_diag_skip: bool = True
  qk_diag_grid: int = 8
  sv_diag_skip: bool = True
  diag_band_skip: bool = True
  dq_reduction_steps: int | None = 1
  q_prescaled_log2e: bool = False
  capacity_factor: float = 2.0
  ragged_buffer_factor: float = 1.2
  quant: quantization.QuantConfig | None = None
  max_async_overlap_transform: bool = True
  expert_permutation: str = "none"


@dataclasses.dataclass(frozen=True)
class DSv3TrainingConfig:
  """Training objective knobs."""

  load_balance_loss_weight: float = 0.0
  megatron_seq_aux_loss: bool = False
  routed_bias_update_rate: float = 0.0
  z_loss_multiplier: float = 0.0
  mtp_loss_scaling_factor: float = 0.1
  mtp_eval_target_module: int = 0
  mtp_reuse_input_embedding: bool = False


@dataclasses.dataclass(frozen=True)
class DSv3Config:
  """Full configuration of the Lineage DSv3 model."""

  model: DSv3ModelConfig
  sharding: dsv3_sharding.DSv3ShardingConfig
  kernels: DSv3KernelConfig
  training: DSv3TrainingConfig = DSv3TrainingConfig()
  eval_sharding: dsv3_sharding.DSv3ShardingConfig | None = None

  def validate(self) -> None:
    """Checks the constraints of the Lineage layers."""
    m, k, t = self.model, self.kernels, self.training
    if m.num_dense_layers < 2:
      raise ValueError("Lineage's dense layer scan needs at least 2 dense layers, got" f" {m.num_dense_layers}.")
    if m.num_sparse_layers < 3:
      raise ValueError("Lineage's sparse layer scan needs at least 3 sparse layers, got" f" {m.num_sparse_layers}.")
    if m.num_mtp_layers < 0:
      raise ValueError(f"num_mtp_layers must be >= 0, got {m.num_mtp_layers}.")
    if m.topk_routing_group <= 0 or (m.num_experts_per_tok % m.topk_routing_group):
      raise ValueError(
          f"num_experts_per_tok ({m.num_experts_per_tok}) must be a positive"
          f" multiple of topk_routing_group ({m.topk_routing_group})."
      )
    if m.num_kv_heads != m.num_query_heads:
      raise ValueError(f"num_kv_heads ({m.num_kv_heads}) must equal num_query_heads" f" ({m.num_query_heads}).")
    if k.dq_reduction_steps not in (None, 1, 3):
      raise ValueError("dq_reduction_steps must be one of (None, 1, 3), got" f" {k.dq_reduction_steps}.")
    if k.capacity_factor <= 0:
      raise ValueError(f"capacity_factor must be > 0, got {k.capacity_factor}.")
    if k.ragged_buffer_factor <= 0.0:
      raise ValueError(f"ragged_buffer_factor must be > 0, got {k.ragged_buffer_factor}.")
    if k.expert_permutation not in (dsv3_expert_shuffle.EXPERT_PERMUTATION_ALGORITHMS):
      raise ValueError(
          "expert_permutation must be one of"
          f" {dsv3_expert_shuffle.EXPERT_PERMUTATION_ALGORITHMS}, got"
          f" {k.expert_permutation!r}."
      )
    if t.mtp_eval_target_module < 0 or (t.mtp_eval_target_module > m.num_mtp_layers):
      raise ValueError(
          f"mtp_eval_target_module ({t.mtp_eval_target_module}) must be in" f" [0, num_mtp_layers={m.num_mtp_layers}]."
      )
    for name, sharding in (("sharding", self.sharding), ("eval_sharding", self.eval_sharding)):
      if sharding is None:
        continue
      missing = [axis for axis in dsv3_sharding.LOGICAL_AXES if axis not in sharding.axis_mapping]
      if missing:
        raise ValueError(f"{name}.axis_mapping lacks logical axes {missing}.")
