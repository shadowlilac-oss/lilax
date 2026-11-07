"""Pydantic-based configuration system for MaxText, organized into modular classes."""

# pylint: disable=too-many-lines

import datetime
from jinja2 import Environment, TemplateSyntaxError
import logging
from math import prod
import os
import yaml
from typing import Any, Literal, NewType

import jax
from lilax._backend.configs.batch_sizes import derive_batch_sizes
from lilax._backend.configs.checkpoint_validation import validate_checkpoint_source_config, validate_checkpoint_storage_config
from lilax._backend.configs.context_parallel_validation import validate_context_parallelism
from lilax._backend.configs.elastic_validation import validate_elastic_backup
from lilax._backend.configs.diagnostics_fields import ManagedMLDiagnostics
from lilax._backend.configs.inference_fields import (
    Decoding, InferenceBenchmark, InferenceGeneral, InferenceLayout, InferenceServer, PrefixCaching, SamplingStrategy,
)
from lilax._backend.configs.positional_embedding import PositionalEmbedding, Rope, YarnRope
from lilax._backend.configs.mesh_parallelism import (
    DcnParallelism, IciParallelism, _resolve_parallelism, _resolved_fsdp_size, resolve_parallelism_lists,
    validate_optimizer_sharding,
)
from lilax._backend.configs import moe_routing
from lilax._backend.configs import moe_quantization_validation
from lilax._backend.configs import attention_validation
from lilax._backend.configs import deepseek_attention_validation
from lilax._backend.configs.training_validation import validate_training_objective
from lilax._backend.configs.splash_defaults import resolve_local_splash_defaults
from lilax._backend.configs.precision_types import DType
from lilax._backend.configs.quantization_types import JAX_MX_QUANTIZATIONS, QuantizationType, TeCommGemmOverlapPolicy
from lilax._backend.configs.remat_types import RematLocation
from lilax._backend.configs.dataset_types import DatasetType
from lilax._backend.configs.profiling_types import ProfilerType, XProfTPUPowerTraceMode
from lilax._backend.configs.profiling_fields import Profiling, ProfilingCore
from lilax._backend.configs.run_fields import RunInfo
from lilax._backend.configs.checkpointing_fields import Checkpointing, EmergencyCheckpointing, OrbaxStorage
from lilax._backend.configs.precision_fields import DataTypes
from lilax._backend.configs.quantization_fields import Quantization
from lilax._backend.configs.model_architecture_fields import ModelArchitecture
from lilax._backend.configs.mtp.fields import MTP
from lilax._backend.configs.logits_fields import LogitsAndLoss
from lilax._backend.configs.moe_fields import MoEGeneral
from lilax._backend.configs.moe_kernel_fields import MoEKernels
from lilax._backend.configs.deepseek_moe_fields import DeepSeekMoE
from lilax._backend.configs.model_family_fields import Ming, Qwen3Next, Qwen3_8FlashNext
from lilax._backend.configs.hardware_mesh_fields import HardwareAndMesh
from lilax._backend.configs.sharding_fields import LayoutAndSharding
from lilax._backend.configs.pipeline_fields import PipelineParallelism
from lilax._backend.configs.remat_fields import RematAndOffload
from lilax._backend.configs.tokenizer_fields import Tokenizer, TokenizerType
from lilax._backend.configs.dataset_fields import (
    DatasetGeneral, GrainDataset, HfDataset, MMapDataset, OlmoGrainDataset, TfdsDataset,
)
from lilax._backend.configs.fine_tuning_fields import DPO, FineTuning, LoRA
from lilax._backend.configs.distillation_fields import Distillation
from lilax._backend.configs.training_loop_fields import TrainingLoop
from lilax._backend.configs.hyper_connection_fields import ManifoldConstrainedHyperConnections
from lilax._backend.configs.diloco_fields import DilocoParams
from lilax._backend.configs.optimizer_fields import AdamW, LearningRateScheduleType, Muon, Optimizer
from lilax._backend.configs.compilation_fields import AOT
from lilax._backend.configs.debugging_fields import DevelopmentAndDebugging
from lilax._backend.configs.hlo_dump_fields import HloDump
from lilax._backend.configs.monitoring_fields import GcpMonitoring, Goodput, Metrics, Tensorboard
from lilax._backend.configs.elastic_fields import ElasticTraining
from lilax._backend.configs.multimodal_fields import AudioEncoder, MultimodalGeneral, VisionProjector, VisionTower
from lilax._backend.configs.vllm_fields import VLLM
from lilax._backend.configs.rl_fields import RL, RLCluster, RLDataset, RLEvaluation, RLReward, RLSpecialTokens
from lilax._backend.configs.engram.fields import Engram
from lilax._backend.configs.scaled_model_dimension_fields import ScaledModelDimensions
from lilax._backend.configs.resolved_mesh_fields import ResolvedMeshTopology
from lilax._backend.configs.batch_size_fields import BatchSizeRampupEnd, BatchSizeRampupIncrements, GlobalBatchSizes
from lilax._backend.configs.output_directory_fields import RunOutputDirectories
from lilax._backend.configs.custom_remat_fields import CustomRematTensors
from lilax._backend.configs.model_scaling import get_individual_scales
from lilax._backend.configs.mtp.validation import validate_mtp_main_hidden_state
from lilax._backend.configs.training_quantization_validation import validate_training_quantization
from lilax._backend.configs.engram.validation import validate_engram
from lilax._backend.configs.moe_architecture_validation import validate_moe_architecture
from lilax._backend.configs.moe_kernel_validation import validate_gmm_v2, validate_moe_kernel_options
from lilax._backend.configs.diloco_settings import (
    resolve_diloco_replicas, validate_diloco_grouped_matmul, validate_streaming_diloco,
)
from lilax._backend.configs.multimodal_validation import validate_multimodal_support
from lilax._backend.configs.alignment_validation import validate_alignment_objectives
from lilax._backend.configs.explicit_sharding_validation import validate_explicit_sharding
from lilax._backend.configs.decoder_block_validation import validate_decoder_block_constraints
from lilax._backend.configs.splash_attention_validation import validate_splash_attention_flags
from lilax._backend.configs.lineage_validation import validate_lineage
from lilax._backend.common.common_types import DecoderBlockType, ShardMode, CustomRule, is_fp8_dtype
from lilax._backend.storage import gcs
from lilax._backend.observability import max_logging
from lilax._backend.utils import elastic_utils as elastic
from lilax._backend.utils.globals import MAXTEXT_ASSETS_ROOT, HF_IDS
from lilax._backend.distributed.mesh_topology import accelerator_to_spec_map
from lilax._backend.distributed.mesh_topology import slice_count
from pydantic.config import ConfigDict
from pydantic.fields import Field
from pydantic.functional_validators import model_validator
from pydantic.main import BaseModel
from pydantic.types import NonNegativeFloat, NonNegativeInt, PositiveInt
from pydantic import ValidationInfo


logger = logging.getLogger(__name__)


def validate_and_set_hlo_dump_defaults(config):
  if not config.run_name:
    if os.environ.get("JOBSET_NAME"):
      config.run_name = os.environ.get("JOBSET_NAME")
    else:
      config.run_name = f'{config.model_name}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}'
  if os.environ.get("XLA_FLAGS") and config.dump_hlo_xla_flags:
    raise ValueError("You must set either XLA_FLAGS or dump_hlo_xla_flags to dump HLO, but not both.")
  if not os.environ.get("XLA_FLAGS") and not config.dump_hlo_xla_flags:
    config.dump_hlo_xla_flags = f"--xla_dump_to={config.dump_hlo_local_dir} --xla_dump_large_constants"
    if config.dump_hlo_local_module_name:
      config.dump_hlo_xla_flags = (
          f"{config.dump_hlo_xla_flags} --xla_dump_hlo_module_re={config.dump_hlo_local_module_name}"
      )
  if not config.dump_hlo_gcs_dir:
    config.dump_hlo_gcs_dir = os.path.join(config.base_output_directory, config.run_name, "xla_dump")
  else:
    config.dump_hlo_gcs_dir = gcs.add_trailing_slash(config.dump_hlo_gcs_dir)
  if not config.dump_jaxpr_gcs_dir:
    config.dump_jaxpr_gcs_dir = os.path.join(config.base_output_directory, config.run_name, "jaxpr_dump")
  else:
    config.dump_jaxpr_gcs_dir = gcs.add_trailing_slash(config.dump_jaxpr_gcs_dir)
  if not os.environ.get("XLA_FLAGS"):
    os.environ["XLA_FLAGS"] = config.dump_hlo_xla_flags


def prepare_runtime_hlo_defaults(prepared: dict[str, Any], config_class) -> dict[str, Any]:
  config = config_class.model_construct(**prepared)
  validate_and_set_hlo_dump_defaults(config)
  return {
      **prepared,
      **{name: getattr(config, name) for name in (
          "run_name", "dump_hlo_xla_flags", "dump_hlo_gcs_dir", "dump_jaxpr_gcs_dir"
      )},
  }


# ----------------------------------------------------------------------------
# Type Aliases
# ----------------------------------------------------------------------------

AxisNames = NewType("AxisNames", str)


class Attention(BaseModel):
  """General configuration for the attention mechanism."""

  attention: str = Field(
      "autoselected",
      description="The attention algorithm to use (dot_product, flash, cudnn_flash_te, vllm_rpa, vllm_batched_rpa, etc).",
  )
  attention_type: Literal["global", "local_sliding", "chunk", "mla", "full", "compressed", "block_diffusion"] = Field(
      "global", description="The variant of attention to use."
  )
  share_kv_projections: bool = Field(
      False,
      description="If True, for global attention, Key and Value projections share the same weights.",
  )
  global_num_kv_heads: int = Field(
      0,
      description="If greater than 0, sets the number of KV heads for global attention.",
  )
  hidden_size_per_layer_input: NonNegativeInt = Field(
      0,
      description=(
          "Per-Layer-Embedding (PLE) inner dim used by Gemma 4 small models. "
          "When > 0, each decoder layer adds an extra PLE sub-block at the end."
      ),
  )
  vocab_size_per_layer_input: NonNegativeInt = Field(
      0,
      description=(
          "Vocab size of the per-layer input embedding table used by Gemma 4 small models. "
          "Defaults to 0 (disabled); typically set equal to `vocab_size`."
      ),
  )
  num_kv_shared_layers: NonNegativeInt = Field(
      0,
      description=(
          "Number of trailing decoder layers that reuse K/V from the last non-shared layer "
          "of the same attention type (sliding↔sliding, full↔full). Used by Gemma 4 small."
      ),
  )
  use_double_wide_mlp: bool = Field(
      False,
      description=(
          "When True, KV-shared layers double their MLP intermediate size to compensate for "
          "the missing K/V projections. Used by Gemma 4 small."
      ),
  )
  attention_sink: bool = Field(False, description="If True, enables attention sinks.")
  float32_qk_product: bool = Field(False, description="In dot-product attention, cast query-key product to fp32.")
  float32_logits: bool = Field(
      False,
      description="In dot-product attention, cast logits to fp32 before softmax.",
  )
  sliding_window_size: NonNegativeInt = Field(0, description="The size of the sliding window for local attention.")
  chunk_attn_window_size: NonNegativeInt = Field(0, description="The window size for chunked attention.")
  causal_block_size: PositiveInt = Field(
      32,
      description="The number of token positions in each bidirectional block for block-causal attention.",
  )
  attn_logits_soft_cap: None | NonNegativeFloat = Field(
      None, description="Soft-cap value for attention logits. None means no cap."
  )
  use_post_attn_norm: bool = Field(False, description="Apply LayerNorm after the attention block.")
  use_post_ffw_norm: bool = Field(False, description="Apply LayerNorm after the feed-forward block.")
  use_ragged_attention: bool = Field(False, description="Whether to use ragged attention kernels.")
  ragged_block_size: int = Field(256, description="Block size for ragged attention.")
  enable_padding_causal_mask: bool = Field(True, description="Temporary flag for Transformer Engine padding.")
  use_tokamax_splash: bool = Field(False, description="Whether to use tokamax splash attention.")
  use_jax_splash: bool = Field(False, description="Whether to use jax splash attention.")
  force_q_layout: bool = Field(False, description="Force the Q layout")
  use_qk_clip: bool = Field(False, description="Whether to use QK-Clip (MuonClip) for training stability.")
  qk_clip_threshold: float = Field(100.0, description="Threshold for QK-Clip (tau).")
  mla_qk_head_chunk_size: int = Field(
      0,
      ge=0,
      description=(
          "Chunk size over heads dimension for QK attention dot product in mla.  "
          "Default is 0 (no chunking). Reduces memory footprint at the cost of time."
      ),
  )
  csa_qk_head_chunk_size: int = Field(
      0,
      ge=0,
      description=(
          "Chunk size over heads dimension for QK attention dot product in CSA (DeepSeek-V4).  "
          "Default is 0 (no chunking). Reduces memory footprint at the cost of time."
      ),
  )


class MoBa(BaseModel):
  """Configuration for Mixture of Block Attention (MoBA)."""

  moba: bool = Field(False, description="If True, enables Mixture of Block Attention.")
  moba_chunk_size: int = Field(1024, description="The chunk size for MoBA.")
  moba_topk: int = Field(8, description="The number of top-k chunks to select in MoBA.")


class MlaAttention(BaseModel):
  """Configuration for Multi-Layer Attention (MLA)."""

  mla_naive_kvcache: bool = Field(True, description="Whether to use naive kvcache for MLA attention.")
  q_lora_rank: NonNegativeInt = Field(0, description="Query LoRA rank for MLA.")
  kv_lora_rank: NonNegativeInt = Field(512, description="Key/Value LoRA rank for MLA.")
  qk_nope_head_dim: NonNegativeInt = Field(128, description="Dimension for non-RoPE part of QK heads in MLA.")
  qk_rope_head_dim: NonNegativeInt = Field(64, description="Dimension for RoPE part of QK heads in MLA.")
  v_head_dim: NonNegativeInt = Field(128, description="Dimension of V heads in MLA.")
  use_sliced_mla_proj: bool = Field(
      False,
      description=(
          "Whether to slice projection kernel weights before contraction in MLA"
          " instead of running full projection + jnp.split."
      ),
  )
  use_mla_absorbed_mqa: bool = Field(
      False,
      description=(
          "Whether to use Absorbed-Query Latent MQA mode in MLA. Trades higher prefill/training FLOPs "
          "for 57x smaller KV cache footprint and memory bandwidth during autoregressive decode."
      ),
  )


class CompressedAttention(BaseModel):
  """Configuration for Compressed Attention."""

  o_lora_rank: NonNegativeInt = Field(0, description="Output LoRA rank for Compressed Attention.")
  o_groups: NonNegativeInt = Field(0, description="Output groups for Compressed Attention.")
  compress_ratios: list[int] = Field(
      default_factory=list,
      description="Per-layer compression ratios (0, 4, 128, etc).",
  )
  compressed_rope_max_timescale: int = Field(
      160000, description="If positive, used for Compressed Sparse/Heavy Attention."
  )
  use_csa_streamindex_kernel: bool = Field(
      False,
      description="Whether to use Pallas TPU kernel for CSA StreamIndex score computation.",
  )


class AttentionIndexer(BaseModel):
  """Configuration for DeepSeek Sparse Attention (DSA): MLA or Compressed Attention with indexer."""

  use_indexer: bool = Field(False, description="Whether to use sparse indexer for MLA or Compressed Attention.")
  indexer_head_dim: NonNegativeInt = Field(128, description="Head dim for indexer query and key.")
  indexer_n_heads: NonNegativeInt = Field(64, description="Number of query heads in indexer.")
  indexer_topk: NonNegativeInt = Field(2048, description="Number of tokens selected by the query token in indexer.")
  indexer_sparse_training: bool = Field(
      False,
      description="Determines the training strategy for the indexer: Dense Warm-up or Sparse Training stage.",
  )
  indexer_loss_scaling_factor: float = Field(0.0, description="Multiplier for the indexer KL divergence loss.")
  indexer_use_approx_top_k: bool = Field(
      False, description="Whether to use approximate top-k selection for the indexer on TPU."
  )
  indexer_approx_top_k_recall: float = Field(0.95, description="Recall target for approximate top-k selection.")
  indexer_mask_exact_topk: bool = Field(
      True,
      description=(
          "When True, enforces that exactly k elements are unmasked by the indexer under boundary ties."
          " When False, uses raw thresholding which is faster but may unmask > k elements"
          " during ties."
      ),
  )


class Llama4Attention(BaseModel):
  """Configuration specific to Llama4-style models."""

  use_qk_norm: bool = Field(
      False,
      description="Whether to apply L2 normalization to Query/Key vectors after RoPE.",
  )
  temperature_tuning: bool = Field(
      False,
      description="Dynamically scale attention temperature based on sequence length.",
  )


class SplashAttention(BaseModel):
  """Tunable block sizes for Splash Attention kernels."""

  sa_block_q: int = Field(512, description="Block size for Q in splash attention.")
  sa_block_kv: int = Field(512, description="Block size for KV in splash attention.")
  sa_block_kv_compute: int = Field(512, description="Block size for KV compute in splash attention.")
  eval_sa_block_q: int = Field(512, description="Block size for Q in splash attention during evaluation.")
  eval_sa_block_kv: int = Field(512, description="Block size for KV in splash attention during evaluation.")
  eval_sa_block_kv_compute: int = Field(
      512, description="Block size for KV compute in splash attention during evaluation."
  )
  sa_block_q_dkv: int = Field(512, description="Block size for Q_dkv in splash attention.")
  sa_block_kv_dkv: int = Field(512, description="Block size for KV_dkv in splash attention.")
  sa_block_kv_dkv_compute: int = Field(512, description="Block size for KV_dkv compute in splash attention.")
  sa_block_q_dq: int = Field(512, description="Block size for Q_dq in splash attention.")
  sa_block_kv_dq: int = Field(512, description="Block size for KV_dq in splash attention.")
  sa_use_fused_bwd_kernel: bool = Field(False, description="Use fused backward kernel in splash attention.")
  sa_bwd_dkv_megacore: bool = Field(
      False,
      description="Megacore-parallel kv-head groups in the static dkv grid. Needs >1 KV head; useful at local batch 1.",
  )
  sa_q_layout: str = Field("HEAD_DIM_MINOR", description="Layout for Q in splash attention.")
  sa_k_layout: str = Field("HEAD_DIM_MINOR", description="Layout for K in splash attention.")
  sa_v_layout: str = Field("HEAD_DIM_MINOR", description="Layout for V in splash attention.")
  eval_sa_q_layout: str = Field("HEAD_DIM_MINOR", description="Layout for Q in splash attention during evaluation.")
  eval_sa_k_layout: str = Field("HEAD_DIM_MINOR", description="Layout for K in splash attention during evaluation.")
  eval_sa_v_layout: str = Field("HEAD_DIM_MINOR", description="Layout for V in splash attention during evaluation.")
  use_splash_scheduler: bool = Field(False, description="Use experimental splash attention scheduler.")
  ring_scan_unroll: NonNegativeInt = Field(
      1,
      description=(
          "Unroll factor for the Tokamax ring attention scan. 0 fully unrolls; values at or above the ring size are "
          "equivalent to full unrolling. Program size and compile time grow with the factor."
      ),
  )
  sa_fuse_reciprocal: bool = Field(True, description="Maps to fuse_reciprocal in SplashConfig.")
  sa_use_base2_exp: bool = Field(True, description="Maps to use_base2_exp in SplashConfig.")
  sa_qk_diag_skip: bool = Field(False, description="Maps to qk_diag_skip in SplashConfig.")
  sa_sv_diag_skip: bool = Field(False, description="Maps to sv_diag_skip in SplashConfig.")
  sa_qk_diag_grid: int = Field(2, description="Maps to qk_diag_grid in SplashConfig.")
  # If None, each local_sa_* flag inherits from the corresponding sa_* flag.
  local_sa_block_q: int | None = Field(None, description="Block size for Q in local splash attention.")
  local_sa_block_kv: int | None = Field(None, description="Block size for KV in local splash attention.")
  local_sa_block_kv_compute: int | None = Field(None, description="Block size for KV compute in local splash attention.")
  eval_local_sa_block_q: int | None = Field(
      None, description="Block size for Q in local splash attention during evaluation."
  )
  eval_local_sa_block_kv: int | None = Field(
      None, description="Block size for KV in local splash attention during evaluation."
  )
  eval_local_sa_block_kv_compute: int | None = Field(
      None, description="Block size for KV compute in local splash attention during evaluation."
  )
  local_sa_block_q_dkv: int | None = Field(None, description="Block size for Q_dkv in local splash attention.")
  local_sa_block_kv_dkv: int | None = Field(None, description="Block size for KV_dkv in local splash attention.")
  local_sa_block_kv_dkv_compute: int | None = Field(
      None, description="Block size for KV_dkv compute in local splash attention."
  )
  local_sa_block_q_dq: int | None = Field(None, description="Block size for Q_dq in local splash attention.")
  local_sa_block_kv_dq: int | None = Field(None, description="Block size for KV_dq in local splash attention.")
  local_sa_use_fused_bwd_kernel: bool | None = Field(
      None, description="Use fused backward kernel in local splash attention."
  )
  local_sa_q_layout: str | None = Field(None, description="Layout for Q in local splash attention.")
  local_sa_k_layout: str | None = Field(None, description="Layout for K in local splash attention.")
  local_sa_v_layout: str | None = Field(None, description="Layout for V in local splash attention.")
  eval_local_sa_q_layout: str | None = Field(
      None, description="Layout for Q in local splash attention during evaluation."
  )
  eval_local_sa_k_layout: str | None = Field(
      None, description="Layout for K in local splash attention during evaluation."
  )
  eval_local_sa_v_layout: str | None = Field(
      None, description="Layout for V in local splash attention during evaluation."
  )
  local_use_splash_scheduler: bool | None = Field(None, description="Use experimental local splash attention scheduler.")
  local_sa_fuse_reciprocal: bool | None = Field(None, description="Maps to local fuse_reciprocal in SplashConfig.")
  local_sa_use_base2_exp: bool | None = Field(None, description="Maps to local use_base2_exp in SplashConfig.")
  experimental_sa_quant_q_fp8: bool = Field(
      False,
      description=(
          "Experimental flag: If enabled, the Q tensor in splash attention is"
          " quantized to jnp.float8_e4m3fn, without scaling factors."
      ),
  )
  experimental_sa_quant_k_fp8: bool = Field(
      False,
      description=(
          "Experimental flag: If enabled, the K tensor in splash attention is"
          " quantized to jnp.float8_e4m3fn, without scaling factors."
      ),
  )
  sa_dkv_reuse_scaled_q: bool = Field(
      False,
      description=(
          "Experimental flag (maps to SplashConfig.dkv_reuse_scaled_q). With sa_use_base2_exp=True, the"
          " fused dkv backward kernel normally keeps two copies of q live (LOG2E-scaled q for logits and"
          " unscaled q for dK). When True, dkv reuses the LOG2E-scaled q for the dK matmul and divides dK"
          " by LOG2E, keeping only one live q copy. Works with bf16 or fp8 q across all attention types."
          " Subsumed by sa_q_prescaled_log2e; prefer sa_dkv_reuse_scaled_q when q is bf16 or on paths where"
          " sa_q_prescaled_log2e is not supported (non-MLA or use_mla_absorbed_mqa=True)."
      ),
  )
  sa_q_prescaled_log2e: bool = Field(
      False,
      description=(
          "Experimental flag (maps to SplashConfig.q_prescaled_log2e). With sa_use_base2_exp=True and"
          " experimental_sa_quant_q_fp8=True, in-kernel LOG2E scaling rounds q to fp8 a second time after"
          " MaxText's softmax_scale cast. When True, attention_mla.py folds LOG2E into the f32"
          " softmax_scale multiply before the fp8 cast (single fp8 rounding), and the kernel skips internal"
          " q scaling in fwd/dq/dkv and divides dq and dk by LOG2E. Strictly subsumes sa_dkv_reuse_scaled_q"
          " (sa_q_prescaled_log2e > sa_dkv_reuse_scaled_q > both False for non-absorbed MLA with fp8 q)."
      ),
  )
  sa_ignore_segment_ids: bool = Field(
      False,
      description=(
          "Do not pass decoder segment ids to splash attention, which skips the"
          " per-block segment mask. Only valid when every row is a single"
          " segment: dataset_type='c4_mlperf' with stream chunking for both"
          " train and eval. Requires attention='flash' and attention_type in"
          " ('global', 'mla', 'local_sliding', 'chunk')."
      ),
  )
  use_max_logit_estimate: int = Field(
      -1,
      description="-1 means no estimate, any > 0 value will be used as max logit estimate",
  )
  cost_estimate_flops_fwd: int = Field(
      -1,
      description="-1 means using splash default cost estimation, any >= 0 value will be used as cost estimation for splash "
      "to overlap for communication (forward)",
  )
  cost_estimate_flops_bwd: int = Field(
      -1,
      description="-1 means using splash default cost estimation, any >= 0 value will be used as cost estimation for splash "
      "to overlap for communication (backward)",
  )
  dq_reduction_steps: int = Field(
      0,
      description="the number of reduction steps. For now, only 3 or all the kv steps are supported.",
  )


def derive_managed_mldiagnostics_dir(
    base_output_directory: str | None,
    run_name: str | None,
    managed_mldiagnostics_storage_path: str | None,
) -> str | None:
  """Returns the managed ML Diagnostics root for a run, or None if it cannot be derived.

  Layout: `<managed_mldiagnostics_storage_path or base_output_directory>/<run_name>/managed-mldiagnostics`,
  without a trailing slash (the SDK mis-joins paths that end in one, b/454725283). The SDK appends its
  own run identifier below this directory for profiles.

  Args:
    base_output_directory: The run's base output directory; used when no dedicated storage path is set.
    run_name: The run name; the directory cannot be derived without it.
    managed_mldiagnostics_storage_path: Optional dedicated storage root that takes priority over
      `base_output_directory`.

  Returns:
    The derived directory, or None when `run_name` or both base paths are empty.
  """
  telemetry_base = managed_mldiagnostics_storage_path or base_output_directory
  if telemetry_base and run_name:
    return os.path.join(telemetry_base, run_name, "managed-mldiagnostics")
  return None


FORCED_ROUTING_SUPPORTED_DECODER_BLOCKS = (
    DecoderBlockType.QWEN3_5,
    DecoderBlockType.MIXTRAL,
    DecoderBlockType.GEMMA4,
)


def check_forced_routing_support(decoder_block: DecoderBlockType) -> None:
  """Raises NotImplementedError if `decoder_block` does not support router replay."""
  if decoder_block not in FORCED_ROUTING_SUPPORTED_DECODER_BLOCKS:
    raise NotImplementedError(
        "Forced routing (router replay) is only supported for decoder_block in"
        f" {FORCED_ROUTING_SUPPORTED_DECODER_BLOCKS}; got"
        f" decoder_block={decoder_block!r}."
    )


def _normalize_axes(axes: Any) -> tuple[str, ...]:
  """Normalize a logical-rule mapping value to a tuple of axis name strings.

  Args:
    axes: The right-hand side of a logical axis rule entry.  Can be a single
      string, a list/tuple of strings, or an empty list.

  Returns:
    A (possibly empty) tuple of physical axis name strings.
  """
  if axes is None:
    return ()
  if isinstance(axes, str):
    return (axes,)
  if isinstance(axes, (list, tuple)):
    return tuple(a for a in axes if isinstance(a, str))
  return ()


def axes_for_logical(logical_axis_rules: list, logical_axis: str) -> tuple[str, ...]:
  """Return the physical axes *logical_axis* is mapped to by *logical_axis_rules*.

  Args:
    logical_axis_rules: The list of ``[logical_name, physical_axes]`` pairs.
    logical_axis: The logical axis name to look up.

  Returns:
    A (possibly empty) tuple of physical axis name strings.
  """
  for rule in logical_axis_rules:
    if rule and len(rule) >= 2 and rule[0] == logical_axis:
      return _normalize_axes(rule[1])
  return ()


def infer_cp_axes(logical_axis_rules: list) -> tuple[str, ...]:
  """Infer which physical mesh axis/axes serve as Context Parallelism (CP).

  Scans *logical_axis_rules* for the ``activation_length`` logical axis and
  returns the physical axis/axes it is mapped to.

  Args:
    logical_axis_rules: The list of ``[logical_name, physical_axes]`` pairs
      (the ``logical_axis_rules`` config field).

  Returns:
    A tuple of physical axis name strings that act as CP.  Empty if the
    ``activation_length`` logical axis is not found in the rules.
  """
  return axes_for_logical(logical_axis_rules, "activation_length")


def infer_ep_axes(logical_axis_rules: list) -> tuple[str, ...]:
  """Infer which physical mesh axis/axes serve as Expert Parallelism (EP).

  Scans *logical_axis_rules* for the ``exp`` logical axis and returns the
  physical axis/axes it is mapped to.

  Args:
    logical_axis_rules: The list of ``[logical_name, physical_axes]`` pairs
      (the ``logical_axis_rules`` config field).

  Returns:
    A tuple of physical axis name strings that act as EP.  Empty if the
    ``exp`` logical axis is not found in the rules.
  """
  for rule in logical_axis_rules:
    if rule and len(rule) >= 2 and rule[0] == "exp":
      return _normalize_axes(rule[1])
  return ()


def ep_enabled(config):
  """Determine if Expert Parallelism (EP) is enabled based on the config."""
  ep_axes = infer_ep_axes(config.logical_axis_rules)
  ep_rank = 1
  for axis_name in ep_axes:
    ici_val = getattr(config, f"ici_{axis_name}_parallelism", 1)
    dcn_val = getattr(config, f"dcn_{axis_name}_parallelism", 1)
    if ici_val == -1 or dcn_val == -1:
      ep_rank = -1
      break
    ep_rank *= ici_val * max(dcn_val, 1)
  return ep_rank != 1


class MaxTextConfig(
    RunInfo,
    Checkpointing,
    OrbaxStorage,
    EmergencyCheckpointing,
    ElasticTraining,
    DataTypes,
    Quantization,
    ModelArchitecture,
    Engram,
    MTP,
    LogitsAndLoss,
    Attention,
    MlaAttention,
    CompressedAttention,
    MoBa,
    AttentionIndexer,
    Llama4Attention,
    SplashAttention,
    MoEGeneral,
    MoEKernels,
    DeepSeekMoE,
    Qwen3Next,
    Qwen3_8FlashNext,
    Ming,
    HardwareAndMesh,
    LayoutAndSharding,
    DcnParallelism,
    IciParallelism,
    PipelineParallelism,
    RematAndOffload,
    TrainingLoop,
    ManifoldConstrainedHyperConnections,
    DilocoParams,
    Optimizer,
    AdamW,
    Muon,
    FineTuning,
    Distillation,
    VLLM,
    PositionalEmbedding,
    Rope,
    YarnRope,
    DatasetGeneral,
    TfdsDataset,
    HfDataset,
    GrainDataset,
    MMapDataset,
    OlmoGrainDataset,
    Tokenizer,
    InferenceGeneral,
    Decoding,
    InferenceLayout,
    InferenceServer,
    InferenceBenchmark,
    PrefixCaching,
    AOT,
    DevelopmentAndDebugging,
    Profiling,
    HloDump,
    Metrics,
    Goodput,
    GcpMonitoring,
    Tensorboard,
    ManagedMLDiagnostics,
    MultimodalGeneral,
    VisionTower,
    VisionProjector,
    AudioEncoder,
    BatchSizeRampupIncrements,
    CustomRematTensors,
    BatchSizeRampupEnd,
    RunOutputDirectories,
    GlobalBatchSizes,
    ResolvedMeshTopology,
    ScaledModelDimensions,
):
  """
  The main configuration object for MaxText.

  This class aggregates all configuration options from modular `BaseModel` classes
  into a single, validated object. It is populated by the `initialize` function.
  Every field is explicitly defined to prevent misconfigurations (`extra='forbid'`).
  """

  dpo: DPO = Field(
      default_factory=DPO,
      description="Configuration for DPO and ORPO alignment algorithms.",
  )
  lora: LoRA = Field(
      default_factory=LoRA,
      description="Configuration for LoRA / QLoRA adapters.",
  )
  model_config = ConfigDict(extra="forbid", protected_namespaces=())

  @model_validator(mode="before")
  @classmethod
  def load_model_specific_defaults(cls, values: dict[str, Any]) -> dict[str, Any]:
    """This method is a no-op because `pyconfig` handles model-specific config loading."""
    return values

  def _validate_check_vma_is_supported(self):
    """Validates that check_vma is used with supported settings."""
    if not self.check_vma:
      return
    if self.shard_mode != ShardMode.AUTO:
      raise ValueError(f"check_vma requires shard_mode='auto', got shard_mode='{self.shard_mode.value}'.")
    if self.use_tokamax_gmm:
      raise ValueError("check_vma is not yet supported with tokamax gmm kernel.")
    if self.use_ragged_sort:
      raise ValueError("check_vma is not yet supported with ragged sort kernel.")
    if self.use_ring_of_experts:
      raise ValueError("check_vma is not yet supported with ring of experts.")
    _allowed = {"ici_expert_parallelism", "ici_fsdp_parallelism"}
    active = [name for name in IciParallelism.model_fields if name not in _allowed and getattr(self, name) != 1]
    if active:
      raise ValueError(
          f"check_vma=True only supports ici_expert_parallelism and ici_fsdp_parallelism. "
          f"Found other ICI axes enabled: {active}."
      )

  validate_moe_dropless_fallback = moe_routing.validate_moe_dropless_fallback
  validate_retry_dropless_first_steps_and_first_phase_buffer = (
      moe_routing.validate_retry_dropless_first_steps_and_first_phase_buffer
  )
  validate_moe_log_max_load_ratio = moe_routing.validate_moe_log_max_load_ratio

  def validate_moe_tc_ragged_sort(self):
    """Validates that moe_tc_ragged_sort is used with the truncated-buffer ring-of-experts ragged sort."""
    if self.moe_tc_ragged_3d_dispatch and not (self.moe_tc_ragged_sort and self.moe_tc_ragged_3d_gmm):
      raise ValueError("moe_tc_ragged_3d_dispatch=True requires moe_tc_ragged_sort=True and moe_tc_ragged_3d_gmm=True.")
    if self.moe_tc_routing_checkpoint != RematLocation.REMAT and not (
        self.moe_tc_ragged_sort and self.moe_tc_ragged_weights_on_activation
    ):
      raise ValueError(
          f"moe_tc_routing_checkpoint={RematLocation(self.moe_tc_routing_checkpoint).value} requires"
          " moe_tc_ragged_sort=True and moe_tc_ragged_weights_on_activation=True; use moe_tc_routing_checkpoint=remat."
      )
    if self.moe_tc_window_routing and not self.moe_tc_ragged_sort:
      raise ValueError("moe_tc_window_routing=True requires moe_tc_ragged_sort=True.")
    if not self.moe_tc_ragged_sort:
      return
    if not self.use_ring_of_experts:
      raise ValueError("moe_tc_ragged_sort=True requires use_ring_of_experts=True.")
    if not self.use_ragged_sort:
      raise ValueError("moe_tc_ragged_sort=True requires use_ragged_sort=True.")
    if self.ragged_buffer_factor <= 0:
      raise ValueError("moe_tc_ragged_sort=True requires ragged_buffer_factor > 0.0.")
    if self.moe_tc_ragged_3d_gmm:
      if not (self.use_tokamax_gmm and self.use_gmm_v2):
        raise ValueError("moe_tc_ragged_3d_gmm=True requires use_tokamax_gmm=True and use_gmm_v2=True.")
      unsupported = ["mlp_bias"] if self.mlp_bias else []
      if self.ici_tensor_parallelism > 1 or self.dcn_tensor_parallelism > 1:
        unsupported.append("tensor parallelism")
      if self.num_moe_emb_chunks > 0:
        unsupported.append("num_moe_emb_chunks > 0")
      if unsupported:
        raise ValueError(f"moe_tc_ragged_3d_gmm=True does not support: {', '.join(unsupported)}.")

  validate_moe_topk_before_ep_all_gather = moe_routing.validate_moe_topk_before_ep_all_gather
  validate_ragged_buffer_factor = moe_routing.validate_ragged_buffer_factor

  def _validate_te_grouped_gemm(self):
    """Validates that use_te_grouped_gemm is not combined with unsupported backends."""
    if not self.use_te_grouped_gemm:
      return
    if not self.sparse_matmul:
      raise ValueError(
          "use_te_grouped_gemm=True requires sparse_matmul=True; the TransformerEngine "
          "grouped GEMM backend is only reachable from the sparse MoE path, so it would "
          "otherwise be silently ignored. Note that some GPU model configs "
          "(e.g. gpu/models/gpt-oss-20b.yml) set sparse_matmul=false."
      )
    if self.megablox:
      raise ValueError(
          "use_te_grouped_gemm=True is not compatible with megablox=True. "
          "Set megablox=False to use the TransformerEngine grouped GEMM backend."
      )
    if self.use_tokamax_gmm:
      raise ValueError(
          "use_te_grouped_gemm=True is not compatible with use_tokamax_gmm=True. "
          "Disable Tokamax GMM to use the TransformerEngine grouped GEMM backend."
      )
    if self.quantization:
      raise ValueError(
          "use_te_grouped_gemm=True does not support quantization yet "
          f"(got quantization={self.quantization!r}). Set quantization=''."
      )
    if self.moe_tc_ragged_sort:
      raise ValueError(
          "use_te_grouped_gemm=True is not compatible with moe_tc_ragged_sort=True: the TensorCore ragged sort "
          "applies routing weights per buffer row, which TE's per-expert grouped GEMM bias cannot carry."
      )
    if self.moe_flat_fsdp_weights:
      raise ValueError(
          "use_te_grouped_gemm=True is not compatible with moe_flat_fsdp_weights=True: the TransformerEngine "
          "grouped GEMM takes 3D [experts, rows, cols] expert weights, not the flat 2D layout."
      )
    # TE v1 grouped GEMM copies group_sizes D2H and stream-syncs inside the FFI.
    # Under a single-process multi-device shard_map that host thread never
    # reaches the collective rendezvous, so the job hangs instead of failing.
    # Same constraint as te_comm_gemm_overlap: one local device per process.
    if jax.distributed.is_initialized() and jax.local_device_count() > 1:
      raise ValueError(
          "use_te_grouped_gemm=True requires one local device per process "
          f"(got jax.local_device_count()={jax.local_device_count()}). "
          "Launch with one GPU per rank (e.g. SLURM_STEP_GPUS / gpu_per_device_run.sh); "
          "do not pass several devices to a single process."
      )

  def _validate_te_comm_gemm_overlap(self):
    """Validates that te_comm_gemm_overlap is used with supported settings to enable TE Collective GEMM ops."""
    te_has_distributed_env = jax.local_device_count() == 1 and jax.distributed.is_initialized()

    if self.hardware != "gpu_multiprocess" or not te_has_distributed_env:
      raise ValueError(
          "TE Collective GEMM operations are only supported for hardware=gpu_multiprocess with rank per GPU."
      )

    tsp_size = self.ici_tensor_sequence_parallelism * self.dcn_tensor_sequence_parallelism
    tp_size = self.ici_tensor_parallelism * self.dcn_tensor_parallelism

    if tsp_size == 1 or tp_size > 1:
      raise ValueError("TE Collective GEMM operations are only supported for TSP size > 1 and TP size == 1.")

    if not self.quantization.startswith("te_"):
      raise ValueError(
          "TE Collective GEMM operations are only supported for TE quantization recipes (i.e. starting with 'te_')."
      )

  def _validate_usp_context_parallelism(self):
    """Validates the USP (Ulysses over ring) context parallelism configuration."""
    if self.context_parallel_strategy != "usp":
      if self.ici_context_usp_ulysses_parallelism != 1 or self.dcn_context_usp_ulysses_parallelism != 1:
        raise ValueError(
            "ici/dcn_context_usp_ulysses_parallelism was specified, but is only supported when "
            "context_parallel_strategy='usp'."
        )
      return
    if self.hardware != "tpu":
      raise ValueError("USP context parallelism (context_parallel_strategy='usp') is only supported on TPU.")
    if self.context_sharding != "context":
      raise ValueError("TPU USP attention requires context_sharding='context'.")
    usp_sequence_axes = (self.context_sharding, self.ulysses_context_sharding)
    for usp_axis in usp_sequence_axes:
      if usp_axis not in self.mesh_axes:
        raise ValueError(f"TPU USP attention requires mesh axis '{usp_axis}' in mesh_axes.")
    if infer_cp_axes(self.logical_axis_rules) != usp_sequence_axes:
      raise ValueError(
          f"TPU USP attention requires activation_length to map to {usp_sequence_axes} in logical_axis_rules."
      )
    if infer_cp_axes(self.logical_axis_rules_for_eval) != usp_sequence_axes:
      raise ValueError(
          f"TPU USP attention requires activation_length to map to {usp_sequence_axes} in logical_axis_rules_for_eval."
      )
    usp_ring_size = self.ici_context_parallelism
    usp_ulysses_size = self.ici_context_usp_ulysses_parallelism
    if (
        usp_ring_size <= 0
        or usp_ulysses_size <= 0
        or self.dcn_context_parallelism <= 0
        or self.dcn_context_usp_ulysses_parallelism <= 0
    ):
      raise ValueError(
          "TPU USP attention requires explicit positive ici/dcn context parallelism values; "
          "inferred (-1) sizes are not supported."
      )
    if usp_ring_size <= 1:
      raise ValueError("TPU USP attention requires ici_context_parallelism > 1 for the ring dimension.")
    if usp_ulysses_size <= 1:
      raise ValueError("TPU USP attention requires ici_context_usp_ulysses_parallelism > 1 for the Ulysses dimension.")
    if self.dcn_context_parallelism != 1 or self.dcn_context_usp_ulysses_parallelism != 1:
      raise ValueError("TPU USP attention does not support dcn context parallelism yet.")
    if self.attention != "flash":
      raise ValueError("TPU USP attention requires attention=flash.")
    if not self.use_tokamax_splash:
      raise ValueError("TPU USP attention requires use_tokamax_splash=True.")
    if self.use_jax_splash:
      raise ValueError("TPU USP attention requires use_jax_splash=False.")
    if self.use_indexer:
      raise ValueError("TPU USP attention does not support sparse indexer masks.")
    if self.attention_type != "global":
      raise ValueError("TPU USP attention is initially supported only for global causal attention.")
    if self.context_parallel_load_balance and usp_ring_size % 2 != 0:
      raise ValueError(
          "TPU USP attention with context_parallel_load_balance=True requires an even ici_context_parallelism."
      )
    if self.use_ragged_attention:
      raise ValueError("TPU USP attention does not support ragged attention.")
    if self.attention_sink:
      raise ValueError("TPU USP attention does not support attention sinks.")
    if self.use_chunked_prefill:
      raise ValueError("TPU USP attention does not support chunked prefill yet.")
    if self.use_multimodal:
      raise ValueError("TPU USP attention does not support multimodal attention.")
    if self.enable_dropout and self.dropout_rate > 0.0:
      raise ValueError("TPU USP attention does not support dropout yet.")
    if self.dq_reduction_steps not in (0, 3):
      raise ValueError("TPU USP attention requires dq_reduction_steps to be 0 or 3.")
    if self.use_qk_clip:
      raise ValueError("TPU USP attention does not support QK-Clip statistics yet.")
    if self.mtp_num_layers > 0:
      raise ValueError("TPU USP attention does not support multi-token prediction (mtp_num_layers > 0) yet.")
    if self.sa_bwd_dkv_megacore:
      raise ValueError("TPU USP attention does not support sa_bwd_dkv_megacore yet.")
    if self.max_target_length % (usp_ring_size * usp_ulysses_size) != 0:
      raise ValueError(
          "TPU USP attention requires max_target_length "
          f"({self.max_target_length}) to be divisible by the total context parallelism "
          f"({usp_ring_size * usp_ulysses_size})."
      )
    if self.max_target_length % (usp_ring_size * usp_ring_size) != 0:
      raise ValueError("TPU USP attention requires max_target_length to be divisible by ici_context_parallelism squared.")
    if self.num_query_heads % usp_ulysses_size != 0:
      raise ValueError(
          "TPU USP attention requires num_query_heads "
          f"({self.num_query_heads}) to be divisible by ici_context_usp_ulysses_parallelism ({usp_ulysses_size})."
      )
    if self.num_kv_heads == 1:
      raise ValueError("TPU USP attention does not support MQA with ici_context_usp_ulysses_parallelism > 1.")
    if self.num_kv_heads % usp_ulysses_size != 0:
      raise ValueError(
          "TPU USP attention requires num_kv_heads "
          f"({self.num_kv_heads}) to be divisible by ici_context_usp_ulysses_parallelism ({usp_ulysses_size})."
      )

  def validate_mllog(self):
    """
    Warns when MLPerf logging is enabled without evaluation.

    The compliance checker requires at least one eval_accuracy event (REQ: AT_LEAST_ONE in
    common.yaml), so a no-eval run is useful for measuring evaluation overhead on end-to-end
    time but will not pass MLPerf compliance checking.
    """
    if self.enable_mllog and self.eval_interval <= 0:
      max_logging.warning(
          f"enable_mllog=True with eval_interval={self.eval_interval} (<= 0): running without "
          "evaluation. The resulting log will not pass MLPerf compliance (which requires "
          "at least one eval_accuracy event)."
      )

  validate_splash_diag_skip = attention_validation.validate_splash_diag_skip

  def validate_num_moe_emb_chunks(self):
    """
    Validates that num_moe_emb_chunks is used with supported settings.
    """
    if self.num_moe_emb_chunks > 0:
      if not self.use_gmm_v2 or not self.use_ring_of_experts:
        raise ValueError(
            f"num_moe_emb_chunks > 0 requires use_gmm_v2=True and use_ring_of_experts=True. "
            f"Got use_gmm_v2={self.use_gmm_v2}, use_ring_of_experts={self.use_ring_of_experts}."
        )
      # The emb-chunking path (moe_emb_chunking in moe.py) routes per chunk and does not tag its routed input with
      # checkpoint_name("moe_x_sorted"), so a non-remat moe_x_sorted would be silently ignored there.
      if self.moe_x_sorted != RematLocation.REMAT:
        raise ValueError(
            f"moe_x_sorted={RematLocation(self.moe_x_sorted).value} is not supported with num_moe_emb_chunks > 0; "
            "use moe_x_sorted=remat."
        )

  validate_moe_quantize_token_all_gather = moe_quantization_validation.validate_moe_quantize_token_all_gather
  validate_moe_quantize_combine_bwd_method = moe_quantization_validation.validate_moe_quantize_combine_bwd_method

  @staticmethod
  def _load_mesh_config_from_yaml(rule_value: str) -> dict:
    """Helper to load and parse custom mesh YAML configurations."""
    custom_mesh_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "custom_mesh_and_rule",
        f"{rule_value}.yml",
    )

    if not os.path.exists(custom_mesh_path):
      # ValueError is more semantically correct for validation errors than NotImplementedError
      raise ValueError(f"Custom mesh config file not found at {custom_mesh_path}")

    # Explicitly setting encoding removes the need for the pylint disable comment
    with open(custom_mesh_path, "r", encoding="utf-8") as f:
      return yaml.safe_load(f) or {}

  @model_validator(mode="after")
  def validate_shard_embed_moe_on_fsdp(self) -> "MaxTextConfig":
    """Raise ValueError if shard_embed_moe_on_fsdp is used without fixed weight quantization calibration."""
    if self.shard_embed_moe_on_fsdp and (
        self.quantization == "" or not self.weight_quantization_calibration_method.startswith("fixed")
    ):
      raise ValueError(
          "shard_embed_moe_on_fsdp requires quantization to be specified and "
          "weight_quantization_calibration_method to be fixed (static scaling mode)."
      )
    return self

  def _validate_jax_mx_quantization(self) -> None:
    """Rejects jax_mxfp8 and jax_mxfp4 settings that would silently leave GEMMs unquantized or fail obscurely in JAX."""
    name = QuantizationType(self.quantization).value
    if self.use_qwix_quantization:
      raise ValueError(f"quantization='{name}' does not support use_qwix_quantization=True.")
    if self.shard_mode == ShardMode.EXPLICIT:
      raise ValueError(f"quantization='{name}' does not support shard_mode='explicit'.")
    if self.num_experts > 1:
      raise ValueError(f"quantization='{name}' does not support MoE models (num_experts > 1).")
    if is_fp8_dtype(self.weight_dtype):
      raise ValueError(
          f"quantization='{name}' quantizes the weights itself; weight_dtype must not be fp8, got"
          f" '{DType(self.weight_dtype).value}'."
      )
    if self.hardware not in ("gpu", "gpu_multiprocess"):
      logger.warning(
          "quantization='%s' has native MX kernels on GPU only; on hardware='%s' JAX dequantizes before "
          "each GEMM, so steps are slower than bf16.",
          name,
          self.hardware,
      )

  @model_validator(mode="after")
  def resolve_lm_head_weight_grad_in_kernel_order(self) -> "MaxTextConfig":
    """Resolve the tri-state flag, and reject it where it cannot be honored.

    The transpose it removes only exists under explicit sharding, and on an untied
    model removing it has been a win or a wash on every configuration measured
    (docs/guides/optimization/shard_mode_performance.md section 5) -- on qwen3-8b it
    is the difference between +3.25% and -0.41% against `auto`. So the default is
    "on wherever it can do anything", and writing the flag out is only needed to
    reproduce a measurement.
    """
    if self.lm_head_weight_grad_in_kernel_order is None:
      self.lm_head_weight_grad_in_kernel_order = self.shard_mode == ShardMode.EXPLICIT and not self.logits_via_embedding
    elif self.lm_head_weight_grad_in_kernel_order and self.logits_via_embedding:
      raise ValueError(
          "lm_head_weight_grad_in_kernel_order only applies to the untied LM head, but logits_via_embedding is True."
      )
    return self

  @model_validator(mode="after")
  def resolve_dense_weight_grad_in_kernel_order(self) -> "MaxTextConfig":
    """Resolve the in-loop counterpart of the LM-head flag.

    Same barrier, same shape of fix, same default rule: on wherever it can act.
    The two are independent and additive -- on llama2 at 18 layers, explicit
    costs +1.098% against auto with neither flag, +0.621% with this one alone,
    +0.429% with the LM-head flag alone and +0.005% with both
    (docs/guides/optimization/shard_mode_performance.md section 4.8).
    """
    if self.dense_weight_grad_in_kernel_order is None:
      self.dense_weight_grad_in_kernel_order = self.shard_mode == ShardMode.EXPLICIT
    return self

  @model_validator(mode="after")
  def validate_save_fsdp_gathered_weights(self) -> "MaxTextConfig":
    """Reject the three configurations the gathered-weight residual cannot serve."""
    if self.save_fsdp_gathered_weights and self.quantization:
      raise ValueError("save_fsdp_gathered_weights casts kernels to config.dtype, which is not valid when quantizing.")
    if self.save_fsdp_gathered_weights and is_fp8_dtype(self.weight_dtype):
      # Weight-only fp8 leaves `quantization` unset, so the check above misses it. The hoisted
      # gather casts to config.dtype, which would upcast the kernel before the all-gather -- the
      # opposite of what fp8 weights are for -- and separate it from its kernel_scale.
      raise ValueError(
          "save_fsdp_gathered_weights casts kernels to config.dtype, which discards the fp8 encoding "
          f"of weight_dtype={self.weight_dtype}."
      )
    if self.save_fsdp_gathered_weights and self.parameter_memory_host_offload:
      raise ValueError(
          "save_fsdp_gathered_weights keeps gathered kernels in HBM, which defeats parameter_memory_host_offload."
      )
    return self


  @model_validator(mode="after")
  def set_derived_and_validate_values(self, info: ValidationInfo) -> "MaxTextConfig":
    """Compute derived values and validate cross-field configuration."""
    if self.custom_mesh_and_rule is not CustomRule.DEFAULT:
      mesh_config = self._load_mesh_config_from_yaml(self.custom_mesh_and_rule.value)
      for field in ("mesh_axes", "logical_axis_rules", "data_sharding"):
        if field in mesh_config:
          setattr(self, field, mesh_config[field])
      cp_axes = infer_cp_axes(self.logical_axis_rules)
      if cp_axes:
        self.context_sharding = cp_axes[0]

    if not self.override_logical_axis_rules:
      ep_is_enabled = ep_enabled(self)
      if not ep_is_enabled:
        _ep_disabled_flags = {
            "use_random_routing": False,
            "use_ragged_sort": False,
            "moe_dropless_fallback": None,
            "use_ring_of_experts": False,
            "num_moe_emb_chunks": 0,
        }
        if not self.te_moe_block:
          _ep_disabled_flags["ragged_buffer_factor"] = -1.0
          _ep_disabled_flags["eval_ragged_buffer_factor"] = -1.0
        for flag_name, disabled_value in _ep_disabled_flags.items():
          current = getattr(self, flag_name)
          if current != disabled_value:
            raise ValueError(f"When EP rank is 1, {flag_name} must be {disabled_value} (was {current}).")

    if self.custom_mesh_and_rule_for_eval is CustomRule.DEFAULT:
      self.custom_mesh_and_rule_for_eval = self.custom_mesh_and_rule
      self.logical_axis_rules_for_eval = self.logical_axis_rules
    else:
      eval_config = self._load_mesh_config_from_yaml(self.custom_mesh_and_rule_for_eval.value)
      self.logical_axis_rules_for_eval = eval_config.get("logical_axis_rules", self.logical_axis_rules)
      dropped_axes = [axis for axis in eval_config.get("mesh_axes", ()) if axis not in self.mesh_axes]
      if dropped_axes:
        logger.warning(
            "custom_mesh_and_rule_for_eval=%s declares mesh axes %s that are absent from the mesh built for "
            "custom_mesh_and_rule=%s; they are ignored and any eval rule referencing them is treated as replicated.",
            self.custom_mesh_and_rule_for_eval.value,
            dropped_axes,
            self.custom_mesh_and_rule.value,
        )
      train_q_axes = axes_for_logical(self.logical_axis_rules, "activation_q_length")
      eval_q_axes = axes_for_logical(self.logical_axis_rules_for_eval, "activation_q_length")
      if self.context_parallel_load_balance and train_q_axes != eval_q_axes:
        logger.warning(
            "context_parallel_load_balance=True but activation_q_length maps to %s under "
            "custom_mesh_and_rule=%s and to %s under custom_mesh_and_rule_for_eval=%s. The input pipeline "
            "reorders for the train extent only, so load balancing is disabled at eval; eval attention runs "
            "unbalanced (the last CP shard does ~2*cp/(cp+1) of the average work). Everything outside the "
            "sequence-quadratic term stays balanced.",
            list(train_q_axes),
            self.custom_mesh_and_rule.value,
            list(eval_q_axes),
            self.custom_mesh_and_rule_for_eval.value,
        )

    if not self.run_name:
      if os.environ.get("JOBSET_NAME"):
        self.run_name = os.environ.get("JOBSET_NAME")
      else:
        self.run_name = f'{self.model_name}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}'

    if self.run_name and self.base_output_directory:
      output_dir = os.path.join(self.base_output_directory, self.run_name)
      self.checkpoint_dir = os.path.join(output_dir, "checkpoints", "")
      self.metrics_dir = os.path.join(output_dir, "metrics", "")
      self.tensorboard_dir = os.path.join(output_dir, "tensorboard", "")
      self.managed_mldiagnostics_dir = derive_managed_mldiagnostics_dir(
          self.base_output_directory,
          self.run_name,
          self.managed_mldiagnostics_storage_path,
      )
      if self.enable_mllog:
        if not self.mllog_file:
          self.mllog_file = os.path.join(output_dir, f"mllog_{self.data_shuffle_seed}.log")
        elif not self.mllog_file.startswith("gs://") and not os.path.isabs(self.mllog_file):
          self.mllog_file = os.path.join(output_dir, self.mllog_file)
      else:
        self.mllog_file = ""
    else:
      self.checkpoint_dir, self.metrics_dir, self.tensorboard_dir = (None, None, None)

    tokenizer_path = getattr(self, "tokenizer_path", "")
    if (
        tokenizer_path
        and not os.path.exists(tokenizer_path)
        and not tokenizer_path.startswith("gs://")
        and not self.colocated_python_data_input
    ):
      tokenizer_path = next(
          filter(
              os.path.exists,
              (
                  os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", os.path.basename(tokenizer_path)),
                  os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", tokenizer_path),
              ),
          ),
          tokenizer_path,
      )
      self.tokenizer_path = tokenizer_path

    chat_template = getattr(self, "chat_template", "")
    if chat_template:
      try:
        env = Environment()
        env.parse(chat_template)
      except TemplateSyntaxError as e:
        raise ValueError(f"Specified chat_template is invalid: {e}") from e

    if self.learning_rate_schedule_steps == -1:
      self.learning_rate_schedule_steps = self.steps
    if self.steps == -1:
      self.steps = self.learning_rate_schedule_steps
    if self.deepstack_visual_indexes_for_vit and self.scan_layers:
      raise ValueError(
          "Deepstack visual embedding injection requires scan_layers=False. "
          "Set scan_layers=False in your config to use deepstack features."
      )
    if self.lr_schedule_type == LearningRateScheduleType.WSD:
      total_fraction = self.warmup_steps_fraction + self.wsd_decay_steps_fraction
      if total_fraction > 1.0:
        raise ValueError(
            f"Invalid WSD schedule: warmup_steps_fraction ({self.warmup_steps_fraction}) + "
            f"wsd_decay_steps_fraction ({self.wsd_decay_steps_fraction}) must not exceed 1.0. "
            f"Current sum: {total_fraction}"
        )
    if getattr(self, "eval_per_device_batch_size", 0.0) == 0.0:
      self.eval_per_device_batch_size = self.per_device_batch_size
    if not self.mu_dtype:
      self.mu_dtype = self.weight_dtype
    if self.attn_logits_soft_cap == 0.0:
      self.attn_logits_soft_cap = None
    if self.final_logits_soft_cap == 0.0:
      self.final_logits_soft_cap = None

    if not (info.context or {}).get("hlo_defaults_prepared"):
      validate_and_set_hlo_dump_defaults(self)

    self.validate_distillation()

    emb_scale, num_head_scale, mlp_dim_scale, layer_scale = get_individual_scales(self.global_parameter_scale)
    self.emb_dim = (2**emb_scale) * self.base_emb_dim
    self.num_query_heads = (2**num_head_scale) * self.base_num_query_heads
    self.num_kv_heads = (2**num_head_scale) * self.base_num_kv_heads
    self.mlp_dim = (2**mlp_dim_scale) * self.base_mlp_dim
    self.moe_mlp_dim = (2**mlp_dim_scale) * self.base_moe_mlp_dim
    self.num_decoder_layers = (2**layer_scale) * self.base_num_decoder_layers

    if self.elastic_enabled:
      elastic.ensure_elastic_manager_initialized(self)

    def get_num_target_devices():
      """Get the number of devices for the target topology, handling AOT compilation and single-controller modes."""
      if self.internal_compile:
        if self.internal_compile_num_devices <= 0:
          raise ValueError("Set internal_compile_num_devices to a positive integer.")
        return self.internal_compile_num_devices
      if self.compile_topology:
        spec = accelerator_to_spec_map.get_system_characteristics(self.compile_topology)
        return int(spec.devices_per_slice * self.compile_topology_num_slices)
      elif self.subslice_shape and self.enable_single_controller:
        shape_tuple = tuple(int(x) for x in self.subslice_shape.split(","))
        return prod(shape_tuple)
      elif self.elastic_enabled:
        return len(elastic.live_devices(config=self))
      else:
        return len(jax.devices())

    self.num_target_devices = 1
    try:
      self.num_target_devices = get_num_target_devices()
    except (RuntimeError, IndexError):
      logger.warning("JAX device system not available for config validation. Assuming 1 device.")

    raw_keys_for_num_slices = {
        "num_slices": self.num_slices,
        "hardware": self.hardware,
        "compile_topology_num_slices": self.compile_topology_num_slices,
    }
    self.num_slices = slice_count.get_num_slices(raw_keys_for_num_slices)

    if self.hardware == "cpu" and self.megablox:
      self.megablox = False
    if self.quantization and not self.use_qwix_quantization:
      if self.quantization not in ("fp8", "nanoo_fp8", *JAX_MX_QUANTIZATIONS) and not self.quantization.startswith("te_"):
        logger.warning(
            "WARNING: AQT quantization is deprecated and will be removed in a future release. "
            "Please migrate to Qwix by setting use_qwix_quantization=True."
        )
    if self.enable_orbax_v1:
      max_logging.log(
          "WARNING: enable_orbax_v1 is deprecated and ignored — Orbax v1 is now always used for "
          "checkpointing. Remove the flag from your config; it will be deleted in a future release."
      )
    if self.quantization_local_shard_count == -1:
      try:
        self.quantization_local_shard_count = jax.local_device_count()
      except RuntimeError:
        self.quantization_local_shard_count = 1

    derive_batch_sizes(self)

    # G. CALCULATE/SET OTHER DERIVED VALUES, E.G. PIPELINE CONFIG
    if self.remat_policy == "custom":
      tensors = [
          "decoder_layer_input", "indexer_cutoff_threshold", "context", "mlpwi", "moe_mlpwi_0",
          "moe_x_sorted", "moe_routing_maps", "moe_tc_routing_checkpoint", "moe_mlpwi_1", "moe_mlpwo",
          "te_moe_dispatch_remat_location", "mlpwi_0", "mlpwi_1", "mlpwo", "query_proj", "key_proj",
          "value_proj", "kv_proj", "query_wa_proj", "kv_wa_proj", "mla_kv", "mla_q", "qkv_proj", "attention_out",
          "out_proj",
      ]
      self.tensors_on_device = [t for t in tensors if getattr(self, t) == "device"]
      self.tensors_to_offload = [t for t in tensors if getattr(self, t) == "offload"]

    if self.pipeline_parallel_layers == -1:
      if self.decoder_block == DecoderBlockType.DEEPSEEK:
        moe_layers = self.num_decoder_layers - self.first_num_dense_layers
        self.pipeline_parallel_layers = moe_layers
      else:
        self.pipeline_parallel_layers = self.num_decoder_layers

    self.using_pipeline_parallelism = self.ici_pipeline_parallelism > 1 or self.dcn_pipeline_parallelism > 1
    if self.using_pipeline_parallelism:
      num_stages = int(self.ici_pipeline_parallelism * self.dcn_pipeline_parallelism)
      if self.num_pipeline_repeats == -1:
        num_pipeline_repeats, remainder = divmod(
            self.pipeline_parallel_layers,
            num_stages * self.num_layers_per_pipeline_stage,
        )
        assert not remainder, (
            f"The number of layers per stage ({self.num_layers_per_pipeline_stage}) times the number of stages "
            f"({num_stages}) must divide the number of pipeline_parallel_layers which defaults to decoder layers "
            f"({self.pipeline_parallel_layers}) "
        )
        self.num_pipeline_repeats = num_pipeline_repeats

      if self.pipeline_fsdp_ag_per_repeat:
        assert self.num_pipeline_repeats > 1, "Pipeline weight prefetching only supports circular pipeline."
        assert (
            self.num_layers_per_pipeline_stage == 1
        ), "Pipeline weight prefetching currently only supports one layer per pipeline stage."
        assert (
            not self.pipeline_delay_activation_forwarding
        ), "Pipeline weight prefetching does not support pipeline delay."
        assert not self.scan_layers_per_stage, "Pipeline weight prefetching currently does not support scan."
        # Qwix quantized training (qwix.QtProvider) keeps no per-layer quantization state: weights are staged through
        # the BSW in their original dtype and quantized inside every matmul/GMM, exactly as without pipelining (verified
        # for fp8_full). AQT and the fp8_gpu/fp8_nanoo delayed-scaling providers instead carry amax-history
        # `_overwrite_with_gradient` variables whose "gradient" is the new history, and the BSW scan would sum it over
        # all microbatches of a repeat, so they stay unsupported.
        qwix_qt_types = ("int4", "int8", "fp4", "fp4_e2m1", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_full")
        assert not self.quantization or (self.use_qwix_quantization and self.quantization in qwix_qt_types), (
            "Pipeline weight prefetching (pipeline_fsdp_ag_per_repeat) only supports Qwix quantized training: "
            f"use_qwix_quantization=True and quantization in {qwix_qt_types}."
        )
      else:
        # pipeline_bsw_no_carry is on by default and has nothing to do without weight prefetching.
        bsw_options = ("pipeline_bsw_phase_split", "pipeline_bsw_in_compute_dtype")
        enabled = [name for name in bsw_options if getattr(self, name)]
        assert not enabled, f"{enabled} only apply to pipeline weight prefetching: set pipeline_fsdp_ag_per_repeat=True."

      assert (num_stages * self.num_pipeline_repeats * self.num_layers_per_pipeline_stage) == (
          self.pipeline_parallel_layers
      ), (
          f"The product of pipeline stages ({num_stages}), repeats ({self.num_pipeline_repeats}), and layers "
          f"per stage ({self.num_layers_per_pipeline_stage}) must be equal to pipeline_parallel_layers "
          f"which defaults to decoder layers ({self.pipeline_parallel_layers})"
      )
      if self.num_pipeline_microbatches == -1:
        if self.pipeline_delay_activation_forwarding:
          self.num_pipeline_microbatches = 2 * num_stages
        else:
          self.num_pipeline_microbatches = num_stages

      assert self.num_pipeline_microbatches > 0, "num_pipeline_microbatches must be positive"
      assert self.num_pipeline_microbatches % num_stages == 0, (
          f"The number of microbatches ({self.num_pipeline_microbatches}) must be divisible by the number of "
          f"stages ({num_stages})"
      )
      if self.micro_batch_size_to_train_on > 0:
        assert self.micro_batch_size_to_train_on % self.num_pipeline_microbatches == 0, (
            f"The batch size for a single forward pass ({self.micro_batch_size_to_train_on}) must be divisible "
            f"by the number of microbatches ({self.num_pipeline_microbatches})"
        )
      if self.pipeline_delay_activation_forwarding:
        assert self.num_pipeline_microbatches >= 2 * num_stages, (
            f"Delayed activation forwarding requires at least 2 * num_stages microbatches, but {num_stages} stages "
            f"are used with {self.num_pipeline_microbatches} microbatches"
        )

      # For AOT compilation and correctness, always prioritize the 'stage' axis for sharding when pipelining.
      for rule in self.logical_axis_rules:
        if rule and rule[0] == "activation_embed_and_logits_batch":
          rule[1] = ["stage", "data", "fsdp", "fsdp_transpose", "expert"]
          break

      if "stage" in self.mesh_axes:
        stage_idx = self.mesh_axes.index("stage")
        data_idx = self.mesh_axes.index("data")
        if stage_idx > data_idx:  # Ensure 'stage' comes before 'data' for correct sharding logic.
          self.mesh_axes[stage_idx], self.mesh_axes[data_idx] = (
              self.mesh_axes[data_idx],
              self.mesh_axes[stage_idx],
          )

      # Adjust data_sharding to also prioritize 'stage'.
      if (
          isinstance(self.data_sharding, list)
          and len(self.data_sharding) > 0
          and self.data_sharding[0]
          and "stage" in self.data_sharding[0]
      ):
        self.data_sharding[0].remove("stage")
        self.data_sharding[0].insert(0, "stage")

      # Add sharding for FP8 amax history when using pipeline parallelism.
      if self.quantization and self.quantization in (
          "fp8", "nanoo_fp8", "fp8_gpu", "te_fp8_delayedscaling",
      ):
        self.logical_axis_rules.append(["aqt_amax_history", ("stage",)])

    # H. RESOLVE local_sa_* and eval_local_sa_* FLAGS: inherit from global sa_* / eval_sa_* if not explicitly set.
    resolve_local_splash_defaults(self)

    # I. RUN ALL CROSS-FIELD VALIDATIONS
    validate_checkpoint_source_config(self)
    validate_elastic_backup(self, DatasetType)
    if self.colocated_python_data_input and not self.enable_single_controller:
      raise ValueError(
          "Colocated python data input is only supported with Pathways (single"
          " controller) enabled (`enable_single_controller=True`)."
      )
    if self.grain_use_elastic_iterator and self.grain_file_type != "arrayrecord":
      raise ValueError(
          "`grain_use_elastic_iterator=True` only supports `grain_file_type=arrayrecord`. "
          "tfrecord and parquet pipelines use `InterleaveIterDataset` (a many-to-one "
          "IterDataset transform), which `ElasticIterator` forbids. "
          f"Got grain_file_type={self.grain_file_type}."
      )
    if self.grain_use_elastic_iterator and self.packing:
      raise ValueError("`grain_use_elastic_iterator=True` requires `packing=False`.")
    if self.use_dpo and self.packing:
      raise ValueError("DPO does not support packing. Set `packing=False`.")
    if self.grain_use_elastic_iterator and not self.use_truncation:
      raise ValueError(
          "`grain_use_elastic_iterator=True` requires `use_truncation=True`. "
          "`TokenizeAndChunk` uses `apply`, which produces a many-to-one "
          "IterDataset transform that `ElasticIterator` forbids."
      )
    if self.grain_use_elastic_iterator and (
        self.grain_train_mixture_config_path or ";" in (self.grain_train_files or "")
    ):
      raise ValueError(
          "`grain_use_elastic_iterator=True` does not support dataset mixtures. "
          "Set `grain_train_mixture_config_path` to empty and use a single "
          "`grain_train_files` pattern (no ';' separator)."
      )
    # Only a full-state resume needs the CheckpointManager, which `enable_checkpointing`
    # gates. `load_parameters_path` is a warm start: it restores through its own
    # `ocp.Checkpointer` in `model_creation_utils.from_pretrained`, before the manager is
    # ever consulted, so it stays legal with saving turned off.
    validate_checkpoint_storage_config(self)
    attention_validation.validate_attention_backend_support(self)
    if self.te_mhc and self.hardware not in ("gpu", "gpu_multiprocess"):
      raise ValueError("`te_mhc=True` requires NVIDIA GPU hardware.")
    deepseek_attention_validation.validate_qk_head_chunking(self)
    deepseek_attention_validation.validate_sparse_indexer(self)
    deepseek_attention_validation.validate_mla_absorbed_mqa(self)
    attention_validation.validate_attention_type(self)
    validate_training_objective(self)
    if self.quantize_kvcache and not self.kv_quant_axis:
      raise ValueError("`kv_quant_axis` cannot be empty when quantize_kvcache is True.")
    validate_mtp_main_hidden_state(self)
    validate_training_quantization(self)
    validate_engram(self)
    if self.moe_use_direct_token_gather and self.use_gather_mosaic_kernel:
      raise ValueError("`moe_use_direct_token_gather=True` currently requires `use_gather_mosaic_kernel=False`.")
    if self.te_moe_disable_cudnn_gmm_fusion and not self.te_moe_block:
      raise ValueError("te_moe_disable_cudnn_gmm_fusion=True requires te_moe_block=True.")
    if self.num_experts > 1:
      validate_moe_architecture(self)
      self.validate_ragged_buffer_factor()
      self.validate_moe_dropless_fallback()
      self.validate_moe_log_max_load_ratio()
      self.validate_moe_tc_ragged_sort()
    self.validate_num_moe_emb_chunks()
    self.validate_moe_quantize_token_all_gather()
    self.validate_moe_topk_before_ep_all_gather()
    self.validate_moe_quantize_combine_bwd_method()
    self.validate_mllog()
    self.validate_splash_diag_skip()
    self.validate_retry_dropless_first_steps_and_first_phase_buffer()
    if self.warmup_skip_train_dropless and not self.warmup_programs_in_init:
      raise ValueError("warmup_skip_train_dropless=True requires warmup_programs_in_init=True.")

    validate_streaming_diloco(self)

    # Gemma 4 small (E2B / E4B) uses per-layer KV sharing, which is incompatible with scanned layers.
    if self.model_name in ("gemma4-e2b", "gemma4-e4b") and self.scan_layers:
      raise ValueError(
          f"{self.model_name} requires scan_layers=False (per-layer KV sharing is incompatible with scanned layers)."
      )
    validate_multimodal_support(self)
    validate_alignment_objectives(self)
    validate_explicit_sharding(self)
    if self.context_sharding not in ("context", "expert"):
      raise ValueError(f"Assigned context_sharding f{self.context_sharding} is not supported.")
    if self.ulysses_context_sharding != "context_usp_ulysses":
      raise ValueError(f"Assigned ulysses_context_sharding {self.ulysses_context_sharding} is not supported.")
    if (
        self.per_device_batch_size > 0
        and (self.per_device_batch_size * self.max_target_length) % self.num_vocab_tiling != 0
    ):
      raise ValueError("Per device batch size times sequence length should be divisible by the number of vocab tiles.")
    context_parallel_size = getattr(self, f"ici_{self.context_sharding}_parallelism", 1) * getattr(
        self, f"dcn_{self.context_sharding}_parallelism", 1
    )
    validate_context_parallelism(self, context_parallel_size)
    if self.hardware == "gpu" and self.packing and self.attention == "cudnn_flash_te" and self.max_segments_per_seq <= 0:
      raise ValueError("max_segments_per_seq must be set when using TransformerEngine attention and packing")
    dcn_product = (
        self.dcn_data_parallelism
        * self.dcn_pipeline_parallelism
        * self.dcn_fsdp_parallelism
        * self.dcn_fsdp_transpose_parallelism
        * self.dcn_sequence_parallelism
        * self.dcn_context_parallelism
        * self.dcn_context_usp_ulysses_parallelism
        * self.dcn_tensor_parallelism
        * self.dcn_tensor_sequence_parallelism
        * self.dcn_expert_parallelism
        * self.dcn_context_autoregressive_parallelism
        * self.dcn_autoregressive_parallelism
    )
    if abs(dcn_product) > 1:
      if not self.num_slices > 1:
        raise ValueError("DCN parallelism requested but only one slice available.")
    validate_decoder_block_constraints(self)

    tokenizer_path = getattr(self, "tokenizer_path", None)
    if (
        tokenizer_path
        and "tokenizer_llama3.tiktoken" in tokenizer_path
        and getattr(self, "tokenizer_type", None) != "tiktoken"
    ):
      logger.warning("`tokenizer_type` is not 'tiktoken' when using llama3 tokenizer. Overriding to 'tiktoken'.")
      self.tokenizer_type = TokenizerType.TIKTOKEN
    # Data input validations
    if self.dataset_type == DatasetType.HF:
      train_dataset_provided = (info.context or {}).get("train_dataset_provided", False)
      eval_dataset_provided = (info.context or {}).get("eval_dataset_provided", False)
      if not self.hf_path and (not train_dataset_provided or (self.eval_interval > 0 and not eval_dataset_provided)):
        raise ValueError("hf_path can't be empty when dataset_type=hf")
      if self.hf_eval_files:
        self.hf_eval_split = "train"
      if self.eval_interval > 0 and not self.hf_eval_split and not eval_dataset_provided:
        raise ValueError("Please specify hf_eval_split or set eval_interval to <=0.")
      if self.grain_worker_count > 1:
        raise ValueError("Only supports <= 1 for now, more workers results in duplicated data")
    elif self.dataset_type == DatasetType.GRAIN:
      use_hf_parquet = self.hf_path and self.grain_file_type == "parquet"
      use_tfds_tfrecord_train = (
          self.grain_file_type == "tfrecord" and self.dataset_path and self.dataset_name and self.train_split
      )
      use_tfds_tfrecord_eval = (
          self.grain_file_type == "tfrecord" and self.dataset_path and self.eval_dataset_name and self.eval_split
      )
      if (
          not self.grain_train_files
          and not self.grain_train_mixture_config_path
          and not use_hf_parquet
          and not use_tfds_tfrecord_train
      ):
        raise ValueError(
            "When dataset_type=grain, set grain_train_files, "
            "grain_train_mixture_config_path, use hf_path with grain_file_type=parquet, or use dataset_path, "
            "dataset_name, and train_split with grain_file_type=tfrecord."
        )
      if self.eval_interval > 0 and not self.grain_eval_files and not use_hf_parquet and not use_tfds_tfrecord_eval:
        raise ValueError(
            "Please specify grain_eval_files, use hf_path with grain_file_type=parquet, or use dataset_path, "
            "eval_dataset_name, and eval_split with grain_file_type=tfrecord; otherwise set eval_interval to <=0."
        )
    elif self.dataset_type == DatasetType.TFDS:
      logger.warning(
          "tfds pipeline is deprecated. Use dataset_type=grain and grain_file_type=tfrecord. You can keep the same "
          "dataset_path, dataset_name, train_split, eval_dataset_name, and eval_split settings to automatically construct "
          "the file paths. Alternatively, provide grain_train_files or grain_eval_files for custom file paths."
      )
      if self.use_dpo:
        raise ValueError(
            "TFDS dataset_type=tfds is not supported for DPO training"
            " (config.use_dpo=True). Please use dataset_type=grain or"
            " dataset_type=hf instead."
        )
      if not self.dataset_name:
        raise ValueError("dataset_name can't be empty when dataset_type=tfds")
      if self.eval_interval > 0 and not self.eval_split:
        raise ValueError("Please specify eval_split or set eval_interval to <=0.")

    if self.sharding_tolerance > 1.0 or self.sharding_tolerance < 0.0:
      logger.warning("'sharding_tolerance: allowed percentage of non-sharded parameters' should be between 0.0 and 1.0")

    if self.eval_interval > 0 >= self.eval_steps and self.generate_padding_batch_eval:
      raise ValueError("`eval_steps` must be > 0 when `generate_padding_batch_eval` is True.")
    if self.use_batch_split_schedule:
      if self.quantization and not self.quantization == "fp8_full":
        raise ValueError("Batch split quantization only supports `quantization=fp8_full`")

    if self.opt_type == "muon" and self.decoder_block not in [
        DecoderBlockType.DEEPSEEK, DecoderBlockType.DEEPSEEK4, DecoderBlockType.QWEN3, DecoderBlockType.QWEN3_MOE,
        DecoderBlockType.QWEN3_CUSTOM_MOE, DecoderBlockType.QWEN3_NEXT, DecoderBlockType.GPT_OSS,
        DecoderBlockType.GEMMA3, DecoderBlockType.LLAMA2,
    ]:
      raise ValueError(
          "Muon dimension numbers haven't been tested for this model. Run this command first: "
          f"`python3 -m lilax._backend.optimizers.muon_config.dimensions {self.model_name} True`"
      )
    validate_splash_attention_flags(self)
    if self.share_kv_projections and self.fused_qkv:
      raise ValueError("`share_kv_projections` is not compatible with `fused_qkv`.")
    if self.share_kv_projections and self.attention_type == "mla":
      raise ValueError("`share_kv_projections` is not compatible with `attention_type='mla'`.")
    if self.use_sliced_mla_proj and self.quantization:
      # Qwix interception still quantizes the sliced dot_general; module-level quant
      # (AQT / Fp8 / TE / batch split) would silently fall back to the unsliced path.
      if not self.use_qwix_quantization or self.use_batch_split_schedule:
        raise ValueError(
            "`use_sliced_mla_proj` with quantization is only supported via Qwix interception "
            "(use_qwix_quantization=True, use_batch_split_schedule=False)."
        )
    if self.use_manual_quantization and not self.use_batch_split_schedule:
      raise ValueError("manual quantization is only used when `use_batch_split_schedule=True`.")

    # Validation for serve_fp8_weight (native FP8 compute without dequantization).
    if self.quantization == QuantizationType.SERVE_FP8_WEIGHT:
      if self.weight_dtype not in (DType.FLOAT8_E4M3FN, DType.FLOAT8_E5M2):
        raise ValueError(
            "quantization='serve_fp8_weight' requires weight_dtype to be an FP8 dtype "
            f"('float8_e4m3fn' or 'float8_e5m2'), got weight_dtype={self.weight_dtype!r}."
        )
      if self.use_qwix_quantization:
        raise ValueError("quantization='serve_fp8_weight' is not supported with use_qwix_quantization=True.")

    validate_gmm_v2(self)
    self._validate_te_grouped_gemm()
    if self.use_gmm_v2_heuristic_tiling and not self.use_gmm_v2:
      raise ValueError("`use_gmm_v2_heuristic_tiling=True` requires `use_gmm_v2=True`.")
    for val in self.compress_ratios:
      if val != 0 and val < 4:
        raise ValueError(f"compress_ratio must be 0 (disabled) or >= 4, got {val}")
    if self.num_kv_shared_layers > 0:
      if self.fused_qkv:
        raise ValueError("`num_kv_shared_layers > 0` is not compatible with `fused_qkv`.")
      if self.share_kv_projections:
        raise ValueError("`num_kv_shared_layers > 0` is not compatible with `share_kv_projections`.")
    validate_moe_kernel_options(self)

    validate_lineage(self)

    # The freqs_cis table lookup clamps positions >= max_position_embeddings to the last row, while
    # rope_direct_position_freqs uses the true angle; keep positions in range so both settings agree.
    if self.rope_direct_position_freqs and self.max_target_length > self.max_position_embeddings:
      raise ValueError(
          "rope_direct_position_freqs=True requires max_target_length <= max_position_embeddings, got "
          f"max_target_length={self.max_target_length}, max_position_embeddings={self.max_position_embeddings}."
      )

    # I. FINAL TYPE CONVERSIONS AND DERIVED LISTS
    resolve_parallelism_lists(self)

    validate_optimizer_sharding(self)

    if self.quantization in JAX_MX_QUANTIZATIONS:
      self._validate_jax_mx_quantization()

    # Diloco params
    resolve_diloco_replicas(self)
    validate_diloco_grouped_matmul(self)

    self._validate_check_vma_is_supported()
    if self.defer_small_all_reduces:
      if not (self.use_ring_of_experts and self.sparse_matmul):
        raise ValueError(
            "defer_small_all_reduces requires use_ring_of_experts=True and sparse_matmul=True (the deferral is"
            " implemented on the ring-of-experts sparse_matmul path only)."
        )
      if self.moe_dropless_fallback == "layer":
        raise ValueError(
            "defer_small_all_reduces does not support moe_dropless_fallback='layer' (the in-layer fallback's two"
            " branches would emit per-chunk counts of different chunkings)."
        )

    # Final string-to-enum conversions if they haven't been coerced by pydantic yet.
    if isinstance(self.decoder_block, str):
      self.decoder_block = DecoderBlockType(self.decoder_block.lower())
    if isinstance(self.shard_mode, str):
      self.shard_mode = ShardMode(self.shard_mode.lower())
    if isinstance(self.te_comm_gemm_overlap, str):
      self.te_comm_gemm_overlap = TeCommGemmOverlapPolicy(self.te_comm_gemm_overlap.lower())

      if self.te_comm_gemm_overlap != TeCommGemmOverlapPolicy.DISABLED:
        self._validate_te_comm_gemm_overlap()

    constant_bound_config = getattr(self, "constant_bound_config", None)
    if isinstance(constant_bound_config, str):
      if constant_bound_config:
        self.constant_bound_config = [float(v.strip()) for v in constant_bound_config.split(",")]
      else:
        self.constant_bound_config = []

    if self.decoder_block == DecoderBlockType.QWEN3_CUSTOM_MOE:
      if self.moe_expert_input_dim != self.attention_output_dim:
        raise ValueError(
            f"For qwen3_custom_moe, moe_expert_input_dim ({self.moe_expert_input_dim}) "
            f"must be equal to attention_output_dim ({self.attention_output_dim})"
        )
    if self.model_name == "glm5.3-flash" and not self.use_m3_model:
      raise ValueError(
          "glm5.3-flash requires use_m3_model=True; legacy decoder_block is not" " supported for GLM-5.3-Flash."
      )
    return self


class RLConfig(
    LogitsAndLoss,
    Engram,
    ManifoldConstrainedHyperConnections,
    RematAndOffload,
    Attention,
    Llama4Attention,
    LayoutAndSharding,
    InferenceLayout,
    InferenceGeneral,
    PrefixCaching,
    Decoding,
    IciParallelism,
    DcnParallelism,
    PipelineParallelism,
    DilocoParams,
    HardwareAndMesh,
    ModelArchitecture,
    MTP,
    MoBa,
    MlaAttention,
    CompressedAttention,
    AttentionIndexer,
    SplashAttention,
    Qwen3Next,
    Qwen3_8FlashNext,
    Ming,
    MultimodalGeneral,
    Muon,
    FineTuning,
    Distillation,
    DatasetGeneral,
    TfdsDataset,
    HfDataset,
    GrainDataset,
    OlmoGrainDataset,
    EmergencyCheckpointing,
    ElasticTraining,
    InferenceServer,
    InferenceBenchmark,
    HloDump,
    Goodput,
    GcpMonitoring,
    ManagedMLDiagnostics,
    PositionalEmbedding,
    Rope,
    YarnRope,
    MoEGeneral,
    MoEKernels,
    DeepSeekMoE,
    RunInfo,
    Checkpointing,
    OrbaxStorage,
    DataTypes,
    Tokenizer,
    AdamW,
    Optimizer,
    TrainingLoop,
    Quantization,
    VisionTower,
    VisionProjector,
    AudioEncoder,
    DevelopmentAndDebugging,
    Profiling,
    AOT,
    Metrics,
    Tensorboard,
    BatchSizeRampupIncrements,
    CustomRematTensors,
    BatchSizeRampupEnd,
    RunOutputDirectories,
    GlobalBatchSizes,
    ResolvedMeshTopology,
    ScaledModelDimensions,
    RL,
    RLCluster,
    RLDataset,
    RLEvaluation,
    RLReward,
    RLSpecialTokens,
    VLLM,
):
  """
  Configuration for Reinforcement Learning in MaxText.
  """

  num_epoch: int = Field(1, ge=1, description="Number of epochs to train for.")
  eval_interval: int = Field(
      -1,
      description="Run evaluation every N training steps. -1 disables interval-based evaluation.",
  )
  enable_dropout: bool = Field(True, description="Enables dropout in the model.")
  dropout_rate: float = Field(0.0, ge=0.0, le=1.0, description="The dropout rate.")
  init_weights_seed: int = Field(0, description="Seed for model weight initialization.")
  log_period: int = Field(100, description="Frequency (in steps) to log metrics and flush to Tensorboard.")
  hf_access_token: None | str = Field(None, description="Hugging Face API access token.")
  enable_tunix_perf_metrics: bool = Field(
      False,
      description="Whether to enable Tunix-managed metrics measurement. The metrics will be uploaded to tensorboard.",
  )
  max_target_length: int = Field(2048, description="Maximum sequence length for the model.")
  max_prefill_predict_length: int = Field(64, description="Maximum length for the prefill stage in decoding.")

  debug: bool = Field(False, description="Enable debug mode for RL.")

  cluster: RLCluster = Field(
      default_factory=RLCluster,
      description="Configuration for RL cluster.",
  )

  rl: RL = Field(
      default_factory=RL,
      description="Configuration for RL algorithms like Group Relative Policy Optimization (GRPO).",
  )

  dataset: RLDataset = Field(
      default_factory=RLDataset,
      description="Configuration for RL datasets.",
  )

  reward: RLReward = Field(
      default_factory=RLReward,
      description="Configuration for RL rewards.",
  )

  special_tokens: RLSpecialTokens = Field(
      default_factory=RLSpecialTokens,
      description="Configuration for RL special tokens.",
  )

  evaluation: RLEvaluation = Field(
      default_factory=RLEvaluation,
      description="Configuration for RL evaluation.",
  )

  vllm: VLLM = Field(
      default_factory=VLLM,
      description="Configuration for rollouts.",
  )

  @model_validator(mode="before")
  @classmethod
  def map_flat_fields_to_nested_models(cls, data: Any, info: ValidationInfo) -> Any:
    """Route flat top-level overrides into their matching nested model fields.

    Runs as a pydantic "before" validator so that flat keys in YAML configs or
    CLI overrides can populate a field of the same name inside a nested model
    without callers having to nest the override themselves.

    Args:
      data: The raw input to the model, as passed to pydantic's constructor.
        Only dicts are processed; anything else is returned unchanged.

    Returns:
      `data`, mutated in place so nested model dicts are pre-populated with
      the matching flat overrides. s
    """
    if not isinstance(data, dict) or (info.context and info.context.get("resolved_values")):
      return data

    # Loop over every field declared in RLConfig
    for field_name, field_info in cls.model_fields.items():
      annotation = getattr(field_info, "annotation", None)

      # If the field is a nested Pydantic model
      if hasattr(annotation, "model_fields"):
        nested_dict = data.setdefault(field_name, {})
        if not isinstance(nested_dict, dict):
          nested_dict = {}
          data[field_name] = nested_dict

        # Check if any parameter of the nested model was passed as a flat key in `data`
        for inner_key in annotation.model_fields.keys():
          if inner_key in data:
            nested_dict[inner_key] = data[inner_key]

    # Map legacy step-based profiling flags to RL role-invocation profiling fields
    # when the invocation fields were not explicitly overridden. Because `pyconfig`
    # merges `post_train/rl.yml` into `data["rl"]` before constructing `RLConfig`,
    # `rl_dict` may already contain the `RL` model defaults; query `RL.model_fields`
    # dynamically rather than hardcoding default values.
    rl_dict = data.get("rl")
    if isinstance(rl_dict, dict):
      rl_fields = getattr(RL, "model_fields")
      default_start = rl_fields["profiler_start_invocation"].default
      if (
          "skip_first_n_steps_for_profiler" in data
          and "profiler_start_invocation" not in data
          and rl_dict.get("profiler_start_invocation", default_start) == default_start
      ):
        try:
          skip_steps = int(data["skip_first_n_steps_for_profiler"])
        except (TypeError, ValueError):
          skip_steps = -1
        if skip_steps >= 0:
          rl_dict["profiler_start_invocation"] = skip_steps
          data["profiler_start_invocation"] = skip_steps

      default_num = rl_fields["profiler_num_invocations"].default
      if (
          "profiler_steps" in data
          and "profiler_num_invocations" not in data
          and rl_dict.get("profiler_num_invocations", default_num) == default_num
      ):
        try:
          prof_steps = int(data["profiler_steps"])
        except (TypeError, ValueError):
          prof_steps = 0
        if prof_steps >= 1:
          rl_dict["profiler_num_invocations"] = prof_steps
          data["profiler_num_invocations"] = prof_steps

    return data

  @model_validator(mode="after")
  def set_derived_values_and_validate(self) -> "RLConfig":
    """Validate required RL config fields and fill in values derived from them.

    Runs as a post-init model validator: checks constraints and computes fields
    that depend on other fields so callers don't have to derive them by hand.

    Returns:
      This RLConfig instance, with derived fields populated in place.

    Raises:
      ValueError: If a required field is missing or a cross-field constraint
        is violated.
    """
    # Validate that model_name is set.
    model_name = getattr(self, "model_name", None)
    if model_name is None:
      raise ValueError("model_name is not set. Please pass model_name in your command.")

    # With sequence packing on, a maximal sequence (prompt cap + generation
    # cap = max_target_length) must fit in one packed row. Tunix checks this
    # too, but only in the learner, after the models are already on the
    # accelerators; fail here, before any of that work.
    if 0 < self.max_seq_token_per_tpu < self.max_target_length:
      raise ValueError(
          f"max_seq_token_per_tpu ({self.max_seq_token_per_tpu}) must be at least "
          f"max_target_length ({self.max_target_length}) when sequence packing is enabled."
      )

    # Set tokenizer_path based on model_name if not explicitly provided.
    tokenizer_path = getattr(self, "tokenizer_path", None)
    if tokenizer_path is None:
      if model_name in HF_IDS:
        self.tokenizer_path = HF_IDS[model_name]
        self.tokenizer_type = TokenizerType.HUGGINGFACE
      else:
        raise ValueError(
            "model_name not found in HF_IDS in maxtext/src/maxtext/utils/globals.py. \
          Please pass tokenizer_path in your command."
        )

    if self.optimizer_memory_host_offload:
      raise ValueError(
          "optimizer_memory_host_offload=True is not supported on the post-training "
          "RL path because the underlying Tunix RLCluster/Trainer does not "
          "support host offloading of the optimizer state."
      )

    if self.num_vocab_tiling > 1:
      raise ValueError(
          f"Vocab Tiling is not supported with RL. "
          f"num_vocab_tiling was configured to {self.num_vocab_tiling}, but it must be 1 when running train_rl."
      )

    # Set checkpoint_dir based on run_name and base_output_directory.
    if self.run_name and self.base_output_directory:
      checkpoint_dir = os.path.join(self.base_output_directory, self.run_name, "checkpoints", "")
    else:
      checkpoint_dir = os.path.join(os.path.abspath("maxtext_output"), "checkpoints", "")
    object.__setattr__(self, "checkpoint_dir", checkpoint_dir)

    # Set tensorboard_dir based on run_name and base_output_directory.
    if self.run_name and self.base_output_directory:
      tensorboard_dir = os.path.join(self.base_output_directory, self.run_name, "tensorboard", "")
    else:
      tensorboard_dir = os.path.join(os.path.abspath("maxtext_output"), "tensorboard", "")
    object.__setattr__(self, "tensorboard_dir", tensorboard_dir)

    # Managed ML Diagnostics: derive the run directory and fail at config time
    # on combinations that would only surface mid-run.
    object.__setattr__(
        self,
        "managed_mldiagnostics_dir",
        derive_managed_mldiagnostics_dir(
            self.base_output_directory,
            self.run_name,
            self.managed_mldiagnostics_storage_path,
        ),
    )
    if self.managed_mldiagnostics:
      if not self.managed_mldiagnostics_dir or not self.managed_mldiagnostics_dir.startswith("gs://"):
        raise ValueError(
            "managed_mldiagnostics=True requires a gs:// run directory: set"
            " run_name and either managed_mldiagnostics_storage_path or"
            " base_output_directory to a gs:// path (derived"
            f" managed_mldiagnostics_dir={self.managed_mldiagnostics_dir!r})."
        )
      if self.profiler == ProfilerType.XPLANE and self.managed_mldiagnostics_on_demand_profiling:
        raise ValueError(
            "managed_mldiagnostics_on_demand_profiling=True cannot be combined"
            " with profiler=xplane on the RL path: an on-demand capture during"
            " a programmatic trace session fails with 'profile already"
            " started'. Set managed_mldiagnostics_on_demand_profiling=False (or"
            " profiler='')."
        )

    if self.rl.profiler_max_num_hosts > 1:
      if self.profiler != ProfilerType.XPLANE:
        raise ValueError(f"rl.profiler_max_num_hosts > 1 requires profiler=xplane, got profiler={self.profiler!r}.")
      import pathwaysutils  # pylint: disable=import-outside-toplevel

      if not pathwaysutils.is_pathways_backend_used():
        raise ValueError("rl.profiler_max_num_hosts > 1 is only supported on the Pathways backend (JAX_PLATFORMS=proxy).")

    # Slice configuration
    if not (
        (self.cluster.num_trainer_slices == -1 and self.cluster.num_samplers_slices == -1)
        or (self.cluster.num_trainer_slices > 0 and self.cluster.num_samplers_slices > 0)
    ):
      raise ValueError("`num_trainer_slices` and `num_samplers_slices` must be both -1 or both positive.")
    self.num_slices = 1 if self.cluster.num_trainer_slices == -1 else -1

    # Set train_steps based on the number of batches, iterations, train fraction, and epochs.
    object.__setattr__(
        self,
        "train_steps",
        int(self.dataset.num_batches * self.rl.num_iterations * self.dataset.train_fraction * self.num_epoch),
    )

    # Set learning_rate_schedule_steps based on train_steps if not explicitly provided.
    if self.learning_rate_schedule_steps == -1:
      self.learning_rate_schedule_steps = self.train_steps

    # Rollout parallelism checks
    for name, val in [
        ("rollout_tensor_parallelism", self.cluster.rollout_tensor_parallelism),
        ("rollout_data_parallelism", self.cluster.rollout_data_parallelism),
        ("rollout_expert_parallelism", self.cluster.rollout_expert_parallelism),
    ]:
      if val != -1 and val <= 0:
        raise ValueError(f"`{name}` must be -1 or > 0, got {val}.")

    # Dynamically inject model dimensions.
    emb_scale, num_head_scale, mlp_dim_scale, layer_scale = get_individual_scales(self.global_parameter_scale)
    self.emb_dim = int((2**emb_scale) * self.base_emb_dim)
    self.num_query_heads = int((2**num_head_scale) * self.base_num_query_heads)
    self.num_kv_heads = int((2**num_head_scale) * self.base_num_kv_heads)
    self.mlp_dim = int((2**mlp_dim_scale) * self.base_mlp_dim)
    self.moe_mlp_dim = int((2**mlp_dim_scale) * getattr(self, "base_moe_mlp_dim", 0))
    self.num_decoder_layers = int((2**layer_scale) * self.base_num_decoder_layers)

    # Mirror into internal MaxText fields for backward compatibility.
    train_micro_batch_size = getattr(self.dataset, "train_micro_batch_size", -1)
    batch_size = getattr(self.dataset, "batch_size", 1)
    if train_micro_batch_size <= 0:
      train_micro_batch_size = batch_size
    self.micro_batch_size_to_train_on = train_micro_batch_size
    self.steps = getattr(self, "train_steps", getattr(self, "num_batches", 10))

    if self.remat_policy == "custom":
      tensors = [
          "decoder_layer_input",
          "indexer_cutoff_threshold",
          "context",
          "mlpwi",
          "moe_mlpwi_0",
          "moe_x_sorted",
          "moe_routing_maps",
          "moe_tc_routing_checkpoint",
          "moe_mlpwi_1",
          "moe_mlpwo",
          "te_moe_dispatch_remat_location",
          "mlpwi_0",
          "mlpwi_1",
          "mlpwo",
          "query_proj",
          "key_proj",
          "value_proj",
          "query_wa_proj",
          "kv_wa_proj",
          "mla_kv",
          "mla_q",
          "qkv_proj",
          "attention_out",
          "out_proj",
      ]
      self.tensors_on_device = [t for t in tensors if getattr(self, t) == "device"]
      self.tensors_to_offload = [t for t in tensors if getattr(self, t) == "offload"]

    def get_parallelism_map(prefix: str) -> dict[str, int]:
      return {
          "diloco": getattr(self, f"{prefix}_diloco_parallelism"),
          "data": getattr(self, f"{prefix}_data_parallelism"),
          "stage": getattr(self, f"{prefix}_pipeline_parallelism"),
          "fsdp": getattr(self, f"{prefix}_fsdp_parallelism"),
          "fsdp_transpose": getattr(self, f"{prefix}_fsdp_transpose_parallelism"),
          "sequence": getattr(self, f"{prefix}_sequence_parallelism"),
          "context": getattr(self, f"{prefix}_context_parallelism"),
          "context_usp_ulysses": getattr(self, f"{prefix}_context_usp_ulysses_parallelism"),
          "context_autoregressive": getattr(self, f"{prefix}_context_autoregressive_parallelism"),
          "tensor": getattr(self, f"{prefix}_tensor_parallelism"),
          "tensor_sequence": getattr(self, f"{prefix}_tensor_sequence_parallelism"),
          "model": getattr(self, f"{prefix}_tensor_parallelism"),
          "expert": getattr(self, f"{prefix}_expert_parallelism"),
          "autoregressive": getattr(self, f"{prefix}_autoregressive_parallelism"),
          "attn_dp": 1,
          "attn_dp_expert": 1,
          "dcp": 1,
          "pcp": 1,
      }

    ici_map = get_parallelism_map("ici")
    self.ici_parallelism = _resolve_parallelism(self.ici_parallelism, ici_map, self.mesh_axes)

    dcn_map = get_parallelism_map("dcn")
    self.dcn_parallelism = _resolve_parallelism(self.dcn_parallelism, dcn_map, self.mesh_axes)

    return self
