"""Inert configuration fields migrated from MaxText's configuration types."""

import copy
import logging
import os
from enum import Enum, IntEnum
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Literal

from jinja2 import Environment, TemplateSyntaxError
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt, PositiveInt, PrivateAttr, ValidationInfo, field_validator, model_validator
import yaml

from lilax._backend.utils.globals import HF_IDS, MAXTEXT_ASSETS_ROOT, MAXTEXT_PKG_DIR
from lilax.configuration.domain_types import DType, DecoderBlockType, KvQuantAxis, MatmulPrecision, ModelName, QuantizationType, RematPolicy, TeCommGemmOverlapPolicy
from lilax.configuration.domain_types import DatasetType, LearningRateScheduleType, OptimizerType, ProfilerType, RopeType, SamplingStrategy, TokenizerType, VisionEncoderBlockType, WsdDecayStyle, XProfTPUPowerTraceMode
from lilax.configuration.domains.data_aot import AOT
from lilax.configuration.domains.data_debugging import DevelopmentAndDebugging, HloDump
from lilax.configuration.domains.data_datasets import DatasetGeneral, GrainDataset, HfDataset, MMapDataset, OlmoGrainDataset, TfdsDataset
from lilax.configuration.domains.data_derived import DerivedValues
from lilax.configuration.domains.data_diloco import DilocoParams
from lilax.configuration.domains.data_distillation import Distillation
from lilax.configuration.domains.data_fine_tuning import DPO, FineTuning, LoRA
from lilax.configuration.domains.data_inference import Decoding, InferenceBenchmark, InferenceGeneral, InferenceLayout, InferenceServer, PrefixCaching
from lilax.configuration.domains.data_mhc import ManifoldConstrainedHyperConnections
from lilax.configuration.domains.data_position import PositionalEmbedding, Rope, YarnRope
from lilax.configuration.domains.data_profiling import Profiling, ProfilingCore
from lilax.configuration.domains.data_rl import RL, RLDataset, RLEvaluation, RLReward, RLSpecialTokens
from lilax.configuration.domains.data_rl_rollout import RLCluster, VLLM
from lilax.configuration.domains.data_tokenizer import Tokenizer
from lilax.configuration.domains.data_qat import QwixQATConfig, QwixQATRule
from lilax.configuration.domains.execution_loss import LogitsAndLoss, MTP
from lilax.configuration.domains.execution_mesh import CustomRule, HardwareAndMesh, ReorderStrategy, ShardMode
from lilax.configuration.domains.execution_monitoring import GcpMonitoring, Goodput, ManagedMLDiagnostics, Metrics, Tensorboard
from lilax.configuration.domains.execution_multimodal import AudioEncoder, Multimodal, MultimodalGeneral, VisionProjector, VisionTower
from lilax.configuration.domains.execution_optimizer import AdamW, Muon, Optimizer
from lilax.configuration.domains.execution_parallelism import DcnParallelism, IciParallelism, PipelineParallelism
from lilax.configuration.domains.execution_remat import RematAndOffload
from lilax.configuration.domains.execution_sharding import LayoutAndSharding
from lilax.configuration.domains.execution_training import ElasticTraining, TrainingLoop
from lilax.configuration.domains.model_checkpointing import Checkpointing, EmergencyCheckpointing, OrbaxStorage
from lilax.configuration.domains.model_architecture import ModelArchitecture
from lilax.configuration.domains.model_attention import Attention
from lilax.configuration.domains.model_attention_variants import AttentionIndexer, CompressedAttention, Llama4Attention, MlaAttention, MoBa
from lilax.configuration.domains.model_data_types import DataTypes
from lilax.configuration.domains.model_engram import Engram
from lilax.configuration.domains.model_ming import Ming
from lilax.configuration.domains.model_moe import DeepSeekMoE, MoEGeneral, MoEKernels, RematLocation, TEGroupedGemmQuantizationType
from lilax.configuration.domains.model_qwen import Qwen3Next, Qwen3_8FlashNext
from lilax.configuration.domains.model_quantization import Quantization
from lilax.configuration.domains.model_run import RunInfo
from lilax.configuration.domains.model_splash import SplashAttention
from lilax.configuration.validators.attention_indexer import validate_attention_indexer
from lilax.configuration.validators.block_diffusion import validate_block_diffusion_options
from lilax.configuration.validators.defaults import resolve_inert_defaults
from lilax.configuration.validators.explicit_sharding import validate_explicit_sharding_options
from lilax.configuration.validators.mesh import resolve_custom_mesh_rules
from lilax.configuration.validators.moe import validate_moe_retry_options, validate_moe_routing_options, validate_shard_embed_moe_on_fsdp
from lilax.configuration.validators.quantization import validate_quantization_options
from lilax.configuration.validators.remaining_attention import validate_attention_options
from lilax.configuration.validators.remaining_attention_kernel import validate_attention_kernel_options
from lilax.configuration.validators.remaining_context_parallel import validate_context_parallel_options
from lilax.configuration.validators.remaining_expert import validate_expert_options
from lilax.configuration.validators.remaining_input_checkpoint import validate_input_and_checkpoint_options
from lilax.configuration.validators.remaining_moe import validate_moe_options
from lilax.configuration.validators.weight_layout import validate_weight_layout_options
from lilax.configuration.values import load_config_values

logger = logging.getLogger(__name__)


class MaxTextConfig(
    RunInfo, Checkpointing, OrbaxStorage, EmergencyCheckpointing, ElasticTraining, DataTypes, Quantization,
    ModelArchitecture, Engram, MTP, LogitsAndLoss, Attention, MlaAttention, CompressedAttention, MoBa,
    AttentionIndexer, Llama4Attention, SplashAttention, MoEGeneral, MoEKernels, DeepSeekMoE, Qwen3Next,
    Qwen3_8FlashNext, Ming, HardwareAndMesh, LayoutAndSharding, DcnParallelism, IciParallelism,
    PipelineParallelism, RematAndOffload, TrainingLoop, ManifoldConstrainedHyperConnections, DilocoParams,
    Optimizer, AdamW, Muon, FineTuning, Distillation, VLLM, PositionalEmbedding, Rope, YarnRope,
    DatasetGeneral, TfdsDataset, HfDataset, GrainDataset, MMapDataset, OlmoGrainDataset, Tokenizer,
    InferenceGeneral, Decoding, InferenceLayout, InferenceServer, InferenceBenchmark, PrefixCaching, AOT,
    DevelopmentAndDebugging, Profiling, HloDump, Metrics, Goodput, GcpMonitoring, Tensorboard,
    ManagedMLDiagnostics, MultimodalGeneral, VisionTower, VisionProjector, AudioEncoder, DerivedValues,
):
  """
  The main configuration object for MaxText.

  This class aggregates all configuration options from modular `BaseModel` classes
  into a single, validated object. It is populated by the `initialize` function.
  Every field is explicitly defined to prevent misconfigurations (`extra='forbid'`).
  """

  model_config = ConfigDict(extra="forbid", protected_namespaces=())
  _explicit_keys: frozenset[str] = PrivateAttr(default_factory=frozenset)
  _eval_rule_was_default: bool = PrivateAttr(default=False)
  _pipeline_input_layout: dict[str, Any] = PrivateAttr(default_factory=dict)
  _learning_rate_schedule_steps_input: int = PrivateAttr(default=-1)
  _source_inputs: tuple[str | Path | None, dict[str, Any]] | None = PrivateAttr(default=None)
  dpo: DPO = Field(default_factory=DPO, description="Configuration for DPO and ORPO alignment algorithms.")
  lora: LoRA = Field(default_factory=LoRA, description="Configuration for LoRA / QLoRA adapters.")
  qwix_qat_config: QwixQATConfig | None = None

  def __init__(self, **data: Any) -> None:
    source_data = copy.deepcopy(data)
    source_root = source_data.pop("base_config", None)
    values = load_config_values(**data)
    self.__pydantic_validator__.validate_python(values, self_instance=self)
    explicit_keys = set(data) | {key for key in values if "M_" + key.upper() in os.environ}
    self._explicit_keys = frozenset(explicit_keys)
    self._source_inputs = (source_root, source_data)
    object.__setattr__(self, "__pydantic_fields_set__", explicit_keys)

  validate_shard_embed_moe_on_fsdp = model_validator(mode="after")(validate_shard_embed_moe_on_fsdp)

  validate_weight_layout_options = model_validator(mode="after")(validate_weight_layout_options)

  resolve_custom_mesh_rules = model_validator(mode="after")(resolve_custom_mesh_rules)

  validate_moe_options = model_validator(mode="after")(validate_moe_options)

  validate_moe_routing_options = model_validator(mode="after")(validate_moe_routing_options)

  validate_expert_options = model_validator(mode="after")(validate_expert_options)

  validate_moe_retry_options = model_validator(mode="after")(validate_moe_retry_options)

  validate_quantization_options = model_validator(mode="after")(validate_quantization_options)

  resolve_inert_defaults = model_validator(mode="after")(resolve_inert_defaults)

  @model_validator(mode="after")
  def validate_attention_options(self) -> "MaxTextConfig":
    return validate_attention_options(self)

  validate_attention_indexer = model_validator(mode="after")(validate_attention_indexer)

  validate_attention_kernel_options = model_validator(mode="after")(validate_attention_kernel_options)

  validate_block_diffusion_options = model_validator(mode="after")(validate_block_diffusion_options)

  validate_explicit_sharding_options = model_validator(mode="after")(validate_explicit_sharding_options)

  @model_validator(mode="after")
  def validate_context_parallel_options(self) -> "MaxTextConfig":
    return validate_context_parallel_options(self)

  @model_validator(mode="after")
  def validate_input_and_checkpoint_options(self) -> "MaxTextConfig":
    return validate_input_and_checkpoint_options(self)

  @classmethod
  def _from_resolved(cls, values: dict[str, Any], explicit_keys) -> "MaxTextConfig":
    config = cls.__new__(cls)
    cls.__pydantic_validator__.validate_python(values, self_instance=config)
    config._explicit_keys = frozenset(explicit_keys)
    object.__setattr__(config, "__pydantic_fields_set__", set(explicit_keys))
    return config

  def model_post_init(self, __context: Any) -> None:
    self._explicit_keys = frozenset(self.model_fields_set)
    self._learning_rate_schedule_steps_input = self.learning_rate_schedule_steps

  def model_copy(self, *, update: dict[str, Any] | None = None, deep: bool = False) -> "MaxTextConfig":
    config = super().model_copy(update=update, deep=deep)
    if update:
      config._explicit_keys = self._explicit_keys | update.keys()
      layout_updates = {key: update[key] for key in self._pipeline_input_layout if key in update}
      if layout_updates:
        config._pipeline_input_layout = copy.deepcopy(self._pipeline_input_layout)
        config._pipeline_input_layout.update(copy.deepcopy(layout_updates))
      if "custom_mesh_and_rule_for_eval" in update:
        config._eval_rule_was_default = update["custom_mesh_and_rule_for_eval"] in (CustomRule.DEFAULT, "")
    if update and "learning_rate_schedule_steps" in update:
      config._learning_rate_schedule_steps_input = update["learning_rate_schedule_steps"]
    if self._source_inputs is not None:
      if update and ("base_config" in update or set(update) & DerivedValues.model_fields.keys()):
        config._source_inputs = None
      else:
        root, overrides = copy.deepcopy(self._source_inputs)
        if update:
          overrides.update(copy.deepcopy(update))
        config._source_inputs = (root, overrides)
    return config

  def get_keys(self) -> dict[str, Any]:
    return self.model_dump()

  def get_source_inputs(self) -> tuple[str | Path | None, dict[str, Any]] | None:
    return copy.deepcopy(self._source_inputs)

  def get_source_values(self) -> dict[str, Any]:
    values = self.model_dump()
    values.update(copy.deepcopy(self._pipeline_input_layout))
    values["learning_rate_schedule_steps"] = self._learning_rate_schedule_steps_input
    if self._eval_rule_was_default:
      values["custom_mesh_and_rule_for_eval"] = ""
    return values


class RLConfig(
    LogitsAndLoss, Engram, ManifoldConstrainedHyperConnections, RematAndOffload, Attention, Llama4Attention,
    LayoutAndSharding, InferenceLayout, InferenceGeneral, PrefixCaching, Decoding, IciParallelism, DcnParallelism,
    PipelineParallelism, DilocoParams, HardwareAndMesh, ModelArchitecture, MTP, MoBa, MlaAttention,
    CompressedAttention, AttentionIndexer, SplashAttention, Qwen3Next, Qwen3_8FlashNext, Ming, MultimodalGeneral,
    Muon, FineTuning, Distillation, DatasetGeneral, TfdsDataset, HfDataset, GrainDataset, OlmoGrainDataset,
    EmergencyCheckpointing, ElasticTraining, InferenceServer, InferenceBenchmark, HloDump, Goodput, GcpMonitoring,
    ManagedMLDiagnostics, PositionalEmbedding, Rope, YarnRope, MoEGeneral, MoEKernels, DeepSeekMoE, RunInfo,
    Checkpointing, OrbaxStorage, DataTypes, Tokenizer, AdamW, Optimizer, TrainingLoop, Quantization, VisionTower,
    VisionProjector, AudioEncoder, DevelopmentAndDebugging, Profiling, AOT, Metrics, Tensorboard, DerivedValues,
    RL, RLCluster, RLDataset, RLEvaluation, RLReward, RLSpecialTokens, VLLM,
):
  """
  Configuration for Reinforcement Learning in MaxText.
  """

  model_config = ConfigDict(extra="ignore", protected_namespaces=())
  _explicit_keys: frozenset[str] = PrivateAttr(default_factory=frozenset)
  _profiler_synthesized: frozenset[str] = PrivateAttr(default_factory=frozenset)

  num_epoch: int = Field(1, ge=1, description="Number of epochs to train for.")
  eval_interval: int = Field(-1, description="Run evaluation every N training steps. -1 disables interval-based evaluation.")
  enable_dropout: bool = Field(True, description="Enables dropout in the model.")
  dropout_rate: float = Field(0.0, ge=0.0, le=1.0, description="The dropout rate.")
  init_weights_seed: int = Field(0, description="Seed for model weight initialization.")
  log_period: int = Field(100, description="Frequency (in steps) to log metrics and flush to Tensorboard.")
  hf_access_token: str | None = Field(None, description="Hugging Face API access token.")
  enable_tunix_perf_metrics: bool = Field(False, description="Whether to enable Tunix-managed metrics measurement. The metrics will be uploaded to tensorboard.")
  max_target_length: int = Field(2048, description="Maximum sequence length for the model.")
  max_prefill_predict_length: int = Field(64, description="Maximum length for the prefill stage in decoding.")
  debug: bool = Field(False, description="Enable debug mode for RL.")
  cluster: RLCluster = Field(default_factory=RLCluster, description="Configuration for RL cluster.")
  rl: RL = Field(default_factory=RL, description="Configuration for RL algorithms like Group Relative Policy Optimization (GRPO).")
  dataset: RLDataset = Field(default_factory=RLDataset, description="Configuration for RL datasets.")
  reward: RLReward = Field(default_factory=RLReward, description="Configuration for RL rewards.")
  special_tokens: RLSpecialTokens = Field(default_factory=RLSpecialTokens, description="Configuration for RL special tokens.")
  evaluation: RLEvaluation = Field(default_factory=RLEvaluation, description="Configuration for RL evaluation.")
  vllm: VLLM = Field(default_factory=VLLM, description="Configuration for rollouts.")

  @model_validator(mode="before")
  @classmethod
  def map_flat_fields_to_nested_models(cls, data: Any, info: ValidationInfo) -> Any:
    if not isinstance(data, dict) or (info.context and info.context.get("resolved_values")):
      return data
    for field_name, field_info in cls.model_fields.items():
      annotation = field_info.annotation
      if hasattr(annotation, "model_fields"):
        nested = data.setdefault(field_name, {})
        if isinstance(nested, BaseModel):
          nested = nested.model_dump()
          data[field_name] = nested
        if not isinstance(nested, dict):
          nested = {}
          data[field_name] = nested
        for inner_key in annotation.model_fields:
          if inner_key in data:
            nested[inner_key] = data[inner_key]

    rl = data.get("rl")
    if isinstance(rl, dict):
      for legacy, current, minimum in (
          ("skip_first_n_steps_for_profiler", "profiler_start_invocation", 0),
          ("profiler_steps", "profiler_num_invocations", 1),
      ):
        default = RL.model_fields[current].default
        if legacy in data and current not in data and rl.get(current, default) == default:
          try:
            value = int(data[legacy])
          except (TypeError, ValueError):
            continue
          if value >= minimum:
            rl[current] = value
            data[current] = value
    return data

  @model_validator(mode="wrap")
  @classmethod
  def capture_profiler_provenance(cls, data: Any, handler: Any) -> "RLConfig":
    input_keys = set(data) if isinstance(data, dict) else set()
    config = handler(data)
    if isinstance(data, dict):
      config._profiler_synthesized = frozenset(
          {"profiler_start_invocation", "profiler_num_invocations"} & (config.model_fields_set - input_keys)
      )
    return config

  @model_validator(mode="after")
  def validate_rl_constraints(self) -> "RLConfig":
    if 0 < self.max_seq_token_per_tpu < self.max_target_length:
      raise ValueError(
          f"max_seq_token_per_tpu ({self.max_seq_token_per_tpu}) must be at least "
          f"max_target_length ({self.max_target_length}) when sequence packing is enabled."
      )
    if self.tokenizer_path is None:
      if self.model_name not in HF_IDS:
        raise ValueError(
            "model_name not found in HF_IDS in maxtext/src/maxtext/utils/globals.py. "
            "Please pass tokenizer_path in your command."
        )
      self.tokenizer_path = HF_IDS[self.model_name]
      self.tokenizer_type = TokenizerType.HUGGINGFACE
    if self.optimizer_memory_host_offload:
      raise ValueError(
          "optimizer_memory_host_offload=True is not supported on the post-training "
          "RL path because the underlying Tunix RLCluster/Trainer does not "
          "support host offloading of the optimizer state."
      )
    if self.num_vocab_tiling > 1:
      raise ValueError(
          f"Vocab Tiling is not supported with RL. num_vocab_tiling was configured to {self.num_vocab_tiling}, "
          "but it must be 1 when running train_rl."
      )
    if self.run_name and self.base_output_directory:
      self.checkpoint_dir = os.path.join(self.base_output_directory, self.run_name, "checkpoints", "")
      self.tensorboard_dir = os.path.join(self.base_output_directory, self.run_name, "tensorboard", "")
    else:
      self.checkpoint_dir = os.path.join(os.path.abspath("maxtext_output"), "checkpoints", "")
      self.tensorboard_dir = os.path.join(os.path.abspath("maxtext_output"), "tensorboard", "")
    telemetry_base = self.managed_mldiagnostics_storage_path or self.base_output_directory
    self.managed_mldiagnostics_dir = (
        os.path.join(telemetry_base, self.run_name, "managed-mldiagnostics") if telemetry_base and self.run_name else None
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
    if self.rl.profiler_max_num_hosts > 1 and self.profiler != ProfilerType.XPLANE:
      raise ValueError(f"rl.profiler_max_num_hosts > 1 requires profiler=xplane, got profiler={self.profiler!r}.")
    if not (
        (self.cluster.num_trainer_slices == -1 and self.cluster.num_samplers_slices == -1)
        or (self.cluster.num_trainer_slices > 0 and self.cluster.num_samplers_slices > 0)
    ):
      raise ValueError("`num_trainer_slices` and `num_samplers_slices` must be both -1 or both positive.")
    self.num_slices = 1 if self.cluster.num_trainer_slices == -1 else -1
    train_steps = int(self.dataset.num_batches * self.rl.num_iterations * self.dataset.train_fraction * self.num_epoch)
    object.__setattr__(self, "train_steps", train_steps)
    if self.learning_rate_schedule_steps == -1:
      self.learning_rate_schedule_steps = train_steps
    for name, value in (
        ("rollout_tensor_parallelism", self.cluster.rollout_tensor_parallelism),
        ("rollout_data_parallelism", self.cluster.rollout_data_parallelism),
        ("rollout_expert_parallelism", self.cluster.rollout_expert_parallelism),
    ):
      if value != -1 and value <= 0:
        raise ValueError(f"`{name}` must be -1 or > 0, got {value}.")
    if self.global_parameter_scale == 0:
      raise ValueError("Global parameter scale cannot be zero.")
    if self.global_parameter_scale < 0:
      raise ValueError("math domain error")
    if self.global_parameter_scale & (self.global_parameter_scale - 1):
      raise ValueError(
          "Global parameter scale should be a power of two. If you want finer grained control of the model sizes "
          "then you can explicitly set base_embed_dim, base_num_query_heads, base_num_kv_heads, "
          "base_mlp_dim, base_num_decoder_layers and/or head_dim."
      )
    base_scale, remainder = divmod(self.global_parameter_scale.bit_length() - 1, 3)
    head_scale = base_scale + int(remainder > 0)
    self.emb_dim = (2 ** (base_scale + int(remainder > 1))) * self.base_emb_dim
    self.num_query_heads = (2 ** head_scale) * self.base_num_query_heads
    self.num_kv_heads = (2 ** head_scale) * self.base_num_kv_heads
    self.mlp_dim = (2 ** head_scale) * self.base_mlp_dim
    self.moe_mlp_dim = (2 ** head_scale) * self.base_moe_mlp_dim
    self.num_decoder_layers = (2 ** base_scale) * self.base_num_decoder_layers
    self.micro_batch_size_to_train_on = (
        self.dataset.train_micro_batch_size if self.dataset.train_micro_batch_size > 0 else self.dataset.batch_size
    )
    self.steps = train_steps
    if self.remat_policy == "custom":
      tensors = [
          "decoder_layer_input", "indexer_cutoff_threshold", "context", "mlpwi", "moe_mlpwi_0", "moe_x_sorted",
          "moe_routing_maps", "moe_tc_routing_checkpoint", "moe_mlpwi_1", "moe_mlpwo", "te_moe_dispatch_remat_location",
          "mlpwi_0", "mlpwi_1", "mlpwo", "query_proj", "key_proj", "value_proj", "query_wa_proj",
          "kv_wa_proj", "mla_kv", "mla_q", "qkv_proj", "attention_out", "out_proj",
      ]
      self.tensors_on_device = [tensor for tensor in tensors if getattr(self, tensor) == "device"]
      self.tensors_to_offload = [tensor for tensor in tensors if getattr(self, tensor) == "offload"]
    for prefix in ("ici", "dcn"):
      axis_map = {
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
          "attn_dp": 1, "attn_dp_expert": 1, "dcp": 1, "pcp": 1,
      }
      configured = getattr(self, f"{prefix}_parallelism")
      if configured and len(configured) == len(self.mesh_axes) and any(axis not in axis_map for axis in self.mesh_axes):
        parallelism = list(configured)
      else:
        parallelism = [axis_map.get(axis, 1) for axis in self.mesh_axes]
      setattr(self, f"{prefix}_parallelism", parallelism)
    return self

  def model_post_init(self, __context: Any) -> None:
    self._explicit_keys = frozenset(self.model_fields_set)

  def model_copy(self, *, update: dict[str, Any] | None = None, deep: bool = False) -> "RLConfig":
    config = super().model_copy(update=update, deep=deep)
    if update:
      config._explicit_keys = self._explicit_keys | update.keys()
      config._profiler_synthesized = self._profiler_synthesized - update.keys()
      if "rl" in update:
        config._explicit_keys = config._explicit_keys - config._profiler_synthesized
        config._profiler_synthesized = frozenset()
      for legacy, current in (
          ("skip_first_n_steps_for_profiler", "profiler_start_invocation"),
          ("profiler_steps", "profiler_num_invocations"),
      ):
        if legacy in update and current in config._profiler_synthesized:
          default = RL.model_fields[current].default
          config.rl = (config.rl | {current: default} if isinstance(config.rl, dict)
                       else config.rl.model_copy(update={current: default}))
          setattr(config, current, default)
          config._profiler_synthesized = config._profiler_synthesized - {current}
          config._explicit_keys = config._explicit_keys - {current}
      for field_name, field_info in type(self).model_fields.items():
        annotation = field_info.annotation
        if hasattr(annotation, "model_fields"):
          nested_updates = {key: update[key] for key in annotation.model_fields if key in update}
          if nested_updates:
            nested = getattr(config, field_name)
            if isinstance(nested, BaseModel):
              nested = nested.model_copy(update=nested_updates)
            elif isinstance(nested, dict):
              nested = nested | nested_updates
            else:
              nested = nested_updates
            setattr(config, field_name, nested)
      for legacy, current, minimum in (
          ("skip_first_n_steps_for_profiler", "profiler_start_invocation", 0),
          ("profiler_steps", "profiler_num_invocations", 1),
      ):
        default = RL.model_fields[current].default
        nested_value = config.rl.get(current, default) if isinstance(config.rl, dict) else getattr(config.rl, current)
        if legacy in update and current not in config._explicit_keys and nested_value == default:
          try:
            value = int(update[legacy])
          except (TypeError, ValueError):
            continue
          if value >= minimum:
            config.rl = (config.rl | {current: value} if isinstance(config.rl, dict)
                         else config.rl.model_copy(update={current: value}))
            setattr(config, current, value)
            config._profiler_synthesized = config._profiler_synthesized | {current}
    return config

  def get_keys(self) -> dict[str, Any]:
    return self.model_dump()
