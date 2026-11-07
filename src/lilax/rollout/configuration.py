"""Rollout engine settings, device allocation and weight synchronization."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from lilax.configuration.domains.data_rl_rollout import RLCluster, VLLM
from lilax.generation.configuration import GenerationConfig


class RolloutClusterConfig(RLCluster):
  model_config = ConfigDict(extra="forbid")


class VLLMRolloutConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  kv_cache_buffer: int = VLLM.model_fields["kv_cache_buffer"]
  hbm_utilization_vllm: float = VLLM.model_fields["hbm_utilization_vllm"]
  swap_space_vllm_gb: int = VLLM.model_fields["swap_space_vllm_gb"]
  enable_dp_attention: bool = VLLM.model_fields["enable_dp_attention"]
  enable_expert_parallel: bool = VLLM.model_fields["enable_expert_parallel"]
  async_scheduling: bool = VLLM.model_fields["async_scheduling"]
  max_num_batched_tokens: int | None = VLLM.model_fields["max_num_batched_tokens"]
  max_num_seqs: int | None = VLLM.model_fields["max_num_seqs"]
  vllm_block_size: int | None = VLLM.model_fields["vllm_block_size"]
  stop_strings: list[str] | None = VLLM.model_fields["stop_strings"]
  additional_config: dict[str, Any] = Field(default_factory=dict)
  hf_overrides: dict[str, Any] = Field(default_factory=dict)
  vllm_hf_config_path: str = VLLM.model_fields["vllm_hf_config_path"]

  def _source_values(self):
    return {"vllm_additional_config" if key == "additional_config" else "vllm_hf_overrides" if key == "hf_overrides"
            else key: value for key, value in self.model_dump().items()}


class WeightSyncConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  free_kv_cache_during_weight_sync: bool = VLLM.model_fields["free_kv_cache_during_weight_sync"]
  use_standalone_converter: bool = VLLM.model_fields["use_standalone_converter"]
  use_weight_converter: bool = VLLM.model_fields["use_weight_converter"]
  use_raiden_ffi: bool | None = VLLM.model_fields["use_raiden_ffi"]
  rollout_backend: Literal["maxtext", "vllm_torchax"] = VLLM.model_fields["rollout_backend"]
  rollout_weight_dtype: str = VLLM.model_fields["rollout_weight_dtype"]
  weight_sync_debug: bool = VLLM.model_fields["weight_sync_debug"]
  log_weight_sync_time: bool = VLLM.model_fields["log_weight_sync_time"]
  kv_tp_size: int = VLLM.model_fields["kv_tp_size"]
  moe_mlp_tp_size: int = VLLM.model_fields["moe_mlp_tp_size"]
  vllm_load_format: str = VLLM.model_fields["vllm_load_format"]
  debug_converter: bool = VLLM.model_fields["debug_converter"]
  gcs_debug_path: str = VLLM.model_fields["gcs_debug_path"]


class RolloutConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  generation_config: GenerationConfig | None = None
  max_prompt_length: int | None = None
  engine: Literal["maxtext", "vllm"] | None = None
  reshard_chunk_size: int | None = None
  cluster: RolloutClusterConfig | None = None
  vllm: VLLMRolloutConfig | None = None
  weight_sync: WeightSyncConfig | None = None
