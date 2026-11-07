"""Serving placement, cache and endpoint settings."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lilax.configuration.domains.data_inference import InferenceLayout
from lilax.distributed import ParallelismConfig


class InferenceCacheConfig(InferenceLayout):
  model_config = ConfigDict(extra="forbid")
  load_from_prefill_dir: bool = False
  prefill_cache_dir: str = ""


class JetStreamConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  inference_server: str = "MaxtextInterleavedServer"
  prefill_slice: str = "v5e-16"
  generate_slice: str = "v5e-16"
  use_chunked_prefill: bool = False
  prefill_chunk_size: int = 256
  enable_prefix_caching: bool = False
  prefix_caching_hbm_byte: int = 10_000_000_000
  prefix_caching_dram_byte: int = 100_000_000_000
  enable_model_warmup: bool = False
  lora_input_adapters_path: str = ""
  multi_sampling: bool = False
  prometheus_port: int = 0
  parallelism_config: ParallelismConfig | None = None
  jax_distributed_initialization_timeout: int = 300
  cache_config: InferenceCacheConfig | None = None


class VLLMConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  hbm_utilization_vllm: float = 0.72
  enable_dp_attention: bool = False
  async_scheduling: bool = False
  max_num_batched_tokens: int | None = None
  max_num_seqs: int | None = None
  vllm_hf_config_path: str = ""
  hf_overrides: dict[str, Any] = Field(default_factory=dict)
  additional_config: dict[str, Any] = Field(default_factory=dict)
  parallelism_config: ParallelismConfig | None = None
  jax_distributed_initialization_timeout: int = 300


class EvaluationServerConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  max_model_len: int
  host: str = "localhost"
  port: int = 8000
  tensor_parallel_size: int = 4
  expert_parallel_size: int = 1
  data_parallel_size: int = 1
  max_num_batched_tokens: int | None = None
  max_num_seqs: int | None = None
  hbm_memory_utilization: float = 0.3
  concurrency: int | None = None
  enable_chat_api: bool = False
