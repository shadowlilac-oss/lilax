"""Native inference layout, server, and prefix-cache configuration fields."""

from pydantic.fields import Field
from pydantic.main import BaseModel


class InferenceLayout(BaseModel):
  """Configuration for KV cache and compute layouts during inference."""

  stack_prefill_result_cache: bool = Field(False, description="Stack prefill cache across layers to reduce latency.")
  prefill_cache_axis_order: str = Field("1,2,0,3", description="Axis order for the prefill KV cache.")
  ar_cache_axis_order: str = Field("1,2,0,3", description="Axis order for the autoregressive KV cache.")
  compute_axis_order: str = Field("0,1,2,3", description="Axis order for compute operations.")
  reshape_q: bool = Field(False, description="Reshape Q tensor in attention.")


class InferenceServer(BaseModel):
  """Configuration for running as an inference server."""

  inference_server: str = Field("MaxtextInterleavedServer", description="Inference server to start.")
  prefill_slice: str = Field("v5e-16", description="Slice to use for prefill in disaggregation mode.")
  generate_slice: str = Field("v5e-16", description="Slice to use for generatation in disaggregation mode.")


class PrefixCaching(BaseModel):
  """Configuration for Prefix Caching in JetStream."""

  enable_prefix_caching: bool = Field(False, description="Enable prefix caching.")
  prefix_caching_hbm_byte: int = Field(10_000_000_000, description="HBM memory allocation for prefix caching in bytes.")
  prefix_caching_dram_byte: int = Field(
      100_000_000_000,
      description="DRAM memory allocation for prefix caching in bytes.",
  )
