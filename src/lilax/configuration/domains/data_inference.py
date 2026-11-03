from pydantic import BaseModel, Field

from lilax.configuration.domain_types import SamplingStrategy


class InferenceGeneral(BaseModel):
  """General configuration for inference."""

  max_target_length: int = Field(2048, description="Maximum sequence length for the model.")
  max_prefill_predict_length: int = Field(64, description="Maximum length for the prefill stage in decoding.")
  prompt: str = Field("I love to", description="The default prompt for sampling.")
  system_prompt: str = Field("", description=(
      "Optional system prompt prepended to the chat message list when "
      "use_chat_template=True. Required for the gemma4-e2b / gemma4-e4b -it "
      "checkpoints which need a system role to produce coherent output."
  ))
  load_from_prefill_dir: bool = Field(False, description="Reads prefill cache from directory instead of computing it.")
  prefill_cache_dir: str = Field("", description="Directory for the prefill cache.")
  autoregressive_decode_assert: str = Field("", description="Value to assert against during autoregressive decoding, for testing.")
  model_call_mode: str = Field("", description="Mode for model call, e.g., 'inference'.")
  use_chunked_prefill: bool = Field(False, description="Use chunked prefilling for long sequences.")
  prefill_chunk_size: int = Field(256, description="The chunk size for chunked prefilling.")
  enable_model_warmup: bool = Field(False, description="Run a warmup cycle before starting the server.")
  enable_llm_inference_pool: bool = Field(False, description="Launch inference server for llm_inference_gateway.")
  multi_sampling: bool = Field(False, description="Enable multiple sampling configurations.")
  return_log_prob: bool = Field(False, description="Return log probabilities during inference.")


class Decoding(BaseModel):
  """Configuration for decoding and sampling strategies."""

  decode_sampling_strategy: SamplingStrategy = Field(SamplingStrategy.GREEDY, description="The strategy for decoding.")
  decode_sampling_nucleus_p: int | float = Field(-1.0, description="Nucleus (top-p) sampling probability. -1 to disable.")
  decode_sampling_top_k: int = Field(0, description="Top-k sampling value. 0 to disable.")
  decode_sampling_temperature: float = Field(1.0, description="Sampling temperature.")


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


class InferenceBenchmark(BaseModel):
  """Configuration for running inference microbenchmarks."""

  inference_microbenchmark_prefill_lengths: str = Field("64,128,256,512,1024", description="Prefill lengths to benchmark.")
  inference_microbenchmark_stages: str = Field("prefill,generate", description="Stages to benchmark.")
  inference_microbenchmark_loop_iters: int = Field(10, description="Number of iterations for the benchmark loop.")
  inference_microbenchmark_log_file_path: str = Field("", description="Path to log benchmark results.")
  inference_microbenchmark_num_samples: list[int] = Field([1, 2, 3, 4, 5], description="Number of samples to benchmark.")
  inference_metadata_file: str = Field("", description="Path to a JSON file with inference metadata.")
  inference_benchmark_test: bool = Field(False, description="Flag to indicate a benchmark test run.")


class PrefixCaching(BaseModel):
  """Configuration for Prefix Caching in JetStream."""

  enable_prefix_caching: bool = Field(False, description="Enable prefix caching.")
  prefix_caching_hbm_byte: int = Field(10_000_000_000, description="HBM memory allocation for prefix caching in bytes.")
  prefix_caching_dram_byte: int = Field(100_000_000_000, description="DRAM memory allocation for prefix caching in bytes.")
