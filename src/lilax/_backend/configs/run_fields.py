"""Native run identity, model selection, and output configuration fields."""

from typing import Literal

from lilax._backend.configs.path_types import PathStr
from pydantic.fields import Field
from pydantic.main import BaseModel


ModelName = Literal[
    "default",
    "llama2-7b",
    "llama2-13b",
    "llama2-70b",
    "llama3-8b",
    "llama3.1-8b-Instruct",
    "llama3-70b",
    "llama3.1-70b-Instruct",
    "llama3.1-8b",
    "llama3.1-70b",
    "llama3.1-405b",
    "llama3.3-70b",
    "mistral-7b",
    "mixtral-8x7b",
    "mixtral-8x22b",
    "deepseek2-16b",
    "deepseek2-236b",
    "deepseek3-671b",
    "deepseek3-671b-2dfsdp",
    "deepseek3-671b-batchsplit",
    "deepseek3-671b-lineage",
    "deepseek3-test",
    "deepseek3-tiny",
    "deepseek3.2-671b",
    "deepseek4-tiny",
    "deepseek4-284b",
    "deepseek4-1600b",
    "deepseek-custom",
    "kimi-k2-1t",
    "gemma-7b",
    "gemma-2b",
    "gemma2-2b",
    "gemma2-9b",
    "gemma2-27b",
    "gemma3-4b",
    "gemma3-12b",
    "gemma3-27b",
    "gemma4-26b",
    "gemma4-31b",
    "gemma4-e2b",
    "gemma4-e4b",
    "maxtext-omni-gemma3-qwen3",
    "qwen2.5-1.5b",
    "qwen2.5-7b",
    "qwen2.5-14b",
    "qwen3-0.6b",
    "qwen3-1.7b",
    "qwen3-1.7b-base",
    "qwen3-4b",
    "qwen3-4b-base",
    "qwen3-4b-thinking-2507",
    "qwen3-8b",
    "qwen3-8b-base",
    "qwen3-14b",
    "qwen3-14b-base",
    "qwen3-32b",
    "qwen3-235b-a22b",
    "qwen3-30b-a3b",
    "qwen3-30b-a3b-base",
    "qwen3-480b-a35b",
    "qwen3-vl-2b",
    "qwen3-vl-4b",
    "qwen3-vl-30b-a3b",
    "weaver-mini",
    "weaver-max",
    "weaver-mini-diffuser",
    "qwen3-next-80b-a3b",
    "qwen3.8-flash-next",
    "qwen3.8-flash-next-mini",
    "qwen3-omni-30b-a3b",
    "qwen3-custom-30b-a3b",
    "qwen3.5-9b",
    "qwen3.5-35b-a3b",
    "qwen3.5-35b-a3b-fp8",
    "qwen3.5-35b-fp8",
    "qwen3.5-397b-a17b",
    "qwen3.5-397b-a17b-fp8",
    "qwen3.8-2.4t-a95b",
    "gpt3-175b",
    "gpt3-22b",
    "gpt3-6b",
    "gpt3-52k",
    "gpt-oss-20b",
    "gpt-oss-120b",
    "llama4-17b-16e",
    "llama4-17b-128e",
    "olmo3-7b",
    "olmo3-7b-pt",
    "olmo3-32b",
    "envy-test",
    "envy-switch-base",
    "envy-switch-large",
    "envy-switch-xxl",
    "glm5.3-flash",
    "ming-3.0-audio-tiny",
    "hy4-preview",
]


class RunInfo(BaseModel):
  """Configuration for the overall run, model identity, and logging."""

  base_config: None | str = Field(
      None,
      description="Base config to inherit from. This is a meta-field and is consumed by the config loading system.",
  )
  run_name: str = Field(
      "",
      description="The name of the run. Checkpoints will be stored under this name.",
  )
  model_name: ModelName = Field("default", description="The name of the model configuration to use.")
  use_m3_model: bool = Field(False, description="Use the m3 backend for a registered model.")
  override_model_config: bool = Field(False, description="If True, allows overriding model parameters via CLI.")
  override_logical_axis_rules: bool = Field(
      False,
      description="If True, logical_axis_rules will be overridden instead of merged.",
  )
  log_config: bool = Field(
      True,
      description="If True, prints the final configuration after initialization.",
  )
  debug_sharding: bool = Field(False, description="If True, print model weight sharding details.")
  base_output_directory: PathStr = Field("", description="Base directory for all outputs, typically a GCS path.")
  enable_mllog: bool = Field(False, description="If True, enables MLPerf logging (mllog).")
  mllog_file: None | PathStr = Field(
      "",
      description="Optional filename or path for mllog export in base_output_directory "
      "(defaults to 'mllog_<data_shuffle_seed>.log').",
  )
  sharding_strategy: None | Literal["experimental"] = Field(
      None,
      description="Experimental sharding strategy used for some inference configs.",
  )
