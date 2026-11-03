"""Run identity configuration."""

from typing import Literal

from pydantic import BaseModel, Field

from lilax.configuration.domain_types import ModelName


class RunInfo(BaseModel):
  """Configuration for the overall run, model identity, and logging."""

  base_config: None | str = Field(None, description="Base config to inherit from. This is a meta-field and is consumed by the config loading system.")
  run_name: str = Field("", description="The name of the run. Checkpoints will be stored under this name.")
  model_name: ModelName = Field("default", description="The name of the model configuration to use.")
  use_m3_model: bool = Field(False, description="Use the m3 backend for a registered model.")
  override_model_config: bool = Field(False, description="If True, allows overriding model parameters via CLI.")
  override_logical_axis_rules: bool = Field(False, description="If True, logical_axis_rules will be overridden instead of merged.")
  log_config: bool = Field(True, description="If True, prints the final configuration after initialization.")
  debug_sharding: bool = Field(False, description="If True, print model weight sharding details.")
  base_output_directory: str = Field("", description="Base directory for all outputs, typically a GCS path.")
  enable_mllog: bool = Field(False, description="If True, enables MLPerf logging (mllog).")
  mllog_file: None | str = Field("", description="Optional filename or path for mllog export in base_output_directory "
                                "(defaults to 'mllog_<data_shuffle_seed>.log').")
  sharding_strategy: None | Literal["experimental"] = Field(None, description="Experimental sharding strategy used for some inference configs.")
