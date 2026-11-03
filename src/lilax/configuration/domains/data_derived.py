from pydantic import BaseModel, Field


class DerivedValues(BaseModel):
  """Holds all fields that are derived from other config values for perfect legacy compatibility."""
  __module__ = "lilax.configuration.schema"

  emb_dim: int | None = Field(None, description="Effective embedding dimension, scaled by `global_parameter_scale`.")
  mlp_dim: int | None = Field(None, description="Effective MLP dimension, scaled by `global_parameter_scale`.")
  moe_mlp_dim: int | None = Field(None, description="Effective MLP dimension for MoE layers, scaled by `global_parameter_scale`.")
  num_decoder_layers: int | None = Field(None, description="Effective number of decoder layers, scaled by `global_parameter_scale`.")
  num_kv_heads: int | None = Field(None, description="Effective number of key/value heads, scaled by `global_parameter_scale`.")
  num_query_heads: int | None = Field(None, description="Effective number of query heads, scaled by `global_parameter_scale`.")
  num_diloco_replicas: int | None = Field(None, description="The number of diloco replicas, derived from ICI and DCN values.")
  ici_parallelism: list[int] | None = Field(None, description="Aggregated list of all ICI parallelism values for legacy compatibility.")
  dcn_parallelism: list[int] | None = Field(None, description="Aggregated list of all DCN parallelism values for legacy compatibility.")
  using_pipeline_parallelism: bool | None = Field(None, description="Boolean flag indicating if pipeline parallelism is active across ICI or DCN.")
  num_target_devices: int | None = Field(None, description="The number of devices computed from topology in train_compile or jax.devices() in train")
  global_batch_size_to_train_on: int | None = Field(
      None, description="The total batch size for training across all devices. Derived from `per_device_batch_size` and data" "parallelism."
  )
  global_batch_size_to_eval_on: int | None = Field(
      None, description="The total batch size for evaluation across all devices. Derived from `eval_per_device_batch_size` and" " data parallelism."
  )
  global_batch_size_to_load: int | None = Field(None, description="The global batch size for the training dataloader, potentially scaled by `expansion_factor_real_data`.")
  global_batch_size_to_load_eval: int | None = Field(None, description="The global batch size for the evaluation dataloader, potentially scaled by `expansion_factor_real_data`.")
  micro_batch_size_to_train_on: int | None = Field(None, description="The size of each micro-batch for training, used in pipeline parallelism. Derived from `global_batch_size_to_train_on`.")
  micro_batch_size_to_eval_on: int | None = Field(None, description="The size of each micro-batch for evaluation, used in pipeline parallelism. Derived from `global_batch_size_to_eval_on`.")
  checkpoint_dir: str | None = Field(None, description="The full path to the checkpoint directory, derived from `run_name`.")
  metrics_dir: str | None = Field(None, description="The full path to the metrics directory, derived from `run_name`.")
  tensorboard_dir: str | None = Field(None, description="The full path to the tensorboard directory, derived from `run_name`.")
  managed_mldiagnostics_dir: str | None = Field(None, description="The full path to the managed mldiagnostics directory, derived from `run_name`.")
  rampup_end_step: int | None = Field(None, description="The step at which the batch size ramp-up phase concludes.")
  tensors_on_device: list[str] | None = Field(None, description="List of tensors to keep on device memory for custom remat.")
  tensors_to_offload: list[str] | None = Field(None, description="List of tensors to offload to host memory for custom remat.")
  global_batch_size_to_load_start: int | None = Field(None, description="Starting global batch size for rampup.")
  global_batch_size_to_load_increment: int | None = Field(None, description="Increment for global batch size during rampup.")
  rampup_samples_per_increment_to_load: float | None = Field(None, description="Samples per increment for rampup.")
