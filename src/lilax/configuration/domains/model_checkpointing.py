"""Checkpointing configuration."""

from typing import Literal

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt


class Checkpointing(BaseModel):
  """Core configuration for checkpointing and run restoration."""

  convert_checkpoint_if_possible: bool = Field(False, description="Whether to convert checkpoint on the fly if not provided via\
        load_parameters_path or base_output_directory")
  load_parameters_path: str = Field("", description="Loads only model parameters from a specific checkpoint path.")
  lora_input_adapters_path: str = Field("", description="Input GCS path for LoRA adapters.")
  hf_lora_adapter_path: str = Field("", description="HuggingFace LoRA adapter repo ID (e.g., 'username/adapter-repo') or local "
                                    "path to directory containing adapter_model.safetensors.")
  load_full_state_path: str = Field("", description="Loads the complete training state from a checkpoint path.")
  enable_checkpointing: bool = Field(True, description="If True, enables saving checkpoints during training.")
  load_checkpoint_only_once: bool = Field(False, description="If True, deep copy the reference model to the actor model.")
  async_checkpointing: bool = Field(True, description="If True, uses an asynchronous checkpointer for performance.")
  checkpoint_period: int = Field(10_000, description="The frequency (in steps) at which to save checkpoints.")
  max_num_checkpoints_to_keep: int | None = Field(None, description="Maximum number of checkpoints to keep.")
  enable_single_replica_ckpt_restoring: bool = Field(False, description="One replica reads and broadcasts the checkpoint.")
  checkpoint_todelete_subdir: str | None = Field(None, description="Subdirectory to move checkpoints to before deletion. (Ignored if directory is prefixed with gs://)")
  checkpoint_todelete_full_path: str | None = Field(None, description="Full path to move checkpoints to before deletion.")
  standalone_checkpointer_per_step_interval: float = Field(0.0, description="Interval in seconds between iterations in standalone checkpointer benchmark loop.")
  standalone_checkpointer_drop_page_cache_before_restore: bool = Field(
      False, description="Whether to execute sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' before restoring a checkpoint in"
      " standalone_checkpointer loop (use for storage benchmarking only).")
  standalone_checkpointer_enable_restore_in_loop: bool = Field(
      True, description="In standalone_checkpointer loop, whether to restore checkpoint after saving in each step (defaults to True"
      " for bidirectional storage read/write benchmarking).")
  standalone_checkpointer_start_from_checkpoint: bool = Field(
      False, description="In standalone_checkpointer, whether to start by attempting to load an existing checkpoint before setting"
      " up training state (for checkpoint restore benchmarking).")
  force_unroll: bool = Field(False, description="During param-only checkpoint generation, whether to unroll the loop.")
  checkpoint_is_quantized: bool = Field(False, description="Set to True if reading from a saved AQT quantized checkpoint.")
  save_quantized_params_path: str = Field("", description="Path to save params quantized on the fly.")
  enable_orbax_v1: bool = Field(False, description="DEPRECATED: Orbax v1 is always used for checkpointing; this flag is ignored.")
  checkpoint_conversion_fn: str | None = Field(None, description="Function for processing loaded checkpoint dict.")
  source_checkpoint_layout: Literal["orbax", "safetensors", "safetensors_dynamic"] = Field("orbax", description="The layout of the source checkpoint to load.")
  save_checkpoint_on_start: bool = Field(True, description="If True, saves an initial checkpoint upon training start.")
  save_checkpoint_on_completion: bool = Field(True, description="If True, saves a final checkpoint upon training completion.")
  enable_continuous_checkpointing: bool = Field(False, description="If True, enables continuous checkpointing.")
  colocated_python_checkpointing: bool = Field(False, description="If True, enables checkpointing from remote TPU VMs instead of head node on pathways.")
  enable_autocheckpoint: bool = Field(False, description="If True, enables autocheckpoint or preemption induced checkpointing.")


class OrbaxStorage(BaseModel):
  """Configuration for Orbax checkpoint storage options."""

  checkpoint_storage_target_data_file_size_bytes: int = Field(2147483648, description="Target file size for chunking large arrays in Orbax.")
  checkpoint_storage_use_ocdbt: bool = Field(True, description="Whether to use the OCDbT storage format for checkpoints.")
  checkpoint_storage_use_zarr3: bool = Field(True, description="Whether to use Zarr3 with OCDbT. Requires use_ocdbt=True.")
  checkpoint_storage_concurrent_gb: int = Field(96, description="Concurrent GB for I/O operations during checkpointing.")
  checkpoint_storage_device_host_concurrent_gb: int | None = Field(8, description="Concurrent GB for device->host staging during checkpoint save. None = unbounded.")


class EmergencyCheckpointing(BaseModel):
  """Configuration for emergency (local) checkpointing."""

  enable_multi_tier_checkpointing: bool = Field(False, description="Enables multi-tier checkpointing (local and persistent).")
  local_checkpoint_directory: str = Field("", description="Local directory for emergency checkpoints.")
  local_checkpoint_period: NonNegativeInt = Field(0, description="Frequency (in steps) for local emergency checkpoints.")
  multi_tier_checkpointing_backup_interval_minutes: PositiveInt | None = Field(None, description="Interval in minutes to back up local checkpoints to persistent storage.")
  multi_tier_checkpointing_backup_interval_steps: PositiveInt | None = Field(None, description="Interval in steps to back up local checkpoints to persistent storage.")
  mtc_data_parallelism: int = Field(0, description="Number of identical pipelines in the job for multi-tier checkpointing. 0 defaults to num_slices.")
  enable_emergency_checkpoint: bool = Field(False, description="Legacy flag for enabling emergency checkpointing. Prefer `enable_multi_tier_checkpointing`.")
  use_replicator_service: bool = Field(False, description="Whether to use emergency checkpointing with the replicator service.")
  replicator_backup_interval_minutes: NonNegativeInt = Field(0, description="Interval in minutes to back up local checkpoints.")
