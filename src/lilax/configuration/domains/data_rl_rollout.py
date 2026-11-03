from typing import Any, Literal

from pydantic import BaseModel, Field


class VLLM(BaseModel):
  """vLLM-specific configuration for rollouts."""

  kv_cache_buffer: int = Field(256, gt=0, description="Buffer for KV cache.")
  hbm_utilization_vllm: float = Field(0.72, gt=0.0, le=1.0, description="Target HBM utilization for vLLM.")
  swap_space_vllm_gb: int = Field(2, ge=0, description="Swap space in GB for vLLM.")
  free_kv_cache_during_weight_sync: bool = Field(
      True, description="Maps to tunix RolloutConfig.rollout_vllm_free_kv_cache_during_weight_sync -> "
      "VllmConfig.free_kv_cache_during_weight_sync (ignored, with a log line, while the installed tunix "
      "predates it). The weight sync materializes a second copy of the sampler weights on HBM; freeing the KV "
      "cache first makes room for it. False keeps the pool (hbm_utilization_vllm of each chip) allocated across "
      "the sync and only resets its prefix-cache entries, saving the free + re-allocation (about 2 s per step "
      "on a colocated Qwen3-0.6B run at hbm_utilization_vllm=0.3). With False the sync runs out of HBM when "
      "that second copy does not fit next to the pool: check the headroom before disabling, colocated setups "
      "and a large hbm_utilization_vllm have the least."
  )
  enable_dp_attention: bool = Field(False, description="Enable the attn_dp mesh axis in vLLM.")
  enable_expert_parallel: bool = Field(False, description="Enable expert parallelism in vLLM.")
  async_scheduling: bool = Field(False, description="Enable asynchronous scheduling in vLLM.")
  max_num_batched_tokens: int | None = Field(None, description="Max number of batched tokens in vLLM.")
  max_num_seqs: int | None = Field(None, description="Max number of sequences in vLLM.")
  vllm_block_size: int | None = Field(None, gt=0, description="KV-cache block (page) size for vLLM. None lets the backend pick it from the engine shape.")
  stop_strings: list[str] | None = Field(None, description="List of stop strings for vLLM decoding.")
  vllm_additional_config: dict[str, Any] = Field(default_factory=dict, description="Additional vLLM config options.")
  vllm_hf_overrides: dict[str, Any] = Field(default_factory=dict, description="Overrides for HuggingFace model config for MaxText model.")
  vllm_hf_config_path: str = Field("", description="Path to HuggingFace model config for MaxText model.")
  use_standalone_converter: bool = Field(False, description="Use the standalone MaxText->torchax vLLM converter")
  use_weight_converter: bool = Field(True, description="Use an explicit weight converter for trainer->rollout weight sync instead of "
                                     "the legacy transfer_state_directly / transfer_state_with_mappings paths.")
  use_raiden_ffi: bool | None = Field(None, description="Use Raiden FFI transport for weight sync.")
  rollout_tensor_parallelism: int = Field(-1, description="Tensor parallelism per replica for rollout. If not specified, it will be auto-determined.")
  rollout_backend: Literal["maxtext", "vllm_torchax"] = Field(
      "maxtext", description="Rollout backend for trainer-side weight converter ('maxtext' or 'vllm_torchax')."
  )
  rollout_weight_dtype: str = Field("", description="Target weight dtype for trainer->rollout weight conversion. When empty, defaults to weight_dtype.")
  weight_sync_debug: bool = Field(
      False, description="Log the device placement of every operand during trainer->rollout weight "
      "conversion, and barrier between conversion steps so a device or sharding "
      "failure names the parameter that caused it. Serializes the sync; use only "
      "when debugging a weight-sync crash."
  )
  log_weight_sync_time: bool = Field(
      False, description="Log the wall time of every trainer->rollout weight sync, blocking on the "
      "rollout state so the number is execution rather than dispatch. Sync 0 is the "
      "initial load_checkpoint and pays XLA compilation; syncs 1+ reuse those "
      "executables, so the gap between them is how much of a sync is compilation "
      "rather than data movement. Adds one barrier per sync."
  )
  kv_tp_size: int = Field(1, ge=1, description="Degree of tensor parallelism for KV cache / attention heads in rollout.")
  moe_mlp_tp_size: int = Field(1, ge=1, description="Degree of tensor parallelism for MoE MLP dimension in rollout.")
  vllm_load_format: str = Field("dummy", description="Weight load format for vLLM in converter validation. Options:'auto', 'dummy'.")
  debug_converter: bool = Field(False, description="If True, run key coverage check, weight stats, and GCS upload.")
  gcs_debug_path: str = Field("", description="If set and debug_converter=True, upload converted layer-0 and global tensors as .npy files to GCS")


class RLCluster(BaseModel):
  """Cluster configurations specific to RL training."""

  trainer_devices_fraction: float = Field(0.5, gt=0.0, le=1.0, description="Fraction of devices to use for the trainer.")
  sampler_devices_fraction: float = Field(0.5, gt=0.0, le=1.0, description="Fraction of devices to use for the sampler.")
  chips_per_vm: int = Field(4, description="Number of accelerator chips per VM.")
  use_pathways: bool = Field(True, description="Whether to use Pathways for multihost orchestration.")
  num_trainer_slices: int = Field(-1, description="Number of slices for the trainer.")
  num_samplers_slices: int = Field(-1, description="Number of slices for the samplers.")
  rollout_data_parallelism: int = Field(-1, description="Total model replicas for rollout. It should only be specified when you would like to use more "
                                      "than one model replica in rollout.")
  rollout_tensor_parallelism: int = Field(-1, description="Tensor parallelism per replica for rollout. If not specified, it will be auto-determined.")
  rollout_expert_parallelism: int = Field(1, description="Expert parallelism per replica for rollout")
  inference_replicas: int = Field(1, description="Legacy experimental GRPO: number of inference (sampler) replicas.")
  inference_devices_per_replica: int = Field(4, description="Legacy experimental GRPO: devices per inference replica (single-controller device split).")
  inference_rollouts: int = Field(1, description="Legacy experimental GRPO: refresh rollouts every N steps (step % inference_rollouts).")
  use_pathways_reshard: bool = Field(True, description="Legacy experimental GRPO: use Pathways resharding to move policy params to the sampler.")
  gc_collect_after_weight_sync: bool = Field(
      True, description="Run a full host gc.collect() after every trainer->sampler weight sync "
      "(tunix ClusterConfig.gc_collect_after_weight_sync). Each sync leaves another copy of the weights on "
      "HBM until Python's garbage collector releases it; with this disabled the copies accumulate and the "
      "run can OOM over time. The collection costs about one second per step on a colocated setup, so "
      "disable it only when there is enough HBM headroom."
  )
