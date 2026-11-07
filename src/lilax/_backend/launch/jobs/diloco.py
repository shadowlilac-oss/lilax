"""Submit the original DiLoCo pretraining workload to a GKE cluster."""

from datetime import datetime
import json
import shlex
import subprocess
import warnings


_XLA_FLAGS = """ \
  --xla_tpu_scoped_vmem_limit_kib=65536 \
--xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
--xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
--xla_tpu_enable_all_gather_offload_tracing=true \
--xla_tpu_use_tc_device_shape_on_sc=True \
--xla_sc_disable_megacore_partitioning=True \
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false \
--xla_enable_async_all_gather=true \
--xla_tpu_prefer_async_allgather_to_allreduce=true \
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true \
--xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
--xla_tpu_enable_concurrent_sparse_core_offloading=true \
--xla_tpu_aggressive_opt_barrier_removal=true \
--xla_tpu_enable_offloading_gather_to_sparsecore=true \
--xla_tpu_sparse_core_all_gather_latency_multiplier=1 \
--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3 \
--xla_tpu_enable_sparse_core_collective_aggregator=true \
--xla_tpu_enable_latency_hiding_layer_scheduler=true \
--xla_tpu_scheduler_percent_shared_memory_limit=150 \
--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true \
--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true \
--xla_tpu_pcie_bandwidth_multiplier=0.03 \
--xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true \
--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false \
  --xla_tpu_enable_3d_reduce_scatter_decomposer=false }"""


def submit_diloco_job(
    *, project_id: str, cluster_name: str, location: str, base_output_directory: str,
    dataset_path: str, docker_image: str, tpu_type: str, num_slices: int, model_name: str,
    workload_name: str | None = None, per_device_batch_size: float = 8, max_target_length: int = 2048,
    training_steps: int = 20, diloco_sync_period: int = 10, diloco_outer_lr: float = 0.1,
    diloco_outer_momentum: float = 0.9, xla_flags: str = _XLA_FLAGS,
    enable_streaming_diloco: bool = False, num_diloco_fragments: int = 37,
    use_sequential_layers: bool = False, num_communication_overlapping_steps: int = 2,
    communication_overlapping_alpha: float = 0.0, run_name: str | None = None,
    enable_checkpointing: bool = True, checkpoint_period: int = 100,
    save_checkpoint_on_completion: bool = True, async_checkpointing: bool = False,
    load_full_state_path: str = "", profiler: str = "xplane", skip_first_n_steps_for_profiler: int = 5,
    profiler_steps: int = 5, upload_all_profiler_results: bool = True,
    jax_distributed_initialization_timeout: int = 3600, priority: str = "medium", max_restarts: int = 0,
) -> tuple[str, int]:
  """Submit a GKE job and return its name and submission exit code.

  The image must contain Lilax and the dependencies for TPU pretraining.
  The GKE cluster and gcluster must already be configured as in the source script.
  """
  workload_name = workload_name or f"diloco-{datetime.now().strftime('%m%d-%H%M%S')}"
  if len(workload_name) > 28:
    raise ValueError(f"WORKLOAD_NAME '{workload_name}' exceeds the 28-character limit for gcluster workload names.")
  if num_slices < 2:
    warnings.warn("NUM_SLICES is less than 2. DiLoCo will not take effect.", stacklevel=2)
  if enable_streaming_diloco and xla_flags == _XLA_FLAGS:
    xla_flags = _XLA_FLAGS.replace(" }", " ")

  settings = {
      "run_name": workload_name, "save_config_to_gcs": True, "base_output_directory": base_output_directory,
      "dataset_type": "grain", "grain_file_type": "tfrecord", "dataset_path": dataset_path,
      "dataset_name": "c4/en:3.0.1", "eval_dataset_name": "c4/en:3.0.1", "model_name": model_name,
      "tokenizer_type": "huggingface", "tokenizer_path": "maxtext/assets/tokenizers/qwen3-tokenizer",
      "per_device_batch_size": per_device_batch_size, "max_target_length": max_target_length,
      "enable_diloco": True, "dcn_diloco_parallelism": num_slices, "diloco_sync_period": diloco_sync_period,
      "diloco_outer_lr": diloco_outer_lr, "diloco_outer_momentum": diloco_outer_momentum, "steps": training_steps,
  }
  if enable_streaming_diloco:
    settings.update({
        "run_name": run_name or workload_name, "enable_streaming_diloco": True,
        "num_diloco_fragments": num_diloco_fragments, "use_sequential_layers": use_sequential_layers,
        "num_communication_overlapping_steps": num_communication_overlapping_steps,
        "communication_overlapping_alpha": communication_overlapping_alpha,
        "enable_checkpointing": enable_checkpointing, "checkpoint_period": checkpoint_period,
        "save_checkpoint_on_completion": save_checkpoint_on_completion, "async_checkpointing": async_checkpointing,
        "load_full_state_path": load_full_state_path, "profiler": profiler,
        "skip_first_n_steps_for_profiler": skip_first_n_steps_for_profiler, "profiler_steps": profiler_steps,
        "upload_all_profiler_results": upload_all_profiler_results,
        "jax_distributed_initialization_timeout": jax_distributed_initialization_timeout,
    })
  payload = json.dumps(settings)
  python = (
      "import json; from lilax import MaxTextConfig; "
      "from lilax.trainers.pretraining.trainer import train_pretraining; "
      f"train_pretraining(MaxTextConfig(**json.loads({payload!r})))"
  )
  unset_xla_flags = "unset XLA_FLAGS && " if enable_streaming_diloco else ""
  command = f"{unset_xla_flags}LIBTPU_INIT_ARGS={shlex.quote(xla_flags)} python3 -c {shlex.quote(python)}"
  args = [
      "gcluster", "job", "submit", f"--cluster={cluster_name}", f"--project={project_id}",
      f"--location={location}", f"--priority={priority}", f"--restarts={max_restarts}", f"--compute-type={tpu_type}",
      f"--num-slices={num_slices}", f"--image={docker_image}", f"--name={workload_name}", f"--command={command}",
  ]
  return workload_name, subprocess.run(args, check=False).returncode


def submit_streaming_diloco_job(
    *, project_id: str, cluster_name: str, location: str, base_output_directory: str, docker_image: str,
    dataset_path: str = "gs://maxtext-dataset", tpu_type: str = "v5p-128", num_slices: int = 2,
    model_name: str = "qwen3-8b", run_name: str | None = None, workload_name: str | None = None,
    per_device_batch_size: float = 8, max_target_length: int = 2048, training_steps: int = 100,
    diloco_sync_period: int = 37, diloco_outer_lr: float = 0.1, diloco_outer_momentum: float = 0.9,
    num_diloco_fragments: int = 37, use_sequential_layers: bool = False,
    num_communication_overlapping_steps: int = 2, communication_overlapping_alpha: float = 0.0,
    enable_checkpointing: bool = True, checkpoint_period: int = 100, save_checkpoint_on_completion: bool = True,
    async_checkpointing: bool = False, load_full_state_path: str = "", profiler: str = "xplane",
    skip_first_n_steps_for_profiler: int = 5, profiler_steps: int = 5, upload_all_profiler_results: bool = True,
    jax_distributed_initialization_timeout: int = 3600, priority: str = "medium", max_restarts: int = 0,
    xla_flags: str = _XLA_FLAGS.replace(" }", " "),
) -> tuple[str, int]:
  """Submit the streaming DiLoCo workload using an existing Lilax image."""
  run_name = run_name or f"spmd-dlco-{datetime.now().strftime('%H%M')}"
  return submit_diloco_job(
      project_id=project_id, cluster_name=cluster_name, location=location, base_output_directory=base_output_directory,
      dataset_path=dataset_path, docker_image=docker_image, tpu_type=tpu_type, num_slices=num_slices,
      model_name=model_name, workload_name=workload_name or run_name, per_device_batch_size=per_device_batch_size,
      max_target_length=max_target_length, training_steps=training_steps, diloco_sync_period=diloco_sync_period,
      diloco_outer_lr=diloco_outer_lr, diloco_outer_momentum=diloco_outer_momentum, xla_flags=xla_flags,
      enable_streaming_diloco=True, num_diloco_fragments=num_diloco_fragments,
      use_sequential_layers=use_sequential_layers, num_communication_overlapping_steps=num_communication_overlapping_steps,
      communication_overlapping_alpha=communication_overlapping_alpha, run_name=run_name,
      enable_checkpointing=enable_checkpointing, checkpoint_period=checkpoint_period,
      save_checkpoint_on_completion=save_checkpoint_on_completion, async_checkpointing=async_checkpointing,
      load_full_state_path=load_full_state_path, profiler=profiler,
      skip_first_n_steps_for_profiler=skip_first_n_steps_for_profiler, profiler_steps=profiler_steps,
      upload_all_profiler_results=upload_all_profiler_results,
      jax_distributed_initialization_timeout=jax_distributed_initialization_timeout, priority=priority,
      max_restarts=max_restarts,
  )
