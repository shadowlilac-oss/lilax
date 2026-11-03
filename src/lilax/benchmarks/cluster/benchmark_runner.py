"""Run configured training benchmarks locally or on a Cluster Toolkit cluster."""

import json
import os
import shlex
import subprocess
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

from omegaconf import OmegaConf

from lilax._backend.benchmarks.catalog import AVAILABLE_MODELS, C4_MODELS
from lilax._backend.benchmarks.cluster_runner import (
    ClusterConfig, MaxTextModel, PathwaysConfig, WorkloadConfig, assign_workload_name, submit_workload,
    wait_for_workload_completion,
)
from lilax._backend.benchmarks.metrics_upload import BenchmarkRun, upload_benchmark_metrics
from lilax._backend.utils.globals import MAXTEXT_CONFIGS_DIR
from lilax.benchmarks.cluster._benchmark_runner.command import build_metrics_upload_command, build_training_command
from lilax.benchmarks.cluster._benchmark_runner.healthscan import healthscan_workload, select_model


def run_healthscan(device_type: str, base_output_directory: str, num_steps: int = 20) -> None:
  """Run the source healthscan model for a supported TPU device type."""
  workload = healthscan_workload(device_type, base_output_directory, num_steps, select_model, WorkloadConfig)
  run_on_device([workload])


def upload_workload_metrics(workload: WorkloadConfig):
  """Upload metrics from the Pathways controller or McJAX worker zero."""
  is_pathways = os.environ.get("JAX_PLATFORMS", "") == "proxy"
  is_worker_zero = int(os.environ.get("TPU_WORKER_ID") or os.environ.get("JOB_COMPLETION_INDEX") or -1) == 0
  if not (is_pathways or is_worker_zero):
    return None
  config_file = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")
  config = OmegaConf.load(config_file)
  tuning = workload.model.tuning_params
  chips = workload.num_devices_per_slice * workload.num_slices
  image = workload.pathways_config.runner_image if workload.pathways_config and workload.pathways_config.runner_image else workload.base_docker_image
  run = BenchmarkRun(
      metrics_gcs_file=os.path.join(workload.base_output_directory, workload.run_name, "metrics"),
      model_id=workload.model.model_type, hardware_id=workload.hardware_id, software_id="jax_maxtext",
      hardware_num_slices=workload.num_slices, number_of_chips=chips, container_image_name=image,
      global_batch_size=tuning.get("per_device_batch_size", config.per_device_batch_size) * chips,
      precision=tuning.get("matmul_precision", config.matmul_precision), optimizer=tuning.get("opt_type", config.opt_type),
      seq_length=tuning.get("max_target_length", config.max_target_length),
      number_of_steps=tuning.get("steps", workload.num_steps) if workload.num_steps == -1 else workload.num_steps,
      xla_flags=workload.model.xla_flags.strip().replace(" ", ","),
      dataset=tuning.get("dataset_type", config.dataset_type), run_type="maxtext-xpk", config_file=config_file,
      topology=workload.topology, tuning_params=tuning, db_project=workload.db_project, db_dataset=workload.db_dataset,
  )
  return upload_benchmark_metrics(run)


def run_on_device(workloads: list[WorkloadConfig]) -> None:
  """Run workloads sequentially using the native Python pretraining API."""
  for workload in workloads:
    if workload.run_name is None:
      workload.run_name = os.environ.get("M_RUN_NAME") or f"{workload.model.model_name}-{time.strftime('%Y%m%d')}"
    subprocess.run(build_training_command(workload), shell=True, text=True, check=True)


def run_cluster_benchmarks(
    cluster: ClusterConfig, workloads: list[WorkloadConfig], *, user: str | None = None,
    disruption_manager=None, wait: bool = False, timeout: str = "-1s",
) -> list[tuple[str, int]]:
  """Submit workloads and optionally wait for each submitted JobSet to complete."""
  results = []
  for workload in workloads:
    name = assign_workload_name(workload, user)
    if workload.disruption_configs and disruption_manager is not None:
      disruption_manager.add_workload(name, workload.disruption_configs)
    name, code = submit(cluster, workload, workload_name=name, user=user)
    if code and disruption_manager is not None:
      disruption_manager.remove_workload(name)
    if code == 0 and wait:
      code = wait_for_workload_completion(cluster, name, timeout)
    results.append((name, code))
  return results


def submit(cluster_config, workload_config, *, workload_name=None, user=None) -> tuple[str, int]:
  """Submit a recipe workload and return its name and submission status."""
  if workload_name is not None:
    workload_config.run_name = workload_name
  assign_workload_name(workload_config, user)
  headless = workload_config.pathways_config and workload_config.pathways_config.headless
  command = "" if headless else build_training_command(workload_config)
  if workload_config.generate_metrics_and_upload_to_big_query and not headless:
    command += f" && {build_metrics_upload_command(workload_config)}"
  return submit_workload(cluster_config, workload_config, command, user=user)


def wait(cluster_config, workload_names):
  """Yield workload names and completion statuses in completion order."""
  with ThreadPoolExecutor(max_workers=max(1, len(workload_names))) as executor:
    futures = {executor.submit(wait_for_workload_completion, cluster_config, name): name for name in workload_names}
    for future in as_completed(futures):
      yield futures[future], future.result()


def cancel(cluster_config, workload_name) -> int:
  """Cancel a Cluster Toolkit workload."""
  return subprocess.run([
      "gcluster", "job", "cancel", workload_name, f"--project={cluster_config.project}",
      f"--cluster={cluster_config.cluster_name}", f"--location={cluster_config.location or cluster_config.zone}",
  ], check=False).returncode
