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


_PATHWAYS_TUNING = {
    "checkpoint_storage_use_ocdbt": False,
    "checkpoint_storage_use_zarr3": False,
    "enable_pathways_goodput": True,
    "enable_goodput_recording": True,
    "enable_single_controller": True,
    "metrics_file": "metrics.txt",
    "goodput_upload_interval_seconds": 30,
}


def select_model(name: str) -> MaxTextModel:
  """Select a named benchmark model using the original catalog precedence."""
  for device_type in ("v6e", "v5litepod", "v5p"):
    if name in AVAILABLE_MODELS[device_type]:
      return deepcopy(AVAILABLE_MODELS[device_type][name])
  if name in C4_MODELS:
    return deepcopy(C4_MODELS[name])
  raise KeyError(name)


def run_healthscan(device_type: str, base_output_directory: str, num_steps: int = 20) -> None:
  """Run the source healthscan model for a supported TPU device type."""
  models = {
      "v5p-128": "llama2_70b_v5p_128", "v5p-256": "llama4_scout_dropless_v5p_256",
      "v5p-512": "deepseek_v3_ep_256_v5p_512", "v5p-1024": "deepseek_v3_ep_256_v5p_512",
      "v5p-2048": "deepseek_v3_ep_256_v5p_512", "v6e-8": "llama2_7b_4096",
      "v6e-16": "llama2_7b_4096", "v6e-32": "llama2_7b_4096", "v6e-64": "llama2_7b_4096",
      "v6e-128": "llama2_70b_4096", "v6e-256": "llama2_70b_4096",
  }
  model = select_model(models[device_type])
  model.tuning_params["gcs_metrics"] = False
  workload = WorkloadConfig(
      model=model, base_output_directory=base_output_directory, device_type=device_type, num_steps=num_steps,
      run_name=f"{time.strftime('%Y%m%d')}-health-test",
  )
  run_on_device([workload])


def build_training_command(workload: WorkloadConfig) -> str:
  """Build a command that invokes the Python training API in the workload image."""
  settings = workload.model.tuning_params.copy()
  pathways = workload.pathways_config is not None
  if pathways:
    settings.update(_PATHWAYS_TUNING)
    settings.update(workload.model.pathways_tuning_params or {})
  if workload.generate_metrics_and_upload_to_big_query:
    settings["gcs_metrics"] = True
  settings.update(model_name=workload.model.model_type, steps=workload.num_steps,
                  base_output_directory=workload.base_output_directory)
  if workload.run_name is not None:
    settings["run_name"] = workload.run_name
  if not pathways:
    settings.update(use_vertex_tensorboard=False, vertex_tensorboard_project="", vertex_tensorboard_region="")
  payload = json.dumps(settings)
  python = (
      "import json; from lilax import MaxTextConfig; "
      "from lilax.trainers.pre_training import train_pretraining; "
      f"train_pretraining(MaxTextConfig(**json.loads({payload!r})))"
  )
  commands = []
  if not pathways and workload.libtpu_type == "nightly-libtpu":
    version = f"=={workload.libtpu_nightly_version}" if workload.libtpu_nightly_version else " --pre"
    commands.append(f"python3 -m pip install libtpu{version} -f https://storage.googleapis.com/libtpu-wheels/index.html")
  elif not pathways and workload.libtpu_type == "custom":
    commands.append("mv libtpu.so /lib/")
  if not pathways:
    commands.append(f"export LIBTPU_INIT_ARGS={shlex.quote(workload.model.xla_flags)}")
    commands.append("export TPU_WORKER_ID=${TPU_WORKER_ID:-${JOB_COMPLETION_INDEX:--1}}")
  commands.extend(["export ENABLE_PATHWAYS_PERSISTENCE=1", f"export JAX_PLATFORMS={'proxy' if pathways else 'tpu,cpu'}",
                   "export ENABLE_PJRT_COMPATIBILITY=true"])
  if workload.hlo_dump:
    commands.append("export XLA_FLAGS='--xla_dump_large_constants --xla_dump_to=/tmp/xla_dump'")
  commands.append(f"python3 -c {shlex.quote(python)}")
  if workload.hlo_dump:
    destination = os.path.join(workload.base_output_directory, workload.run_name, "hlo_dump")
    commands.append(f"gcloud storage cp -r /tmp/xla_dump {shlex.quote(destination)}")
  return " && ".join(commands)


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
    payload = json.dumps({
        "model": asdict(workload_config.model), "base_output_directory": workload_config.base_output_directory,
        "device_type": workload_config.device_type, "num_slices": workload_config.num_slices,
        "base_docker_image": workload_config.base_docker_image, "num_steps": workload_config.num_steps,
        "run_name": workload_config.run_name, "compute_type": workload_config.compute_type,
        "topology": workload_config.topology, "pathways_config": (
            asdict(workload_config.pathways_config) if workload_config.pathways_config else None
        ), "db_project": workload_config.db_project, "db_dataset": workload_config.db_dataset,
    })
    python = (
        "import json; from lilax.operations.benchmark_runner import "
        "MaxTextModel, PathwaysConfig, WorkloadConfig, upload_workload_metrics; "
        f"data = json.loads({payload!r}); data['model'] = MaxTextModel(**data['model']); "
        "data['pathways_config'] = PathwaysConfig(**data['pathways_config']) if data['pathways_config'] else None; "
        "upload_workload_metrics(WorkloadConfig(**data))"
    )
    command += f" && python3 -c {shlex.quote(python)}"
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
