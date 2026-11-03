"""Construct training and metrics-upload commands for benchmark workloads."""

import json
import os
import shlex
from dataclasses import asdict

from lilax._backend.benchmarks.cluster_runner import WorkloadConfig


_PATHWAYS_TUNING = {
    "checkpoint_storage_use_ocdbt": False,
    "checkpoint_storage_use_zarr3": False,
    "enable_pathways_goodput": True,
    "enable_goodput_recording": True,
    "enable_single_controller": True,
    "metrics_file": "metrics.txt",
    "goodput_upload_interval_seconds": 30,
}


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


def build_metrics_upload_command(workload_config: WorkloadConfig) -> str:
  """Build the post-training metrics upload command."""
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
  return f"python3 -c {shlex.quote(python)}"
