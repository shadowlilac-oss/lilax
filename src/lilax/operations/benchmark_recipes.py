"""Python workflows for Cluster Toolkit benchmark workloads.

The cluster runner supplies submission, completion, and cancellation. Workload
and Pathways configurations are the runner's dataclasses.
"""

from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime
import logging
from pathlib import Path

from google.api_core.exceptions import Conflict, Forbidden, NotFound, PermissionDenied
from google.cloud import storage
from omegaconf import OmegaConf

from lilax._backend.benchmarks.catalog import AVAILABLE_MODELS
from lilax._backend.benchmarks.cluster_runner import MaxTextModel, PathwaysConfig, WorkloadConfig
from lilax._backend.benchmarks.recipes.user_configs import BenchmarkRecipeConfig, build_user_models
from lilax.operations import benchmark_disruption, benchmark_metrics, benchmark_runner


def _validate_bigquery(config: BenchmarkRecipeConfig) -> bool:
  if config.bq_enable and (not config.bq_db_project or not config.bq_db_dataset):
    logging.error("Validation FAILED: BigQuery is enabled, but 'bq_db_project' or 'bq_db_dataset' is missing.")
    return False
  return True


def run_benchmarks(cluster_config, workload_configs, *, user=None) -> int:
  """Submit benchmark workloads without waiting for their completion."""
  status = 0
  for workload in workload_configs:
    _, code = benchmark_runner.submit(cluster_config, workload, workload_name=workload.run_name, user=user)
    if code != 0:
      status = 1
  return status


def run_headless(cluster_config, workload_config, *, workload_name, user=None) -> int:
  """Start a two-slice Pathways server and proxy without a training command."""
  workload = replace(
      workload_config,
      model=None,
      num_slices=2,
      max_restarts=0,
      base_docker_image=None,
      libtpu_type=None,
      libtpu_nightly_version="",
      pathways_config=replace(workload_config.pathways_config, headless=True),
  )
  _, status = benchmark_runner.submit(cluster_config, workload, workload_name=workload_name, user=user)
  return status


def run_long_running(cluster_config, workload_config, *, project, region, user=None) -> int:
  """Submit the original long-running Pathways or McJAX benchmark."""
  tuning = dict(workload_config.model.tuning_params)
  tuning.update(use_vertex_tensorboard=True, vertex_tensorboard_project=project, vertex_tensorboard_region=region)
  if workload_config.pathways_config is not None:
    tuning["profiler"] = "xplane"
  model = replace(workload_config.model, tuning_params=tuning)
  workload = replace(workload_config, model=model, num_slices=2, max_restarts=10_000, num_steps=10_000_000,
                     priority="medium")
  _, status = benchmark_runner.submit(cluster_config, workload, workload_name=workload.run_name, user=user)
  return status


def run_checkpoint_benchmarks(
    cluster_config, workload_configs, *, test_restore=False, restore_steps=20, user=None
) -> int:
  """Submit initial runs and optionally relaunch successful runs for restoration.

  Initial completion is observed in completion order. Restore jobs are submitted
  after their initial run succeeds, and are not waited on, matching the recipe.
  """
  submitted = {}
  for workload in workload_configs:
    name, code = benchmark_runner.submit(cluster_config, workload, workload_name=workload.run_name, user=user)
    if code == 0:
      submitted[name] = workload

  status = 0
  for name, code in benchmark_runner.wait(cluster_config, list(submitted)):
    if code != 0:
      status = 1
      continue
    if test_restore:
      cancel_code = benchmark_runner.cancel(cluster_config, name)
      if cancel_code != 0:
        print(f"Warning: Deleting completed workload '{name}' returned {cancel_code}.")
      workload = submitted[name]
      restored = replace(workload, num_steps=workload.num_steps + restore_steps)
      _, code = benchmark_runner.submit(cluster_config, restored, workload_name=name, user=user)
      if code != 0:
        status = 1
  return status


def checkpoint_pathways_config(*, server_image, proxy_server_image, runner_image):
  """Build the source checkpoint recipe's Pathways configuration."""
  return PathwaysConfig(
      server_image=server_image, proxy_server_image=proxy_server_image, runner_image=runner_image,
      server_flags="--enable_metrics_collection=true", proxy_flags="--enable_metrics_collection=true",
      worker_flags="--enable_metrics_collection=true",
  )


def checkpoint_workloads(workload_type, cluster_config, pathways_config, models=None, *, base_output_directory,
                         runner_image, resume_checkpoints=None, now=None):
  """Build the initial checkpoint workloads using the source recipe's limits."""
  if models is None:
    model = AVAILABLE_MODELS["v6e"]["llama3_1_70b_8192_synth"]
    models = {
        "pathways": {"models": [model], "num_slices_list": [2]},
        "mcjax": {"models": [model], "num_slices_list": [2]},
    }
  checkpoint_tuning = {
      "enable_checkpointing": True,
      "async_checkpointing": True,
      "checkpoint_period": 20,
      "enable_checkpoint_cloud_logger": True,
      "enable_goodput_recording": True,
  }
  timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
  resumes = resume_checkpoints or {}
  workloads = []
  for framework, selection in models.items():
    for model in selection["models"]:
      if model.pathways_tuning_params is None:
        model = replace(model, pathways_tuning_params=checkpoint_tuning.copy())
      for num_slices in selection["num_slices_list"]:
        resume = resumes.get(framework, {}).get(num_slices)
        output = (f"{base_output_directory}{timestamp}_{framework}_{num_slices}_slice_"
                  f"{cluster_config.device_type}_{model.model_name}/")
        workloads.append(workload_type(
            model=model,
            num_slices=num_slices, device_type=cluster_config.device_type,
            base_output_directory=resume["base_output_directory"] if resume else output,
            max_restarts=100, libtpu_type=None, libtpu_nightly_version="",
            base_docker_image=runner_image if framework == "mcjax" else None,
            pathways_config=pathways_config if framework == "pathways" else None,
            num_steps=resume["num_steps"] if resume else 41, priority="medium",
            run_name=resume["run_name"] if resume else None,
        ))
  return workloads


def standard_workloads(config: BenchmarkRecipeConfig, *, disruption_method=None, disruptions=None) -> list[WorkloadConfig]:
  """Build workloads for every selected framework, model, and slice count."""
  workloads = []
  for framework, models in config.models.items():
    for model in models:
      for num_slices in config.num_slices_list:
        workloads.append(WorkloadConfig(
            model=model,
            num_slices=num_slices, device_type=config.cluster_config.device_type,
            base_output_directory=(
                f"{config.base_output_directory}{framework}_{num_slices}_slice_"
                f"{config.device_type}_{model.model_name}/"
            ),
            max_restarts=config.max_restarts, libtpu_type=None, libtpu_nightly_version="",
            base_docker_image=config.runner if framework == "mcjax" else None,
            pathways_config=config.pathways_config if framework == "pathways" else None,
            run_name=config.workload_id, num_steps=config.benchmark_steps,
            priority=config.priority, skip_validation=config.skip_validation,
            generate_metrics_and_upload_to_big_query=config.bq_enable,
            db_project=config.bq_db_project, db_dataset=config.bq_db_dataset,
            disruption_configs=(
                benchmark_disruption.construct_disruption_configs(framework, disruption_method, disruptions)
                if disruptions else None
            ),
        ))
  return workloads


def upload_workload_metrics(workload: WorkloadConfig, *, db_project: str, db_dataset: str):
  """Upload a completed workload's metrics from its training process environment."""
  config_file = str(Path(__file__).resolve().parents[1] / "_backend" / "configs" / "base.yml")
  config = OmegaConf.load(config_file)
  tuning = workload.model.tuning_params
  batch_size = tuning.get("per_device_batch_size", config.per_device_batch_size)
  steps = tuning.get("steps", workload.num_steps) if workload.num_steps == -1 else workload.num_steps
  image = workload.pathways_config.runner_image if workload.pathways_config and workload.pathways_config.runner_image else workload.base_docker_image
  run = benchmark_metrics.BenchmarkRun(
      metrics_gcs_file=f"{workload.base_output_directory.rstrip('/')}/{workload.run_name}/metrics",
      model_id=workload.model.model_type, hardware_id=workload.hardware_id, software_id="jax_maxtext",
      hardware_num_slices=workload.num_slices,
      number_of_chips=workload.num_devices_per_slice * workload.num_slices,
      container_image_name=image,
      global_batch_size=batch_size * workload.num_devices_per_slice * workload.num_slices,
      precision=tuning.get("matmul_precision", config.matmul_precision),
      optimizer=tuning.get("opt_type", config.opt_type),
      seq_length=tuning.get("max_target_length", config.max_target_length),
      number_of_steps=steps, xla_flags=workload.model.xla_flags.strip().replace(" ", ","),
      dataset=tuning.get("dataset_type", config.dataset_type), run_type="maxtext-xpk",
      config_file=config_file, topology=workload.topology, tuning_params=tuning,
      db_project=db_project, db_dataset=db_dataset,
  )
  return benchmark_metrics.upload_benchmark_metrics(run)


def run_elastic_training(config: BenchmarkRecipeConfig, disruption_manager=None, *, compare_with_mcjax=True) -> int:
  """Run the source SIGILL disruption scenario using a configured Kubernetes client."""
  frameworks = ["pathways", "mcjax"] if compare_with_mcjax else ["pathways"]
  config = replace(config, max_restarts=10, headless=False, selected_model_framework=frameworks)
  if not _validate_bigquery(config):
    return 1
  workloads = standard_workloads(
      config, disruption_method=benchmark_disruption.DisruptionMethod.SIGILL,
      disruptions={"time_seconds": [120, 600]},
  )
  if workloads:
    manager_context = (nullcontext(disruption_manager) if disruption_manager is not None
                       else benchmark_disruption.cluster_disruption_manager(config.cluster_config))
    with manager_context as manager:
      benchmark_runner.run_cluster_benchmarks(
          config.cluster_config, workloads, user=config.user, disruption_manager=manager,
      )
      manager.start_disruptions_and_wait_for_completion()
  return 0


def run_suspend_resume(config: BenchmarkRecipeConfig, disruption_manager=None) -> int:
  """Run the source SIGTERM disruption scenario at training step three."""
  config = replace(config, max_restarts=3, headless=False)
  if not _validate_bigquery(config):
    return 1
  workloads = standard_workloads(
      config, disruption_method=benchmark_disruption.DisruptionMethod.SIGTERM, disruptions={"step": [3]},
  )
  if workloads:
    manager_context = (nullcontext(disruption_manager) if disruption_manager is not None
                       else benchmark_disruption.cluster_disruption_manager(config.cluster_config))
    with manager_context as manager:
      benchmark_runner.run_cluster_benchmarks(
          config.cluster_config, workloads, user=config.user, disruption_manager=manager,
      )
      manager.start_disruptions_and_wait_for_completion()
  return 0


def check_and_create_bucket(storage_client, bucket_name, region):
  """Get the output bucket or create it in the requested region."""
  print(f"Checking GCS bucket: {bucket_name}...")
  try:
    bucket = storage_client.get_bucket(bucket_name)
    print(f"GCS bucket '{bucket_name}' already exists. No creation needed.")
    return bucket
  except NotFound:
    print(f"GCS bucket '{bucket_name}' not found. Attempting to create...")
    try:
      bucket = storage_client.create_bucket(bucket_or_name=bucket_name, location=region)
      print(f"Successfully created GCS bucket: '{bucket.name}' in region: {bucket.location}.")
      return bucket
    except (Conflict, Forbidden, PermissionDenied) as error:
      print(f"Failed to create GCS bucket due to {type(error).__name__}! Error message: {error}")
      return None
  except (Forbidden, PermissionDenied) as error:
    print(f"Failed to access GCS bucket due to {type(error).__name__}! Check your credentials. Error message: {error}")
    return None


def run_standard_benchmark(config: BenchmarkRecipeConfig) -> int:
  """Create or access the output bucket and submit the selected workloads."""
  client = storage.Client(project=config.project)
  bucket_name = config.base_output_directory.removeprefix("gs://").split("/")[0]
  if not check_and_create_bucket(client, bucket_name, config.region):
    print("Exiting due to failure in GCS bucket creation or access.")
    return 1
  if not _validate_bigquery(config):
    return 1
  for workload in standard_workloads(config):
    benchmark_runner.submit(config.cluster_config, workload, workload_name=workload.run_name, user=config.user)
  return 0


def run_remote_python_sidecar(cluster_config, *, user: str) -> int:
  """Submit the original one-slice Pathways remote Python sidecar workload."""
  location = cluster_config.location or cluster_config.zone
  parts = location.split("-")
  region = "-".join(parts[:-1]) if len(parts) >= 3 else location
  image_prefix = f"us-docker.pkg.dev/cloud-tpu-v2-images/pathways/gke/{user}"
  pathways = PathwaysConfig(
      server_image=f"{image_prefix}/server:latest", proxy_server_image=f"{image_prefix}/proxy_server:latest",
      runner_image=f"gcr.io/{cluster_config.project}/{user}_latest:latest",
      colocated_python_sidecar_image=f"gcr.io/{cluster_config.project}/{user}/colocated_python_sidecar_latest:latest",
  )
  model = MaxTextModel(
      model_name="default-basic-1", model_type="default", xla_flags="",
      tuning_params={
          "per_device_batch_size": 1, "remat_policy": "full", "global_parameter_scale": 1,
          "attention": "flash", "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic",
          "reuse_example_batch": 1, "enable_checkpointing": False, "profiler": "xplane",
      },
  )
  workload = WorkloadConfig(
      model=model,
      num_slices=1, device_type=cluster_config.device_type, base_output_directory=f"gs://{user}-{region}/{user}/",
      max_restarts=0, libtpu_type=None, libtpu_nightly_version="", base_docker_image=None,
      pathways_config=pathways, num_steps=1_000_000,
  )
  _, status = benchmark_runner.submit(cluster_config, workload, user=user)
  return status
