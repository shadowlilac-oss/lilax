"""Aggregate benchmark metrics and persist a run summary in BigQuery."""

import dataclasses
import datetime
import fnmatch
import getpass
import json
import logging
import os
import pprint
import statistics
import uuid
from typing import Optional, get_args, get_origin

from google.api_core import exceptions
from google.cloud import bigquery, storage
from omegaconf import OmegaConf


logger = logging.getLogger(__name__)


HARDWARE_BF16_TFLOPS = {
    "v4": 275, "v5e": 197, "v5p": 459, "v6e": 918, "v6e-8": 918, "v6e-1": 918,
    "a3mega": 989, "a3ultra": 989,
}


@dataclasses.dataclass
class Metrics:
  avg_tflops_per_sec: float
  avg_tokens_per_sec: float
  median_step_time: float
  e2e_step_time: float


@dataclasses.dataclass
class BenchmarkRun:
  metrics_gcs_file: str
  model_id: str
  hardware_id: str
  software_id: str
  number_of_chips: int
  container_image_name: str
  global_batch_size: int
  precision: str
  optimizer: str
  seq_length: int
  number_of_steps: int
  xla_flags: str
  dataset: str
  run_type: str
  config_file: str
  topology: str
  tuning_params: dict
  db_project: str
  db_dataset: str
  hardware_num_slices: Optional[int] = None


@dataclasses.dataclass
class WorkloadBenchmarkV2:
  run_id: str
  model_id: str
  software_id: str
  hardware_id: str
  hardware_num_chips: int
  result_success: bool
  update_person_ldap: str
  configs_framework: Optional[str] = None
  configs_container_version: Optional[str] = None
  logs_artifact_directory: Optional[str] = None
  configs_env: Optional[str] = None
  hardware_num_nodes: Optional[int] = None
  storage_id: Optional[str] = None
  run_source: str = "manual"
  is_run_externally_visible: bool = False
  run_type: str = "perf_optimization"
  run_release_status: Optional[str] = "local"
  k8_jobset_yaml_file_path: Optional[str] = None
  benchmark_type: Optional[str] = None
  experiment_id: Optional[str] = None
  workload_gbs: Optional[int] = None
  workload_mbs: Optional[int] = None
  workload_precision: Optional[str] = None
  workload_optimizer: Optional[str] = None
  workload_others: Optional[str] = None
  workload_manager: Optional[str] = None
  workload_type: str = "training"
  workload_sequence_length: Optional[int] = None
  metrics_step_time: Optional[float] = None
  metrics_mfu: Optional[float] = None
  metrics_tokens_per_second: Optional[float] = None
  metrics_e2e_time: Optional[float] = None
  metrics_num_steps: Optional[int] = None
  metrics_other: Optional[str] = None
  metrics_tflops_per_second: Optional[float] = None
  hardware_num_superblocks: Optional[str] = None
  hardware_num_slices: Optional[int] = None
  hardware_topology: Optional[str] = None
  hardware_num_cores: Optional[int] = None
  result_error: Optional[str] = None
  hardware_nccl_driver_nickname: Optional[str] = None
  configs_xla_flags: Optional[str] = None
  configs_dataset: Optional[str] = None
  configs_reviewer: Optional[str] = None
  configs_other: Optional[str] = None
  logs_profile: Optional[str] = None
  logs_cloud_logs: Optional[str] = None
  logs_comments: Optional[str] = None
  logs_other: Optional[str] = None
  checkpointing_async: Optional[bool] = None
  checkpointing_interval_every_n_steps: Optional[int] = None
  checkpointing_size_in_gibs: Optional[float] = None
  checkpointing_individual_file_size: Optional[int] = None
  checkpointing_file_format: Optional[str] = None
  max_epochs: Optional[int] = None
  max_steps: Optional[int] = None
  training_dataset_samples: Optional[int] = None
  data_loader_num_workers: Optional[int] = None
  data_loader_prefetch_factor: Optional[int] = None
  training_dataset_file_format: Optional[str] = None
  start_time: Optional[datetime.datetime] = None
  end_time: Optional[datetime.datetime] = None
  gcs_metrics_bucket: Optional[str] = None
  gcsfuse_csi_driver: Optional[str] = None
  cloud_region: Optional[str] = None
  source_bucket: Optional[str] = None
  cluster_name: Optional[str] = None
  reviewer_ldap: str = ""
  update_timestamp: Optional[datetime.datetime] = datetime.datetime.now()


def parse_metrics(metrics_text: str, total_steps: int, last_n_steps: int = 10) -> Metrics:
  """Aggregate the last matching JSONL records for each source metric."""
  records = [json.loads(line) for line in metrics_text.splitlines()]

  def last_values(target, n):
    values = []
    for record in reversed(records):
      if target in record:
        values.append(record[target])
        if len(values) >= n:
          break
    return values

  tflops = last_values("perf/per_device_tflops_per_sec", last_n_steps)
  tokens = last_values("perf/per_device_tokens_per_sec", last_n_steps)
  step_times = last_values("perf/step_time_seconds", last_n_steps)
  all_step_times = last_values("perf/step_time_seconds", total_steps)
  return Metrics(sum(tflops) / len(tflops), sum(tokens) / len(tokens), statistics.median(step_times), sum(all_step_times))


def _schema():
  types = {str: "STRING", int: "INT64", float: "FLOAT64", bool: "BOOL", datetime.datetime: "TIMESTAMP"}
  fields = []
  for field in dataclasses.fields(WorkloadBenchmarkV2):
    optional = get_origin(field.type) is not None
    field_type = next(t for t in get_args(field.type) if t is not type(None)) if optional else field.type
    fields.append(bigquery.SchemaField(field.name, types[field_type], mode="NULLABLE" if optional else "REQUIRED"))
  return fields


def upload_benchmark_metrics(run: BenchmarkRun) -> Optional[Metrics]:
  """Upload the first matching Cloud Storage object for worker zero or Pathways.

  The original uploader selects the first matching local directory entry after
  downloading the files. Cloud Storage listing order may select a different file.
  """
  is_pathways = os.environ.get("JAX_PLATFORMS", "") == "proxy"
  is_worker_zero = int(os.environ.get("TPU_WORKER_ID") or os.environ.get("JOB_COMPLETION_INDEX") or -1) == 0
  if not (is_pathways or is_worker_zero):
    return None

  bucket_name, prefix = run.metrics_gcs_file.removeprefix("gs://").split("/", 1)
  prefix = prefix.rstrip("/") + "/"
  blobs = storage.Client().list_blobs(bucket_name, prefix=prefix)
  blob = next(blob for blob in blobs if "/" not in blob.name[len(prefix):] and fnmatch.fnmatch(
      blob.name[len(prefix):], "metrics_step*.txt"
  ))
  steps = int(run.number_of_steps)
  metrics = parse_metrics(blob.download_as_text(), steps, steps - 10 if steps - 10 > 0 else steps)
  chips = int(run.number_of_chips)
  nodes = chips // 4 if run.hardware_id.startswith("v") else chips // 8 if run.hardware_id in ("a3mega", "a3ultra") else chips

  config = OmegaConf.load(run.config_file)
  for key, value in run.tuning_params.items():
    OmegaConf.update(config, key, value, merge=True)
  framework_config = json.dumps({"config": OmegaConf.to_container(config, resolve=True)})
  summary = WorkloadBenchmarkV2(
      run_id=f"run-{uuid.uuid4()}", model_id=run.model_id, software_id=run.software_id,
      hardware_id=run.hardware_id, hardware_num_chips=chips, hardware_num_nodes=nodes,
      hardware_num_slices=run.hardware_num_slices, result_success=True, configs_framework=framework_config,
      configs_env=json.dumps({"env_vars": dict(os.environ)}), configs_container_version=run.container_image_name,
      configs_xla_flags=run.xla_flags.replace(",", " "), configs_dataset=run.dataset, logs_artifact_directory="",
      update_person_ldap=getpass.getuser(), run_source="automation", run_type=run.run_type,
      run_release_status="local", workload_precision=run.precision, workload_gbs=int(run.global_batch_size),
      workload_optimizer=run.optimizer, workload_sequence_length=int(run.seq_length),
      metrics_e2e_time=metrics.e2e_step_time, metrics_mfu=metrics.avg_tflops_per_sec / HARDWARE_BF16_TFLOPS[run.hardware_id],
      metrics_step_time=metrics.median_step_time, metrics_tokens_per_second=metrics.avg_tokens_per_sec,
      metrics_num_steps=steps, metrics_other="", hardware_nccl_driver_nickname=None,
      hardware_topology=run.topology, hardware_num_superblocks=0, logs_comments="",
  )
  client = bigquery.Client(project=run.db_project)
  table_id = f"{run.db_project}.{run.db_dataset}.run_summary"
  schema = _schema()
  try:
    table = client.get_table(table_id)
  except exceptions.NotFound:
    table = client.create_table(bigquery.Table(table_id, schema=schema))
  table_fields = {field.name: field for field in table.schema}
  schema_fields = {field.name: field for field in schema}
  for name in schema_fields:
    if name not in table_fields:
      raise ValueError(f"Column: {name} is absent in table whereas it's present in dataclass.")
  for name, field in table_fields.items():
    if name not in schema_fields and field.mode != "NULLABLE":
      raise ValueError(f"Column: {name} is absent in dataclass whereas it's present in table & is of Required type.")
  serialized_rows = [dataclasses.asdict(summary)]
  try:
    logger.info("Writing to BigQuery: %d rows", len(serialized_rows))
    errors = client.insert_rows(table=table, rows=serialized_rows)
    if errors:
      logger.error("There were errors while writing to Bigquery:\n%s", pprint.pformat(errors))
    else:
      logger.info("Successfully wrote to BigQuery")
  except exceptions.GoogleAPICallError as error:
    logger.exception("Failed to write to BigQuery with error %s", error)
  return metrics
