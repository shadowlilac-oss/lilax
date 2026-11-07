"""Submit benchmark workloads to Cluster Toolkit and wait for their JobSets."""

from dataclasses import dataclass, field
import os
import random
import string
import subprocess
import time


_V5E_V6E_TOPOLOGIES = {1: "1x1", 4: "2x2", 8: "2x4", 16: "4x4", 32: "4x8", 64: "8x8", 128: "8x16", 256: "16x16"}
_V5P_TOPOLOGIES = {8: "2x2x1", 16: "2x2x2", 32: "2x2x4", 64: "2x4x4", 128: "4x4x4", 256: "4x4x8", 512: "4x8x8", 1024: "8x8x8", 2048: "8x8x16", 4096: "8x16x16"}


@dataclass(kw_only=True)
class ClusterConfig:
  cluster_name: str
  project: str
  device_type: str
  zone: str = ""
  location: str | None = None
  compute_type: str | None = None
  topology: str | None = None


@dataclass(kw_only=True)
class PathwaysConfig:
  server_image: str | None = None
  proxy_server_image: str | None = None
  runner_image: str | None = None
  colocated_python_sidecar_image: str | None = None
  server_flags: str = ""
  proxy_flags: str = ""
  worker_flags: str = ""
  headless: bool = False


@dataclass
class MaxTextModel:
  model_name: str
  model_type: str
  tuning_params: dict
  xla_flags: str
  pathways_tuning_params: dict | None = None
  pathways_xla_flag_options: dict | None = None


@dataclass(kw_only=True)
class WorkloadConfig:
  model: MaxTextModel | None
  base_output_directory: str
  device_type: str | None = None
  num_slices: int | None = None
  base_docker_image: str | None = None
  num_steps: int = 20
  max_restarts: int = 0
  priority: str = "medium"
  run_name: str | None = None
  compute_type: str | None = None
  topology: str | None = None
  mounts: list[str] = field(default_factory=list)
  pathways_config: PathwaysConfig | None = None
  hlo_dump: bool = False
  skip_validation: bool = False
  libtpu_type: str = "maxtext-docker"
  libtpu_nightly_version: str = ""
  generate_metrics_and_upload_to_big_query: bool = False
  db_project: str = ""
  db_dataset: str = ""
  disruption_configs: list | None = None
  num_devices_per_slice: int | None = field(init=False, default=None)
  hardware_id: str = field(init=False, default="v6e")

  def __post_init__(self):
    if self.device_type is None:
      if self.generate_metrics_and_upload_to_big_query:
        raise ValueError(
            "device_type is None and generate_metrics_and_upload_to_big_query is enabled. "
            "Device_type is required for uploading run results to BigQuery"
        )
      return
    size = int(self.device_type.split("-")[-1])
    if self.device_type.startswith(("v6e", "v5e", "v5litepod")):
      if size not in _V5E_V6E_TOPOLOGIES and not self.topology:
        raise ValueError(f"Unsupported v5e or v6e size: {size}")
      self.num_devices_per_slice = size
      self.topology = self.topology or _V5E_V6E_TOPOLOGIES[size]
      prefix = "v6e" if self.device_type.startswith("v6e") else "v5litepod"
      self.compute_type = self.compute_type or f"{prefix}-{size if size in (1, 8) else 4}"
    elif self.device_type.startswith("v5p"):
      if size not in _V5P_TOPOLOGIES and not self.topology:
        raise ValueError(f"Unsupported v5p size: {size}")
      self.num_devices_per_slice = size // 2
      self.topology = self.topology or _V5P_TOPOLOGIES[size]
      self.compute_type = self.compute_type or "v5p-8"
    else:
      self.num_devices_per_slice = size // 2
      self.topology = self.topology or ""
      self.compute_type = self.compute_type or self.device_type
    self.hardware_id = self.device_type.split("-")[0]
    if self.hardware_id == "v5litepod":
      self.hardware_id = "v5e"


def assign_workload_name(workload: WorkloadConfig, user: str | None = None) -> str:
  """Assign the original benchmark's generated JobSet name when needed."""
  user = user or os.environ.get("USER", "user")
  pw = workload.pathways_config
  if not workload.run_name:
    suffix = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(3))
    model = workload.model.model_name.replace("_", "-")
    if pw:
      workload.run_name = f"{user[:3]}-pw-{model[:7]}-{workload.num_slices}-{suffix}"
    else:
      workload.run_name = f"{user[:3]}-{model[:10]}-{workload.num_slices}-{time.strftime('%m%d%H')}-{suffix}"
  return workload.run_name


def submit_workload(cluster: ClusterConfig, workload: WorkloadConfig, command: str, *, user: str | None = None) -> tuple[str, int]:
  """Submit a workload and return its JobSet name and submission exit code."""
  assert cluster.device_type == workload.device_type
  assign_workload_name(workload, user)
  pw = workload.pathways_config
  location = cluster.location or cluster.zone
  args = ["gcluster", "job", "submit"]
  if workload.skip_validation:
    args.append("--skip-prereqs")
  args.extend([f"--cluster={cluster.cluster_name}", f"--project={cluster.project}", f"--location={location}"])
  for flag, value in (("compute-type", cluster.compute_type or workload.compute_type), ("topology", cluster.topology or workload.topology)):
    if value:
      args.append(f"--{flag}={value}")
  for mount in workload.mounts:
    if ";" not in mount:
      raise ValueError(f"Invalid mount specification '{mount}'; expected '<src>;<dest>[;<mode>]' ")
    args.append(f"--mount={mount}")
  args.append(f"--num-slices={workload.num_slices}")
  if not (pw and pw.headless):
    args.append(f"--command={command}")
  if pw:
    args.extend(["--pathways", "--grace-period=300s", f"--pathways-gcs-location={workload.base_output_directory}"])
    for flag, value in (("server-image", pw.server_image), ("proxy-server-image", pw.proxy_server_image),
                        ("colocated-python-sidecar-image", pw.colocated_python_sidecar_image)):
      if value is not None:
        args.append(f"--pathways-{flag}={value}")
    options = workload.model.pathways_xla_flag_options or {} if workload.model else {}
    proxy = (pw.proxy_flags + (" " + workload.model.xla_flags if not pw.headless else "")).split()
    if not pw.headless:
      proxy = [flag for flag in proxy if flag not in options.get("remove", [])]
      proxy.extend(options.get("add_proxy", "").split())
    args.extend([f"--pathways-server-args={' '.join((pw.server_flags + (' ' + options.get('add_server', '') if not pw.headless else '')).split())}",
                 f"--pathways-proxy-args={' '.join(proxy)}",
                 f"--pathways-worker-args={' '.join((pw.worker_flags + (' ' + options.get('add_worker', '') if not pw.headless else '')).split())}"])
    if pw.headless:
      args.append("--pathways-headless")
    if pw.runner_image:
      args.append(f"--image={pw.runner_image}")
  elif workload.base_docker_image:
    image = workload.base_docker_image
    if "/" in image:
      args.append(f"--image={image}")
    else:
      parts = location.split("-")
      region = "-".join(parts[:2]) if len(parts) == 3 else location
      tag = image if ":" in image else f"{image}:latest"
      remote = f"{region}-docker.pkg.dev/{cluster.project}/{os.environ.get('GCLUSTER_IMAGE_REPO', 'maxtext-images')}/{tag}"
      subprocess.run(["docker", "tag", image, remote], check=True)
      subprocess.run(["docker", "push", remote], check=True)
      args.extend([f"--base-image={remote}", "--build-context=."])
  if not pw and workload.libtpu_type == "custom":
    args.append("--env=TPU_LIBRARY_PATH=/lib/libtpu.so")
  args.extend(["--verbose", f"--name={workload.run_name}", f"--priority={workload.priority}", f"--restarts={workload.max_restarts}"])
  return workload.run_name, subprocess.run(args, check=False).returncode


def wait_for_workload_completion(cluster: ClusterConfig, workload_name: str, timeout: str = "-1s") -> int:
  """Wait for JobSet terminal state and report success only for Completed."""
  result = subprocess.run(["gcloud", "container", "clusters", "get-credentials", cluster.cluster_name,
                           f"--location={cluster.location or cluster.zone}", f"--project={cluster.project}"], check=False)
  if result.returncode:
    return result.returncode
  for attempt in range(30):
    result = subprocess.run(["kubectl", "get", f"jobset/{workload_name}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if result.returncode == 0:
      break
    if attempt < 29:
      time.sleep(2)
  result = subprocess.run(["kubectl", "wait", "--for=jsonpath={.status.terminalState}", f"jobset/{workload_name}", f"--timeout={timeout}"], check=False)
  if result.returncode:
    return result.returncode
  return subprocess.run(["kubectl", "wait", "--for=condition=Completed", f"jobset/{workload_name}", "--timeout=0s"], check=False).returncode
