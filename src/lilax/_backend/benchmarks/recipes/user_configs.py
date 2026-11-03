"""Configuration and model selection for Pathways and McJAX benchmarks."""

from dataclasses import dataclass, field

from lilax._backend.benchmarks.catalog import AVAILABLE_MODELS
from lilax._backend.benchmarks.cluster_runner import ClusterConfig, PathwaysConfig


def build_user_models(frameworks, names, device_base_type, available_models):
  """Select benchmark model configurations for the requested accelerator."""
  for framework in frameworks:
    if framework not in ("mcjax", "pathways"):
      raise ValueError(f"Model framework '{framework}' not available. Available model frameworks are: ['mcjax', 'pathways']")
  if device_base_type not in available_models:
    raise ValueError(f"Unknown device base type: {device_base_type}. Original device type was: {device_base_type}")
  models = available_models[device_base_type]
  for name in names:
    if name not in models:
      raise ValueError(
          f"Model name '{name}' not available for device type '{device_base_type}'. "
          f"Available model names are: {list(models.keys())}"
      )
  return {framework: [models[name] for name in names] for framework in frameworks}


@dataclass
class BenchmarkRecipeConfig:
  """User settings for the standard Pathways and McJAX benchmark."""

  available_models: dict = field(default_factory=lambda: AVAILABLE_MODELS)
  user: str = "user_name"
  cluster_name: str = "v6e-256-cluster"
  project: str = "tpu-prod-env-cluster"
  zone: str = "us-east5-b"
  location: str | None = None
  device_type: str = "v6e-256"
  priority: str = "medium"
  base_output_directory: str | None = None
  server_image: str = "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server"
  proxy_image: str = "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server"
  runner: str = "us-docker.pkg.dev/path/to/maxtext_runner"
  colocated_python_image: str | None = None
  worker_flags: str = ""
  proxy_flags: str = ""
  server_flags: str = ""
  benchmark_steps: int = 20
  headless: bool = False
  selected_model_framework: list[str] = field(default_factory=lambda: ["pathways"])
  selected_model_names: list[str] = field(default_factory=lambda: ["llama3_1_8b_8192"])
  num_slices_list: list[int] = field(default_factory=lambda: [2])
  bq_enable: bool = False
  bq_db_project: str = ""
  bq_db_dataset: str = ""
  max_restarts: int = 0
  workload_id: str | None = None
  skip_validation: bool = False
  cluster_config: ClusterConfig = field(init=False)
  pathways_config: PathwaysConfig = field(init=False)
  region: str = field(init=False)
  headless_workload_name: str = field(init=False)
  models: dict = field(init=False)

  def __post_init__(self):
    if self.location:
      self.zone = self.location if len(self.location.split("-")) == 3 else ""
    self.cluster_config = ClusterConfig(
        cluster_name=self.cluster_name, project=self.project, zone=self.zone,
        location=self.location, device_type=self.device_type,
    )
    location = self.location or self.zone
    parts = location.split("-")
    self.region = "-".join(parts[:-1]) if len(parts) >= 3 else location
    self.pathways_config = PathwaysConfig(
        server_image=self.server_image, proxy_server_image=self.proxy_image, runner_image=self.runner,
        colocated_python_sidecar_image=self.colocated_python_image, headless=self.headless,
        server_flags=self.server_flags, proxy_flags=self.proxy_flags, worker_flags=self.worker_flags,
    )
    self.headless_workload_name = f"{self.user[:3]}-headless"
    if not self.base_output_directory:
      self.base_output_directory = f"gs://{self.user}-{self.region}/{self.user}-"
    elif not self.base_output_directory.endswith("/"):
      self.base_output_directory += "/"
    self.models = build_user_models(
        self.selected_model_framework, self.selected_model_names,
        self.device_type.split("-", maxsplit=1)[0], self.available_models,
    )
