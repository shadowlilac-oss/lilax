"""Stage and run distillation workloads with XPK."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


@dataclass
class DistillationJob:
  cluster: str
  project: str
  zone: str
  device_type: str
  workload: str
  run_name: str
  base_output_directory: str
  image: str
  config_path: str
  config_gcs: str
  libtpu_init_args: str
  trainer_module: str = "lilax.trainers.post_training.distillation"
  trainer_function: str = "run_distillation"
  overrides: dict = field(default_factory=dict)
  priority: str = "high"
  num_slices: int = 1
  dataset_bucket: str = "maxtext-dataset"
  dataset_subpath: str = "array-record/climbmix/*.arrayrecord"
  use_gcsfuse: bool = True
  tokenizer_gcs: str | None = None
  tokenizer_local: str | None = None
  hf_cache_dir: str = "/dev/shm/hf"

  @property
  def output_directory(self) -> str:
    return f"{self.base_output_directory.rstrip('/')}/{self.workload}"


@dataclass(frozen=True)
class ResumeResult:
  completed: bool
  attempts: int
  checkpoint_step: int
  terminal_state: str


def stage(job: DistillationJob) -> None:
  """Copy the local distillation YAML to the configured GCS location."""
  subprocess.run(["gcloud", "storage", "cp", job.config_path, job.config_gcs], check=True)


def build_command(job: DistillationJob, *, steps: int | None = None,
                  checkpoint_period: int | None = None) -> str:
  """Build the remote command for a Python trainer integration.

  The integration callable receives config_path, overrides, and
  resume_from_checkpoint as keyword arguments. It must load the staged YAML,
  apply the overrides, and run the native distillation trainer.
  """
  if (job.tokenizer_gcs is None) != (job.tokenizer_local is None):
    raise ValueError("tokenizer_gcs and tokenizer_local must be supplied together")
  dataset = (f"/tmp/gcsfuse/{job.dataset_subpath}" if job.use_gcsfuse
             else f"gs://{job.dataset_bucket.removeprefix('gs://')}/{job.dataset_subpath}")
  overrides = dict(job.overrides)
  overrides.update(run_name=job.run_name, base_output_directory=job.output_directory,
                   grain_train_files=dataset, save_checkpoint_on_completion=True)
  if steps is not None:
    overrides.update(steps=steps, learning_rate_schedule_steps=steps)
  if checkpoint_period is not None:
    overrides["checkpoint_period"] = checkpoint_period
  payload = json.dumps({"config_path": job.config_path, "overrides": overrides, "resume_from_checkpoint": True})
  python = (
      "import importlib, json; "
      f"trainer = getattr(importlib.import_module({job.trainer_module!r}), {job.trainer_function!r}); "
      f"trainer(**json.loads({payload!r}))"
  )
  quote = shlex.quote
  environment = {
      "PYTHONPATH": "/deps/src:/app/src", "BASE_OUTPUT_DIRECTORY": job.output_directory,
      "LIBTPU_INIT_ARGS": " ".join(job.libtpu_init_args.split()), "TMPDIR": "/dev/shm",
      "JAX_COMPILATION_CACHE_DIR": "/dev/shm/jax_cache", "HF_HOME": job.hf_cache_dir,
      "HF_DATASETS_CACHE": f"{job.hf_cache_dir}/datasets",
  }
  commands = [f"export {key}={quote(value)}" for key, value in environment.items()]
  commands.extend([
      f"mkdir -p {quote(job.hf_cache_dir + '/datasets')} {quote(str(Path(job.config_path).parent))}",
      f"gcloud storage cp {quote(job.config_gcs)} {quote(job.config_path)}",
  ])
  if job.tokenizer_gcs is not None:
    commands.extend([f"mkdir -p {quote(job.tokenizer_local)}",
                     f"gcloud storage rsync {quote(job.tokenizer_gcs)} {quote(job.tokenizer_local)}"])
  if job.use_gcsfuse:
    bucket = job.dataset_bucket.removeprefix("gs://")
    commands.extend([
        "mkdir -p /tmp/gcsfuse",
        "gcsfuse -o ro --implicit-dirs --log-severity=debug --type-cache-max-size-mb=-1 "
        "--stat-cache-max-size-mb=-1 --kernel-list-cache-ttl-secs=-1 --metadata-cache-ttl-secs=-1 "
        f"{quote(bucket)} /tmp/gcsfuse",
    ])
  commands.append(f"python3 -c {quote(python)}")
  return " && ".join(commands)


def submit(job: DistillationJob, *, stage_config: bool = True, steps: int | None = None,
           checkpoint_period: int | None = None) -> str:
  """Submit an XPK workload and return its name."""
  command = build_command(job, steps=steps, checkpoint_period=checkpoint_period)
  if stage_config:
    stage(job)
  image_flag = "--docker-image" if "/" in job.image else "--base-docker-image"
  subprocess.run([
      "xpk", "workload", "create", "--cluster", job.cluster, "--workload", job.workload,
      f"--priority={job.priority}", f"--tpu-type={job.device_type}", f"--num-slices={job.num_slices}",
      f"--project={job.project}", f"--zone={job.zone}", f"{image_flag}={job.image}", "--command", command,
  ], check=True)
  return job.workload


def delete(job: DistillationJob) -> None:
  """Delete the XPK workload so its name can be reused."""
  subprocess.run([
      "xpk", "workload", "delete", f"--workload={job.workload}", f"--cluster={job.cluster}",
      f"--project={job.project}", f"--zone={job.zone}", "--force",
  ], check=True)


def monitor(job: DistillationJob, *, log_path: str | Path | None = None) -> None:
  """Wait for a running pod and stream all workload logs to stdout and a file."""
  selector = f"jobset.sigs.k8s.io/jobset-name={job.workload}"
  while True:
    result = subprocess.run(["kubectl", "get", "pods", "-l", selector, "-o", "json"],
                            capture_output=True, text=True, check=True)
    pods = json.loads(result.stdout)["items"]
    phases = [pod["status"].get("phase", "") for pod in pods]
    if "Running" in phases:
      break
    if phases and not any(phase in ("Pending", "ContainerCreating") for phase in phases):
      if pods:
        subprocess.run(["kubectl", "logs", pods[0]["metadata"]["name"], "--all-containers", "--tail=40"],
                       check=False)
      raise RuntimeError(f"No running pod for {job.workload}; phases: {phases}")
    time.sleep(15)
  destination = Path(log_path) if log_path is not None else Path.home() / f"training-{job.workload}.log"
  with destination.open("w", encoding="utf-8") as output:
    process = subprocess.Popen([
        "kubectl", "logs", "-f", "-l", selector, "--all-containers",
        f"--max-log-requests={len(pods) * 4}", "--prefix",
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
      sys.stdout.write(line)
      output.write(line)
      output.flush()
    code = process.wait()
  if code:
    raise subprocess.CalledProcessError(code, process.args)


def resume_until_done(job: DistillationJob, *, steps: int, max_retries: int = 10,
                      checkpoint_period: int | None = None) -> ResumeResult:
  """Resubmit the same workload until its latest checkpoint reaches steps."""
  last_step = 0
  terminal = ""
  for attempt in range(1, max_retries + 1):
    submit(job, stage_config=False, steps=steps, checkpoint_period=checkpoint_period)
    while not terminal:
      time.sleep(60)
      result = subprocess.run([
          "kubectl", "get", "jobset", job.workload, "-o", "jsonpath={.status.terminalState}",
      ], capture_output=True, text=True, check=True)
      terminal = result.stdout.strip()
    checkpoint_dir = f"{job.output_directory}/{job.run_name}/checkpoints/"
    result = subprocess.run(["gcloud", "storage", "ls", checkpoint_dir], capture_output=True, text=True, check=True)
    names = [line.rstrip("/").rsplit("/", 1)[-1] for line in result.stdout.splitlines() if line.endswith("/")]
    last_step = max((int(name) for name in names if name.isdigit()), default=0)
    if last_step >= steps:
      return ResumeResult(True, attempt, last_step, terminal)
    delete(job)
    if attempt < max_retries:
      time.sleep(60)
      terminal = ""
  return ResumeResult(False, max_retries, last_step, terminal)
