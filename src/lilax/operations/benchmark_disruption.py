"""Monitor benchmark workloads and signal selected Kubernetes workers."""

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import base64
import copy
import os
import re
import subprocess
import tempfile
import threading
import time

from kubernetes import client, config
from kubernetes.stream import stream
import yaml


MCJAX_STANDARD_TARGET_POD_REGEX_SUFFIX = ".*(main-job|slice-job)-0-0.*"
MCJAX_STANDARD_STEP_POD_REGEX_SUFFIX = ".*(main-job|slice-job)-0-0.*"
PATHWAYS_STANDARD_TARGET_POD_REGEX_SUFFIX = ".*worker-0-0.*"
PATHWAYS_STANDARD_STEP_POD_REGEX_SUFFIX = ".*head-0-0.*"
MCJAX_WORKER_CONTAINER_NAME = "workload-container"
PATHWAYS_WORKER_CONTAINER_NAME = "pathways-worker"
STANDARD_STEP_LOG_REGEX = r".*completed step: (\d+), .*"
POLLING_INTERVAL_SECONDS = 10


@contextmanager
def cluster_disruption_manager(cluster_config):
  """Create a disruption manager using isolated gcloud cluster credentials."""
  with tempfile.TemporaryDirectory(prefix="kubeconfig_") as directory:
    kubeconfig = os.path.join(directory, "config")
    environment = os.environ.copy()
    environment["KUBECONFIG"] = kubeconfig
    location = cluster_config.location or cluster_config.zone
    subprocess.run([
        "gcloud", "container", "clusters", "get-credentials", cluster_config.cluster_name,
        f"--location={location}", f"--project={cluster_config.project}",
    ], env=environment, check=True)
    with open(kubeconfig, encoding="utf-8") as config_file:
      kubeconfig_data = yaml.safe_load(config_file)
    current_context = next(item["context"] for item in kubeconfig_data["contexts"]
                           if item["name"] == kubeconfig_data["current-context"])
    cluster = next(item["cluster"] for item in kubeconfig_data["clusters"]
                   if item["name"] == current_context["cluster"])
    if "certificate-authority-data" in cluster:
      ca_path = os.path.join(directory, "ca.crt")
      with open(ca_path, "wb") as ca_file:
        ca_file.write(base64.b64decode(cluster.pop("certificate-authority-data")))
      cluster["certificate-authority"] = ca_path

    def create_client():
      return config.new_client_from_config_dict(config_dict=copy.deepcopy(kubeconfig_data), persist_config=False)

    api_client = create_client()
    try:
      namespace = current_context.get("namespace") or "default"
      yield DisruptionManager(client.CoreV1Api(api_client), namespace, api_client_factory=create_client)
    finally:
      api_client.close()
      api_client.rest_client.pool_manager.clear()


class TriggerType(Enum):
  TIME_SECONDS = "time_seconds"
  STEP = "step"


class DisruptionMethod(Enum):
  SIGILL = "sigill"
  SIGTERM = "sigterm"


class RecoverMethod(Enum):
  SIGTERM_METADATA_REMOVE = "sigterm_metadata_remove"


@dataclass
class DisruptionConfig:
  name: str
  trigger_type: TriggerType
  trigger_value: int
  disruption_method: DisruptionMethod
  worker_container_name: str
  recover_trigger_type: TriggerType | None = None
  recover_trigger_value: int | None = None
  recover_method: RecoverMethod | None = None
  target_pod_regex: str = MCJAX_STANDARD_TARGET_POD_REGEX_SUFFIX
  step_pod_regex: str = MCJAX_STANDARD_STEP_POD_REGEX_SUFFIX


def construct_disruption_configs(framework: str, disruption_method: DisruptionMethod, disruptions: dict):
  """Build the original McJAX or Pathways disruption configurations."""
  if framework == "pathways":
    target_regex = PATHWAYS_STANDARD_TARGET_POD_REGEX_SUFFIX
    step_regex = PATHWAYS_STANDARD_STEP_POD_REGEX_SUFFIX
    container = PATHWAYS_WORKER_CONTAINER_NAME
  elif framework == "mcjax":
    target_regex = MCJAX_STANDARD_TARGET_POD_REGEX_SUFFIX
    step_regex = MCJAX_STANDARD_STEP_POD_REGEX_SUFFIX
    container = MCJAX_WORKER_CONTAINER_NAME
  else:
    raise ValueError(f"Unsupported framework: {framework}")

  return [
      DisruptionConfig(
          name=f"{value}_{trigger_type}",
          trigger_type=TriggerType.TIME_SECONDS if trigger_type == "time_seconds" else TriggerType.STEP,
          trigger_value=value,
          disruption_method=disruption_method,
          worker_container_name=container,
          target_pod_regex=target_regex,
          step_pod_regex=step_regex,
      )
      for trigger_type, values in disruptions.items()
      for value in values
  ]


class DisruptionManager:
  """Monitor workloads using a configured Kubernetes CoreV1Api and namespace.

  Recovery settings are retained in the configuration but are not implemented,
  matching the original disruption manager.
  """

  def __init__(self, api, namespace: str, *, api_client_factory=None):
    self.api = api
    self.namespace = namespace
    self.api_client_factory = api_client_factory
    self.threads_to_monitor: defaultdict[str, list[threading.Thread]] = defaultdict(list)

  def add_workload(self, workload_name: str, disruption_configs: list[DisruptionConfig]) -> None:
    for config in disruption_configs:
      self.threads_to_monitor[workload_name].append(
          threading.Thread(target=self._monitor_and_disrupt_workload, args=(workload_name, config), daemon=True)
      )

  def remove_workload(self, workload_name: str) -> None:
    if workload_name in self.threads_to_monitor:
      self.threads_to_monitor[workload_name].clear()

  def start_disruptions_and_wait_for_completion(self) -> None:
    for threads in self.threads_to_monitor.values():
      for thread in threads:
        thread.start()
    for threads in self.threads_to_monitor.values():
      for thread in threads:
        thread.join()

  def _find_pod(self, api, pod_regex: str):
    pods = api.list_namespaced_pod(self.namespace).items
    return next((pod for pod in pods if re.search(pod_regex, pod.metadata.name)), None)

  def _monitor_and_disrupt_workload(self, workload_name: str, config: DisruptionConfig) -> None:
    api_client = self.api_client_factory() if self.api_client_factory is not None else None
    api = client.CoreV1Api(api_client) if api_client is not None else self.api
    try:
      self._run_disruption(api, workload_name, config)
    finally:
      if api_client is not None:
        api_client.close()
        api_client.rest_client.pool_manager.clear()

  def _run_disruption(self, api, workload_name: str, config: DisruptionConfig) -> None:
    if config.trigger_type == TriggerType.TIME_SECONDS:
      time.sleep(config.trigger_value)
    elif config.trigger_type == TriggerType.STEP:
      step_regex = f"{workload_name}{config.step_pod_regex}"
      while True:
        pod = self._find_pod(api, step_regex)
        if pod and api.read_namespaced_pod_status(pod.metadata.name, self.namespace).status.phase == "Running":
          break
        time.sleep(POLLING_INTERVAL_SECONDS)
      logs = api.read_namespaced_pod_log(pod.metadata.name, self.namespace, follow=True, _preload_content=False)
      try:
        for line in logs:
          match = re.search(STANDARD_STEP_LOG_REGEX, line.decode("utf-8").strip())
          if match and int(match.group(1)) >= config.trigger_value:
            break
        else:
          return
      finally:
        logs.close()
        logs.release_conn()
    else:
      raise ValueError(f"Unsupported trigger type: {config.trigger_type}")

    pod = self._find_pod(api, f"{workload_name}{config.target_pod_regex}")
    if not pod:
      return
    if config.disruption_method not in (DisruptionMethod.SIGILL, DisruptionMethod.SIGTERM):
      raise ValueError(f"Unsupported disruption method: {config.disruption_method}")
    exec_client = client.ApiClient(api.api_client.configuration)
    try:
      stream(
          client.CoreV1Api(exec_client).connect_get_namespaced_pod_exec,
          pod.metadata.name,
          self.namespace,
          container=config.worker_container_name,
          command=["/bin/sh", "-c", f"kill -s {config.disruption_method.value.upper()} 1"],
          stderr=True,
          stdin=False,
          stdout=True,
          tty=False,
      )
    finally:
      exec_client.close()
      exec_client.rest_client.pool_manager.clear()
