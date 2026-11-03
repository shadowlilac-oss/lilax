"""Google Cloud project lookup for Vertex AI TensorBoard."""

import os
import subprocess

from lilax._backend.common.cloud import is_decoupled
from lilax._backend.utils import max_logging


def get_project():
  if is_decoupled():
    return os.environ.get("LOCAL_GCLOUD_PROJECT", "local-maxtext-project")
  try:
    completed_command = subprocess.run(["gcloud", "config", "get", "project"], check=True, capture_output=True)
    project_outputs = completed_command.stdout.decode().strip().split("\n")
    if len(project_outputs) < 1 or project_outputs[-1] == "":
      max_logging.log("You must specify config.vertex_tensorboard_project or set 'gcloud config set project <project>'")
      return None
    return project_outputs[-1]
  except (FileNotFoundError, subprocess.CalledProcessError) as ex:
    max_logging.log(f"Unable to retrieve gcloud project (decoupled={is_decoupled()}): {ex}")
    return None
