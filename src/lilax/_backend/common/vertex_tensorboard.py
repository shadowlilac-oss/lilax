"""Utilities for Tensorboard in Vertex AI."""

import os

import jax
from cloud_accelerator_diagnostics import tensorboard, uploader

from lilax._backend.common.cloud import is_decoupled
from lilax._backend.utils import max_logging
from lilax._backend.utils.tensorboard_logging import gcloud_project


class VertexTensorboardManager:
  """Create Vertex AI Tensorboard and upload logs to it."""

  def __init__(self):
    self.uploader_flag = False

  def __del__(self):
    if self.uploader_flag:
      uploader.stop_upload_to_tensorboard()

  def setup(self):
    max_logging.log("Setting up Tensorboard and Experiment in Vertex AI.")
    project = os.environ.get("TENSORBOARD_PROJECT")
    region = os.environ.get("TENSORBOARD_REGION")
    if not project or not region:
      max_logging.log("Either config.vertex_tensorboard_project or config.vertex_tensorboard_region is not set.")
      return None
    name = os.environ.get("TENSORBOARD_NAME")
    instance_id = tensorboard.create_instance(project=project, location=region, tensorboard_name=name)
    if instance_id is None:
      return None
    _, url = tensorboard.create_experiment(
        project=project, location=region, experiment_name=os.environ.get("EXPERIMENT_NAME"), tensorboard_name=name
    )
    return url

  def upload_data(self, tensorboard_dir):
    project = os.environ.get("TENSORBOARD_PROJECT")
    region = os.environ.get("TENSORBOARD_REGION")
    name = os.environ.get("TENSORBOARD_NAME")
    experiment = os.environ.get("EXPERIMENT_NAME")
    if not project or not region or not name or not experiment:
      max_logging.log("Vertex Tensorboard configurations are not set. Data will not be uploaded to Vertex AI.")
      self.uploader_flag = False
    max_logging.log(
        f"Data will be uploaded to Vertex Tensorboard instance: {name} and Experiment: {experiment} in {region}."
    )
    uploader.start_upload_to_tensorboard(
        project=project, location=region, experiment_name=experiment, tensorboard_name=name, logdir=tensorboard_dir
    )
    self.uploader_flag = True

  def configure_vertex_tensorboard(self, config):
    if is_decoupled():
      max_logging.log("Decoupled mode -> Skipping Vertex Tensorboard configuration.")
      return
    if jax.process_index() == 0:
      if not os.environ.get("TENSORBOARD_PROJECT"):
        os.environ["TENSORBOARD_PROJECT"] = config.vertex_tensorboard_project or gcloud_project.get_project()
      if not os.environ.get("TENSORBOARD_REGION"):
        os.environ["TENSORBOARD_REGION"] = config.vertex_tensorboard_region
      if not os.environ.get("TENSORBOARD_NAME"):
        os.environ["TENSORBOARD_NAME"] = f"{os.environ.get('TENSORBOARD_PROJECT')}-tb-instance"
      if not os.environ.get("EXPERIMENT_NAME"):
        os.environ["EXPERIMENT_NAME"] = config.run_name
      if config.use_vertex_tensorboard:
        url = self.setup()
        if url is None:
          raise ValueError("Unable to create Tensorboard and Experiment in Vertex AI.")
        max_logging.log(f"View your Vertex AI Tensorboard at: {url}")
        self.upload_data(config.tensorboard_dir)
      elif os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"):
        self.upload_data(config.tensorboard_dir)
