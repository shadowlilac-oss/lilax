"""Standalone native pretraining initialization and retry orchestration."""

import os

import jax
import pathwaysutils
import tensorflow as tf

from lilax._backend.common.goodput import (
    RECORD_JOB_START_TIME, create_goodput_recorder, maybe_monitor_goodput, record_goodput,
)
from lilax._backend.common.vertex_tensorboard import VertexTensorboardManager
from lilax._backend.configs.runtime import resolve_runtime_config
from lilax._backend.configs.types import TeCommGemmOverlapPolicy
from lilax._backend.trainers.pre_train import run as native_run
from lilax._backend.utils.gpu_collective_overlap_pass.registration import (
    register as register_gpu_collective_overlap_pass,
)
from lilax._backend.common.telemetry import mllog_utils
from lilax._backend.utils import train_utils
from lilax._backend.observability.diagnostics import system_info
from lilax._backend.integration.transformer_engine import cgemm


def train_pretraining(config, *, devices=None, resume_from_checkpoint=None):
  """Run native pretraining from an inert configuration and return the training state."""
  if resume_from_checkpoint is None or resume_from_checkpoint is True:
    checkpoint_step = -1
  elif (
      isinstance(resume_from_checkpoint, int) and not isinstance(resume_from_checkpoint, bool)
      and resume_from_checkpoint >= 0
  ):
    checkpoint_step = resume_from_checkpoint
  else:
    raise ValueError("resume_from_checkpoint must be None, True, or a nonnegative checkpoint step.")

  run_name = None

  def initialize():
    pathwaysutils.initialize()
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    tf.config.set_visible_devices([], "GPU")
    if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
      os.environ["LIBTPU_INIT_ARGS"] = (
          os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
      )
    overrides = {"run_name": run_name} if run_name is not None else None
    native_config = resolve_runtime_config(config, overrides=overrides)
    system_info.print_system_information()
    train_utils.validate_train_config(native_config)
    register_gpu_collective_overlap_pass()
    jax.config.update("jax_default_prng_impl", native_config.prng_impl)
    jax.config.update("jax_use_shardy_partitioner", native_config.shardy)
    jax.config.update("jax_remove_size_one_mesh_axis_from_type", native_config.remove_size_one_mesh_axis_from_type)
    os.environ["TFDS_DATA_DIR"] = native_config.dataset_path or ""
    vertex_tensorboard_manager = VertexTensorboardManager()
    if native_config.use_vertex_tensorboard or os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"):
      vertex_tensorboard_manager.configure_vertex_tensorboard(native_config)
    if native_config.te_comm_gemm_overlap != TeCommGemmOverlapPolicy.DISABLED:
      cgemm.bootstrap_transformer_engine_cgemm(native_config)
    mllog_utils.init_start(native_config)
    return native_config, create_goodput_recorder(native_config)

  native_config, recorder = initialize()
  run_name = native_config.run_name
  initial_attempt = True

  def setup_attempt(attempt_config, attempt_recorder, retry_cache):
    nonlocal initial_attempt
    restore_step = checkpoint_step if initial_attempt else -1
    setup = train_utils.setup_train_loop(
        attempt_config, attempt_recorder, devices=devices, elastic_retry_cache=retry_cache,
        checkpoint_step=restore_step,
    )
    initial_attempt = False
    return setup

  record_goodput(recorder, RECORD_JOB_START_TIME)
  with maybe_monitor_goodput(native_config):
    return native_run.run_elastic(
        native_config, recorder, setup_fn=setup_attempt, reinitialize=initialize, reinitialize_first=True,
    )


