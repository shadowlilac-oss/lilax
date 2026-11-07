"""Native Omni SFT orchestration."""

import os

import jax
import pathwaysutils

from lilax._backend.common import profiler
from lilax._backend.common.checkpoints import snapshotting
from lilax._backend.common.goodput import (
    RECORD_JOB_START_TIME, create_goodput_recorder, maybe_monitor_goodput, record_goodput,
)
from lilax._backend.common.vertex_tensorboard import VertexTensorboardManager
from lilax._backend.configs.types import TeCommGemmOverlapPolicy
from lilax._backend.trainers.pre_train import train
from lilax._backend.distributed import elastic
from lilax._backend.observability import max_logging
from lilax._backend.utils.gpu_collective_overlap_pass.registration import (
    register as register_gpu_collective_overlap_pass,
)
from lilax._backend.common.telemetry import mllog_utils
from lilax._backend.utils import train_utils
from lilax._backend.observability.diagnostics import system_info
from lilax._backend.integration.transformer_engine import cgemm, mesh_context
from lilax._backend.configs.inert import MaxTextConfig
from lilax.runtime import Runtime
import tensorflow as tf


def train_sft_omni(config_path, *, devices=None, **overrides):
  """Load an Omni SFT configuration and return the native training state."""
  options = {
      "use_sft": True,
      "use_tunix_gradient_accumulation": False,
      "override_model_config": True,
      **overrides,
  }

  def initialize():
    pathwaysutils.initialize()
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    tf.config.set_visible_devices([], "GPU")
    if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
      os.environ["LIBTPU_INIT_ARGS"] = (
          os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
      )
    runtime = Runtime(MaxTextConfig(base_config=config_path, **options), devices=devices)
    runtime.resolve_config()
    config = runtime.config
    system_info.print_system_information()
    train_utils.validate_train_config(config)
    register_gpu_collective_overlap_pass()
    jax.config.update("jax_default_prng_impl", config.prng_impl)
    jax.config.update("jax_use_shardy_partitioner", config.shardy)
    jax.config.update("jax_remove_size_one_mesh_axis_from_type", config.remove_size_one_mesh_axis_from_type)
    os.environ["TFDS_DATA_DIR"] = config.dataset_path or ""
    vertex_tensorboard_manager = VertexTensorboardManager()
    if config.use_vertex_tensorboard or os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"):
      vertex_tensorboard_manager.configure_vertex_tensorboard(config)
    if config.te_comm_gemm_overlap != TeCommGemmOverlapPolicy.DISABLED:
      cgemm.bootstrap_transformer_engine_cgemm(config)
    mllog_utils.init_start(config)
    return runtime, create_goodput_recorder(config)

  retry_cache = None
  runtime, recorder = initialize()
  config = runtime.config
  record_goodput(recorder, RECORD_JOB_START_TIME)

  def run_attempt(attempt_runtime, attempt_recorder):
    attempt_config = attempt_runtime.config
    with mesh_context.maybe_get_transformer_engine_context(attempt_config), profiler.nnx_named_scope_context():
      setup = train_utils.setup_train_loop(
          attempt_config, attempt_recorder, elastic_retry_cache=retry_cache, runtime=attempt_runtime,
      )
      return train.train_loop(attempt_config, attempt_recorder, elastic_retry_cache=retry_cache, setup=setup)

  if config.elastic_enabled:
    max_logging.log("Elastic utils: Elastic training enabled.")
    retry_cache = elastic.RetryCache()
    retry_cache.snapshotter = snapshotting.create_snapshotter(config)

    def on_elastic_event(scale_up):
      elastic.record_elastic_event_start(recorder, scale_up=scale_up)
      if retry_cache.snapshotter is not None and not scale_up:
        retry_cache.snapshotter = snapshotting.reset_snapshotter(retry_cache.snapshotter)

    def on_slices_ready():
      elastic.record_elastic_wait_end_and_reinit_start(recorder)

    def elastic_attempt():
      attempt_runtime, attempt_recorder = elastic.get_or_build(retry_cache, "initialize", initialize)
      return run_attempt(attempt_runtime, attempt_recorder)

    train_func = elastic.elastic_retry(
        config, callback_fn=on_elastic_event, pre_callback_fn=on_slices_ready, retry_cache=retry_cache,
    )(elastic_attempt)
  else:
    def train_func():
      return run_attempt(runtime, recorder)

  with maybe_monitor_goodput(config):
    return train_func()
