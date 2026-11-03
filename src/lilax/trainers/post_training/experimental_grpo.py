"""Experimental MaxText GRPO training with offline generation."""

import os
from contextlib import nullcontext
from functools import partial

import jax
import pathwaysutils
import tensorflow as tf

from lilax._backend.common.goodput import (
    RECORD_JOB_START_TIME, create_goodput_recorder, maybe_monitor_goodput, record_goodput,
)
from lilax._backend.common.vertex_tensorboard import VertexTensorboardManager
from lilax._backend.common.profiler import xprof_server
from lilax._backend.configs.runtime import resolve_runtime_config
from lilax._backend.experimental.rl import grpo_trainer
from lilax._backend.utils import train_utils
from lilax._backend.observability.diagnostics import system_info
from ..telemetry.reporting import MetricReporter
from ..arguments.training_args import TrainingArguments


def train_grpo(config, inference_config, *, report_to=None, launch_xprof=False, xprof_port=8791):
  """Run the experimental native GRPO loop with training and inference configurations."""
  pathwaysutils.initialize()
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  tf.config.set_visible_devices([], "GPU")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )
  args = TrainingArguments(report_to=report_to)
  config = resolve_runtime_config(config, overrides=args._overrides())
  inference_config = resolve_runtime_config(inference_config)
  if not config.use_grpo:
    raise ValueError("Please set the value of use_grpo to True")
  if config.inference_rollouts < 1 or config.inference_rollouts > config.steps:
    raise ValueError(
        f"Please set the value of inference_rollouts to be less than {config.steps} or greater than 1. "
        f"Current value: {config.inference_rollouts}"
    )
  if config.decode_sampling_strategy == "greedy" or config.decode_sampling_temperature == 0.0:
    raise ValueError("Please set decode_sampling_strategy as 'weighted' and decode_sampling_temperature as a positive number")
  if config.inference_devices_per_replica * config.inference_replicas >= jax.device_count():
    raise ValueError(
        f"Invalid value chosen for {config.inference_devices_per_replica=} and {config.inference_replicas=} "
        f"with {jax.device_count()} devices"
    )
  if config.per_device_batch_size < 1.0 or inference_config.per_device_batch_size < 1.0:
    raise ValueError("GRPO does not support setting per_device_batch_size < 1.0")
  jax.config.update("jax_use_shardy_partitioner", config.shardy)
  system_info.print_system_information()
  train_utils.validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  vertex_tensorboard_manager = VertexTensorboardManager()
  if config.use_vertex_tensorboard or os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"):
    vertex_tensorboard_manager.configure_vertex_tensorboard(config)
  recorder = create_goodput_recorder(config)
  record_goodput(recorder, RECORD_JOB_START_TIME)
  with maybe_monitor_goodput(config):
    reporter_factory = partial(MetricReporter, report_to=args.report_to) if args.report_to else None
    report_inference_captures = bool(
        args.report_to and {"trackio", "wandb"} & set(args.report_to)
        and inference_config.profiler == "xplane" and not inference_config.managed_mldiagnostics
        and jax.process_index() == 0
    )
    server = xprof_server(config.tensorboard_dir, xprof_port, jax.process_index()) if launch_xprof else nullcontext()
    with server:
      return grpo_trainer.train_loop(
          config, inference_config, recorder, reporter_factory=reporter_factory,
          report_inference_captures=report_inference_captures,
      )
