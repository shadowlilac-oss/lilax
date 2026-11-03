"""Native pretraining run entry points with optional profiler server and elastic retry."""

import contextlib
import os

import jax

from lilax._backend.common import profiler
from lilax._backend.common.checkpoints import snapshotting
from lilax._backend.trainers.pre_train.train import train_loop
from lilax._backend.utils import elastic_utils
from lilax._backend.integration.transformer_engine import mesh_context


def run(config, recorder, elastic_retry_cache=None, *, setup=None, setup_fn=None, reporter_factory=None,
        launch_xprof=False, xprof_port=8791, training_graph=None):
  with mesh_context.maybe_get_transformer_engine_context(config), profiler.nnx_named_scope_context():
    if setup_fn is not None:
      setup = setup_fn(config, recorder, elastic_retry_cache)
    server = (
        profiler.xprof_server(os.path.join(config.tensorboard_dir, ""), xprof_port, jax.process_index())
        if launch_xprof else contextlib.nullcontext()
    )
    with server:
      return train_loop(
          config, recorder, elastic_retry_cache=elastic_retry_cache, setup=setup, reporter_factory=reporter_factory,
          training_graph=training_graph,
      )


def run_elastic(config, recorder, *, setup=None, setup_fn=None, reinitialize=None, reinitialize_first=False,
                reporter_factory=None, launch_xprof=False, xprof_port=8791, training_graph=None):
  if not config.elastic_enabled:
    return run(config, recorder, setup=setup, setup_fn=setup_fn, reporter_factory=reporter_factory,
               launch_xprof=launch_xprof, xprof_port=xprof_port, training_graph=training_graph)

  retry_cache = elastic_utils.RetryCache()
  retry_cache.snapshotter = snapshotting.create_snapshotter(config)

  def on_elastic_event(scale_up):
    elastic_utils.record_elastic_event_start(recorder, scale_up=scale_up)
    if retry_cache.snapshotter is not None and not scale_up:
      retry_cache.snapshotter = snapshotting.reset_snapshotter(retry_cache.snapshotter)

  def on_slices_ready():
    elastic_utils.record_elastic_wait_end_and_reinit_start(recorder)

  initialized = False

  def initialize_attempt():
    nonlocal initialized
    if (initialized or reinitialize_first) and reinitialize is not None:
      initialized = True
      return reinitialize()
    initialized = True
    return config, recorder

  def attempt():
    attempt_config, attempt_recorder = elastic_utils.get_or_build(retry_cache, "initialize", initialize_attempt)
    return run(attempt_config, attempt_recorder, elastic_retry_cache=retry_cache, setup=setup, setup_fn=setup_fn,
               reporter_factory=reporter_factory, launch_xprof=launch_xprof, xprof_port=xprof_port,
               training_graph=training_graph)

  return elastic_utils.elastic_retry(
      config, callback_fn=on_elastic_event, pre_callback_fn=on_slices_ready, retry_cache=retry_cache,
  )(attempt)()
