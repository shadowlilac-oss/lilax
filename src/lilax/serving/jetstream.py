"""JetStream server lifecycle for MaxText configurations."""

import os
import jax
from jetstream.core import server_lib

from lilax._backend.inference.maxengine import server as maxengine_server
from .configuration import JetStreamConfig
from ._model_source import serving_model_source, serving_overrides


def start_server(model, *, server_config: JetStreamConfig | None = None) -> server_lib.JetStreamServer:
  """Start JetStream and return its native server for stopping or waiting."""
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  options = server_config or JetStreamConfig()
  config, resource = serving_model_source(model, overrides=serving_overrides(options))
  try:
    server = maxengine_server.start_server(config)
  except BaseException:
    if resource is not None:
      resource.cleanup()
    raise
  server._lilax_checkpoint_directory = resource
  return server


def serve(model, *, server_config: JetStreamConfig | None = None) -> None:
  """Start JetStream and wait for termination."""
  server = start_server(model, server_config=server_config)
  try:
    server.wait_for_termination()
  finally:
    if server._lilax_checkpoint_directory is not None:
      server._lilax_checkpoint_directory.cleanup()


__all__ = ["serve", "start_server"]
