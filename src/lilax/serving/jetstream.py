"""JetStream server lifecycle for MaxText configurations."""

import os
from typing import Any

import jax
from jetstream.core import server_lib

from lilax._backend.configs.runtime import resolve_runtime_config
from lilax._backend.inference.maxengine import server as maxengine_server
from lilax.configuration import MaxTextConfig, RLConfig


def start_server(config: Any) -> server_lib.JetStreamServer:
  """Start JetStream and return its native server for stopping or waiting."""
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if isinstance(config, (MaxTextConfig, RLConfig)):
    config = resolve_runtime_config(config)
  return maxengine_server.start_server(config)


def serve(config: Any) -> None:
  """Start JetStream and wait for termination."""
  start_server(config).wait_for_termination()


__all__ = ["serve", "start_server"]
