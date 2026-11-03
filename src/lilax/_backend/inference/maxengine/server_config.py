"""Configure MaxEngine for JetStream."""

import functools
from typing import Any

import jax
from jetstream.core import config_lib
from jetstream.engine import engine_api

from lilax._backend.common.cloud import is_decoupled
from lilax._backend.inference.maxengine import maxengine


def create_exp_maxengine(devices: Any, config: Any) -> engine_api.Engine:
  if is_decoupled():
    return maxengine.MaxEngine(config)
  return maxengine.MaxEngine(config=config, devices=devices)


def create_maxengine(devices: Any, config: Any) -> engine_api.Engine:
  del devices
  return maxengine.MaxEngine(config)


def get_server_config(config_str: str, config: Any) -> config_lib.ServerConfig:
  """Get the JetStream server configuration for a MaxText configuration."""
  match config_str:
    case "MaxtextInterleavedServer":
      return config_lib.ServerConfig(
          prefill_slices=(),
          generate_slices=(),
          interleaved_slices=("tpu=" + str(jax.device_count()),),
          prefill_engine_create_fns=(),
          generate_engine_create_fns=(),
          interleaved_engine_create_fns=(functools.partial(create_maxengine, config=config),),
      )
    case "ExperimentalMaxtextDisaggregatedServer":
      return config_lib.ServerConfig(
          prefill_slices=(config.prefill_slice,),
          generate_slices=(config.generate_slice,),
          interleaved_slices=(),
          prefill_engine_create_fns=(functools.partial(create_exp_maxengine, config=config),),
          generate_engine_create_fns=(functools.partial(create_exp_maxengine, config=config),),
          interleaved_engine_create_fns=(),
      )
    case _:
      raise NotImplementedError
