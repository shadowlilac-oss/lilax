"""Start a JetStream server with a native MaxText configuration."""

from typing import Any

from jetstream.core import config_lib, server_lib
import pathwaysutils

from lilax._backend.inference.maxengine import server_config as maxengine_config


def _create_prefix_caching_config(config: Any) -> config_lib.PrefixCachingConfig | None:
  if not config.enable_prefix_caching:
    return None
  if not config.use_chunked_prefill:
    raise ValueError("Prefix caching requires chunked prefill.")
  return config_lib.PrefixCachingConfig(
      max_hbm_byte=config.prefix_caching_hbm_byte,
      max_dram_byte=config.prefix_caching_dram_byte,
  )


def start_server(config: Any) -> server_lib.JetStreamServer:
  """Initialize Pathways and return the running JetStream server."""
  pathwaysutils.initialize()
  devices = server_lib.get_devices()
  server_config = maxengine_config.get_server_config(config.inference_server, config)
  metrics_server_config = None
  if config.prometheus_port != 0:
    metrics_server_config = config_lib.MetricsServerConfig(port=config.prometheus_port)

  return server_lib.run(
      threads=256,
      port=9000,
      config=server_config,
      devices=devices,
      metrics_server_config=metrics_server_config,
      enable_jax_profiler=config.enable_jax_profiler if config.enable_jax_profiler else False,
      jax_profiler_port=config.jax_profiler_port if config.jax_profiler_port else 9999,
      enable_model_warmup=config.enable_model_warmup if config.enable_model_warmup else False,
      lora_input_adapters_path=config.lora_input_adapters_path,
      multi_sampling=config.multi_sampling if config.multi_sampling else False,
      prefix_caching_config=_create_prefix_caching_config(config),
  )


def main(config: Any) -> None:
  """Start the server and wait for termination."""
  start_server(config).wait_for_termination()
