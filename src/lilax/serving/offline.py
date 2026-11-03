"""Native continuous-batching offline inference."""

from typing import Any

import jax
from jax.sharding import Mesh

from lilax._backend.configs.runtime import resolve_runtime_config
from lilax._backend.inference.offline_engine import CompletionOutput, InputData
from lilax._backend.inference.offline_engine import OfflineEngine as _OfflineEngine
from lilax.configuration import MaxTextConfig, RLConfig
from lilax.runtime import Runtime


class OfflineEngine(_OfflineEngine):
  """Run native offline inference using a public or resolved configuration."""

  @staticmethod
  def create_mesh(devices, config):
    if isinstance(config, Runtime):
      config.resolve_config()
      config = config.config
    elif isinstance(config, (MaxTextConfig, RLConfig)):
      config = resolve_runtime_config(config)
    return _OfflineEngine.create_mesh(devices, config)

  def __init__(
      self,
      config: Any,
      params: Any = None,
      enable_batch_prefill: bool = False,
      min_decode_steps: int = 10,
      tokenizer: Any = None,
      eos_ids: list[int] | None = None,
      prefill_lengths: list[int] | str = "auto",
      batch_prefill_max_batch_size: int = 16,
      mesh: Mesh = None,
      rng: jax.random.PRNGKey = None,
      debug: bool = False,
  ):
    if isinstance(config, Runtime):
      config.resolve_config()
      if mesh is None and config.devices is not None:
        mesh = _OfflineEngine.create_mesh(config.devices, config.config)
      config = config.config
    elif isinstance(config, (MaxTextConfig, RLConfig)):
      config = resolve_runtime_config(config)
    super().__init__(
        config=config,
        params=params,
        enable_batch_prefill=enable_batch_prefill,
        min_decode_steps=min_decode_steps,
        tokenizer=tokenizer,
        eos_ids=eos_ids,
        prefill_lengths=prefill_lengths,
        batch_prefill_max_batch_size=batch_prefill_max_batch_size,
        mesh=mesh,
        rng=rng,
        debug=debug,
    )


__all__ = ["CompletionOutput", "InputData", "OfflineEngine"]
