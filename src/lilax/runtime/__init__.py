"""Explicit runtime initialization and training mesh contexts."""

from contextlib import contextmanager
from functools import partial

import jax
from flax.core.spmd import logical_axis_rules

from lilax._backend.configs.runtime import resolve_initialized_config, resolve_runtime_config
from lilax._backend.distributed import device_mesh
from lilax._backend.utils import elastic_utils
from lilax._backend.trainers.post_train.rl.setup import device_allocation
from lilax.configuration.schema import RLConfig


class Runtime:
  """Own distributed initialization and the typed training mesh."""

  def __init__(self, config, devices=None):
    self.public_config = config
    self.config = config
    self.devices = devices
    self.mesh = None
    self._overrides = {}

  @classmethod
  def for_rl(cls, config, *, devices=None):
    """Partition RL devices before initializing the trainer runtime."""
    if not isinstance(config, RLConfig):
      raise TypeError("RL runtime configuration must be an inert RLConfig.")
    native_config = resolve_runtime_config(config)
    trainer_config, sampler_config, trainer_devices, sampler_devices = (
        device_allocation.setup_configs_and_devices_from_config(
            native_config, resolve_config=lambda overrides: resolve_runtime_config(config, overrides=overrides),
            devices=devices,
        )
    )
    runtime = cls(config, devices=trainer_devices)
    runtime.config = trainer_config
    return runtime.initialize(), sampler_config, sampler_devices

  def resolve_config(self) -> "Runtime":
    """Resolve native configuration without creating the training mesh."""
    if self.config is self.public_config:
      self.config = resolve_runtime_config(self.public_config)
    return self

  def initialize(
      self, elastic_retry_cache=None, *, overrides=None, train_dataset_provided=False, eval_dataset_provided=False,
  ):
    """Resolve native configuration and build or reuse the training mesh."""
    combined_overrides = {**self._overrides, **(overrides or {})}
    if self.config is self.public_config:
      config = resolve_runtime_config(
          self.public_config, overrides=combined_overrides, train_dataset_provided=train_dataset_provided,
          eval_dataset_provided=eval_dataset_provided,
      )
    elif overrides or train_dataset_provided or eval_dataset_provided:
      config = resolve_initialized_config(
          self.public_config, self.config, overrides=combined_overrides, train_dataset_provided=train_dataset_provided,
          eval_dataset_provided=eval_dataset_provided,
      )
    else:
      config = self.config
    if self.mesh is not None and any(
        getattr(config, name) != getattr(self.config, name)
        for name in (
            "mesh_axes", "ici_parallelism", "dcn_parallelism", "data_sharding", "logical_axis_rules",
            "num_slices", "subslice_shape", "enable_single_controller", "allow_split_physical_axes", "custom_mesh",
            "optimize_mesh_for_tpu_v6e", "use_lineage", "elastic_enabled", "inference_benchmark_test", "shard_mode",
        )
    ):
      raise ValueError("Training configuration cannot change an initialized mesh or its sharding.")
    self.config = config
    self._overrides = combined_overrides
    if self.mesh is None:
      self.mesh = elastic_utils.get_or_build(
          elastic_retry_cache, "mesh", partial(device_mesh.get_mesh_from_config, self.config, self.devices)
      )
    return self

  @contextmanager
  def context(self, *, evaluation=False):
    """Activate the training mesh and the selected logical axis rules."""
    rules = self.config.logical_axis_rules_for_eval if evaluation else self.config.logical_axis_rules
    with jax.set_mesh(self.mesh), logical_axis_rules(rules):
      yield
