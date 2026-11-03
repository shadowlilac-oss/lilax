"""Staging trained weights for rollout through Raiden."""

from __future__ import annotations

from typing import Any, Optional

from absl import logging
import jax
import jax.numpy as jnp
from lilax._backend.configs import pyconfig
from lilax._backend.integration.tunix.weight_mapping import raiden_unscan
from lilax._backend.integration.vllm.convert_utils import (
    is_verify_weights_enabled,
    resolve_prefuse_moe_weights,
    resolve_rollout_tp,
)
from tunix.dist_rl.weight_sync import raiden_synchronizer


class WeightSyncStager:
  """Converts parameter state for the rollout backend and stages it once per train step."""

  def __init__(self, config: pyconfig.HyperParameters) -> None:
    self._config = config
    self._raiden_sync: Any = None
    self._last_staged_step: Optional[int] = None
    self._staged_metadata: Any = None
    self._target_state: Any = None
    self._use_weight_converter = bool(self._config.use_weight_converter)
    self._rollout_backend = self._config.rollout_backend
    rollout_tp = resolve_rollout_tp(self._config)
    kv_tp = self._config.kv_tp_size or rollout_tp
    moe_tp = self._config.moe_mlp_tp_size or rollout_tp
    prefuse_moe = resolve_prefuse_moe_weights(self._config)
    if self._use_weight_converter:
      from lilax._backend.integration.vllm.weight_converter import WeightConverter

      self._weight_converter = WeightConverter(
          config=self._config,
          tp=rollout_tp,
          kv_tp_size=kv_tp,
          moe_mlp_tp_size=moe_tp,
          prefuse_moe_weights=prefuse_moe,
          rollout_backend=self._rollout_backend,
          debug=self._config.weight_sync_debug,
      )
    else:
      self._weight_converter = None

  def set_target_state(self, target_state: Any) -> None:
    self._target_state = target_state

  def reuse_staged(self, train_step: int) -> Any:
    """Returns the metadata already staged for `train_step`, or None when it must be staged again."""
    if self._raiden_sync is not None and self._last_staged_step == train_step and self._staged_metadata is not None:
      logging.info(
          "Trainer reusing staged weight sync for step %d (%d variables)",
          train_step, sum(len(metadata.variables) for metadata in self._staged_metadata),
      )
      return self._staged_metadata
    return None

  def _convert(self, params_state: Any) -> Any:
    if self._use_weight_converter:
      return self._weight_converter.convert(params_state, target_state=self._target_state)
    params_state = jax.tree_util.tree_map(
        lambda value: value.astype(jnp.bfloat16)
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating) else value,
        params_state,
    )
    if self._config.scan_layers:
      return raiden_unscan.unscan_layers(
          params_state,
          num_layers=self._config.num_decoder_layers,
          scan_axis=self._config.param_scan_axis,
          cycle_interval=self._config.inhomogeneous_layer_cycle_interval,
      )
    return params_state

  def stage(self, params_state: Any, train_step: int) -> Any:
    converted_state = self._convert(params_state)
    del params_state
    if self._raiden_sync is None:
      self._raiden_sync = raiden_synchronizer.RaidenSynchronizer(
          job_name="trainer", worker_index=jax.process_index(), auto_h2d=False, parallelism=4
      )
    self._raiden_sync.bind(converted_state)
    del converted_state
    if self._raiden_sync.active:
      self._raiden_sync.d2h()
    verify_weights = is_verify_weights_enabled()
    if verify_weights:
      logging.info("Source weights checksums: %s", self._raiden_sync.checksums())
    all_metadata = self._raiden_sync.work_unit_metadata_all()
    total_variables = sum(len(metadata.variables) for metadata in all_metadata)
    logging.info(
        "Trainer prepared weight sync for step %d: registered %d work unit(s) with %d variables on mesh %s",
        train_step, len(all_metadata), total_variables, all_metadata[0].mesh_axes if all_metadata else (),
    )
    self._last_staged_step = train_step
    self._staged_metadata = all_metadata
    return all_metadata

  def release(self) -> None:
    self._last_staged_step = None
    self._staged_metadata = None
    if self._raiden_sync:
      logging.vlog(1, "Trainer Raiden metrics: %s", self._raiden_sync.metrics())

  def close(self) -> None:
    if self._raiden_sync:
      if hasattr(self._raiden_sync, "close"):
        self._raiden_sync.close()
      self._raiden_sync = None
    self._last_staged_step = None
    self._staged_metadata = None
