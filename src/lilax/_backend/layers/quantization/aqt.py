"""AQT quantized dot_general and einsum (github.com/google/aqt) for Linen modules."""

import functools
import re
from typing import ClassVar, Tuple, Sequence
from dataclasses import dataclass

from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2.flax import aqt_flax
from aqt.jax.v2 import tiled_dot_general

import jax.numpy as jnp

import flax.linen as nn

from lilax._backend.layers.quantization.aqt_configs import DEFAULT


def _tiling_fn(lhs, rhs, dimension_numbers, tile_size):
  """apply tiling function"""
  del lhs, rhs

  (lhs_ca, rhs_ca), _ = dimension_numbers
  ret = tiled_dot_general.Cfg(
      lhs=tiled_dot_general.TensorTiling(contraction_axes=[], remaining_axes=[]),
      rhs=tiled_dot_general.TensorTiling(contraction_axes=[], remaining_axes=[]),
  )

  for lhs_idx, rhs_idx in zip(lhs_ca, rhs_ca):
    ret.lhs.contraction_axes.append(tiled_dot_general.AxisTiling(axis=lhs_idx, tile_size=tile_size, tile_count=None))
    ret.rhs.contraction_axes.append(tiled_dot_general.AxisTiling(axis=rhs_idx, tile_size=tile_size, tile_count=None))

  return ret


def _rhs_axis_metadata_wrapper(
    x: jnp.ndarray,
    tile_map,
    no_sharding_axis: Sequence[int],
    mesh_axes: Tuple[str, ...],
    is_tiled: bool,
    replicate_scale: bool = False,
):
  """right-hand-side axis metadata wrapper"""
  if replicate_scale:
    # Temporarily using the shape to identify the scale.
    # TODO: remove the replication once the 2d sharding quantization
    # works as expected.
    if len(x.shape) == 1:
      return nn.with_logical_partitioning((lambda: x), tuple(None for _ in mesh_axes))()

  mesh_axes = list(mesh_axes)  # pyrefly: ignore[bad-assignment]
  if is_tiled:
    # tile_map is a mapping between original rank and a list of new, tiled rank.
    if len(mesh_axes) < len(tile_map):
      mesh_axes = [None] * (len(tile_map) - len(mesh_axes)) + mesh_axes  # pyrefly: ignore[unsupported-operation]
    new_mesh_axes = [None] * len(x.shape)
    for orig_rank, new_rank in tile_map.items():
      assert new_rank
      assert len(new_rank) <= 2
      new_mesh_axes[new_rank[-1]] = mesh_axes[orig_rank]
    mesh_axes = new_mesh_axes  # pyrefly: ignore[bad-assignment]

  if mesh_axes is not None and len(mesh_axes) > 0:
    for no_shard_idx in no_sharding_axis:
      if no_shard_idx < len(mesh_axes):
        mesh_axes[no_shard_idx] = None  # pyrefly: ignore[unsupported-operation]

  return nn.with_logical_partitioning((lambda: x), mesh_axes)()


@dataclass
class AqtQuantization:
  """Configures AQT quantization github.com/google/aqt."""

  # Declared, not inherited: this class sits outside the Quantization hierarchy. AQT
  # sets rng_type="jax.uniform" and stochastic rounding draws at apply time.
  needs_apply_rngs: ClassVar[bool] = True

  quant_dg: aqt_config.DotGeneral
  quant_mode: aqt_flax.QuantMode = aqt_flax.QuantMode.TRAIN
  replicate_scale: bool = False

  def _get_mixed_precision_cfg(self):
    """get configuration for mixed precision"""
    quant_dg = None
    is_tiled = False
    tiling_fn = None
    # pylint: disable=protected-access
    module_path = "/".join(nn.module._context.module_stack[-1].path)  # pyrefly: ignore[missing-attribute]
    tile_size = -1
    for layer_name_re, layer_quant_dg in self.quant_dg.items():  # pyrefly: ignore[missing-attribute]
      if re.fullmatch(layer_name_re, module_path):
        quant_dg, tile_size = layer_quant_dg
    if quant_dg is None:
      quant_dg, tile_size = self.quant_dg[DEFAULT]  # pyrefly: ignore[bad-index]
    if tile_size != -1:
      is_tiled = True
      tiling_fn = functools.partial(_tiling_fn, tile_size=tile_size)
    return quant_dg, is_tiled, tiling_fn

  def _get_rhs_axis_metadata_wrapper(
      self, mesh_axes: Tuple[str, ...] = (), is_tiled: bool = False, replicate_scale: bool = False
  ):
    if self.quant_mode == aqt_flax.QuantMode.CONVERT:
      return None
    return functools.partial(
        _rhs_axis_metadata_wrapper, mesh_axes=mesh_axes, is_tiled=is_tiled, replicate_scale=replicate_scale
    )

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns dot_general configured with aqt params."""
    if isinstance(self.quant_dg, dict):
      quant_dg, is_tiled, tiling_fn = self._get_mixed_precision_cfg()
    else:
      quant_dg, is_tiled, tiling_fn = self.quant_dg, False, None
    rhs_axis_metadata_wrapper = self._get_rhs_axis_metadata_wrapper(
        mesh_axes, is_tiled, replicate_scale=self.replicate_scale
    )
    # module_path = "/".join(nn.module._context.module_stack[-1].path)
    # print(f"quant_dg: {quant_dg}, is_tiled: {is_tiled}, module_path: {module_path}")
    aqt_dg_cls = functools.partial(
        aqt_flax.AqtDotGeneral,
        quant_dg,
        rhs_quant_mode=self.quant_mode,
        lhs_freeze_mode=aqt_flax.FreezerMode.NONE,
        rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
        rhs_axis_metadata_wrapper=rhs_axis_metadata_wrapper,
        use_legacy_freezer=False,
        tiling_fn=tiling_fn,
    )
    return aqt_dg_cls

  def einsum(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns einsum configured with aqt params."""
    if isinstance(self.quant_dg, dict):
      quant_dg, is_tiled, tiling_fn = self._get_mixed_precision_cfg()
    else:
      quant_dg, is_tiled, tiling_fn = self.quant_dg, False, None

    rhs_axis_metadata_wrapper = self._get_rhs_axis_metadata_wrapper(
        mesh_axes, is_tiled, replicate_scale=self.replicate_scale
    )
    aqt_einsum = functools.partial(
        aqt_flax.AqtEinsum(
            cfg=quant_dg,
            rhs_quant_mode=self.quant_mode,
            lhs_freeze_mode=aqt_flax.FreezerMode.NONE,
            rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
            rhs_axis_metadata_wrapper=rhs_axis_metadata_wrapper,
            use_legacy_freezer=False,
            tiling_fn=tiling_fn,
        )
    )
    return aqt_einsum
