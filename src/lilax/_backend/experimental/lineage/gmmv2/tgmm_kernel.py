# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TGMM kernel."""

import dataclasses
import functools
import math
from typing import Any, Callable, Tuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from lilax._backend.experimental.lineage.gmmv2 import gmmv2_kernel as gmm_v2
from lilax._backend.experimental.lineage.gmmv2 import relayout


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class OperandRef:
  """Bundles a kernel operand with its optional scale."""

  value: Any
  scale: Any | None = None
  quant_scale: Any | None = None


TileTgmmFn = Callable[..., gmm_v2.TileSizes]


def get_scope_name(cfgs: gmm_v2.GmmConfigs) -> str:
  dims = cfgs.dims
  tiles = cfgs.tiles
  return (
      f"tgmm_v2-g_{dims.size_group}-m_{dims.size_m}-k_{dims.size_k}-act_{cfgs.fuse_act}"
      f"-n_{dims.size_n}-tm_{tiles.tile_m}-tk_{tiles.tile_k}-tn_{tiles.tile_n}"
  )


def get_cost_estimate(cfgs: gmm_v2.GmmConfigs) -> pl.CostEstimate:
  dims = cfgs.dims
  flops = 2 * dims.size_m * dims.size_k * dims.size_n
  lhs_bytes = dims.size_m * dims.size_k * cfgs.lhs_cfgs.dtype.itemsize
  rhs_bytes = dims.size_m * dims.size_n * cfgs.rhs_cfgs.dtype.itemsize
  out_bytes = dims.size_group * dims.size_k * dims.size_n * cfgs.out_dtype.itemsize
  return pl.CostEstimate(flops=flops, bytes_accessed=lhs_bytes + rhs_bytes + out_bytes, transcendentals=0)


def calculate_tgmm_tiling(
    dims: gmm_v2.Dimensions, lhs_cfgs: gmm_v2.InputConfigs, rhs_cfgs: gmm_v2.InputConfigs,
    vmem_limit_bytes: int, out_dtype: jnp.dtype, acc_dtype: jnp.dtype, target_zero_ref_bytes: int,
) -> gmm_v2.TileSizes:
  """Calculate optimal tile sizes for TGMM kernel."""
  bf16_bf16_tile_m = 256
  tile_m = min(bf16_bf16_tile_m, dims.size_m)
  tile_m = max(tile_m, dims.size_lhs_sublane)
  num_lanes = pltpu.get_tpu_info().num_lanes

  def _align_3d(size: int) -> int:
    return 8 * num_lanes if size >= 8 * num_lanes else size

  k_align = _align_3d(dims.size_k) if lhs_cfgs.is_3d else num_lanes
  n_align = _align_3d(dims.size_n) if rhs_cfgs.is_3d else num_lanes
  tile_n = gmm_v2.align_to(dims.size_n, n_align)
  tile_n_lower_bound = pltpu.get_tpu_info().mxu_column_size * 2
  tile_n_lower_bound = min(tile_n_lower_bound, dims.size_n)
  tile_n_lower_bound = max(tile_n_lower_bound, n_align)
  tile_k = gmm_v2.align_to(dims.size_k, k_align)

  def _slab_padded(tile: int, dtype) -> int:
    sublane_tiling = pltpu.get_tpu_info().get_sublane_tiling(dtype)
    return gmm_v2.align_to(tile // num_lanes, sublane_tiling) * num_lanes

  def within_vmem_limit(tile_m, tile_k, tile_n):
    acc_bytes = jax.dtypes.itemsize_bits(acc_dtype) // 8
    out_bytes = jax.dtypes.itemsize_bits(out_dtype) // 8
    lhs_bytes = jax.dtypes.itemsize_bits(lhs_cfgs.dtype) // 8
    rhs_bytes = jax.dtypes.itemsize_bits(rhs_cfgs.dtype) // 8
    num_buffers = 2
    tk_vmem = _slab_padded(tile_k, lhs_cfgs.dtype) if lhs_cfgs.is_3d else tile_k
    tn_vmem = _slab_padded(tile_n, rhs_cfgs.dtype) if rhs_cfgs.is_3d else tile_n
    budget = (
        tile_k * tile_n * (acc_bytes + num_buffers * out_bytes)
        + num_buffers * (tile_m * tk_vmem * lhs_bytes)
        + 2 * (tile_m * tile_k * lhs_bytes)
        + num_buffers * (tile_m * tn_vmem * rhs_bytes)
        + 1 * (tile_m * tile_n * rhs_bytes) + target_zero_ref_bytes
    )
    return budget <= vmem_limit_bytes

  def _shrink_dim(size: int, align: int, lower_bound: int, fits_fn: Callable[[int], bool]) -> int:
    aligned_size = gmm_v2.align_to(size, align)
    num_units = aligned_size // align
    for n_tiles in range(1, num_units + 1):
      if num_units % n_tiles != 0:
        continue
      candidate = (num_units // n_tiles) * align
      if candidate < lower_bound:
        break
      if fits_fn(candidate):
        return candidate
    curr_tile = aligned_size
    n_tiles = 1
    prev_tile = curr_tile
    while not fits_fn(curr_tile):
      n_tiles += 1
      curr_tile = gmm_v2.align_to(size, n_tiles * align) // n_tiles
      if curr_tile < lower_bound or curr_tile >= prev_tile:
        break
      prev_tile = curr_tile
    if curr_tile < lower_bound:
      n_tiles -= 1
      curr_tile = gmm_v2.align_to(size, n_tiles * align) // n_tiles
    return curr_tile

  if lhs_cfgs.is_3d and not rhs_cfgs.is_3d:
    tile_k = _shrink_dim(dims.size_k, k_align, k_align, lambda tk: within_vmem_limit(tile_m, tk, tile_n))
    tile_n = _shrink_dim(dims.size_n, n_align, tile_n_lower_bound, lambda tn: within_vmem_limit(tile_m, tile_k, tn))
  else:
    tile_n = _shrink_dim(dims.size_n, n_align, tile_n_lower_bound, lambda tn: within_vmem_limit(tile_m, tile_k, tn))
    tile_k = _shrink_dim(dims.size_k, k_align, k_align, lambda tk: within_vmem_limit(tile_m, tk, tile_n))
  if not within_vmem_limit(tile_m, tile_k, tile_n):
    raise ValueError(
        f"Could not find valid tile sizes for tgmm. dims={dims},"
        f" tiles=({tile_m},{tile_k},{tile_n}), vmem={vmem_limit_bytes}"
    )
  max_num_buckets = 4
  bucket_base = tile_m
  for _ in range(1, max_num_buckets):
    new_tile_m = tile_m + bucket_base
    if new_tile_m > dims.size_m:
      break
    if not within_vmem_limit(new_tile_m, tile_k, tile_n):
      break
    tile_m = new_tile_m
  return gmm_v2.TileSizes(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, bucket_base=bucket_base)


def _operand_size(x: jax.Array, name: str) -> int:
  """Returns the flattened minor size of a 2D or 3D operand."""
  if x.ndim == 2:
    return x.shape[1]
  num_lanes = pltpu.get_tpu_info().num_lanes
  if x.ndim != 3 or x.shape[-1] != num_lanes:
    raise ValueError(f"tgmm {name} must be [M, S] or [M, S // {num_lanes}, {num_lanes}];" f" got {x.shape}.")
  if x.dtype != jnp.bfloat16 and not gmm_v2.is_fp8(x.dtype):
    raise ValueError(
        f"3D tgmm {name} is only supported for bfloat16 and fp8 (the in-VMEM"
        f" relayout packs 16 / 8-bit rows into words); got {x.dtype}."
    )
  return x.shape[1] * num_lanes


def _sublane_size(lhs_dtype, rhs_dtype, size_m: int, lhs_quant_dtype=None, rhs_quant_dtype=None) -> int:
  """Row block (sublane tiling) the kernel reshapes both operands by."""
  tpu_info = pltpu.get_tpu_info()
  dtypes = [dt for dt in (lhs_dtype, rhs_dtype, lhs_quant_dtype, rhs_quant_dtype) if dt is not None]
  sublane = max(tpu_info.get_sublane_tiling(dt) for dt in dtypes)
  return min(sublane, size_m)


def make_tgmm_configs(
    lhs: jax.Array, rhs: jax.Array, rhs_scale: jax.Array, group_sizes: jax.Array, num_actual_groups: int,
    *, tile_info: gmm_v2.TileSizes | TileTgmmFn, vmem_limit_bytes: int | None, out_dtype: jnp.dtype,
    acc_dtype: jnp.dtype | None, target_zero_ref_bytes: int, disable_multi_core_mode: bool = False,
    lhs_scale: jax.Array | None = None, lhs_quant_dtype: jnp.dtype | None = None,
    rhs_quant_scale: jax.Array | None = None, rhs_quant_dtype: jnp.dtype | None = None,
):
  """Fills the GMM config for the TGMM kernel."""
  assert out_dtype, "out_dtype cannot be None"
  assert lhs.shape[0] == rhs.shape[0], (
      f"lhs and rhs m-dim mismatch: {lhs.shape[0]}!={rhs.shape[0]} {lhs.shape}" f" vs {rhs.shape}"
  )
  size_m = lhs.shape[0]
  size_k = _operand_size(lhs, "lhs")
  size_n = _operand_size(rhs, "rhs")
  if rhs_scale is not None:
    if rhs_scale.ndim == 3 and rhs_scale.shape[0] > 1:
      raise NotImplementedError(
          "tgmm_v2 only supports per-N rhs_scale with shape (1, 1, size_n);"
          f" got {rhs_scale.shape}, which implies {rhs_scale.shape[0]}"
          " sub-channel quant blocks along the m (reduction) dimension."
          " Sub-channel quantization is not implemented."
      )
    assert rhs_scale.shape == (1, 1, size_n), (
        "expecting rhs_scale.shape to be (1, 1, size_n) but got" f" {rhs_scale.shape}"
    )
  if (lhs_scale is None) != (lhs_quant_dtype is None):
    raise ValueError("lhs_scale and lhs_quant_dtype must be given together.")
  if lhs_scale is not None and lhs_scale.shape != (1, 1):
    raise ValueError(f"lhs_scale must be [1, 1]; got {lhs_scale.shape}.")
  if (rhs_quant_scale is None) != (rhs_quant_dtype is None):
    raise ValueError("rhs_quant_scale and rhs_quant_dtype must be given together.")
  if rhs_quant_scale is not None and rhs_quant_scale.shape != (1, 1):
    raise ValueError(f"rhs_quant_scale must be [1, 1]; got {rhs_quant_scale.shape}.")
  size_lhs_sublane = _sublane_size(lhs.dtype, rhs.dtype, size_m, lhs_quant_dtype=lhs_quant_dtype, rhs_quant_dtype=rhs_quant_dtype)
  dims = gmm_v2.Dimensions(
      size_m=size_m, size_k=size_k, size_n=size_n, size_group=num_actual_groups,
      size_lhs_group=group_sizes.shape[0], size_lhs_sublane=size_lhs_sublane,
  )
  rhs_cfgs = gmm_v2.InputConfigs(
      quant_dtype=rhs_quant_dtype, quant_block_size=size_m, dtype=rhs.dtype,
      has_scale=(rhs_scale is not None), is_3d=rhs.ndim == 3,
  )
  lhs_cfgs = gmm_v2.InputConfigs(
      quant_dtype=lhs_quant_dtype, quant_block_size=-1, dtype=lhs.dtype,
      has_scale=lhs_scale is not None, is_3d=lhs.ndim == 3,
  )
  fuse_act = None
  if acc_dtype is None:
    acc_dtype = jnp.float32.dtype
  if isinstance(tile_info, gmm_v2.TileSizes):
    tiles = tile_info
  else:
    tiles = tile_info(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, out_dtype, acc_dtype, target_zero_ref_bytes)
  assert tiles.tile_m % tiles.bucket_base == 0, (
      f"tile_m ({tiles.tile_m}) must be divisible by bucket_base" f" ({tiles.bucket_base})"
  )
  num_lanes = pltpu.get_tpu_info().num_lanes
  if lhs_cfgs.is_3d:
    if tiles.tile_k % num_lanes != 0:
      raise ValueError(f"3D lhs requires tile_k % {num_lanes} == 0; got {tiles.tile_k=}.")
    relayout.check_tile_d0(tiles.tile_k // num_lanes, full_d0=dims.size_k // num_lanes, pack=relayout.packing(lhs.dtype))
  if rhs_cfgs.is_3d:
    if tiles.tile_n % num_lanes != 0:
      raise ValueError(f"3D rhs requires tile_n % {num_lanes} == 0; got {tiles.tile_n=}.")
    relayout.check_tile_d0(tiles.tile_n // num_lanes, full_d0=dims.size_n // num_lanes, pack=relayout.packing(rhs.dtype))
  return gmm_v2.GmmConfigs(
      dims=dims, tiles=tiles, lhs_cfgs=lhs_cfgs, rhs_cfgs=rhs_cfgs, out_dtype=jnp.dtype(out_dtype),
      acc_dtype=jnp.dtype(acc_dtype), zero_init=False, fuse_act=fuse_act,
      disable_multi_core_mode=disable_multi_core_mode,
  )


def tgmm_inner_kernel(
    tiled_lhs_ref: OperandRef, tiled_rhs_ref: OperandRef, tiled_out_ref: jax.Array, acc_ref: jax.Array,
    metadata_ref: gmm_v2.MetadataRef, acc_in_ref: jax.Array | None = None, acc_in_sem: jax.Array | None = None,
    *, cfgs: gmm_v2.GmmConfigs,
):
  """Inner kernel for TGMM computation."""
  tiled_rhs_scale_ref = tiled_rhs_ref.scale
  tiled_rhs_value_ref = tiled_rhs_ref.value
  tiled_lhs_value_ref = tiled_lhs_ref.value
  tile_k = cfgs.tiles.tile_k
  tile_n = cfgs.tiles.tile_n
  num_lanes = pltpu.get_tpu_info().num_lanes

  def _load_lhs(bucket_m: int) -> jax.Array:
    if cfgs.lhs_cfgs.is_3d:
      return relayout.load_3d_as_2d(tiled_lhs_value_ref, bucket_m, tile_k // num_lanes)
    return tiled_lhs_value_ref.reshape(-1, tile_k)[:bucket_m]

  def _load_rhs(bucket_m: int) -> jax.Array:
    if cfgs.rhs_cfgs.is_3d:
      return relayout.load_3d_as_2d(tiled_rhs_value_ref, bucket_m, tile_n // num_lanes)
    return tiled_rhs_value_ref.reshape(-1, tile_n)[:bucket_m]

  gm_id = pl.program_id(2)
  m_start = metadata_ref.gm_id_to_m_offset[gm_id]
  m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
  m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane
  m_start_local = m_start - m_offset
  m_end_local = m_end - m_offset

  def _acc_in_copy():
    assert acc_in_ref is not None and acc_in_sem is not None
    n_start = pl.program_id(0) * tile_n
    size_n = jnp.minimum(tile_n, acc_in_ref.shape[2] - n_start)
    size_n = pl.multiple_of(size_n, num_lanes)
    return pltpu.make_async_copy(
        acc_in_ref.at[cur_group_id, pl.ds(pl.program_id(1) * tile_k, tile_k), pl.ds(n_start, size_n)],
        tiled_out_ref.at[:, pl.ds(0, size_n)], acc_in_sem,
    )

  def _matmul(is_new_group: bool, is_group_changing: bool, bucket_m: int):
    if acc_in_ref is not None and is_new_group:
      _acc_in_copy().start()
    lhs_val = _load_lhs(bucket_m)
    if is_new_group and is_group_changing:
      lhs_iota = lax.broadcasted_iota(jnp.int32, (bucket_m, tile_k), 0)
      lhs_mask = jnp.logical_and(m_start_local <= lhs_iota, lhs_iota < m_end_local)
      lhs_masked = jnp.where(lhs_mask, lhs_val, 0)
    elif is_new_group:
      lhs_iota = lax.broadcasted_iota(jnp.int32, (bucket_m, tile_k), 0)
      lhs_masked = jnp.where(m_start_local <= lhs_iota, lhs_val, 0)
    elif is_group_changing:
      lhs_iota = lax.broadcasted_iota(jnp.int32, (bucket_m, tile_k), 0)
      lhs_masked = jnp.where(lhs_iota < m_end_local, lhs_val, 0)
    else:
      lhs_masked = lhs_val
    if cfgs.lhs_cfgs.has_scale:
      q_dtype = cfgs.lhs_cfgs.quant_dtype
      q_max = float(jnp.finfo(q_dtype).max)
      assert tiled_lhs_ref.scale is not None
      inv_scale = 1.0 / tiled_lhs_ref.scale[...]
      lhs_masked = jnp.clip(lhs_masked.astype(jnp.float32) * inv_scale, -q_max, q_max).astype(q_dtype)
    rhs_val = _load_rhs(bucket_m)
    if is_new_group and is_group_changing:
      rhs_iota = lax.broadcasted_iota(jnp.int32, (bucket_m, tile_n), 0)
      rhs_mask = jnp.logical_and(m_start_local <= rhs_iota, rhs_iota < m_end_local)
      rhs_masked = jnp.where(rhs_mask, rhs_val, 0)
    elif is_new_group:
      rhs_iota = lax.broadcasted_iota(jnp.int32, (bucket_m, tile_n), 0)
      rhs_masked = jnp.where(m_start_local <= rhs_iota, rhs_val, 0)
    elif is_group_changing:
      rhs_iota = lax.broadcasted_iota(jnp.int32, (bucket_m, tile_n), 0)
      rhs_masked = jnp.where(rhs_iota < m_end_local, rhs_val, 0)
    else:
      rhs_masked = rhs_val
    if cfgs.rhs_cfgs.quant_dtype is not None:
      q_dtype = cfgs.rhs_cfgs.quant_dtype
      q_max = float(jnp.finfo(q_dtype).max)
      assert tiled_rhs_ref.quant_scale is not None
      inv_scale = 1.0 / tiled_rhs_ref.quant_scale[...]
      rhs_masked = jnp.clip(rhs_masked.astype(jnp.float32) * inv_scale, -q_max, q_max).astype(q_dtype)
    acc = jax.lax.dot_general(lhs_masked, rhs_masked, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    if not is_new_group:
      acc += acc_ref[...]
    if is_group_changing:
      if cfgs.lhs_cfgs.has_scale:
        assert tiled_lhs_ref.scale is not None
        acc *= tiled_lhs_ref.scale[...]
      if cfgs.rhs_cfgs.has_scale:
        scale_slice = tiled_rhs_scale_ref[0]
        acc *= scale_slice
      if acc_in_ref is None:
        tiled_out_ref[...] = acc.astype(tiled_out_ref.dtype)
        return
      acc_ref[...] = acc
      _acc_in_copy().wait()
      chunk = math.gcd(tile_k, 512)

      @pl.loop(0, tile_k, step=chunk)
      def _(i):
        rows = pl.ds(pl.multiple_of(i, chunk), chunk)
        tiled_out_ref[rows] = (acc_ref[rows] + tiled_out_ref[rows].astype(jnp.float32)).astype(tiled_out_ref.dtype)
    else:
      acc_ref[...] = acc

  prev_gm_id = jnp.where(gm_id > 0, gm_id - 1, 0)
  is_first_gm = gm_id == 0
  group_id_changed = metadata_ref.gm_id_to_group_id[gm_id] != metadata_ref.gm_id_to_group_id[prev_gm_id]
  new_group = jnp.logical_or(is_first_gm, group_id_changed)
  is_last_gm = gm_id == (pl.num_programs(2) - 1)
  next_gm_id = jnp.where(is_last_gm, gm_id, gm_id + 1)
  next_group_id = metadata_ref.gm_id_to_group_id[next_gm_id]
  cur_group_id = metadata_ref.gm_id_to_group_id[gm_id]
  group_is_changing = jnp.logical_or(is_last_gm, cur_group_id != next_group_id)
  bucket_base = cfgs.tiles.bucket_base
  num_buckets = cfgs.tiles.tile_m // bucket_base
  bucket_idx = jnp.maximum(0, (m_end_local - 1) // bucket_base)

  def run_changing_step(bucket_m: int):
    @jax.named_scope(f"bm{bucket_m}_matmul_new_group_and_changing")
    def matmul_new_group_and_changing():
      _matmul(is_new_group=True, is_group_changing=True, bucket_m=bucket_m)

    @jax.named_scope(f"bm{bucket_m}_matmul_group_changing")
    def matmul_group_changing():
      _matmul(is_new_group=False, is_group_changing=True, bucket_m=bucket_m)

    lax.cond(new_group, matmul_new_group_and_changing, matmul_group_changing)

  def run_full_tile_step():
    full_m = cfgs.tiles.tile_m

    @jax.named_scope(f"bm{full_m}_matmul_new_group")
    def matmul_new_group():
      _matmul(is_new_group=True, is_group_changing=False, bucket_m=full_m)

    @jax.named_scope(f"bm{full_m}_matmul")
    def matmul():
      _matmul(is_new_group=False, is_group_changing=False, bucket_m=full_m)

    lax.cond(new_group, matmul_new_group, matmul)

  lax.cond(
      group_is_changing,
      lambda: lax.switch(bucket_idx, [functools.partial(run_changing_step, bucket_m=bucket_base * (i + 1)) for i in range(num_buckets)]),
      run_full_tile_step,
  )


class TgmmIndexMaps:
  """Index maps for TGMM kernel."""

  def __init__(self, metadata_ref: gmm_v2.MetadataRef, cfgs: gmm_v2.GmmConfigs):
    self.metadata_ref = metadata_ref
    self.cfgs = cfgs

  def _rows(self, gm_id: jax.Array):
    m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]
    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    row_size = row_end - row_start
    return pl.ds(row_start, row_size)

  def lhs_index_map(self, n_id: jax.Array, k_id: jax.Array, gm_id: jax.Array):
    del n_id
    if self.cfgs.lhs_cfgs.is_3d:
      return (self._rows(gm_id), 0, k_id, 0)
    return (self._rows(gm_id), 0, k_id)

  def rhs_index_map(self, n_id: jax.Array, k_id: jax.Array, gm_id: jax.Array):
    del k_id
    if self.cfgs.rhs_cfgs.is_3d:
      return (self._rows(gm_id), 0, n_id, 0)
    return (self._rows(gm_id), 0, n_id)

  def rhs_scale_index_map(self, n_id: jax.Array, k_id: jax.Array, gm_id: jax.Array):
    del k_id, gm_id
    return (0, 0, n_id)

  def out_index_map(self, n_id: jax.Array, k_id: jax.Array, gm_id: jax.Array):
    group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
    return (group_id, k_id, n_id)


def generate_tgmm_block_specs(
    metadata_ref: gmm_v2.MetadataRef, cfgs: gmm_v2.GmmConfigs,
) -> Tuple[Tuple[OperandRef, OperandRef], pl.BlockSpec]:
  """Generates block specs for the given lhs, rhs, and out refs."""
  index_map = TgmmIndexMaps(metadata_ref, cfgs)
  num_lanes = pltpu.get_tpu_info().num_lanes
  bounded_slice_gm = pl.BoundedSlice(cfgs.tiles.tile_m // cfgs.dims.size_lhs_sublane)
  sublane = cfgs.dims.size_lhs_sublane
  if cfgs.lhs_cfgs.is_3d:
    lhs_block_shape = (bounded_slice_gm, sublane, cfgs.tiles.tile_k // num_lanes, num_lanes)
  else:
    lhs_block_shape = (bounded_slice_gm, sublane, cfgs.tiles.tile_k)
  lhs_block_spec = pl.BlockSpec(lhs_block_shape, index_map.lhs_index_map)
  lhs_scale_block_spec = None
  if cfgs.lhs_cfgs.has_scale:
    lhs_scale_block_spec = pl.BlockSpec((1, 1), lambda *_: (0, 0))
  lhs_spec = OperandRef(value=lhs_block_spec, scale=lhs_scale_block_spec)
  if cfgs.rhs_cfgs.is_3d:
    rhs_block_shape = (bounded_slice_gm, sublane, cfgs.tiles.tile_n // num_lanes, num_lanes)
  else:
    rhs_block_shape = (bounded_slice_gm, sublane, cfgs.tiles.tile_n)
  rhs_block_spec = pl.BlockSpec(rhs_block_shape, index_map.rhs_index_map)
  rhs_scale_block_spec = None
  if cfgs.rhs_cfgs.has_scale:
    rhs_scale_block_spec = pl.BlockSpec((1, 1, cfgs.tiles.tile_n), index_map.rhs_scale_index_map)
  rhs_quant_scale_block_spec = None
  if cfgs.rhs_cfgs.quant_dtype is not None:
    rhs_quant_scale_block_spec = pl.BlockSpec((1, 1), lambda *_: (0, 0))
  rhs_spec = OperandRef(value=rhs_block_spec, scale=rhs_scale_block_spec, quant_scale=rhs_quant_scale_block_spec)
  out_block_spec = pl.BlockSpec((None, cfgs.tiles.tile_k, cfgs.tiles.tile_n), index_map.out_index_map)
  return (lhs_spec, rhs_spec), out_block_spec


def zero_out_start(lhs_group_sizes_ref, group_offset_ref, out_ref, zero_ref, semaphore_ref):
  """If group_sizes[i]==0, kick off async DMAs to zero out drhs[i]."""
  num_actual_groups, aligned_k, aligned_n = out_ref.shape
  tile_zero_k = zero_ref.shape[0]
  zero_ref = zero_ref.reshape(1, tile_zero_k, -1)
  num_lanes = pltpu.get_tpu_info().num_lanes
  assert aligned_n % num_lanes == 0
  zero_ref[...] = jnp.zeros_like(zero_ref)

  def fill_zero(local_group_id, should_copy):
    should_copy_int = should_copy.astype(int)
    for i in range(pl.cdiv(aligned_k, tile_zero_k)):
      size_k_to_copy = min(tile_zero_k, aligned_k - i * tile_zero_k)
      for j in range(aligned_n // num_lanes):
        src = zero_ref.at[pl.ds(0, should_copy_int), pl.ds(0, size_k_to_copy)]
        dst = out_ref.at[
            pl.ds(local_group_id, should_copy_int), pl.ds(i * tile_zero_k, size_k_to_copy),
            pl.ds(j * num_lanes, num_lanes),
        ]
        pltpu.make_async_copy(src_ref=src, dst_ref=dst, sem=semaphore_ref.at[0]).start(priority=1)
    return 1

  num_groups_to_zero = 0
  group_offset = group_offset_ref[0]
  for local_group_id in range(num_actual_groups):
    global_group_id = local_group_id + group_offset
    should_copy = lhs_group_sizes_ref[global_group_id] == 0
    num_groups_to_zero += should_copy.astype(int)
    fill_zero(local_group_id, should_copy)
  return num_groups_to_zero


def zero_out_end(num_groups_to_zero, out_ref, semaphore_ref):
  """Drain the DMAs started by zero_out_start."""
  dst = out_ref.at[pl.ds(0, num_groups_to_zero),]
  src = dst
  pltpu.make_async_copy(src_ref=src, dst_ref=dst, sem=semaphore_ref.at[0]).wait()


def tgmm_kernel_main(
    lhs_group_sizes_ref, group_offset_ref, lhs_ref, rhs_ref, out_ref,
    acc_ref: jax.Array, metadata_ref: gmm_v2.MetadataRef, zero_ref: jax.Array, semaphore_ref: jax.Array,
    *, cfgs, acc_in_ref: jax.Array | None = None,
):
  """Main kernel function for TGMM computation."""
  out_ref = out_ref.reshape(cfgs.dims.size_group, -1, out_ref.shape[-1])
  if acc_in_ref is not None:
    acc_in_ref = acc_in_ref.reshape(out_ref.shape)
  num_groups_to_zero = None
  if acc_in_ref is None:
    num_groups_to_zero = zero_out_start(lhs_group_sizes_ref, group_offset_ref, out_ref, zero_ref, semaphore_ref)
  num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
  num_n = pl.cdiv(cfgs.dims.size_n, cfgs.tiles.tile_n)
  num_gm = gmm_v2.fill_metadata(lhs_group_sizes_ref, group_offset_ref, metadata_ref, cfgs=cfgs)
  in_specs, out_specs = generate_tgmm_block_specs(metadata_ref, cfgs)
  core_axis_name = None if cfgs.disable_multi_core_mode else "core"
  dimension_semantics = None if cfgs.disable_multi_core_mode else (pltpu.PARALLEL, pltpu.PARALLEL, pltpu.ARBITRARY)
  pipeline_fn = pltpu.emit_pipeline(
      functools.partial(tgmm_inner_kernel, cfgs=cfgs), grid=(num_n, num_k, num_gm),
      in_specs=in_specs, out_specs=out_specs, core_axis_name=core_axis_name, dimension_semantics=dimension_semantics,
  )
  sublane = cfgs.dims.size_lhs_sublane
  lhs_value = lhs_ref.value
  lhs_in = OperandRef(value=lhs_value.reshape(-1, sublane, *lhs_value.shape[1:]), scale=lhs_ref.scale)
  rhs_value = rhs_ref.value
  rhs_in = rhs_value.reshape(-1, sublane, *rhs_value.shape[1:])
  rhs_operand = OperandRef(value=rhs_in, scale=rhs_ref.scale, quant_scale=rhs_ref.quant_scale)
  scratches = [acc_ref, metadata_ref]
  if acc_in_ref is not None:
    scratches += [acc_in_ref, semaphore_ref.at[0]]
  pipeline_fn(lhs_in, rhs_operand, out_ref, scratches=scratches)
  if num_groups_to_zero is not None:
    zero_out_end(num_groups_to_zero, out_ref, semaphore_ref)


def _tgmm_kernel_main_into_acc(
    lhs_group_sizes_ref: jax.Array, group_offset_ref: jax.Array, lhs_ref: OperandRef, rhs_ref: OperandRef,
    acc_in_ref: jax.Array, out_ref: jax.Array, *scratch_refs, cfgs: gmm_v2.GmmConfigs,
):
  """`tgmm_kernel_main` accumulating into `acc_in_ref`, aliased with `out_ref`."""
  tgmm_kernel_main(
      lhs_group_sizes_ref, group_offset_ref, lhs_ref, rhs_ref, out_ref, *scratch_refs, cfgs=cfgs, acc_in_ref=acc_in_ref,
  )


def validate_tgmm_inputs(
    group_sizes: jax.Array, num_actual_groups: int, group_offset: jax.Array | None = None,
) -> None:
  """Validates inputs to 'tgmm_v2' eagerly."""
  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  elif jnp.isscalar(group_offset):
    assert group_offset.size == 1
    if jnp.isscalar(group_offset):
      group_offset = group_offset[None]
  if group_sizes.size < group_offset[0] + num_actual_groups:
    raise ValueError(
        f"group_sizes.size ({group_sizes.size}) must be >= group_offset"
        f" ({group_offset[0]}) + num_actual_groups ({num_actual_groups})"
    )


@jax.jit(
    static_argnames=[
        "num_actual_groups", "tile_info", "vmem_limit_bytes", "precision", "preferred_element_type",
        "acc_dtype", "disable_multi_core_mode", "lhs_quant_dtype", "rhs_quant_dtype",
    ],
)
def tgmm_v2(
    lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, num_actual_groups: int,
    rhs_scale: jax.Array | None = None, group_offset: jax.Array | None = None, *,
    tile_info: gmm_v2.TileSizes | TileTgmmFn = calculate_tgmm_tiling, vmem_limit_bytes: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT, preferred_element_type: jnp.dtype | None = None,
    acc_dtype: jnp.dtype | None = None, disable_multi_core_mode: bool = False, acc: jax.Array | None = None,
    lhs_scale: jax.Array | None = None, lhs_quant_dtype: jnp.dtype | None = None,
    rhs_quant_scale: jax.Array | None = None, rhs_quant_dtype: jnp.dtype | None = None,
    out_scale: jax.Array | None = None,
):
  """Computes a transposed grouped matrix multiplication."""
  del precision
  if preferred_element_type is None:
    preferred_element_type = lhs.dtype
  if acc is not None and not disable_multi_core_mode:
    raise ValueError("`acc` requires disable_multi_core_mode=True.")
  size_m = lhs.shape[0]
  size_sublane = _sublane_size(lhs.dtype, rhs.dtype, size_m)
  padded_size_m = gmm_v2.align_to(size_m, size_sublane)
  if padded_size_m != size_m:
    pad = padded_size_m - size_m
    lhs = jnp.pad(lhs, ((0, pad),) + ((0, 0),) * (lhs.ndim - 1))
    rhs = jnp.pad(rhs, ((0, pad),) + ((0, 0),) * (rhs.ndim - 1))
  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  else:
    if jnp.isscalar(group_offset):
      group_offset = group_offset[None]
  if vmem_limit_bytes is None:
    vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
  target_zero_ref_bytes = 2 * 1024 * 1024
  if rhs_quant_scale is not None:
    if rhs_quant_scale.shape != (1, 1):
      raise ValueError(f"rhs_quant_scale must be [1, 1]; got {rhs_quant_scale.shape}.")
    out_scale = rhs_quant_scale if out_scale is None else rhs_quant_scale * out_scale
  if out_scale is not None:
    if out_scale.shape != (1, 1):
      raise ValueError(f"out_scale must be [1, 1]; got {out_scale.shape}.")
    size_n = _operand_size(rhs, "rhs")
    out_scale = jnp.broadcast_to(out_scale.astype(jnp.float32).reshape(1, 1, 1), (1, 1, size_n))
    rhs_scale = out_scale if rhs_scale is None else rhs_scale * out_scale
  cfgs = make_tgmm_configs(
      lhs, rhs, rhs_scale, group_sizes, num_actual_groups, tile_info=tile_info, vmem_limit_bytes=vmem_limit_bytes,
      out_dtype=preferred_element_type, acc_dtype=acc_dtype, target_zero_ref_bytes=target_zero_ref_bytes,
      disable_multi_core_mode=disable_multi_core_mode, lhs_scale=lhs_scale, lhs_quant_dtype=lhs_quant_dtype,
      rhs_quant_scale=rhs_quant_scale, rhs_quant_dtype=rhs_quant_dtype,
  )
  dims = cfgs.dims
  tiles = cfgs.tiles
  num_lanes = pltpu.get_tpu_info().num_lanes
  aligned_n = gmm_v2.align_to(dims.size_n, num_lanes)
  aligned_k = gmm_v2.align_to(dims.size_k, tiles.tile_k)
  out_init = jax.ShapeDtypeStruct(
      (num_actual_groups * aligned_k, aligned_n), cfgs.out_dtype, manual_axis_type=jax.typeof(lhs).manual_axis_type,
  )

  def _unflatten(out: jax.Array) -> jax.Array:
    out = out.reshape(num_actual_groups, aligned_k, aligned_n)
    return out[:, : dims.size_k, : dims.size_n]

  max_num_gm = dims.size_group + pl.cdiv(dims.size_m, tiles.tile_m) - 1
  scratch_shapes = [
      pltpu.VMEM((tiles.tile_k, tiles.tile_n), cfgs.acc_dtype),
      gmm_v2.MetadataRef(
          gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
          gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
      ),
  ]
  out_bytes = jnp.dtype(cfgs.out_dtype).itemsize
  tile_zero_k = target_zero_ref_bytes // num_lanes // out_bytes
  tile_zero_k = min(tile_zero_k, dims.size_k)
  size_out_sublane = pltpu.get_tpu_info().get_sublane_tiling(cfgs.out_dtype)
  tile_zero_k = (tile_zero_k // size_out_sublane) * size_out_sublane
  assert tile_zero_k > 0
  scratch_shapes += [pltpu.VMEM((tile_zero_k, num_lanes), cfgs.out_dtype), pltpu.SemaphoreType.DMA((1,))]
  if rhs_scale is not None:
    rhs_scale = rhs_scale.astype(jnp.float32)
    pad_n = aligned_n - dims.size_n
    if pad_n > 0:
      rhs_scale = jnp.pad(rhs_scale, ((0, 0), (0, 0), (0, pad_n)))
  if rhs_quant_scale is not None:
    rhs_quant_scale = rhs_quant_scale.astype(jnp.float32)
  rhs = OperandRef(value=rhs, scale=rhs_scale, quant_scale=rhs_quant_scale)
  if lhs_scale is not None:
    lhs_scale = lhs_scale.astype(jnp.float32)
  lhs = OperandRef(value=lhs, scale=lhs_scale)
  if acc is not None:
    want_shape = (num_actual_groups, dims.size_k, dims.size_n)
    if acc.shape != want_shape or acc.dtype != cfgs.out_dtype:
      raise ValueError(f"`acc` must be {want_shape} of dtype {cfgs.out_dtype}; got" f" {acc.shape=}, {acc.dtype=}.")
    pad_k, pad_n = aligned_k - dims.size_k, aligned_n - dims.size_n
    if pad_k or pad_n:
      acc = jnp.pad(acc, ((0, 0), (0, pad_k), (0, pad_n)))
    acc = acc.reshape(out_init.shape)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    scalars = (group_sizes, group_offset)
    inputs = (lhs, rhs, acc)
    out = pl.pallas_call(
        functools.partial(_tgmm_kernel_main_into_acc, cfgs=cfgs),
        out_shape=jax.ShapeDtypeStruct(acc.shape, acc.dtype, manual_axis_type=jax.typeof(acc).manual_axis_type),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalars), in_specs=[jax.tree.map(lambda _: hbm, lhs), jax.tree.map(lambda _: hbm, rhs), hbm],
            out_specs=hbm, scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes, disable_bounds_checks=True),
        input_output_aliases={len(jax.tree.leaves((scalars, inputs))) - 1: 0}, name=get_scope_name(cfgs),
        cost_estimate=get_cost_estimate(cfgs), metadata=gmm_v2.get_metadata(cfgs),
    )(*scalars, *inputs)
    return _unflatten(out)
  if cfgs.disable_multi_core_mode:
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    in_specs = [jax.tree.map(lambda _: hbm_spec, lhs), jax.tree.map(lambda _: hbm_spec, rhs)]
    out = pl.pallas_call(
        functools.partial(tgmm_kernel_main, cfgs=cfgs), out_shape=out_init,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2, in_specs=in_specs, out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes, disable_bounds_checks=True),
        name=get_scope_name(cfgs), cost_estimate=get_cost_estimate(cfgs), metadata=gmm_v2.get_metadata(cfgs),
    )(group_sizes, group_offset, lhs, rhs)
    return _unflatten(out)
  group_sizes = pltpu.with_memory_space_constraint(group_sizes, pltpu.SMEM)
  group_offset = pltpu.with_memory_space_constraint(group_offset, pltpu.SMEM)
  out = pl.kernel(
      functools.partial(tgmm_kernel_main, cfgs=cfgs), out_type=out_init,
      mesh=pltpu.TensorCoreMesh(axis_name="core"), scratch_types=scratch_shapes,
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes, disable_bounds_checks=True),
      name=get_scope_name(cfgs), cost_estimate=get_cost_estimate(cfgs), metadata=gmm_v2.get_metadata(cfgs),
  )(group_sizes, group_offset, lhs, rhs)
  return _unflatten(out)
