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
"""Pallas TPU grouped matrix multiplication (gmm_v2) kernel."""

import abc
import dataclasses
import functools
import math
from typing import Any, Callable, Tuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from lilax._backend.experimental.lineage import ops
from lilax._backend.experimental.lineage.gmmv2 import relayout


def swigluoai(gate: jax.Array, up: jax.Array, *, alpha: float = 1.702, limit: float = 7.0) -> jax.Array:
  """Activation used in some models such as GPT-OSS."""
  gate = jnp.clip(gate, max=limit)
  up = jnp.clip(up, min=-limit, max=limit)
  glu = gate * jax.nn.sigmoid(alpha * gate)
  return (up + 1.0) * glu


def silu_and_mul_with_clamp(gate: jax.Array, up: jax.Array, limit: float = 10.0) -> jax.Array:
  """Activation used in some models DeepSeek V4."""
  gate = jnp.clip(gate, max=limit)
  up = jnp.clip(up, min=-limit, max=limit)
  return jax.nn.silu(gate) * up


def situ_and_mul(gate: jax.Array, up: jax.Array, beta: float, linear_beta: float | None) -> jax.Array:
  gate = beta * jnp.tanh(gate / beta) * jax.nn.sigmoid(gate)
  if linear_beta is not None:
    up = linear_beta * jnp.tanh(up / linear_beta)
  return gate * up


def interleave_lane(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
  """Interleaves two arrays along lane dim at zero-cost."""
  assert lhs.shape == rhs.shape
  chunk_size = pltpu.get_tpu_info().num_lanes
  num_chunks = lhs.shape[-1] // chunk_size
  lhs_chunks = jnp.split(lhs, num_chunks, axis=-1)
  rhs_chunks = jnp.split(rhs, num_chunks, axis=-1)
  interleaved = []
  for i in range(num_chunks):
    interleaved += [lhs_chunks[i], rhs_chunks[i]]
  return jnp.concat(interleaved, axis=-1)


def deinterleave_lane(val: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Deinterleaves an array along lane dim at zero-cost."""
  chunk_size = pltpu.get_tpu_info().num_lanes
  num_chunks = val.shape[-1] // chunk_size
  chunks = jnp.split(val, num_chunks, axis=-1)
  lhs = jnp.concat(chunks[0::2], axis=-1)
  rhs = jnp.concat(chunks[1::2], axis=-1)
  return lhs, rhs


def apply_act_fn(acc: jax.Array, fuse_act: str | None):
  """Applies a fused activation function to the accumulator."""
  if fuse_act is None:
    return acc
  acc_gate, acc_up = deinterleave_lane(acc)
  match fuse_act:
    case "silu":
      return jax.nn.silu(acc_gate) * acc_up
    case "gelu":
      return jax.nn.gelu(acc_gate) * acc_up
    case "gelu_tanh":
      return jax.nn.gelu(acc_gate, approximate=True) * acc_up
    case "swigluoai":
      return swigluoai(acc_gate, acc_up)
    case "silu_and_mul_with_clamp":
      return silu_and_mul_with_clamp(acc_gate, acc_up)
    case str() if fuse_act.startswith("situ:"):
      _, beta, linear_beta = fuse_act.split(":")
      linear_beta = None if linear_beta == "none" else float(linear_beta)
      return situ_and_mul(acc_gate, acc_up, float(beta), linear_beta)
    case _:
      raise NotImplementedError(f"Unsupported activation function: {fuse_act}")


def align_to(x, a):
  return pl.cdiv(x, a) * a


def _get_lhs_sublane_size(dtype: jnp.dtype, size_m: int) -> int:
  """Sublane block size the kernel processes lhs rows in."""
  size_lhs_sublane = pltpu.get_tpu_info().get_sublane_tiling(dtype)
  size_lhs_sublane = min(size_lhs_sublane, size_m)
  return size_lhs_sublane


def get_packing_factor(storage_dtype: jnp.dtype, quant_dtype: jnp.dtype | None) -> int:
  """Returns the number of quantized elements packed per storage element."""
  if quant_dtype is None or quant_dtype == storage_dtype:
    return 1
  storage_bits = jax.dtypes.itemsize_bits(storage_dtype)
  quant_bits = jax.dtypes.itemsize_bits(quant_dtype)
  packing_factor, remainder = divmod(storage_bits, quant_bits)
  if remainder != 0:
    raise ValueError(f"Storage dtype {storage_dtype} is not divisible by " f"quant dtype {quant_dtype}")
  return packing_factor


class RhsRef(abc.ABC):
  """Abstract class that defines interfaces for rhs values."""

  @abc.abstractmethod
  def get_weight(self) -> jax.Array:
    ...

  @abc.abstractmethod
  def get_scale(self, replicate_size: int | None = None) -> jax.Array:
    """Returns scale array, optionally replicated across sublanes."""
    ...

  @abc.abstractmethod
  def get_bias(self) -> jax.Array:
    ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WeightsRef(RhsRef):
  """Dataclass for a single weights."""

  weight: Any
  scale: Any | None
  bias: Any | None

  def get_weight(self) -> jax.Array:
    return self.weight[...]

  def get_scale(self, replicate_size: int | None = None) -> jax.Array:
    assert self.scale is not None
    if replicate_size is not None:
      return self.scale[:, pl.ds(0, replicate_size, 0), :]
    return self.scale[...]

  def get_bias(self) -> jax.Array:
    assert self.bias is not None
    return self.bias[...]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FusedWeightsRef(RhsRef):
  """Dataclass for gate and up weights used in fused activation."""

  gate: WeightsRef
  up: WeightsRef

  def get_weight(self) -> jax.Array:
    w_gate = self.gate.get_weight()
    w_up = self.up.get_weight()
    return interleave_lane(w_gate, w_up)

  def get_scale(self, replicate_size: int | None = None) -> jax.Array:
    s_gate = self.gate.get_scale(replicate_size)
    s_up = self.up.get_scale(replicate_size)
    return interleave_lane(s_gate, s_up)

  def get_bias(self) -> jax.Array:
    b_gate = self.gate.get_bias()
    b_up = self.up.get_bias()
    return interleave_lane(b_gate, b_up)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LhsRef:
  """Dataclass for the lhs value and its optional quantization scale."""

  value: Any
  scale: Any | None
  glu_coeffs: Any | None = None

  def get_value(self) -> jax.Array:
    return self.value[...]

  def get_scale(self) -> jax.Array:
    assert self.scale is not None
    return self.scale[...]

  def get_glu_coeffs(self) -> jax.Array:
    assert self.glu_coeffs is not None
    return self.glu_coeffs[...]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class OutRef:
  """Bundles the main GMM output ref and optional fused GLU activation ref."""

  out: Any
  act: Any | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
  gm_id_to_group_id: jax.Array
  gm_id_to_m_offset: jax.Array


@dataclasses.dataclass(frozen=True)
class TileSizes:
  tile_m: int
  tile_k: int
  tile_n: int
  bucket_base: int


@dataclasses.dataclass(frozen=True)
class Dimensions:
  size_m: int
  size_k: int
  size_n: int
  size_group: int
  size_lhs_group: int
  size_lhs_sublane: int


@dataclasses.dataclass(frozen=True)
class InputConfigs:
  """Configuration for an LHS or RHS input operand."""

  quant_dtype: jnp.dtype | None
  quant_block_size: int | None
  dtype: jnp.dtype
  has_bias: bool = False
  has_scale: bool = False
  is_transposed: bool = False
  is_3d: bool = False

  @property
  def should_unpack(self) -> bool:
    return get_packing_factor(self.dtype, self.quant_dtype) > 1

  @property
  def should_use_external_scale(self) -> bool:
    return self.has_scale and self.quant_dtype is not None and self.dtype != self.quant_dtype

  @property
  def should_dequantize_before_matmul(self) -> bool:
    if not self.has_scale:
      return False
    assert self.quant_block_size is not None
    mxu_size = pltpu.get_tpu_info().mxu_column_size
    return self.quant_block_size < mxu_size

  @property
  def should_dequantize_after_matmul(self) -> bool:
    return self.has_scale and not self.should_dequantize_before_matmul

  @property
  def should_quantize(self) -> bool:
    if self.quant_dtype is None:
      return False
    return self.quant_dtype != self.dtype


@dataclasses.dataclass(frozen=True)
class GmmConfigs:
  """Aggregated configuration for the GMM/TGMM Pallas kernel."""

  tiles: TileSizes
  dims: Dimensions
  lhs_cfgs: InputConfigs
  rhs_cfgs: InputConfigs
  out_dtype: jnp.dtype
  acc_dtype: jnp.dtype
  zero_init: bool
  fuse_act: str | None
  transpose_rhs: bool = False
  disable_multi_core_mode: bool = False
  out_is_3d: bool = False
  flat_rhs: bool = False
  has_out_scale: bool = False
  has_glu_coeffs: bool = False
  glu_out_dtype: jnp.dtype | None = None
  has_glu_out_scale: bool = False
  lhs_gather: bool = False

  @property
  def num_quant_blocks_per_tile_k(self) -> int:
    return pl.cdiv(self.tiles.tile_k, self.rhs_cfgs.quant_block_size)

  @property
  def out_size_n(self) -> int:
    if self.fuse_act is None:
      return self.dims.size_n
    return self.dims.size_n // 2

  @property
  def rhs_row_blocks_per_group(self) -> int:
    if self.transpose_rhs:
      return self.dims.size_n // self.tiles.tile_n
    return self.dims.size_k // self.tiles.tile_k

  @property
  def lhs_is_3d(self) -> bool:
    return self.lhs_cfgs.is_3d

  @property
  def lhs_tile_d0(self) -> int:
    assert self.lhs_is_3d
    return self.tiles.tile_k // pltpu.get_tpu_info().num_lanes

  @property
  def out_tile_d0(self) -> int:
    assert self.out_is_3d
    return self.tiles.tile_n // pltpu.get_tpu_info().num_lanes


TileFn = Callable[..., TileSizes]


def is_fp8(dtype: jax.typing.DTypeLike) -> bool:
  return jnp.dtype(dtype) in (jnp.dtype(jnp.float8_e4m3fn), jnp.dtype(jnp.float8_e5m2))


class IndexMaps:
  """Index maps for GMM kernel."""

  def __init__(self, metadata_ref: MetadataRef, cfgs: GmmConfigs, out_offset_ref: jax.Array | None = None):
    self.metadata_ref = metadata_ref
    self.cfgs = cfgs
    self.out_offset_ref = out_offset_ref

  def lhs_index_map(self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array):
    m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]
    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    row_size = row_end - row_start
    if self.cfgs.lhs_is_3d:
      return (pl.ds(row_start, row_size), 0, k_id, 0)
    return (pl.ds(row_start, row_size), 0, k_id)

  def lhs_scale_index_map(self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array):
    del gm_id, k_id
    return (0, 0)

  def glu_coeffs_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array):
    del n_id, k_id
    m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]
    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    row_size = row_end - row_start
    return (pl.ds(row_start, row_size), 0, 0)

  def rhs_weight_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array):
    group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
    row_id, col_id = (n_id, k_id) if self.cfgs.transpose_rhs else (k_id, n_id)
    if self.cfgs.flat_rhs:
      return (group_id * self.cfgs.rhs_row_blocks_per_group + row_id, col_id)
    return (group_id, row_id, col_id)

  def rhs_bias_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array):
    group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
    return (group_id, 0, n_id)

  def rhs_scale_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array):
    group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
    k_row = k_id * self.cfgs.tiles.tile_k
    b_row = k_row // self.cfgs.rhs_cfgs.quant_block_size
    b_tile_id = b_row // self.cfgs.num_quant_blocks_per_tile_k
    return (group_id, b_tile_id, 0, n_id)

  def out_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array):
    is_last_gm = gm_id == (pl.num_programs(1) - 1)
    m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]
    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    capped_row_end = m_end // self.cfgs.dims.size_lhs_sublane
    last_row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    row_end = jnp.where(is_last_gm, last_row_end, capped_row_end)
    row_size = row_end - row_start
    if self.out_offset_ref is not None:
      row_start += self.out_offset_ref[0] // self.cfgs.dims.size_lhs_sublane
    if self.cfgs.out_is_3d:
      return (pl.ds(row_start, row_size), 0, n_id, 0)
    return (pl.ds(row_start, row_size), 0, n_id)

  def act_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array):
    is_last_gm = gm_id == (pl.num_programs(1) - 1)
    m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]
    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    capped_row_end = m_end // self.cfgs.dims.size_lhs_sublane
    last_row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    row_end = jnp.where(is_last_gm, last_row_end, capped_row_end)
    row_size = row_end - row_start
    return (pl.ds(row_start, row_size), 0, n_id)


def generate_block_specs(
    metadata_ref: MetadataRef, cfgs: GmmConfigs, out_offset_ref: jax.Array | None = None,
) -> Tuple[Tuple[LhsRef, WeightsRef], OutRef]:
  """Generates block specs for the given lhs, rhs, and out refs."""
  index_map = IndexMaps(metadata_ref, cfgs, out_offset_ref)
  bounded_slice_gm = pl.BoundedSlice(cfgs.tiles.tile_m // cfgs.dims.size_lhs_sublane)
  if cfgs.lhs_gather:
    lhs_value_spec = None
  elif cfgs.lhs_is_3d:
    lhs_value_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.lhs_tile_d0, pltpu.get_tpu_info().num_lanes),
        index_map.lhs_index_map,
    )
  else:
    lhs_value_spec = pl.BlockSpec((bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_k), index_map.lhs_index_map)
  lhs_scale_spec = None
  if cfgs.lhs_cfgs.has_scale:
    lhs_scale_spec = pl.BlockSpec((1, 1), index_map.lhs_scale_index_map)
  glu_coeffs_spec = None
  if cfgs.has_glu_coeffs:
    glu_coeffs_spec = pl.BlockSpec((bounded_slice_gm, cfgs.dims.size_lhs_sublane, 1), index_map.glu_coeffs_index_map)
  lhs_block_spec = LhsRef(value=lhs_value_spec, scale=lhs_scale_spec, glu_coeffs=glu_coeffs_spec)
  if cfgs.transpose_rhs:
    rhs_tile = (cfgs.tiles.tile_n, cfgs.tiles.tile_k)
  else:
    rhs_tile = (cfgs.tiles.tile_k, cfgs.tiles.tile_n)
  rhs_weight_spec = pl.BlockSpec(
      rhs_tile if cfgs.flat_rhs else (None, *rhs_tile), index_map.rhs_weight_index_map,
      pipeline_mode=pl.Buffered(buffer_count=3),
  )
  rhs_scale_block_spec = rhs_bias_block_spec = None
  if cfgs.rhs_cfgs.has_bias:
    rhs_bias_block_spec = pl.BlockSpec((None, 1, cfgs.tiles.tile_n), index_map.rhs_bias_index_map)
  if cfgs.rhs_cfgs.has_scale:
    rhs_scale_block_spec = pl.BlockSpec(
        (None, cfgs.num_quant_blocks_per_tile_k, 1, cfgs.tiles.tile_n), index_map.rhs_scale_index_map,
    )
  rhs_block_spec = WeightsRef(weight=rhs_weight_spec, scale=rhs_scale_block_spec, bias=rhs_bias_block_spec)
  if cfgs.out_is_3d:
    out_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.out_tile_d0, pltpu.get_tpu_info().num_lanes),
        index_map.out_index_map,
    )
  else:
    out_block_spec = pl.BlockSpec((bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_n), index_map.out_index_map)
  act_block_spec = None
  if cfgs.has_glu_coeffs:
    act_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_n // 2), index_map.act_index_map,
    )
  return (lhs_block_spec, rhs_block_spec), OutRef(out=out_block_spec, act=act_block_spec)


_GATHER_IDX_ALIGN = 128
_GATHER_ISSUE_UNROLL = 16


def gather_idx_window(tile_m: int) -> int:
  """Number of gather indices staged in SMEM per gm tile."""
  return align_to(tile_m, _GATHER_IDX_ALIGN) + _GATHER_IDX_ALIGN


@dataclasses.dataclass(frozen=True)
class GatherRefs:
  """Refs of the in-kernel lhs row gather."""

  start_ref: jax.Array
  idx_hbm_ref: jax.Array
  x_hbm_ref: jax.Array
  buf_ref: jax.Array
  row_sems: jax.Array
  idx_smem_ref: jax.Array
  idx_sems: jax.Array
  num_k: int
  idx_window: int
  max_win_start: int

  def lhs_block(self, slot: jax.Array, k_id: jax.Array) -> Any:
    return self.buf_ref.at[slot * self.num_k + k_id]

  def _window(self, metadata_ref: MetadataRef, gm_id: jax.Array | int) -> tuple[jax.Array, jax.Array]:
    row = self.start_ref[0] + metadata_ref.gm_id_to_m_offset[gm_id]
    win_start = jnp.minimum(row // _GATHER_IDX_ALIGN * _GATHER_IDX_ALIGN, self.max_win_start)
    return win_start, row - win_start

  def idx_copy(self, metadata_ref: MetadataRef, gm_id: jax.Array | int, slot: jax.Array | int):
    win_start, _ = self._window(metadata_ref, gm_id)
    return pltpu.make_async_copy(
        self.idx_hbm_ref.at[pl.ds(pl.multiple_of(win_start, _GATHER_IDX_ALIGN), self.idx_window)],
        self.idx_smem_ref.at[pl.ds(slot * self.idx_window, self.idx_window)], self.idx_sems.at[slot],
    )

  def start_rows(
      self, metadata_ref: MetadataRef, gm_id: jax.Array | int, slot: jax.Array | int, size_lhs_sublane: int,
  ):
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    num_rows = metadata_ref.gm_id_to_m_offset[gm_id + 1] - m_start
    _, win_off = self._window(metadata_ref, gm_id)
    idx_base = slot * self.idx_window + win_off
    dst_base = m_start % size_lhs_sublane
    sem = self.row_sems.at[slot]

    def _copy(src_row, dst_row):
      pltpu.make_async_copy(
          self.x_hbm_ref.at[pl.ds(src_row * self.num_k, self.num_k)],
          self.buf_ref.at[pl.ds(slot * self.num_k, self.num_k), pl.ds(dst_row, 1)], sem,
      ).start()

    unroll = _GATHER_ISSUE_UNROLL
    num_full = num_rows // unroll

    @pl.loop(0, num_full)
    def _(i):
      base = i * unroll
      src_rows = [self.idx_smem_ref[idx_base + base + u] for u in range(unroll)]
      for u in range(unroll):
        _copy(src_rows[u], dst_base + base + u)

    @pl.loop(num_full * unroll, num_rows)
    def _(i):
      _copy(self.idx_smem_ref[idx_base + i], dst_base + i)

  def wait_rows(self, num_rows: jax.Array, slot: jax.Array | int):
    sem = self.row_sems.at[slot]
    tile_m = self.buf_ref.shape[1]
    for b in reversed(range(tile_m.bit_length())):
      size = 1 << b

      @pl.when((num_rows & size) != 0)
      def _(size=size):
        rows = self.buf_ref.at[pl.ds(slot * self.num_k, self.num_k), pl.ds(0, size)]
        pltpu.make_async_copy(rows, rows, sem).wait()


def fill_metadata(
    lhs_group_sizes_ref: jax.Array, group_offset_ref: jax.Array, metadata_ref: MetadataRef, *, cfgs: GmmConfigs,
) -> jax.Array:
  """Fills the metadata for the given lhs group sizes and group offset."""
  group_offset = group_offset_ref[0]
  max_num_group = group_offset + cfgs.dims.size_group
  metadata_ref.gm_id_to_m_offset[0] = 0

  @jax.named_scope("inner_tm_loop")
  def inner_tm_loop(tm_id, curr_m_offset, *, end_m_offset, group_id):
    local_offset = curr_m_offset % cfgs.dims.size_lhs_sublane
    tm_size = jnp.minimum(cfgs.tiles.tile_m - local_offset, end_m_offset - curr_m_offset)
    metadata_ref.gm_id_to_group_id[tm_id] = group_id
    next_m_offset = curr_m_offset + tm_size
    metadata_ref.gm_id_to_m_offset[tm_id] = curr_m_offset
    metadata_ref.gm_id_to_m_offset[tm_id + 1] = next_m_offset
    return next_m_offset

  @jax.named_scope("outer_group_loop")
  def outer_group_loop(lhs_group_id, carry):
    num_gm, start_m_offset = carry
    group_id = lhs_group_id - group_offset
    group_size = lhs_group_sizes_ref[lhs_group_id]
    end_m_offset = start_m_offset + group_size
    local_offset = start_m_offset % cfgs.dims.size_lhs_sublane
    aligned_group_size = group_size + local_offset
    curr_num_gm = pl.cdiv(aligned_group_size, cfgs.tiles.tile_m)
    should_process = jnp.logical_and(group_size > 0, group_id >= 0)
    curr_num_gm = jnp.where(should_process, curr_num_gm, 0)
    next_num_gm = num_gm + curr_num_gm
    tm_loop_fn = functools.partial(inner_tm_loop, end_m_offset=end_m_offset, group_id=group_id)
    lax.fori_loop(num_gm, next_num_gm, tm_loop_fn, start_m_offset)
    return next_num_gm, end_m_offset

  num_gm, _ = lax.fori_loop(0, max_num_group, outer_group_loop, (0, 0))
  return num_gm


def zero_out_start(
    out_ref: jax.Array, zero_ref: jax.Array, semaphore_ref: jax.Array, metadata_ref: MetadataRef,
    num_gm: jax.Array, *, dims: Dimensions,
):
  """Zero out output rows that are not used in the computation."""
  num_lanes = pltpu.get_tpu_info().num_lanes
  assert num_lanes == zero_ref.shape[-1]
  zero_ref[...] = jnp.zeros_like(zero_ref)
  out_is_3d = out_ref.ndim == 3
  if out_is_3d:
    zero_dma = zero_ref.reshape(-1, dims.size_lhs_sublane, *zero_ref.shape[1:])
    out_dma = out_ref.reshape(-1, dims.size_lhs_sublane, *out_ref.shape[1:])
  else:
    zero_dma = zero_ref.reshape(-1, dims.size_lhs_sublane, num_lanes)
    out_dma = out_ref.reshape(-1, dims.size_lhs_sublane, out_ref.shape[-1])
  row_size = zero_dma.shape[0]
  compute_start = metadata_ref.gm_id_to_m_offset[0]
  compute_end = metadata_ref.gm_id_to_m_offset[num_gm]
  left_zero_start = 0
  left_zero_end = compute_start // dims.size_lhs_sublane
  left_zero_size = left_zero_end - left_zero_start
  left_num_loops = pl.cdiv(left_zero_size, row_size)
  right_zero_start = pl.cdiv(compute_end, dims.size_lhs_sublane)
  right_zero_end = out_dma.shape[0]
  right_zero_size = right_zero_end - right_zero_start
  right_num_loops = pl.cdiv(right_zero_size, row_size)

  def fill_zero(i, zero_size, *, start, end):
    dma_start = start + i * row_size
    dma_end = jnp.minimum(dma_start + row_size, end)
    dma_size = dma_end - dma_start
    if out_is_3d:
      pltpu.make_async_copy(
          src_ref=zero_dma.at[pl.ds(0, dma_size)], dst_ref=out_dma.at[pl.ds(dma_start, dma_size)],
          sem=semaphore_ref.at[0],
      ).start(priority=1)
    else:
      for n_start in range(0, out_dma.shape[-1], num_lanes):
        n_end = n_start + num_lanes
        pltpu.make_async_copy(
            src_ref=zero_dma.at[pl.ds(0, dma_size)], dst_ref=out_dma.at[pl.ds(dma_start, dma_size), :, n_start:n_end],
            sem=semaphore_ref.at[0],
        ).start(priority=1)
    return zero_size + dma_size

  @jax.named_scope("left_fill_zero")
  def left_fill_zero(i, zero_size):
    return fill_zero(i, zero_size, start=left_zero_start, end=left_zero_end)

  @jax.named_scope("right_fill_zero")
  def right_fill_zero(i, zero_size):
    return fill_zero(i, zero_size, start=right_zero_start, end=right_zero_end)

  zero_size = lax.fori_loop(0, left_num_loops, left_fill_zero, 0)
  zero_size = lax.fori_loop(0, right_num_loops, right_fill_zero, zero_size)
  return zero_size


def zero_out_end(out_ref: jax.Array, semaphore_ref: jax.Array, zero_size: jax.Array, *, dims: Dimensions):
  out_dma = out_ref.reshape(-1, dims.size_lhs_sublane, *out_ref.shape[1:])
  pltpu.make_async_copy(src_ref=out_dma.at[pl.ds(0, zero_size)], dst_ref=out_dma.at[pl.ds(0, zero_size)], sem=semaphore_ref.at[0]).wait()


def inner_kernel(
    tiled_lhs_ref: LhsRef,
    tiled_rhs_ref: RhsRef,
    tiled_out_ref: OutRef,
    partial_out_ref: jax.Array,
    acc_ref: jax.Array,
    metadata_ref: MetadataRef,
    gather_refs: GatherRefs | None = None,
    *,
    cfgs: GmmConfigs,
    out_scale_ref: jax.Array | None = None,
    glu_out_scale_ref: jax.Array | None = None,
    partial_act_ref: jax.Array | None = None,
):
  """Inner kernel invoked by emit_pipeline to perform matmul."""
  gm_id = pl.program_id(1)
  m_start = metadata_ref.gm_id_to_m_offset[gm_id]
  m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
  m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane
  m_start_local = m_start - m_offset
  m_end_local = m_end - m_offset

  gathered_lhs_ref = None
  if gather_refs is not None:
    k_id = pl.program_id(2)
    slot = gm_id % 2

    @pl.when(k_id == 0)
    @jax.named_scope("lhs_gather")
    def _gather():
      gather_refs.wait_rows(m_end - m_start, slot)
      num_gm = pl.num_programs(1)
      next_slot = 1 - slot

      @pl.when(gm_id + 1 < num_gm)
      def _():
        gather_refs.idx_copy(metadata_ref, gm_id + 1, next_slot).wait()
        gather_refs.start_rows(metadata_ref, gm_id + 1, next_slot, cfgs.dims.size_lhs_sublane)

      @pl.when(gm_id + 2 < num_gm)
      def _():
        gather_refs.idx_copy(metadata_ref, gm_id + 2, slot).start()

    gathered_lhs_ref = gather_refs.lhs_block(slot, k_id)

  def _matmul(is_first_k_step: bool, is_last_k_step: bool, bucket_m: int):
    tpu_info = pltpu.get_tpu_info()
    mxu_size = tpu_info.mxu_column_size
    if cfgs.lhs_is_3d:
      tiled_lhs = relayout.load_3d_as_2d(
          tiled_lhs_ref.value if gathered_lhs_ref is None else gathered_lhs_ref, bucket_m, cfgs.lhs_tile_d0,
      )
    else:
      tiled_lhs = tiled_lhs_ref.get_value().reshape(-1, cfgs.tiles.tile_k)[:bucket_m]
    tiled_rhs = tiled_rhs_ref.get_weight()
    if cfgs.transpose_rhs:
      tiled_rhs = jnp.transpose(tiled_rhs)

    rhs_tile_n = tiled_rhs.shape[1]
    rhs_qbs = cfgs.rhs_cfgs.quant_block_size
    if cfgs.rhs_cfgs.should_dequantize_before_matmul:
      if cfgs.transpose_rhs:
        raise NotImplementedError("should_dequantize_before_matmul is not supported with" " transpose_rhs.")
      tiled_rhs_scale = tiled_rhs_ref.get_scale(replicate_size=rhs_qbs).astype(cfgs.lhs_cfgs.dtype)
      num_blocks = cfgs.num_quant_blocks_per_tile_k
      tiled_rhs_dequant = tiled_rhs.astype(cfgs.lhs_cfgs.dtype).reshape(num_blocks, rhs_qbs, rhs_tile_n)
      tiled_rhs_dequant = tiled_rhs_dequant * tiled_rhs_scale
      tiled_rhs = tiled_rhs_dequant.reshape(cfgs.tiles.tile_k, rhs_tile_n)
      rhs_qbs = cfgs.tiles.tile_k

    valid_k = cfgs.dims.size_k % cfgs.tiles.tile_k
    if is_last_k_step and valid_k != 0:
      mask_lhs = lax.broadcasted_iota(jnp.int32, tiled_lhs.shape, 1) < valid_k
      tiled_lhs = jnp.where(mask_lhs, tiled_lhs, 0)
      mask_rhs = lax.broadcasted_iota(jnp.int32, tiled_rhs.shape, 0) < valid_k
      tiled_rhs = jnp.where(mask_rhs, tiled_rhs, 0)

    if cfgs.lhs_cfgs.should_use_external_scale and not cfgs.rhs_cfgs.has_scale:
      assert cfgs.lhs_cfgs.quant_dtype is not None
      inv_scale = 1.0 / tiled_lhs_ref.get_scale().astype(jnp.float32)
      tiled_lhs = ops.quantize_tile(tiled_lhs, inv_scale, cfgs.lhs_cfgs.quant_dtype)

    acc_list = []
    if not cfgs.lhs_cfgs.should_quantize or (cfgs.lhs_cfgs.should_use_external_scale and not cfgs.rhs_cfgs.has_scale):
      for start_n in range(0, rhs_tile_n, mxu_size):
        end_n = min(rhs_tile_n, start_n + mxu_size)
        col_size = end_n - start_n
        acc_n = jnp.zeros((bucket_m, col_size), dtype=acc_ref.dtype)
        for start_k in range(0, cfgs.tiles.tile_k, rhs_qbs):
          end_k = min(cfgs.tiles.tile_k, start_k + rhs_qbs)
          block_acc = lax.dot_general(
              tiled_lhs[:, start_k:end_k], tiled_rhs[start_k:end_k, start_n:end_n],
              (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32,
          ).astype(acc_ref.dtype)
          if cfgs.rhs_cfgs.should_dequantize_after_matmul:
            if cfgs.transpose_rhs:
              raise NotImplementedError("should_dequantize_after_matmul is not supported with" " transpose_rhs.")
            b_id = start_k // rhs_qbs
            rhs_scale_replicated = tiled_rhs_ref.get_scale(replicate_size=bucket_m)[b_id, :, start_n : start_n + col_size]
            block_acc *= rhs_scale_replicated.astype(acc_ref.dtype)
          acc_n += block_acc
        acc_list.append(acc_n)
    else:
      lhs_q_dtype = cfgs.lhs_cfgs.quant_dtype
      q_block_size = cfgs.lhs_cfgs.quant_block_size
      if jnp.issubdtype(lhs_q_dtype, jnp.floating):
        dtype_max = float(jnp.finfo(lhs_q_dtype).max)
        preferred_element_type = jnp.float32
      else:
        dtype_max = float(jnp.iinfo(lhs_q_dtype).max)
        preferred_element_type = jnp.int32
      lhs_scale = lhs_scale_inv = None
      should_use_external_scale = cfgs.lhs_cfgs.should_use_external_scale
      if should_use_external_scale:
        lhs_scale = tiled_lhs_ref.get_scale().astype(acc_ref.dtype)
        lhs_scale_inv = 1.0 / lhs_scale
      ext_lhs_scale_const_k = should_use_external_scale and lhs_scale is not None and lhs_scale.shape[-1] == 1
      step_k = math.gcd(q_block_size, rhs_qbs) if cfgs.rhs_cfgs.should_dequantize_after_matmul else q_block_size
      for start_n in range(0, rhs_tile_n, mxu_size):
        end_n = min(rhs_tile_n, start_n + mxu_size)
        col_size = end_n - start_n
        acc_n = jnp.zeros((bucket_m, col_size), dtype=acc_ref.dtype)
        for start_k in range(0, cfgs.tiles.tile_k, q_block_size):
          end_k = min(cfgs.tiles.tile_k, start_k + q_block_size)
          block_lhs = tiled_lhs[:, start_k:end_k]
          block_rhs = tiled_rhs[start_k:end_k, start_n:end_n]
          if should_use_external_scale:
            assert lhs_scale is not None
            assert lhs_scale_inv is not None
            block_lhs_q = jnp.clip(block_lhs * lhs_scale_inv, -dtype_max, dtype_max).astype(lhs_q_dtype)
            block_scale = lhs_scale
          else:
            block_abs_max = jnp.max(jnp.abs(block_lhs), axis=1, keepdims=True)
            block_scale = block_abs_max / dtype_max
            block_scale_inv = jnp.where(block_scale == 0, 0, 1 / block_scale)
            block_lhs_q = (block_lhs * block_scale_inv).astype(lhs_q_dtype)
          if not tpu_info.is_matmul_supported(lhs_q_dtype, block_rhs.dtype):
            block_rhs = block_rhs.astype(lhs_q_dtype)
          block_len = end_k - start_k
          block_acc = acc_n if ext_lhs_scale_const_k else None
          for sub_k in range(0, block_len, step_k):
            sub_end_k = min(block_len, sub_k + step_k)
            sub_acc = jnp.matmul(
                block_lhs_q[:, sub_k:sub_end_k], block_rhs[sub_k:sub_end_k, :],
                preferred_element_type=preferred_element_type,
            ).astype(acc_ref.dtype)
            if cfgs.rhs_cfgs.should_dequantize_after_matmul:
              if cfgs.transpose_rhs:
                raise NotImplementedError("should_dequantize_after_matmul is not supported with" " transpose_rhs.")
              b_id = (start_k + sub_k) // rhs_qbs
              rhs_scale_replicated = tiled_rhs_ref.get_scale(replicate_size=bucket_m)[b_id, :, start_n : start_n + col_size]
              sub_acc *= rhs_scale_replicated.astype(acc_ref.dtype)
            block_acc = sub_acc if block_acc is None else block_acc + sub_acc
          assert block_acc is not None
          if ext_lhs_scale_const_k:
            acc_n = block_acc
          else:
            block_acc *= block_scale.astype(acc_ref.dtype)
            acc_n += block_acc
        if ext_lhs_scale_const_k:
          assert lhs_scale is not None
          acc_n *= lhs_scale
        acc_list.append(acc_n)

    acc = jnp.concatenate(acc_list, axis=1)
    if not is_first_k_step:
      acc += acc_ref[:bucket_m]
    acc_m = acc.shape[0]
    if is_last_k_step:
      if out_scale_ref is not None:
        acc *= out_scale_ref[0].astype(acc.dtype)
      if cfgs.rhs_cfgs.has_bias:
        tiled_rhs_bias = tiled_rhs_ref.get_bias()
        acc += tiled_rhs_bias.astype(acc.dtype)
      acc = apply_act_fn(acc, cfgs.fuse_act)
      iota = lax.broadcasted_iota(jnp.int32, acc.shape, 0)
      mask = jnp.logical_and(m_start_local <= iota, iota < m_end_local)
      acc_masked = jnp.where(mask, acc, 0)
      out_val_ref = tiled_out_ref.out
      acc_out = acc_masked.astype(out_val_ref.dtype)
      if cfgs.out_is_3d:
        relayout.store_2d_as_3d(out_val_ref, acc_out, acc_m, cfgs.out_tile_d0)
      else:
        tiled_out_2d_ref = out_val_ref.reshape(-1, cfgs.tiles.tile_n)
        tiled_out_2d_ref[:acc_m] = acc_out
      partial_out_zeros = jnp.zeros_like(partial_out_ref)
      out_val_ref[0] += jnp.where(gm_id == 0, partial_out_zeros, partial_out_ref[...])
      last_row = m_end_local // cfgs.dims.size_lhs_sublane
      partial_out_ref[...] = jnp.where(
          m_end_local % cfgs.dims.size_lhs_sublane == 0, partial_out_zeros, out_val_ref[last_row],
      )
      if cfgs.has_glu_coeffs:
        assert tiled_out_ref.act is not None
        assert partial_act_ref is not None
        act_val_ref = tiled_out_ref.act
        half_n = cfgs.tiles.tile_n // 2
        c_tile = jnp.where(mask[:, :1], tiled_lhs_ref.get_glu_coeffs().reshape(-1, 1)[:acc_m], 0.0)
        act_tile = ops.silu_mul_tile(acc_out[:, :half_n], acc_out[:, half_n:], c_tile, out_val_ref.dtype)
        if glu_out_scale_ref is not None:
          inv_scale = 1.0 / glu_out_scale_ref[0].astype(jnp.float32)
          act_out = ops.quantize_tile(act_tile, inv_scale, act_val_ref.dtype)
        else:
          act_out = act_tile.astype(act_val_ref.dtype)
        tiled_act_2d_ref = act_val_ref.reshape(-1, half_n)
        tiled_act_2d_ref[:acc_m] = act_out
        partial_act_zeros = jnp.zeros_like(partial_act_ref)
        prev_partial_act = jnp.where(gm_id == 0, partial_act_zeros, partial_act_ref[...])
        sublane_iota = lax.broadcasted_iota(jnp.int32, partial_act_ref.shape, 0)
        act_val_ref[0] = jnp.where(sublane_iota < m_start_local, prev_partial_act, act_val_ref[0])
        partial_act_ref[...] = jnp.where(
            m_end_local % cfgs.dims.size_lhs_sublane == 0, partial_act_zeros, act_val_ref[last_row],
        )
    else:
      acc_ref[:acc_m] = acc

  def run_matmul_step(bucket_idx: int):
    bucket_m = cfgs.tiles.bucket_base * (bucket_idx + 1)

    @jax.named_scope(f"bm{bucket_m}_first_last")
    def matmul_first_last():
      _matmul(is_first_k_step=True, is_last_k_step=True, bucket_m=bucket_m)

    @jax.named_scope(f"bm{bucket_m}_first")
    def matmul_first():
      _matmul(is_first_k_step=True, is_last_k_step=False, bucket_m=bucket_m)

    @jax.named_scope(f"bm{bucket_m}_mid")
    def matmul_mid():
      _matmul(is_first_k_step=False, is_last_k_step=False, bucket_m=bucket_m)

    @jax.named_scope(f"bm{bucket_m}_last")
    def matmul_last():
      _matmul(is_first_k_step=False, is_last_k_step=True, bucket_m=bucket_m)

    num_k = pl.num_programs(2)
    k_id = pl.program_id(2)
    is_first_k_step = k_id == 0
    is_last_k_step = k_id == (num_k - 1)
    lax.cond(
        is_first_k_step,
        lambda: lax.cond(is_last_k_step, matmul_first_last, matmul_first),
        lambda: lax.cond(is_last_k_step, matmul_last, matmul_mid),
    )

  branches = []
  for bucket_idx in range(cfgs.tiles.tile_m // cfgs.tiles.bucket_base):
    branches.append(functools.partial(run_matmul_step, bucket_idx=bucket_idx))
  bucket_idx = m_end_local // cfgs.tiles.bucket_base
  lax.switch(bucket_idx, branches)


def kernel_main(
    lhs_group_sizes_ref: jax.Array,
    group_offset_ref: jax.Array,
    lhs_ref: LhsRef,
    rhs_ref: WeightsRef,
    out_scale_ref: jax.Array | None,
    glu_out_scale_ref: jax.Array | None,
    out_ref: OutRef,
    partial_out_ref: jax.Array,
    partial_act_ref: jax.Array | None,
    acc_ref: jax.Array,
    metadata_ref: MetadataRef,
    zero_ref: jax.Array | None,
    semaphore_ref: jax.Array | None,
    *,
    cfgs: GmmConfigs,
    out_offset_ref: jax.Array | None = None,
    gather_refs: GatherRefs | None = None,
):
  """Entry point for GMM kernel."""
  num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
  num_n = pl.cdiv(cfgs.out_size_n, cfgs.tiles.tile_n)
  if cfgs.rhs_cfgs.should_unpack:
    rhs_weight = rhs_ref.weight.bitcast(cfgs.rhs_cfgs.quant_dtype)
    rhs_ref = dataclasses.replace(rhs_ref, weight=rhs_weight)
  num_gm = fill_metadata(lhs_group_sizes_ref, group_offset_ref, metadata_ref, cfgs=cfgs)
  if gather_refs is not None:
    @pl.when(num_gm > 0)
    @jax.named_scope("lhs_gather_prologue")
    def _():
      idx_copy = gather_refs.idx_copy(metadata_ref, 0, 0)
      idx_copy.start()
      idx_copy.wait()
      gather_refs.start_rows(metadata_ref, 0, 0, cfgs.dims.size_lhs_sublane)

      @pl.when(num_gm > 1)
      def _():
        gather_refs.idx_copy(metadata_ref, 1, 1).start()

  if cfgs.zero_init:
    zero_size = zero_out_start(out_ref.out, zero_ref, semaphore_ref, metadata_ref, num_gm, dims=cfgs.dims)
  (lhs_spec, rhs_spec), out_spec = generate_block_specs(metadata_ref, cfgs, out_offset_ref)
  if cfgs.fuse_act is not None:
    rhs_up_ref = jax.tree.map(lambda x: x.at[..., cfgs.out_size_n :], rhs_ref)
    rhs_ref = FusedWeightsRef(gate=rhs_ref, up=rhs_up_ref)
    rhs_spec = FusedWeightsRef(gate=rhs_spec, up=rhs_spec)
  core_axis_name = None if cfgs.disable_multi_core_mode else "core"
  dimension_semantics = None if cfgs.disable_multi_core_mode else (pltpu.PARALLEL, pltpu.ARBITRARY, pltpu.ARBITRARY)
  pipeline_fn = pltpu.emit_pipeline(
      functools.partial(
          inner_kernel, cfgs=cfgs, out_scale_ref=out_scale_ref,
          glu_out_scale_ref=glu_out_scale_ref, partial_act_ref=partial_act_ref,
      ),
      grid=(num_n, num_gm, num_k), in_specs=(lhs_spec, rhs_spec), out_specs=out_spec,
      core_axis_name=core_axis_name, dimension_semantics=dimension_semantics,
  )
  if lhs_ref.value is None:
    lhs_value_in = None
  elif cfgs.lhs_is_3d:
    lhs_value_in = lhs_ref.value.reshape(-1, cfgs.dims.size_lhs_sublane, *lhs_ref.value.shape[-2:])
  else:
    lhs_value_in = lhs_ref.value.reshape(-1, cfgs.dims.size_lhs_sublane, lhs_ref.value.shape[-1])
  glu_coeffs_in = None
  if lhs_ref.glu_coeffs is not None:
    glu_coeffs_in = lhs_ref.glu_coeffs.reshape(-1, cfgs.dims.size_lhs_sublane, 1)
  lhs_in = LhsRef(value=lhs_value_in, scale=lhs_ref.scale, glu_coeffs=glu_coeffs_in)
  out_val_in = out_ref.out.reshape(-1, cfgs.dims.size_lhs_sublane, *out_ref.out.shape[1:])
  act_val_in = None
  if out_ref.act is not None:
    act_val_in = out_ref.act.reshape(-1, cfgs.dims.size_lhs_sublane, out_ref.act.shape[-1])
  out_in = OutRef(out=out_val_in, act=act_val_in)
  scratches: list[Any] = [partial_out_ref, acc_ref, metadata_ref]
  if gather_refs is not None:
    scratches.append(gather_refs)
  pipeline_fn(lhs_in, rhs_ref, out_in, scratches=scratches)
  if cfgs.zero_init:
    zero_out_end(out_ref.out, semaphore_ref, zero_size, dims=cfgs.dims)


def _kernel_main_into_out(
    lhs_group_sizes_ref: jax.Array, group_offset_ref: jax.Array, out_offset_ref: jax.Array,
    lhs_ref: LhsRef, rhs_ref: WeightsRef, out_scale_ref: jax.Array | None,
    glu_out_scale_ref: jax.Array | None, aliased_out_ref: jax.Array, out_ref: OutRef, *scratch_refs,
    cfgs: GmmConfigs,
):
  """`kernel_main` writing into an existing `out` aliased with the output."""
  del aliased_out_ref
  kernel_main(
      lhs_group_sizes_ref, group_offset_ref, lhs_ref, rhs_ref, out_scale_ref, glu_out_scale_ref,
      out_ref, *scratch_refs, cfgs=cfgs, out_offset_ref=out_offset_ref,
  )


def _kernel_main_gather_into_out(
    lhs_group_sizes_ref: jax.Array, group_offset_ref: jax.Array, out_offset_ref: jax.Array,
    gather_start_ref: jax.Array, lhs_ref: LhsRef, rhs_ref: WeightsRef, out_scale_ref: jax.Array | None,
    glu_out_scale_ref: jax.Array | None, gather_indices_ref: jax.Array, aliased_out_ref: jax.Array,
    out_ref: OutRef, *scratch_refs, cfgs: GmmConfigs, idx_window: int, max_win_start: int,
):
  """`_kernel_main_into_out` gathering the lhs rows in-kernel."""
  del aliased_out_ref
  *scratch_refs, buf_ref, row_sems, idx_smem_ref, idx_sems = scratch_refs
  gather_refs = GatherRefs(
      start_ref=gather_start_ref, idx_hbm_ref=gather_indices_ref, x_hbm_ref=lhs_ref.value,
      buf_ref=buf_ref, row_sems=row_sems, idx_smem_ref=idx_smem_ref, idx_sems=idx_sems,
      num_k=cfgs.dims.size_k // cfgs.tiles.tile_k, idx_window=idx_window, max_win_start=max_win_start,
  )
  kernel_main(
      lhs_group_sizes_ref, group_offset_ref, dataclasses.replace(lhs_ref, value=None), rhs_ref,
      out_scale_ref, glu_out_scale_ref, out_ref, *scratch_refs, cfgs=cfgs,
      out_offset_ref=out_offset_ref, gather_refs=gather_refs,
  )


def calculate_tiling(
    dims: Dimensions, lhs_cfgs: InputConfigs, rhs_cfgs: InputConfigs, vmem_limit_bytes: int,
    fuse_act: str | None = None, *, out_is_3d: bool = False, flat_rhs: bool = False,
    has_glu_coeffs: bool = False,
) -> TileSizes:
  """Calculate optimal tile sizes for GMM kernel."""
  lhs_dtype = lhs_cfgs.dtype
  rhs_dtype = rhs_cfgs.quant_dtype or rhs_cfgs.dtype
  lhs_bits = jax.dtypes.itemsize_bits(lhs_dtype)
  rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)
  bf16_bf16_tile_m = 256 if rhs_cfgs.is_transposed else 128
  rhs_mod = min(pl.cdiv(16, rhs_bits), 2)
  tile_m = bf16_bf16_tile_m // rhs_mod
  if lhs_cfgs.should_quantize:
    tile_m *= 2
  tile_m = min(tile_m, dims.size_m)
  tile_n_limit = pltpu.get_tpu_info().mxu_column_size * 2
  tile_n_limit = min(tile_n_limit, dims.size_n)
  size_n_per_rhs = dims.size_n
  fuse_act_factor = 1
  if fuse_act is not None:
    fuse_act_factor = 2
    size_n_per_rhs //= fuse_act_factor
    tile_n_limit //= fuse_act_factor

  def _is_tile_k_quant_block_compatible(tk: int) -> bool:
    if tk % rhs_cfgs.quant_block_size != 0 and rhs_cfgs.quant_block_size % tk != 0:
      return False
    return True

  num_k_tiles = num_n_tiles = 1
  num_lanes = pltpu.get_tpu_info().num_lanes

  def _align_3d(size: int) -> int:
    return 8 * num_lanes if size >= 8 * num_lanes else size

  k_align = _align_3d(dims.size_k) if lhs_cfgs.is_3d else num_lanes
  n_align = _align_3d(size_n_per_rhs) if out_is_3d else num_lanes
  tile_k = align_to(dims.size_k, k_align)
  tile_n = align_to(size_n_per_rhs, n_align)
  sublane_tiling = pltpu.get_tpu_info().get_sublane_tiling(lhs_dtype)

  def _slab_padded(tile: int) -> int:
    return align_to(tile // num_lanes, sublane_tiling) * num_lanes

  def _gmm_vmem_estimate(tm: int, tn: int, tk: int) -> int:
    lhs_tile_bytes = lhs_bits // 8
    tk_vmem = _slab_padded(tk) if lhs_cfgs.is_3d else tk
    lhs_vmem = 2 * tm * tk_vmem * lhs_tile_bytes
    if lhs_cfgs.is_3d or out_is_3d:
      lhs_vmem += 2 * 1024 * 1024
    if lhs_cfgs.should_quantize:
      lhs_quant_bits = jax.dtypes.itemsize_bits(lhs_cfgs.quant_dtype)
      lhs_vmem += tm * tk * (lhs_quant_bits // 8)
    rhs_weight_vmem = tk * tn * rhs_bits // 8
    rhs_scale_vmem = 0
    if rhs_cfgs.has_scale and rhs_cfgs.quant_block_size is not None:
      num_quant_blocks_per_tile_k = pl.cdiv(tk, rhs_cfgs.quant_block_size)
      rhs_scale_vmem = num_quant_blocks_per_tile_k * tn * 4
    rhs_bias_vmem = 0
    if rhs_cfgs.has_bias:
      rhs_bias_vmem = tn * 4
    rhs_vmem = fuse_act_factor * (3 * rhs_weight_vmem + 2 * rhs_scale_vmem + 2 * rhs_bias_vmem)
    acc_cols = fuse_act_factor * tn
    acc_dtype_bytes = 2 if lhs_cfgs.quant_dtype is not None else 4
    acc_vmem = tm * acc_cols * acc_dtype_bytes
    out_dtype_bytes = jax.dtypes.itemsize_bits(lhs_cfgs.dtype) // 8
    tn_vmem = _slab_padded(tn) if out_is_3d else tn
    out_vmem = 2 * tm * tn_vmem * out_dtype_bytes
    if has_glu_coeffs:
      out_vmem += 2 * tm * (tn // 2) * 2 + 2 * tm * num_lanes * 2
    return lhs_vmem + rhs_vmem + acc_vmem + out_vmem

  def _is_tile_k_valid(tk: int) -> bool:
    if not _is_tile_k_quant_block_compatible(tk):
      return False
    if flat_rhs and not rhs_cfgs.is_transposed and dims.size_k % tk != 0:
      return False
    return True

  def _is_tile_n_valid(tn: int) -> bool:
    if flat_rhs and rhs_cfgs.is_transposed and size_n_per_rhs % tn != 0:
      return False
    return True

  def _shrink_tile_k():
    nonlocal tile_k, num_k_tiles
    while (_gmm_vmem_estimate(tile_m, tile_n, tile_k) > vmem_limit_bytes or not _is_tile_k_valid(tile_k)) and tile_k > k_align:
      num_k_tiles += 1
      tile_k = align_to(dims.size_k, num_k_tiles * k_align) // num_k_tiles

  tile_n_floor = dims.size_n if has_glu_coeffs else max(tile_n_limit, n_align)
  while (_gmm_vmem_estimate(tile_m, tile_n, tile_k) > vmem_limit_bytes or not _is_tile_n_valid(tile_n)) and tile_n > tile_n_floor:
    num_n_tiles += 1
    tile_n = align_to(size_n_per_rhs, num_n_tiles * n_align) // num_n_tiles
  if tile_n < tile_n_limit:
    num_n_tiles -= 1
    tile_n = align_to(size_n_per_rhs, num_n_tiles * n_align) // num_n_tiles
    _shrink_tile_k()
  elif (out_is_3d or has_glu_coeffs) and _gmm_vmem_estimate(tile_m, tile_n, tile_k) > vmem_limit_bytes:
    _shrink_tile_k()
  if rhs_cfgs.has_scale and rhs_cfgs.quant_block_size is not None and not _is_tile_k_quant_block_compatible(tile_k):
    tile_k = rhs_cfgs.quant_block_size
  if tile_n == 0 or tile_k == 0:
    final_estimate = _gmm_vmem_estimate(tile_m, tile_n, tile_k)
    raise ValueError(f"Could not find valid tile sizes for {dims=} and" f" {final_estimate=} (limit: {vmem_limit_bytes}).")
  max_num_buckets = 4
  bucket_base = tile_m
  for _ in range(1, max_num_buckets):
    new_tile_m = tile_m + bucket_base
    if new_tile_m > dims.size_m:
      break
    if _gmm_vmem_estimate(new_tile_m, tile_n, tile_k) > vmem_limit_bytes:
      break
    tile_m = new_tile_m
  return TileSizes(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, bucket_base=bucket_base)


def is_manually_cast_matmul_dtype_combo(lhs_dtype: jax.typing.DTypeLike, rhs_dtype: jax.typing.DTypeLike) -> bool:
  """Whether the kernel explicitly casts a matmul dtype combination."""
  manual_cast_dtype_combo = {
      (jnp.dtype(jnp.float8_e4m3fn), jnp.dtype(jnp.int4)),
      (jnp.dtype(jnp.float8_e5m2), jnp.dtype(jnp.int4)),
      (jnp.dtype(jnp.float8_e4m3fn), jnp.dtype(jnp.bfloat16)),
      (jnp.dtype(jnp.float8_e5m2), jnp.dtype(jnp.bfloat16)),
  }
  return (jnp.dtype(lhs_dtype), jnp.dtype(rhs_dtype)) in manual_cast_dtype_combo


def validate_inputs(
    lhs: jax.Array, rhs: jax.Array, rhs_scale: jax.Array | None, rhs_bias: jax.Array | None,
    group_sizes: jax.Array, group_offset: jax.Array, fuse_act: str | None = None,
    maybe_quantize_lhs: bool = True, lhs_scale: jax.Array | None = None, transpose_rhs: bool = False,
    lhs_quant_dtype: jnp.dtype | None = None, packing_factor: int = 1,
) -> Dimensions:
  """Validates the inputs for the GMM kernel."""
  size_m = lhs.shape[0]
  if transpose_rhs:
    size_group, size_n, size_k = rhs.shape
  else:
    size_group, size_k, size_n = rhs.shape
  size_k *= packing_factor
  size_lhs_group = group_sizes.shape[0]
  assert size_group <= size_lhs_group
  if lhs.ndim == 3:
    num_lanes = pltpu.get_tpu_info().num_lanes
    if lhs.dtype != jnp.bfloat16 and not is_fp8(lhs.dtype):
      raise ValueError(
          "3D lhs [size_m, size_k // 128, 128] is only supported for bfloat16"
          " and fp8 (the in-VMEM relayout packs 16 / 8-bit rows into words);" f" got {lhs.dtype}."
      )
    if lhs.shape != (size_m, size_k // num_lanes, num_lanes):
      raise ValueError(
          f"3D lhs must have shape ({size_m}, {size_k} // {num_lanes},"
          f" {num_lanes}) to match rhs contracting dim {size_k}; got" f" {lhs.shape}."
      )
    if size_k % num_lanes != 0:
      raise ValueError(f"3D lhs requires {size_k=} % {num_lanes} == 0.")
  else:
    assert lhs.shape == (size_m, size_k)
  if transpose_rhs:
    if rhs_scale is not None:
      raise NotImplementedError("transpose_rhs does not support quantized RHS (rhs_scale is not" " None).")
    if rhs_bias is not None:
      raise NotImplementedError("transpose_rhs does not support RHS bias (rhs_bias is not None).")
    if fuse_act is not None:
      raise NotImplementedError("transpose_rhs does not support fuse_act.")
    assert rhs.shape == (size_group, size_n, size_k // packing_factor)
  else:
    assert rhs.shape == (size_group, size_k // packing_factor, size_n)
  if rhs_bias is not None:
    assert rhs_bias.shape == (size_group, 1, size_n)
  if rhs_scale is not None:
    num_quant_blocks = rhs_scale.shape[1]
    assert rhs_scale.shape == (size_group, num_quant_blocks, 1, size_n), (
        f"rhs_scale shape {rhs_scale.shape}. Expecting ({size_group}," f" {num_quant_blocks}, 1, {size_n})"
    )
    assert size_k % num_quant_blocks == 0
  if lhs_scale is not None:
    assert maybe_quantize_lhs, "lhs_scale requires maybe_quantize_lhs=True."
    assert lhs_scale.shape == (1, 1), "Only per-tensor lhs_scale of shape (1, 1) is supported, got " f"{lhs_scale.shape}."
  assert group_offset.shape == (1,)
  if lhs_quant_dtype is not None:
    if not maybe_quantize_lhs:
      raise ValueError("lhs_quant_dtype cannot be set when maybe_quantize_lhs is False.")
    tpu_info = pltpu.get_tpu_info()
    if (
        not tpu_info.is_matmul_supported(lhs_quant_dtype, rhs.dtype)
        and not is_manually_cast_matmul_dtype_combo(lhs_quant_dtype, rhs.dtype)
        and not (is_fp8(lhs_quant_dtype) and is_fp8(rhs.dtype))
    ):
      raise ValueError(f"lhs_quant_dtype ({lhs_quant_dtype}) and rhs dtype ({rhs.dtype}) " "are not a supported combination.")
  size_lhs_sublane = _get_lhs_sublane_size(lhs.dtype, size_m)
  if fuse_act is not None:
    num_lanes = pltpu.get_tpu_info().num_lanes
    if size_n % (2 * num_lanes) != 0:
      raise ValueError(
          f"{size_n=} should be divisible by 2 * num_lanes when fuse_act is "
          "enabled since we need to split n dimension for gate and up."
      )
  return Dimensions(
      size_m=size_m, size_k=size_k, size_n=size_n, size_group=size_group,
      size_lhs_group=size_lhs_group, size_lhs_sublane=size_lhs_sublane,
  )


def get_cost_estimate(cfgs: GmmConfigs):
  """Returns the cost estimate for the GMM kernel."""
  dims = cfgs.dims
  lhs_dtype = cfgs.lhs_cfgs.quant_dtype or cfgs.lhs_cfgs.dtype
  rhs_dtype = cfgs.rhs_cfgs.quant_dtype or cfgs.rhs_cfgs.dtype
  rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)
  fp32_bytes = jnp.dtype(jnp.float32).itemsize
  flops = 2 * dims.size_m * dims.size_k * dims.size_n
  lhs_bytes = dims.size_m * dims.size_k * jnp.dtype(lhs_dtype).itemsize
  rhs_size = dims.size_group * dims.size_k * dims.size_n
  rhs_bytes = rhs_size * rhs_bits // 8
  if cfgs.rhs_cfgs.has_scale:
    num_quant_blocks = pl.cdiv(dims.size_k, cfgs.rhs_cfgs.quant_block_size)
    rhs_bytes += dims.size_group * num_quant_blocks * dims.size_n * fp32_bytes
  if cfgs.rhs_cfgs.has_bias:
    rhs_bytes += dims.size_group * dims.size_n * fp32_bytes
  out_bytes = dims.size_m * cfgs.out_size_n * cfgs.out_dtype.itemsize
  total_bytes = lhs_bytes + rhs_bytes + out_bytes
  return pl.CostEstimate(flops=flops, bytes_accessed=total_bytes, transcendentals=0)


def get_scope_name(cfgs: GmmConfigs) -> str:
  dims = cfgs.dims
  tiles = cfgs.tiles
  name = (
      f"gmm_v2-g_{dims.size_group}-m_{dims.size_m}-k_{dims.size_k}-act_{cfgs.fuse_act}"
      f"-n_{dims.size_n}-tm_{tiles.tile_m}-tk_{tiles.tile_k}-tn_{tiles.tile_n}"
  )
  if cfgs.has_glu_coeffs:
    name += "-glu"
  if cfgs.lhs_gather:
    name += "-gather"
  return name


def make_gmm_configs(
    lhs: jax.Array, rhs: jax.Array, rhs_scale: jax.Array | None, rhs_bias: jax.Array | None,
    group_sizes: jax.Array, group_offset: jax.Array, *, tile_info: TileSizes | TileFn,
    vmem_limit_bytes: int | None, out_dtype: jnp.dtype | None, acc_dtype: jnp.dtype | None,
    maybe_quantize_lhs: bool, zero_initialize: bool, fuse_act: str | None = None,
    lhs_scale: jax.Array | None = None, transpose_rhs: bool = False, lhs_quant_dtype: jnp.dtype | None = None,
    rhs_quant_dtype: jnp.dtype | None = None, disable_multi_core_mode: bool = False,
    out_is_3d: bool = False, flat_rhs: bool = False, has_out_scale: bool = False,
    has_glu_coeffs: bool = False, glu_out_dtype: jnp.dtype | None = None, has_glu_out_scale: bool = False,
):
  """Fills the GMM config for the GMM kernel."""
  packing_factor = get_packing_factor(rhs.dtype, rhs_quant_dtype)
  dims = validate_inputs(
      lhs, rhs, rhs_scale, rhs_bias, group_sizes, group_offset, fuse_act, maybe_quantize_lhs,
      lhs_scale, transpose_rhs=transpose_rhs, lhs_quant_dtype=lhs_quant_dtype, packing_factor=packing_factor,
  )
  if rhs_scale is not None:
    has_scale = True
    rhs_quant_dtype = rhs_quant_dtype or rhs.dtype
    num_blocks = rhs_scale.shape[1]
    block_size = dims.size_k // num_blocks
  else:
    has_scale = False
    block_size = dims.size_k
  rhs_cfgs = InputConfigs(
      quant_dtype=rhs_quant_dtype, quant_block_size=block_size, dtype=rhs.dtype,
      has_bias=rhs_bias is not None, has_scale=has_scale, is_transposed=transpose_rhs,
  )
  lhs_q_dtype = None
  if maybe_quantize_lhs and rhs_cfgs.should_dequantize_after_matmul:
    if lhs_quant_dtype is not None:
      lhs_q_dtype = lhs_quant_dtype
    else:
      assert rhs_quant_dtype is not None
      is_rhs_float = jnp.issubdtype(rhs_quant_dtype, jnp.floating)
      tpu_info = pltpu.get_tpu_info()
      if tpu_info.fp8_ops_per_second > 0:
        is_rhs_4bits = jax.dtypes.itemsize_bits(rhs_quant_dtype) == 4
        if is_rhs_float or is_rhs_4bits:
          lhs_q_dtype = jnp.float8_e4m3fn.dtype
      if tpu_info.int8_ops_per_second > 0:
        if not is_rhs_float:
          lhs_q_dtype = jnp.int8.dtype
  elif maybe_quantize_lhs and lhs_scale is not None and lhs_quant_dtype is not None:
    lhs_q_dtype = lhs_quant_dtype
  if lhs_scale is not None:
    assert lhs_q_dtype is not None, (
        "lhs_scale requires lhs quantization to engage, but no lhs quant "
        "dtype was selected. Ensure rhs is quantized and the hardware supports " "fp8/int8 matmul."
    )
  has_lhs_scale = lhs_scale is not None and lhs_q_dtype is not None
  lhs_cfgs = InputConfigs(
      quant_dtype=lhs_q_dtype, quant_block_size=dims.size_k if has_lhs_scale else 512,
      dtype=lhs.dtype, has_scale=has_lhs_scale, is_3d=lhs.ndim == 3,
  )
  if out_dtype is None:
    out_dtype = lhs.dtype
  if acc_dtype is None:
    if lhs_cfgs.quant_dtype is None:
      acc_dtype = jnp.float32.dtype
    else:
      acc_dtype = jnp.bfloat16.dtype
  if isinstance(tile_info, TileSizes):
    tiles = tile_info
  else:
    if out_is_3d:
      tile_info = functools.partial(tile_info, out_is_3d=True)
    if flat_rhs:
      tile_info = functools.partial(tile_info, flat_rhs=True)
    if has_glu_coeffs:
      tile_info = functools.partial(tile_info, has_glu_coeffs=True)
    tiles = tile_info(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act)
  num_lanes = pltpu.get_tpu_info().num_lanes
  if lhs_cfgs.is_3d:
    if tiles.tile_k % num_lanes != 0:
      raise ValueError(f"3D lhs requires tile_k % {num_lanes} == 0; got {tiles.tile_k=}.")
    relayout.check_tile_d0(tiles.tile_k // num_lanes, full_d0=dims.size_k // num_lanes, pack=relayout.packing(lhs.dtype))
  if out_is_3d:
    if jnp.dtype(out_dtype) != jnp.bfloat16:
      raise ValueError(
          "3D output [size_m, out_size_n // 128, 128] is only supported for"
          f" bfloat16 (the in-VMEM relayout packs bf16 pairs); got {out_dtype}."
      )
    out_size_n = dims.size_n if fuse_act is None else dims.size_n // 2
    if out_size_n % num_lanes != 0:
      raise ValueError(f"3D output requires out_size_n ({out_size_n}) % {num_lanes} == 0.")
    if tiles.tile_n % num_lanes != 0:
      raise ValueError(f"3D output requires tile_n % {num_lanes} == 0; got {tiles.tile_n=}.")
    relayout.check_tile_d0(tiles.tile_n // num_lanes, full_d0=out_size_n // num_lanes)
  if flat_rhs:
    if packing_factor > 1 or has_scale or fuse_act is not None:
      raise ValueError(
          "flat_rhs supports only unquantized, unpacked rhs without fuse_act;"
          f" got {rhs.dtype=}, {rhs_quant_dtype=}, {has_scale=}, {fuse_act=}."
      )
    if transpose_rhs:
      rows, tile_rows = dims.size_n, tiles.tile_n
    else:
      rows, tile_rows = dims.size_k, tiles.tile_k
    if rows % tile_rows != 0:
      raise ValueError(
          f"flat_rhs requires the rhs row tile ({tile_rows}) to divide the rhs"
          f" rows per group ({rows}); {transpose_rhs=}."
      )
  if has_glu_coeffs:
    if fuse_act is not None or out_is_3d:
      raise ValueError("glu_coeffs is not supported with fuse_act or out_is_3d.")
    if tiles.tile_n != dims.size_n or dims.size_n % (2 * num_lanes) != 0:
      raise ValueError(
          f"glu_coeffs requires tile_n ({tiles.tile_n}) == size_n"
          f" ({dims.size_n}) and size_n divisible by {2 * num_lanes}."
      )
  return GmmConfigs(
      dims=dims, tiles=tiles, lhs_cfgs=lhs_cfgs, rhs_cfgs=rhs_cfgs, out_dtype=jnp.dtype(out_dtype),
      acc_dtype=jnp.dtype(acc_dtype), zero_init=zero_initialize, fuse_act=fuse_act, transpose_rhs=transpose_rhs,
      disable_multi_core_mode=disable_multi_core_mode, out_is_3d=out_is_3d, flat_rhs=flat_rhs,
      has_out_scale=has_out_scale, has_glu_coeffs=has_glu_coeffs,
      glu_out_dtype=(jnp.dtype(glu_out_dtype) if glu_out_dtype is not None else None),
      has_glu_out_scale=has_glu_out_scale,
  )


def get_metadata(cfgs: GmmConfigs) -> dict[str, str | int | float]:
  cfgs_dict = dataclasses.asdict(cfgs)
  ret = {}
  for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
    key = jax.tree_util.keystr(path, simple=True, separator=".")
    if not isinstance(val, str | int | float):
      val = str(val)
    ret[key] = val
  return ret


@jax.jit(
    static_argnames=[
        "tile_info", "vmem_limit_bytes", "precision", "preferred_element_type", "acc_dtype",
        "maybe_quantize_lhs", "lhs_quant_dtype", "rhs_quant_dtype", "zero_initialize", "fuse_act",
        "transpose_rhs", "disable_multi_core_mode", "out_is_3d", "flat_rhs", "glu_out_dtype", "lhs_gather_size_m",
    ],
    donate_argnames=["out"],
)
def gmm_v2(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    group_offset: jax.Array | None = None,
    lhs_scale: jax.Array | None = None,
    out_scale: jax.Array | None = None,
    *,
    tile_info: TileSizes | TileFn = calculate_tiling,
    vmem_limit_bytes: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jnp.dtype | None = None,
    acc_dtype: jnp.dtype | None = None,
    maybe_quantize_lhs: bool = True,
    lhs_quant_dtype: jnp.dtype | None = None,
    rhs_quant_dtype: jnp.dtype | None = None,
    zero_initialize: bool = True,
    fuse_act: str | None = None,
    transpose_rhs: bool = False,
    disable_multi_core_mode: bool = False,
    out_is_3d: bool = False,
    flat_rhs: bool = False,
    out: jax.Array | None = None,
    out_offset: jax.Array | int = 0,
    glu_coeffs: jax.Array | None = None,
    glu_out_dtype: jnp.dtype | None = None,
    glu_out_scale: jax.Array | None = None,
    lhs_gather_indices: jax.Array | None = None,
    lhs_gather_start: jax.Array | int = 0,
    lhs_gather_size_m: int | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """GMM kernel implemented with emit_pipeline."""
  del precision
  if out is not None:
    if not disable_multi_core_mode:
      raise ValueError("`out` requires disable_multi_core_mode=True (`pl.kernel` has no" " input/output aliasing).")
    if out_is_3d or out.ndim != 2:
      raise ValueError(f"`out` must be a 2D output; got {out.shape=}.")
    zero_initialize = False
  gather_src = None
  size_m = lhs.shape[0]
  if lhs_gather_indices is not None:
    if out is None or lhs.ndim != 3 or lhs_scale is not None:
      raise ValueError(
          "`lhs_gather_indices` requires `out` and a 3D lhs without"
          f" `lhs_scale`; got {lhs.shape=}, {out is None=}," f" {lhs_scale is None=}."
      )
    if lhs_gather_indices.ndim != 1:
      raise ValueError(f"`lhs_gather_indices` must be 1D; got {lhs_gather_indices.shape=}.")
    if lhs_gather_size_m is None or lhs_gather_size_m <= 0:
      raise ValueError("`lhs_gather_indices` requires a positive `lhs_gather_size_m`; got" f" {lhs_gather_size_m}.")
    gather_src = lhs
    size_m = lhs_gather_size_m
  if glu_coeffs is not None and glu_coeffs.shape != (size_m, 1):
    raise ValueError(f"glu_coeffs must have shape ({size_m}, 1); got {glu_coeffs.shape}.")
  size_lhs_sublane = _get_lhs_sublane_size(lhs.dtype, size_m)
  padded_size_m = align_to(size_m, size_lhs_sublane)
  if gather_src is not None:
    lhs = jax.ShapeDtypeStruct((padded_size_m, *gather_src.shape[1:]), gather_src.dtype)
    if glu_coeffs is not None and padded_size_m != size_m:
      glu_coeffs = jnp.pad(glu_coeffs, ((0, padded_size_m - size_m), (0, 0)))
  elif padded_size_m != size_m:
    lhs = jnp.pad(lhs, ((0, padded_size_m - size_m),) + ((0, 0),) * (lhs.ndim - 1))
    if glu_coeffs is not None:
      glu_coeffs = jnp.pad(glu_coeffs, ((0, padded_size_m - size_m), (0, 0)))
  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  else:
    if jnp.isscalar(group_offset):
      group_offset = group_offset[None]
  if vmem_limit_bytes is None:
    vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
  cfgs = make_gmm_configs(
      lhs, rhs, rhs_scale, rhs_bias, group_sizes, group_offset, tile_info=tile_info,
      vmem_limit_bytes=vmem_limit_bytes, out_dtype=preferred_element_type, acc_dtype=acc_dtype,
      maybe_quantize_lhs=maybe_quantize_lhs, zero_initialize=zero_initialize, fuse_act=fuse_act,
      lhs_scale=lhs_scale, transpose_rhs=transpose_rhs, lhs_quant_dtype=lhs_quant_dtype,
      rhs_quant_dtype=rhs_quant_dtype, disable_multi_core_mode=disable_multi_core_mode,
      out_is_3d=out_is_3d, flat_rhs=flat_rhs, has_out_scale=out_scale is not None,
      has_glu_coeffs=glu_coeffs is not None, glu_out_dtype=glu_out_dtype, has_glu_out_scale=glu_out_scale is not None,
  )
  dims = cfgs.dims
  tiles = cfgs.tiles
  if gather_src is not None:
    if dims.size_k % tiles.tile_k != 0 or cfgs.lhs_cfgs.quant_dtype is not None or tiles.tile_n < cfgs.out_size_n:
      raise ValueError(
          "`lhs_gather_indices` requires tile_k to divide size_k, an lhs"
          " consumed as is and a single n tile; got"
          f" {tiles=}, {dims.size_k=}, {cfgs.out_size_n=}," f" {cfgs.lhs_cfgs.quant_dtype=}."
      )
    cfgs = dataclasses.replace(cfgs, lhs_gather=True)
  if cfgs.lhs_cfgs.should_use_external_scale and not cfgs.rhs_cfgs.has_scale:
    assert lhs_scale is not None
    out_scale = lhs_scale if out_scale is None else lhs_scale * out_scale
    cfgs = dataclasses.replace(cfgs, has_out_scale=True)
  if out_scale is not None:
    if out_scale.shape != (1, 1):
      raise ValueError(f"out_scale must be [1, 1]; got {out_scale.shape}.")
    out_scale = out_scale.astype(jnp.float32).reshape(1)
  if glu_out_scale is not None:
    if glu_out_scale.shape != (1, 1):
      raise ValueError(f"glu_out_scale must be [1, 1]; got {glu_out_scale.shape}.")
    glu_out_scale = glu_out_scale.astype(jnp.float32).reshape(1)
  smem = pl.BlockSpec(memory_space=pltpu.SMEM)
  out_scale_spec = smem if out_scale is not None else None
  glu_out_scale_spec = smem if glu_out_scale is not None else None
  if cfgs.lhs_cfgs.has_scale:
    assert lhs_scale is not None
    lhs_scale = lhs_scale.astype(jnp.float32)
  else:
    lhs_scale = None
  if rhs_scale is not None:
    rhs_scale = rhs_scale.astype(jnp.float32)
  if rhs_bias is not None:
    rhs_bias = rhs_bias.astype(jnp.float32)
  num_lanes = pltpu.get_tpu_info().num_lanes
  aligned_n = align_to(cfgs.out_size_n, num_lanes)
  out_d0 = aligned_n // num_lanes
  max_num_gm = dims.size_group + pl.cdiv(dims.size_m, tiles.tile_m) - 1
  acc_cols = 2 * tiles.tile_n if cfgs.fuse_act is not None else tiles.tile_n
  if cfgs.out_is_3d:
    partial_out_shape = (dims.size_lhs_sublane, cfgs.out_tile_d0, num_lanes)
  else:
    partial_out_shape = (dims.size_lhs_sublane, tiles.tile_n)
  act_dtype = cfgs.glu_out_dtype or cfgs.out_dtype
  partial_act_scratch = pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n // 2), act_dtype) if cfgs.has_glu_coeffs else None
  scratch_shapes = [
      pltpu.VMEM(partial_out_shape, cfgs.out_dtype), partial_act_scratch,
      pltpu.VMEM((tiles.tile_m, acc_cols), cfgs.acc_dtype),
      MetadataRef(
          gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
          gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
      ),
  ]
  if cfgs.zero_init:
    target_zero_ref_bytes = 2 * 1024 * 1024
    out_bytes = jnp.dtype(cfgs.out_dtype).itemsize
    if cfgs.out_is_3d:
      sublane = dims.size_lhs_sublane
      sublane_tiling = pltpu.get_tpu_info().get_sublane_tiling(cfgs.out_dtype)
      row_bytes = align_to(out_d0, sublane_tiling) * num_lanes * out_bytes
      tile_zero_m = target_zero_ref_bytes // row_bytes // sublane * sublane
      tile_zero_m = max(sublane, min(tile_zero_m, dims.size_m))
      zero_shape = (tile_zero_m, out_d0, num_lanes)
    else:
      tile_zero_m = target_zero_ref_bytes // num_lanes // out_bytes
      tile_zero_m = min(tile_zero_m, dims.size_m)
      zero_shape = (tile_zero_m, num_lanes)
    scratch_shapes += [pltpu.VMEM(zero_shape, cfgs.out_dtype), pltpu.SemaphoreType.DMA((1,))]
  else:
    scratch_shapes += [None, None]
  if cfgs.out_is_3d:
    out_shape = (dims.size_m, out_d0, num_lanes)
    out_slice = (slice(None, size_m),)
  else:
    out_shape = (dims.size_m, aligned_n)
    out_slice = (slice(None, size_m), slice(None, cfgs.out_size_n))
  lhs_manual_axis_type = jax.typeof(lhs if gather_src is None else gather_src).manual_axis_type
  out_init = jax.ShapeDtypeStruct(out_shape, cfgs.out_dtype, manual_axis_type=lhs_manual_axis_type)
  act_init = None
  if cfgs.has_glu_coeffs:
    act_init = jax.ShapeDtypeStruct((dims.size_m, dims.size_n // 2), act_dtype, manual_axis_type=lhs_manual_axis_type)
  lhs_in = LhsRef(value=lhs, scale=lhs_scale, glu_coeffs=glu_coeffs)
  if cfgs.flat_rhs:
    rhs = rhs.reshape(-1, rhs.shape[-1])
  rhs_weights = WeightsRef(weight=rhs, scale=rhs_scale, bias=rhs_bias)
  if out is not None:
    if out.shape[1] != aligned_n or out.shape[0] % dims.size_lhs_sublane != 0 or out.dtype != cfgs.out_dtype:
      raise ValueError(
          f"`out` must be [k * {dims.size_lhs_sublane}, {aligned_n}] of dtype"
          f" {cfgs.out_dtype}; got {out.shape=}, {out.dtype=}."
      )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    out_offset = jnp.asarray(out_offset, jnp.int32)[None]
    scalars = (group_sizes, group_offset, out_offset)
    inputs = [lhs_in, rhs_weights, out_scale, glu_out_scale]
    in_specs = [
        LhsRef(value=hbm, scale=hbm if cfgs.lhs_cfgs.has_scale else None, glu_coeffs=hbm if cfgs.has_glu_coeffs else None),
        WeightsRef(weight=hbm, scale=hbm if rhs_scale is not None else None, bias=hbm if rhs_bias is not None else None),
        out_scale_spec, glu_out_scale_spec,
    ]
    kernel = functools.partial(_kernel_main_into_out, cfgs=cfgs)
    if gather_src is not None:
      assert lhs_gather_indices is not None
      num_k = dims.size_k // tiles.tile_k
      idx_window = gather_idx_window(tiles.tile_m)
      indices = lhs_gather_indices.astype(jnp.int32)
      num_indices = indices.shape[0]
      indices = jnp.pad(indices, (0, max(-num_indices % _GATHER_IDX_ALIGN, idx_window - num_indices)))
      scalars += (jnp.asarray(lhs_gather_start, jnp.int32).reshape(1),)
      x_view = gather_src.reshape(-1, 1, cfgs.lhs_tile_d0, num_lanes)
      inputs[0] = dataclasses.replace(lhs_in, value=x_view)
      inputs.append(indices)
      in_specs.append(hbm)
      kernel = functools.partial(
          _kernel_main_gather_into_out, cfgs=cfgs, idx_window=idx_window, max_win_start=indices.shape[0] - idx_window,
      )
      scratch_shapes = scratch_shapes + [
          pltpu.VMEM((2 * num_k, tiles.tile_m, cfgs.lhs_tile_d0, num_lanes), gather_src.dtype),
          pltpu.SemaphoreType.DMA((2,)), pltpu.SMEM((2 * idx_window,), jnp.int32), pltpu.SemaphoreType.DMA((2,)),
      ]
    inputs.append(out)
    in_specs.append(hbm)
    res = pl.pallas_call(
        kernel,
        out_shape=OutRef(
            out=jax.ShapeDtypeStruct(out.shape, out.dtype, manual_axis_type=jax.typeof(out).manual_axis_type),
            act=act_init,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalars), in_specs=in_specs,
            out_specs=OutRef(out=hbm, act=hbm if cfgs.has_glu_coeffs else None), scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes, disable_bounds_checks=True),
        input_output_aliases={len(jax.tree.leaves((scalars, inputs))) - 1: 0},
        name=get_scope_name(cfgs), cost_estimate=get_cost_estimate(cfgs), metadata=get_metadata(cfgs),
    )(*scalars, *inputs)
    if cfgs.has_glu_coeffs:
      assert res.act is not None
      return res.out, res.act[:size_m]
    return res.out

  if cfgs.disable_multi_core_mode:
    lhs_scale_spec = None
    if cfgs.lhs_cfgs.has_scale:
      lhs_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    glu_coeffs_spec = None
    if cfgs.has_glu_coeffs:
      glu_coeffs_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    rhs_scale_spec = rhs_bias_spec = None
    if rhs_scale is not None:
      rhs_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    if rhs_bias is not None:
      rhs_bias_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    res = pl.pallas_call(
        functools.partial(kernel_main, cfgs=cfgs), out_shape=OutRef(out=out_init, act=act_init),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                LhsRef(value=pl.BlockSpec(memory_space=pltpu.HBM), scale=lhs_scale_spec, glu_coeffs=glu_coeffs_spec),
                WeightsRef(weight=pl.BlockSpec(memory_space=pltpu.HBM), scale=rhs_scale_spec, bias=rhs_bias_spec),
                out_scale_spec, glu_out_scale_spec,
            ],
            out_specs=OutRef(
                out=pl.BlockSpec(memory_space=pltpu.HBM),
                act=(pl.BlockSpec(memory_space=pltpu.HBM) if cfgs.has_glu_coeffs else None),
            ),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes, disable_bounds_checks=True),
        name=get_scope_name(cfgs), cost_estimate=get_cost_estimate(cfgs), metadata=get_metadata(cfgs),
    )(group_sizes, group_offset, lhs_in, rhs_weights, out_scale, glu_out_scale)
    if cfgs.has_glu_coeffs:
      assert res.act is not None
      return res.out[out_slice], res.act[:size_m]
    return res.out[out_slice]

  group_sizes = pltpu.with_memory_space_constraint(group_sizes, pltpu.SMEM)
  group_offset = pltpu.with_memory_space_constraint(group_offset, pltpu.SMEM)
  if out_scale is not None:
    out_scale = pltpu.with_memory_space_constraint(out_scale, pltpu.SMEM)
  if glu_out_scale is not None:
    glu_out_scale = pltpu.with_memory_space_constraint(glu_out_scale, pltpu.SMEM)
  res = pl.kernel(
      functools.partial(kernel_main, cfgs=cfgs), out_type=OutRef(out=out_init, act=act_init),
      mesh=pltpu.TensorCoreMesh(axis_name="core"), scratch_types=scratch_shapes,
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes, disable_bounds_checks=True),
      name=get_scope_name(cfgs), cost_estimate=get_cost_estimate(cfgs), metadata=get_metadata(cfgs),
  )(group_sizes, group_offset, lhs_in, rhs_weights, out_scale, glu_out_scale)
  if cfgs.has_glu_coeffs:
    assert res.act is not None
    return res.out[out_slice], res.act[:size_m]
  return res.out[out_slice]


tokamax_gmmv2 = gmm_v2
gmmv2 = gmm_v2
