# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused FP8 linear GMM and top-k combine for the DSv3 routed-expert MLP."""

from __future__ import annotations

import dataclasses
import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from lilax._backend.experimental.lineage.gmmv2 import gmmv2_kernel
from lilax._backend.experimental.lineage.gmmv2 import relayout

LANES = 128
_SUBLANE = 32
_IDX_ALIGN = 128
_MXU_COLS = 256
_ISSUE_UNROLL = 16
_SEEN_BITS = 32
_TOK_SLOTS = 4


@dataclasses.dataclass(frozen=True)
class Tiles:
  """Tile sizes of `linear_combine_fwd`."""

  tile_m: int = 256
  tile_n: int = 512
  bucket_m: int = 64
  zero_window: int = 256
  zero_rows: int = 32


DEFAULT_TILES = Tiles()


def _count_visits(group_sizes: jax.Array, tile_m: int) -> jax.Array:
  """Returns the number of visits, matching `_fill_metadata`."""
  gs = group_sizes.astype(jnp.int32)
  m_off = jnp.cumsum(gs) - gs
  n_tiles = jnp.where(gs > 0, (gs + (m_off % _SUBLANE) + tile_m - 1) // tile_m, 0)
  return jnp.sum(n_tiles)


def _fill_metadata(
    gs_ref, v2e_ref, v2m_ref, v2w_ref, w2e_ref, *, num_experts: int, tile_m: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Fills the visit tables."""
  v2m_ref[0] = 0

  def fill_expert(e, carry):
    v_idx, wl, m_off = carry
    size = gs_ref[e]
    end_m = m_off + size
    n_tiles = jnp.where(size > 0, pl.cdiv(size + m_off % _SUBLANE, tile_m), 0)

    @pl.when(size > 0)
    def _record_wload():
      w2e_ref[wl] = e

    def fill_tile(v, cur):
      rows = jnp.minimum(tile_m - cur % _SUBLANE, end_m - cur)
      v2e_ref[v] = e
      v2w_ref[v] = wl
      v2m_ref[v] = cur
      v2m_ref[v + 1] = cur + rows
      return cur + rows

    lax.fori_loop(v_idx, v_idx + n_tiles, fill_tile, m_off)
    return v_idx + n_tiles, wl + jnp.where(size > 0, 1, 0), end_m

  return lax.fori_loop(0, num_experts, fill_expert, (jnp.int32(0), jnp.int32(0), jnp.int32(0)))


def _unrolled_loop(n, unroll: int, fn):
  """Calls fn(i) for i in [0, n), unrolling the body `unroll` times."""
  num_full = n // unroll

  @pl.loop(0, num_full)
  def _full(c):
    for u in range(unroll):
      fn(c * unroll + u)

  @pl.loop(num_full * unroll, n)
  def _rest(i):
    fn(i)


def _unrolled_fori(n, unroll: int, body, init):
  """`lax.fori_loop(0, n, body, init)` with the body unrolled `unroll` times."""
  num_full = n // unroll

  def chunk(c, carry):
    for u in range(unroll):
      carry = body(c * unroll + u, carry)
    return carry

  carry = lax.fori_loop(0, num_full, chunk, init)
  return lax.fori_loop(num_full * unroll, n, body, carry)


def _pieces(n: int, parts: int) -> list[tuple[int, int]]:
  """Splits [0, n) into `parts` contiguous static ranges."""
  return [(p * n // parts, (p + 1) * n // parts) for p in range(parts)]


def _wait_rows(sem, ref, num_rows, max_rows: int):
  """Waits on sem for DMAs totalling num_rows leading rows shaped like ref's."""
  assert max_rows <= ref.shape[0], (max_rows, ref.shape)

  @pl.when(num_rows > 0)
  def _wait():
    rows = ref.at[pl.ds(0, num_rows)]
    pltpu.make_async_copy(rows, rows, sem).wait()


def _zero_runs(
    sorted_tokens: jax.Array, num_rows: jax.Array, *, num_out_tokens: int, num_windows: int, window: int, max_run: int,
) -> tuple[jax.Array, jax.Array]:
  """Returns the runs of tokens without rows, for the zero-fill."""
  size = num_windows * window
  tokens = sorted_tokens.astype(jnp.int32)[:size]
  tokens = jnp.pad(tokens, (0, size - tokens.shape[0]))
  idx = jnp.arange(size, dtype=jnp.int32)
  tokens = jnp.where(idx < num_rows, tokens, num_out_tokens)
  prev = jnp.concatenate([jnp.full((1,), -1, jnp.int32), tokens[:-1]])
  length = jnp.maximum(tokens - prev - 1, 0).reshape(num_windows, window)
  start = (prev + 1).reshape(num_windows, window)
  is_long = jnp.any(length > max_run, axis=1).astype(jnp.int32)
  return is_long, jnp.stack([start, length], axis=1).reshape(-1)


@dataclasses.dataclass(frozen=True)
class _Config:
  num_experts: int
  size_f: int
  size_d: int
  num_out_tokens: int
  tile_m: int
  tile_n: int
  tok_window: int
  zero_window: int
  zero_rows: int
  accumulate: bool
  bucket_m: int


def _kernel(gs_ref, start_ref, scale_ref, zmeta_ref, act_hbm, w_hbm, tok_hbm, *refs, cfg: _Config):
  """Kernel body over grid `(max(num_visits, 1),)`."""
  if cfg.accumulate:
    _, y_hbm, *scratch = refs
    zinfo_hbm = None
  else:
    zinfo_hbm, y_hbm, *scratch = refs
  (
      meta_ref, v2e_ref, v2m_ref, v2w_ref, w2e_ref, seen_ref, tok_smem, rmw_rows_smem,
      zwin_smem, zlist_smem, act_buf, w_buf, obuf, rbuf, zbuf, act_sem, tok_sem, w_sem,
      wr_sem, rd_sem, zero_sem, zt_sem,
  ) = scratch
  tile_m, size_f, size_d = cfg.tile_m, cfg.size_f, cfg.size_d
  nbt = cfg.num_out_tokens
  tw = cfg.tok_window
  ze = cfg.zero_window
  zrows = cfg.zero_rows
  num_visits_i, num_wloads_i, num_rows_i, zcount_i = range(4)
  wpending_i = 4
  rmw_count_i = 6
  v = pl.program_id(0)
  last_step = v == pl.num_programs(0) - 1

  def visit(idx):
    m_s = v2m_ref[idx]
    m_e = v2m_ref[idx + 1]
    a_s = m_s // _SUBLANE * _SUBLANE
    return v2e_ref[idx], m_s, m_e, a_s

  def act_dma(idx, slot):
    _, _, m_e, a_s = visit(idx)
    a_s = pl.multiple_of(a_s, _SUBLANE)
    fetch = pl.multiple_of(pl.cdiv(m_e - a_s, _SUBLANE) * _SUBLANE, _SUBLANE)
    return pltpu.make_async_copy(act_hbm.at[pl.ds(a_s, fetch)], act_buf.at[slot, pl.ds(0, fetch)], act_sem.at[slot])

  def tok_win_start(idx):
    row = start_ref[0] + v2m_ref[idx]
    return jnp.minimum(row // _IDX_ALIGN * _IDX_ALIGN, tok_hbm.shape[0] - tw)

  def tok_dma(idx):
    ws = pl.multiple_of(tok_win_start(idx), _IDX_ALIGN)
    slot = jnp.bitwise_and(idx, _TOK_SLOTS - 1)
    return pltpu.make_async_copy(tok_hbm.at[pl.ds(ws, tw)], tok_smem.at[pl.ds(slot * tw, tw)], tok_sem.at[slot])

  def w_dma(wl, slot):
    row = pl.multiple_of(w2e_ref[wl] * size_f, size_f)
    return pltpu.make_async_copy(w_hbm.at[pl.ds(row, size_f)], w_buf.at[slot], w_sem.at[slot])

  def zt_dma(win):
    assert zinfo_hbm is not None
    start = pl.multiple_of(win * 2 * ze, 2 * ze)
    return pltpu.make_async_copy(zinfo_hbm.at[pl.ds(start, 2 * ze)], zwin_smem, zt_sem.at[0])

  def wait_writes(slot):
    _wait_rows(wr_sem.at[slot], y_hbm, meta_ref[wpending_i + slot], tile_m)
    meta_ref[wpending_i + slot] = 0

  def write_row(obuf_row, tok_base, slot, i):
    t = tok_smem[tok_base + i]
    pltpu.make_async_copy(obuf.at[slot, pl.ds(obuf_row + i, 1)], y_hbm.at[pl.ds(t, 1)], wr_sem.at[slot]).start()

  def classify_row(tok_base, rmw_base, num_rows, i, cnt):
    t = tok_smem[tok_base + jnp.minimum(i, num_rows - 1)]
    word = jnp.right_shift(t, 5)
    bit = jnp.left_shift(jnp.int32(1), jnp.bitwise_and(t, _SEEN_BITS - 1))
    seen = seen_ref[word]
    seen_ref[word] = seen | bit
    rmw_rows_smem[rmw_base + cnt] = i
    dup = jnp.logical_and((seen & bit) != 0, i < num_rows)
    return cnt + dup.astype(jnp.int32)

  def visit_tok_base(idx):
    return jnp.bitwise_and(idx, _TOK_SLOTS - 1) * tw + start_ref[0] + v2m_ref[idx] - tok_win_start(idx)

  def classify_slow(idx):
    num_rows = v2m_ref[idx + 1] - v2m_ref[idx]
    meta_ref[rmw_count_i + jnp.bitwise_and(idx, 1)] = _unrolled_fori(
        num_rows, _ISSUE_UNROLL,
        functools.partial(classify_row, visit_tok_base(idx), jnp.bitwise_and(idx, 1) * tile_m, num_rows),
        jnp.int32(0),
    )

  def compact_zero_runs():
    def compact(j, cnt):
      zlist_smem[cnt] = j
      return cnt + (zwin_smem[ze + j] > 0).astype(jnp.int32)

    return _unrolled_fori(ze, _ISSUE_UNROLL, compact, jnp.int32(0))

  def zero_run(k, count):
    j = zlist_smem[k]
    length = zwin_smem[ze + j]
    pltpu.make_async_copy(zbuf.at[pl.ds(0, length)], y_hbm.at[pl.ds(zwin_smem[j], length)], zero_sem.at[0]).start()
    return count + length

  def zero_window_slow():
    def body(j, count):
      start = zwin_smem[j]
      length = zwin_smem[ze + j]
      num_full = length // zrows

      @pl.loop(0, num_full)
      def _full(q):
        pltpu.make_async_copy(zbuf, y_hbm.at[pl.ds(start + q * zrows, zrows)], zero_sem.at[0]).start()

      rem = length - num_full * zrows

      @pl.when(rem > 0)
      def _rest():
        pltpu.make_async_copy(
            zbuf.at[pl.ds(0, rem)], y_hbm.at[pl.ds(start + num_full * zrows, rem)], zero_sem.at[0],
        ).start()

      return count + length

    meta_ref[zcount_i] = lax.fori_loop(0, ze, body, meta_ref[zcount_i])

  @pl.when(v == 0)
  def _prologue():
    num_visits, num_wloads, num_rows = _fill_metadata(
        gs_ref, v2e_ref, v2m_ref, v2w_ref, w2e_ref, num_experts=cfg.num_experts, tile_m=tile_m,
    )
    meta_ref[num_visits_i] = num_visits
    meta_ref[num_wloads_i] = num_wloads
    meta_ref[num_rows_i] = num_rows
    meta_ref[zcount_i] = 0
    for slot in range(2):
      meta_ref[wpending_i + slot] = 0
      meta_ref[rmw_count_i + slot] = 0
    if not cfg.accumulate:
      zbuf[...] = jnp.zeros_like(zbuf)

      @pl.loop(0, seen_ref.shape[0])
      def _clear_seen(i):
        seen_ref[i] = 0

    @pl.when(num_visits > 0)
    def _start_first():
      w_dma(0, 0).start()
      act_dma(0, 0).start()
      tok_dma(0).start()
      if not cfg.accumulate:
        tok_dma(0).wait()
        classify_slow(0)

      @pl.when(num_visits > 1)
      def _second_tokens():
        tok_dma(1).start()

  num_visits = meta_ref[num_visits_i]
  num_wloads = meta_ref[num_wloads_i]
  num_rows = meta_ref[num_rows_i]
  num_windows = pl.cdiv(num_rows + 1, ze)

  @pl.when(v < num_visits)
  def _step():
    slot = jnp.bitwise_and(v, 1)
    e, m_s, m_e, a_s = visit(v)
    nrows = m_e - m_s
    lo = m_s - a_s
    wl = v2w_ref[v]
    wslot = jnp.bitwise_and(wl, 1)
    is_new_expert = jnp.logical_or(v == 0, e != v2e_ref[jnp.maximum(v - 1, 0)])
    has_next = v + 1 < num_visits
    wait_writes(slot)

    @pl.when(is_new_expert)
    def _wait_w():
      w_dma(wl, wslot).wait()

    act_dma(v, slot).wait()
    if cfg.accumulate:
      tok_dma(v).wait()

    @pl.when(has_next)
    def _prefetch_next():
      act_dma(v + 1, 1 - slot).start()

    @pl.when(v + 2 < num_visits)
    def _prefetch_tokens():
      tok_dma(v + 2).start()

    @pl.when(jnp.logical_and(is_new_expert, wl + 1 < num_wloads))
    def _prefetch_w():
      w_dma(wl + 1, 1 - wslot).start()

    if not cfg.accumulate:
      @pl.when(v < num_windows)
      def _load_zero_window():
        zt_dma(v).start()

    tok_base = visit_tok_base(v)
    rmw_base = slot * tile_m
    if cfg.accumulate:
      num_rmw = nrows
      rmw_row = lambda j: j
    else:
      num_rmw = meta_ref[rmw_count_i + slot]
      rmw_row = lambda j: rmw_rows_smem[rmw_base + j]

    @pl.when(is_new_expert)
    def _order_writes():
      wait_writes(1 - slot)

    def read_row(j):
      t = tok_smem[tok_base + rmw_row(j)]
      pltpu.make_async_copy(y_hbm.at[pl.ds(t, 1)], rbuf.at[pl.ds(j, 1)], rd_sem.at[0]).start()

    _unrolled_loop(num_rmw, _ISSUE_UNROLL, read_row)

    if not cfg.accumulate:
      @pl.when(v < num_windows)
      def _zero_window():
        zt_dma(v).wait()

        @pl.when(zmeta_ref[v] != 0)
        def _long_runs():
          zero_window_slow()

        @pl.when(zmeta_ref[v] == 0)
        def _short_runs():
          meta_ref[zcount_i] = _unrolled_fori(compact_zero_runs(), _ISSUE_UNROLL, zero_run, meta_ref[zcount_i])

      @pl.when(has_next)
      def _wait_next_tokens():
        tok_dma(v + 1).wait()

    nxt = jnp.minimum(v + 1, num_visits - 1)
    classify_next = functools.partial(
        classify_row, visit_tok_base(nxt), (1 - slot) * tile_m, v2m_ref[nxt + 1] - v2m_ref[nxt],
    )
    scale = scale_ref[0]
    num_groups = size_d // cfg.tile_n

    def gmm(bucket_m: int):
      cnt = jnp.int32(0)
      for g, piece in enumerate(_pieces(tile_m, num_groups)):
        c0 = g * cfg.tile_n
        lhs = act_buf[slot, :bucket_m]
        parts = []
        for sub in range(0, cfg.tile_n, _MXU_COLS):
          acc = lax.dot_general(
              lhs, w_buf[wslot, :, pl.ds(c0 + sub, _MXU_COLS)], (((1,), (0,)), ((), ())),
              preferred_element_type=jnp.float32,
          )
          parts.append((acc * scale).astype(jnp.bfloat16))
        relayout.store_2d_into_3d(obuf.at[slot], jnp.concatenate(parts, axis=-1), bucket_m, c0 // LANES)
        if not cfg.accumulate:
          for i in range(*piece):
            cnt = classify_next(i, cnt)
      if not cfg.accumulate:
        meta_ref[rmw_count_i + 1 - slot] = cnt

    buckets = list(range(cfg.bucket_m, tile_m + 1, cfg.bucket_m))
    if len(buckets) == 1:
      gmm(tile_m)
    else:
      lax.switch(
          jnp.right_shift(lo + nrows - 1, cfg.bucket_m.bit_length() - 1),
          [functools.partial(gmm, b) for b in buckets],
      )

    _wait_rows(rd_sem.at[0], rbuf, num_rmw, tile_m)

    @pl.loop(0, num_rmw)
    def _add(j):
      r = lo + rmw_row(j)
      obuf[slot, r] = (obuf[slot, r].astype(jnp.float32) + rbuf[j].astype(jnp.float32)).astype(jnp.bfloat16)

    _unrolled_loop(nrows, _ISSUE_UNROLL, functools.partial(write_row, lo, tok_base, slot))
    meta_ref[wpending_i + slot] = nrows

  @pl.when(last_step)
  def _epilogue():
    if not cfg.accumulate:
      @pl.loop(jnp.minimum(num_visits, num_windows), num_windows)
      def _rest_windows(win):
        zt_dma(win).start()
        zt_dma(win).wait()
        zero_window_slow()

      _wait_rows(zero_sem.at[0], y_hbm, meta_ref[zcount_i], nbt)
    wait_writes(0)
    wait_writes(1)


def linear_combine_fwd(
    act: jax.Array,
    w_q: jax.Array,
    group_sizes: jax.Array,
    token_indices: jax.Array,
    start: jax.Array | int,
    sorted_tokens: jax.Array | None,
    *,
    out_scale: jax.Array,
    num_out_tokens: int,
    out: jax.Array | None = None,
    tiles: Tiles | None = None,
    vmem_limit_bytes: int | None = None,
) -> jax.Array:
  """Returns `out` plus the top-k combine of the linear GMM of `act`."""
  if tiles is None:
    tiles = DEFAULT_TILES
  size_c, size_f = act.shape
  num_experts, w_f, size_d = w_q.shape
  if not (gmmv2_kernel.is_fp8(act.dtype) and gmmv2_kernel.is_fp8(w_q.dtype)):
    raise ValueError(f"act and w_q must be fp8, got {act.dtype}, {w_q.dtype}.")
  if w_f != size_f or size_d % LANES or group_sizes.shape != (num_experts,):
    raise ValueError(f"Bad shapes: act {act.shape}, w_q {w_q.shape}, group_sizes" f" {group_sizes.shape}.")
  if token_indices.ndim != 1:
    raise ValueError(f"token_indices must be 1D, got {token_indices.shape}.")
  d0 = size_d // LANES
  out_shape = (num_out_tokens, d0, LANES)
  accumulate = out is not None
  if accumulate:
    if out.shape != out_shape or out.dtype != jnp.bfloat16:
      raise ValueError(f"out must be bf16 {out_shape}, got {out.shape}.")
  elif sorted_tokens is None:
    raise ValueError("sorted_tokens is required without out.")
  tile_m = min(tiles.tile_m, pl.cdiv(size_c, _SUBLANE) * _SUBLANE)
  tile_n = min(tiles.tile_n, size_d)
  bucket_m = min(tiles.bucket_m, tile_m)
  if (
      tile_m % _SUBLANE or size_d % tile_n or tile_n % _MXU_COLS or tile_m % bucket_m
      or bucket_m % _SUBLANE or bucket_m & (bucket_m - 1)
  ):
    raise ValueError(f"Unsupported tiles {tiles} for D={size_d}.")
  if size_c % _SUBLANE:
    raise ValueError(f"act rows ({size_c}) must be a multiple of {_SUBLANE}.")

  tok_window = pl.cdiv(tile_m, _IDX_ALIGN) * _IDX_ALIGN + _IDX_ALIGN
  token_indices = token_indices.astype(jnp.int32)
  num_indices = token_indices.shape[0]
  tok_len = max(pl.cdiv(num_indices, _IDX_ALIGN) * _IDX_ALIGN, tok_window)
  if tok_len != num_indices:
    token_indices = jnp.pad(token_indices, (0, tok_len - num_indices))
  zero_window = tiles.zero_window
  zero_rows = tiles.zero_rows
  num_zero_windows = pl.cdiv(size_c + 1, zero_window)
  inputs = [act, w_q.reshape(num_experts * size_f, size_d), token_indices]
  if accumulate:
    zmeta = jnp.zeros((num_zero_windows,), jnp.int32)
    inputs.append(out)
  else:
    assert sorted_tokens is not None
    zmeta, zinfo = _zero_runs(
        sorted_tokens, jnp.sum(group_sizes, dtype=jnp.int32), num_out_tokens=num_out_tokens,
        num_windows=num_zero_windows, window=zero_window, max_run=zero_rows,
    )
    inputs.append(zinfo)

  num_visits = _count_visits(group_sizes, tile_m)
  max_visits = num_experts + pl.cdiv(size_c, tile_m) + 1
  cfg = _Config(
      num_experts=num_experts, size_f=size_f, size_d=size_d, num_out_tokens=num_out_tokens, tile_m=tile_m,
      tile_n=tile_n, tok_window=tok_window, zero_window=zero_window, zero_rows=zero_rows,
      accumulate=accumulate, bucket_m=bucket_m,
  )
  dma = pltpu.SemaphoreType.DMA
  scratch_shapes = [
      pltpu.SMEM((8,), jnp.int32),
      pltpu.SMEM((max_visits,), jnp.int32),
      pltpu.SMEM((max_visits + 1,), jnp.int32),
      pltpu.SMEM((max_visits,), jnp.int32),
      pltpu.SMEM((num_experts,), jnp.int32),
      pltpu.SMEM((pl.cdiv(num_out_tokens, _SEEN_BITS),), jnp.int32),
      pltpu.SMEM((_TOK_SLOTS * tok_window,), jnp.int32),
      pltpu.SMEM((2 * tile_m,), jnp.int32),
      pltpu.SMEM((2 * zero_window,), jnp.int32),
      pltpu.SMEM((zero_window,), jnp.int32),
      pltpu.VMEM((2, tile_m, size_f), act.dtype),
      pltpu.VMEM((2, size_f, size_d), w_q.dtype),
      pltpu.VMEM((2, tile_m, d0, LANES), jnp.bfloat16),
      pltpu.VMEM((tile_m, d0, LANES), jnp.bfloat16),
      pltpu.VMEM((zero_rows, d0, LANES), jnp.bfloat16),
      dma((2,)), dma((_TOK_SLOTS,)), dma((2,)), dma((2,)), dma((1,)), dma((1,)), dma((1,)),
  ]
  if vmem_limit_bytes is None:
    vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.98)
  hbm = pl.BlockSpec(memory_space=pltpu.HBM)
  scalars = (
      group_sizes.astype(jnp.int32), jnp.asarray(start, jnp.int32).reshape(1),
      out_scale.astype(jnp.float32).reshape(1), zmeta,
  )
  manual_axis_type = (jax.typeof(out) if accumulate else jax.typeof(act)).manual_axis_type
  flops = 2 * size_c * size_f * size_d
  bytes_accessed = num_experts * size_f * size_d + size_c * size_f + 2 * num_out_tokens * size_d * 2
  acc_tag = "-acc" if accumulate else ""
  name = f"linear_combine_fwd-g_{num_experts}-m_{size_c}-k_{size_f}-n_{size_d}" f"-tm_{tile_m}-tn_{tile_n}{acc_tag}"
  return pl.pallas_call(
      functools.partial(_kernel, cfg=cfg),
      out_shape=jax.ShapeDtypeStruct(out_shape, jnp.bfloat16, manual_axis_type=manual_axis_type),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalars), grid=(jnp.maximum(num_visits, 1),),
          in_specs=[hbm] * len(inputs), out_specs=hbm, scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes, disable_bounds_checks=True, dimension_semantics=("arbitrary",),
      ),
      input_output_aliases=({len(scalars) + len(inputs) - 1: 0} if accumulate else {}),
      cost_estimate=pl.CostEstimate(flops=flops, bytes_accessed=bytes_accessed, transcendentals=0),
      name=name,
  )(*scalars, *inputs)
