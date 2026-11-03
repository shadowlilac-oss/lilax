"""Lazy static Splash Attention mask types."""

from typing import Optional

from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from lilax._backend.layers.context_parallel import mask_reorder
import numpy as np


# TODO(agagik): change splash_attention_mask._ComputableMask to be non protected
class ChunkedCausalMask(splash_attention_mask._ComputableMask):  # pylint: disable=protected-access
  """Lazy chunked causal mask.

  Attention is causal within each chunk (0, K), (K, 2K), (2K, 3K), ... tokens
  attend to each other but not across chunks.
  Llama4 models use interleaved chunk attention along with global attention.

  This mask class inherits from splash_attention_mask._ComputableMask and is
  designed to be used with Splash Attention.
  It allows the mask logic to be computed on-the-fly or fused into the attention
  kernel, avoiding the memory cost of
  materializing the full (sequence_length, sequence_length) boolean mask array,
  which can be prohibitive for long sequences.
  """

  #: The size of each attention chunk.
  chunk_size: int

  def __init__(
      self,
      shape: tuple[int, int],
      chunk_size: int,
      shard_count: int = 1,
  ):
    if chunk_size <= 0:
      raise ValueError("chunk_size must be positive")
    self.chunk_size = chunk_size

    # Define the mask function for chunk attention
    def chunked_causal_mask_function(q_ids, kv_ids):
      """Computes the mask logic for the given slice indices."""
      if q_ids.size == 0 or kv_ids.size == 0:
        return np.empty((q_ids.shape[0], kv_ids.shape[1]), dtype=np.bool_)

      # Condition 1: Same chunk
      q_chunk = q_ids // self.chunk_size
      kv_chunk = kv_ids // self.chunk_size
      same_chunk = q_chunk == kv_chunk

      # Condition 2: Causal
      causal = q_ids >= kv_ids

      return same_chunk & causal

    # Initialize the parent ComputableMask with this function
    super().__init__(
        shape=shape,
        mask_function=chunked_causal_mask_function,
        shard_count=shard_count,
    )

  # Implement equality and hashing based on relevant attributes
  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented
    # Compare shape, chunk_size, and the underlying q_sequence array
    return (
        self.shape == other.shape
        and self.chunk_size == other.chunk_size
        and np.array_equal(self.q_sequence, other.q_sequence)
    )

  def __hash__(self):
    return hash(
        (
            type(self),
            self.shape,
            self.chunk_size,
            self.q_sequence.tobytes() if self.q_sequence is not None else None,
        )
    )


class BlockCausalMask(splash_attention_mask._ComputableMask):  # pylint: disable=protected-access,abstract-method
  """Lazy mask with bidirectional attention within causal blocks."""

  causal_block_size: int

  def __init__(
      self,
      shape: tuple[int, int],
      causal_block_size: int,
      shard_count: int = 1,
  ):
    if causal_block_size <= 0:
      raise ValueError("causal_block_size must be positive")
    self.causal_block_size = causal_block_size

    def block_causal_mask_function(q_ids, kv_ids):
      return (q_ids // self.causal_block_size) >= (kv_ids // self.causal_block_size)

    super().__init__(
        shape=shape,
        mask_function=block_causal_mask_function,
        shard_count=shard_count,
    )

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented
    return (
        self.shape == other.shape
        and self.causal_block_size == other.causal_block_size
        and np.array_equal(self.q_sequence, other.q_sequence)
    )

  def __hash__(self):
    return hash(
        (
            type(self),
            self.shape,
            self.causal_block_size,
            self.q_sequence.tobytes() if self.q_sequence is not None else None,
        )
    )


class HCAStaticMask(splash_attention_mask.Mask):
  """Static compile-time mask for DeepSeek-V4 Heavily Compressed Attention (HCA).

  Defines coarse-block coordinates and deduplicated partial block patterns for
  Splash Attention with pre-packed bitmasks loaded into VMEM/SMEM. Evaluates
  analytical boolean slices on host CPU during compilation, eliminating runtime VPU
  arithmetic overhead without materializing dense boolean mask arrays in HBM.

  Attributes:
    shape: (seq_len, aligned_kv_len) tuple.
    local_kv_len: Unpadded local sequence length.
    compressed_kv_len: Number of active compressed tokens.
    pad_kv_total: Prepend tile padding tokens between local and compressed KV.
    compress_ratio: Compression ratio for HCA (e.g. 128). Must be positive.
    local_window: Local sliding-window size (e.g. 128).
  """

  def __init__(
      self,
      shape: tuple[int, int],
      local_kv_len: Optional[int] = None,
      compressed_kv_len: Optional[int] = None,
      pad_kv_total: int = 0,
      compress_ratio: int = 128,
      local_window: Optional[int] = 128,
      shard_count: int = 1,
  ):
    # shard_count is retained for Splash Attention Mask API compatibility;
    # HCA static coordinates do not shard local window math across sequence axes.
    del shard_count
    if compress_ratio <= 0:
      raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    self._shape = shape
    self.local_kv_len = local_kv_len if local_kv_len is not None else shape[0]
    if self._shape[0] > self.local_kv_len and pad_kv_total <= 0:
      raise ValueError(
          f"pad_kv_total must be > 0 when query sequence is padded (shape[0]={self._shape[0]} > local_kv_len={self.local_kv_len})."
      )
    self.compressed_kv_len = (
        compressed_kv_len if compressed_kv_len is not None else max(0, self.local_kv_len // compress_ratio)
    )
    self.pad_kv_total = pad_kv_total
    self.compress_ratio = compress_ratio
    self.local_window = local_window

  @property
  def shape(self) -> tuple[int, int]:
    return self._shape

  def __getitem__(self, idx: tuple[slice, slice]) -> np.ndarray:
    if len(idx) != 2:
      raise NotImplementedError(f"Unsupported slice: {idx}")
    q_slice, kv_slice = idx
    if not isinstance(q_slice, slice) or not isinstance(kv_slice, slice):
      raise NotImplementedError(f"Unsupported slice: {idx}")

    q_slice = splash_attention_mask._fill_slice(q_slice, self.shape[0])
    kv_slice = splash_attention_mask._fill_slice(kv_slice, self.shape[1])

    rows = np.arange(q_slice.start, q_slice.stop, dtype=np.int32)[:, None]
    cols = np.arange(kv_slice.start, kv_slice.stop, dtype=np.int32)[None, :]

    if rows.size == 0 or cols.size == 0:
      return np.empty((rows.shape[0], cols.shape[1]), dtype=np.bool_)

    valid_row = rows < self.local_kv_len

    diff = rows - cols
    if self.local_window is None:
      local_m = valid_row & (diff >= 0) & (cols < self.local_kv_len)
    elif self.local_window > 0:
      local_m = valid_row & (diff >= 0) & (diff < self.local_window) & (cols < self.local_kv_len)
    else:
      local_m = np.zeros((rows.shape[0], cols.shape[1]), dtype=np.bool_)

    c_idx = cols - self.local_kv_len - self.pad_kv_total
    c_thresh = (rows + 1) // self.compress_ratio
    comp_m = valid_row & (c_idx >= 0) & (c_idx < c_thresh) & (c_idx < self.compressed_kv_len)

    # Map each padded query row 1-to-1 to a distinct padding KV column (softmax denominator = 1.0, logsumexp = 0.0).
    # This eliminates 0 * inf = NaN in backward dkv without cross-talk or gradient bleed into real tokens.
    if self.pad_kv_total > 0:
      pad_cols = self.local_kv_len + ((rows - self.local_kv_len) % self.pad_kv_total)
      pad_m = (~valid_row) & (cols == pad_cols)
    else:
      pad_m = np.zeros((rows.shape[0], cols.shape[1]), dtype=np.bool_)

    return local_m | comp_m | pad_m

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented
    return (
        self.shape == other.shape
        and self.local_kv_len == other.local_kv_len
        and self.compressed_kv_len == other.compressed_kv_len
        and self.pad_kv_total == other.pad_kv_total
        and self.compress_ratio == other.compress_ratio
        and self.local_window == other.local_window
    )

  def __hash__(self):
    return hash(
        (
            type(self),
            self.shape,
            self.local_kv_len,
            self.compressed_kv_len,
            self.pad_kv_total,
            self.compress_ratio,
            self.local_window,
        )
    )

  def __bool__(self) -> bool:
    raise NotImplementedError(
        "Conversion to bool is unsupported. Could be caused by using logical" " instead of bitwise operations on masks."
    )


def _load_balanced_q_sequence(shape: tuple[int, int], cp_size: int):
  """Reorders query positions the same way as load-balanced input tokens."""
  arr = np.arange(shape[0])
  return mask_reorder.reorder_mask_load_balancing(arr, cp_size, 0)


class LoadBalancedCausalMask(splash_attention_mask._ComputableMask):
  """Lazy causal mask, prevents the model from attending to future tokens."""

  offset: int
  shape: tuple[int, int]
  cp_size: int

  def __init__(
      self,
      shape: tuple[int, int],
      offset: int = 0,
      shard_count: int = 1,
      cp_size: int = 4,
  ):
    self.offset = offset

    def causal_mask_function(q_ids, kv_ids):
      if self.offset == 0:
        return q_ids >= kv_ids
      else:
        return q_ids + self.offset >= kv_ids

    mask_function = causal_mask_function
    super().__init__(
        shape=shape,
        mask_function=mask_function,
        shard_count=shard_count,
    )
    self.q_sequence = _load_balanced_q_sequence(shape, cp_size)

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented
    return self.shape == other.shape and self.offset == other.offset and np.array_equal(self.q_sequence, other.q_sequence)

  def __hash__(self):
    return hash(
        (
            type(self),
            self.shape,
            self.offset,
            self.q_sequence.tobytes() if self.q_sequence is not None else None,
        )
    )


class LoadBalancedLocalMask(splash_attention_mask.LocalMask):
  """Lazy local mask with load-balanced query positions."""

  def __init__(
      self,
      shape: tuple[int, int],
      window_size: tuple[int | None, int | None],
      offset: int,
      shard_count: int = 1,
      cp_size: int = -1,
  ):
    super().__init__(
        shape=shape,
        window_size=window_size,
        offset=offset,
        shard_count=shard_count,
    )
    self.q_sequence = _load_balanced_q_sequence(shape, cp_size)


class LoadBalancedChunkedCausalMask(ChunkedCausalMask):
  """Lazy chunked mask with load-balanced query positions."""

  def __init__(
      self,
      shape: tuple[int, int],
      chunk_size: int,
      cp_size: int,
      shard_count: int = 1,
  ):
    super().__init__(
        shape=shape,
        chunk_size=chunk_size,
        shard_count=shard_count,
    )
    self.q_sequence = _load_balanced_q_sequence(shape, cp_size)


class LoadBalancedBlockCausalMask(BlockCausalMask):
  """Lazy block-causal mask with load-balanced query positions."""

  def __init__(
      self,
      shape: tuple[int, int],
      causal_block_size: int,
      cp_size: int,
      shard_count: int = 1,
  ):
    super().__init__(
        shape=shape,
        causal_block_size=causal_block_size,
        shard_count=shard_count,
    )
    self.q_sequence = _load_balanced_q_sequence(shape, cp_size)
