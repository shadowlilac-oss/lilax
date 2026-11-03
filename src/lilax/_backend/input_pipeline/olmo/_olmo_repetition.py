"""Periodic repetition filtering for OLMo datasets."""

from __future__ import annotations

from typing import Generator, List, NamedTuple

import numpy as np


class RepetitionTuple(NamedTuple):
  """``arr[start:end]`` is a periodic span of length ``period``,
  ``times = (end - start) // period``."""

  start: int
  end: int
  period: int
  times: int


def _find_end_first_consecutive_true(arr: np.ndarray) -> int:
  """End offset (exclusive) of the leading run of True in ``arr``.

  Returns 0 if ``arr[0]`` is False, ``len(arr)`` if all True.
  """
  if not arr[0]:
    return 0
  prog = np.cumsum(arr)
  if prog[-1] == len(arr):
    return int(len(arr))
  # First index where the cumulative sum stops increasing == start of False run.
  true_locs = np.where(prog[:-1] == prog[1:])[0]
  return int(true_locs[0] + 1)


def _find_start_last_consecutive_true(arr: np.ndarray) -> int:
  """Start offset of the trailing run of True in ``arr``, or -1 if none."""
  reverse = _find_end_first_consecutive_true(arr[::-1])
  return len(arr) - reverse if reverse > 0 else -1


def _group_consecutive_values(arr: np.ndarray, stepsize: int = 1) -> List[np.ndarray]:
  """Split a 1-D array of ints into runs of consecutive values."""
  if len(arr) == 0:
    return []
  return np.split(arr, np.where(np.diff(arr) != stepsize)[0] + 1)


def find_periodic_sequences(
    arr: np.ndarray,
    max_period: int,
    min_period: int = 1,
    mask_value: int = -1,
) -> Generator[RepetitionTuple, None, None]:
  """Yield :class:`RepetitionTuple` for periodic spans of length ≥ 3 in ``arr``.

  ``mask_value`` is reshape padding and must not appear in ``arr``. Default
  -1 is the max uint32 value, above any realistic vocab; pass an
  out-of-vocab sentinel if your vocab hits that id.
  """
  if (arr == mask_value).sum() > 0:
    raise ValueError("`mask_value` is in the array")

  max_period = min(max_period, len(arr) // 3)

  for period in range(min_period, max_period + 1):
    pad = (period - (len(arr) % period)) % period
    padded_arr = np.pad(arr, (0, pad), constant_values=mask_value) if pad else arr
    shaped_arr = padded_arr.reshape(-1, period)

    is_equal_to_prev_row = shaped_arr == np.roll(shaped_arr, shift=1, axis=0)
    rows_with_period = np.where(is_equal_to_prev_row.all(axis=1))[0]
    if len(rows_with_period) == 0:
      continue

    for sequence in _group_consecutive_values(rows_with_period):
      start_row = int(sequence[0])
      end_row = int(sequence[-1])

      start_offset = _find_start_last_consecutive_true(is_equal_to_prev_row[start_row - 1])
      start_offset = period - start_offset if start_offset > 0 else 0

      end_offset = _find_end_first_consecutive_true(is_equal_to_prev_row[(end_row + 1) % shaped_arr.shape[0]])

      start_pos = (start_row - 1) * period - start_offset
      end_pos = ((end_row + 1) * period) + end_offset

      out = RepetitionTuple(
          start=start_pos,
          end=end_pos,
          period=period,
          times=(end_pos - start_pos) // period,
      )
      if out.times > 2:
        yield out


def is_clean_instance(
    input_ids: np.ndarray,
    *,
    repetition_max_period: int = 13,
    repetition_min_period: int = 1,
    repetition_max_count: int = 32,
    mask_value: int = -1,
) -> bool:
  """``False`` iff ``input_ids`` has any periodic span (period ∈
  [min, max]) that repeats ≥ ``repetition_max_count`` times. Defaults
  match OLMo-core's ``_validate_instance``."""
  for m in find_periodic_sequences(
      input_ids,
      max_period=repetition_max_period,
      min_period=repetition_min_period,
      mask_value=mask_value,
  ):
    if m.times >= repetition_max_count:
      return False
  return True
