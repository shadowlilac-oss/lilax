"""Left-shifting of copied token sequences into next-token targets."""

import dataclasses

import grain.python as grain
import numpy as np


def shift_right(x, axis=1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  slices = [slice(None)] * len(x.shape)
  slices[axis] = slice(0, -1)
  padded = np.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(0))
  return padded[tuple(slices)]


def shift_left(x, pad_id, axis=1):
  """Shift to the left and pad."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (0, 1)
  slices = [slice(None)] * len(x.shape)
  slices[axis] = slice(1, None)
  padded = np.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(pad_id))
  return padded[tuple(slices)]


def shift_and_refine(x, ignored_ids, axis=1):
  """Left-shift targets and mask labels equal to one of ``ignored_ids``."""
  x["targets"] = shift_left(x["targets"], ignored_ids[0], axis=axis)
  x["targets_segmentation"] = shift_left(x["targets_segmentation"], 0, axis=axis)
  for ignore_id in ignored_ids:
    x["targets_segmentation"] = np.where(x["targets"] != ignore_id, x["targets_segmentation"], 0)
  return x


@dataclasses.dataclass
class ShiftData(grain.MapTransform):
  """Convert copied token sequences into next-token targets after batching.

  ``targets`` and ``targets_segmentation`` are shifted one position to the
  left along ``axis``. The final target is padded with ``ignored_ids[0]`` and
  receives segmentation value zero. Any shifted target equal to an ignored ID
  also receives segmentation value zero, excluding it from loss computation.

  This transform intentionally leaves positions unchanged: input and target
  positions refer to the same prediction positions after the left shift.
  """

  def __init__(self, ignored_ids, axis=1):
    self.ignored_ids = ignored_ids
    self.axis = axis

  def map(self, element):
    return shift_and_refine(element, ignored_ids=self.ignored_ids, axis=self.axis)
