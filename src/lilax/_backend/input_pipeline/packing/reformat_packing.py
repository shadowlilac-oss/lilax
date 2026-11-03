"""Flattening of packed (tokens, segmentation, position) outputs into feature columns."""

import dataclasses

import grain.python as grain


@dataclasses.dataclass
class ReformatPacking(grain.MapTransform):
  """Reformat packing outputs."""

  def __init__(self, column_names):
    self.column_names = column_names

  def map(self, element):
    ret = {}
    for col in self.column_names:
      ret[f"{col}"] = element[0][col]
      ret[f"{col}_segmentation"] = element[1][col]
      ret[f"{col}_position"] = element[2][col]
    return ret
