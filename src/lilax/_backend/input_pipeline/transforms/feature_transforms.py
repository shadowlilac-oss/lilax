"""Selection, renaming, and normalization of dataset feature columns."""

import dataclasses
from typing import Any

import grain.python as grain
import numpy as np


@dataclasses.dataclass
class NormalizeFeatures(grain.MapTransform):
  """Normalize text feature keys."""

  def __init__(self, column_names, tokenize):
    self.column_names = column_names
    self.tokenize = tokenize

  def map(self, element):
    if self.tokenize:
      return {col: element[col][0].decode() for col in self.column_names}
    else:
      return {col: element[col] for col in self.column_names}


@dataclasses.dataclass
class KeepFeatures(grain.MapTransform):
  """Filter dataset elements to specified features for parquet and other non-proto formats.

  Retains only the keys present in `feature_names`. Validates the stored value
  type against `tokenize`: raises ValueError if `tokenize=True` but a column
  contains integer data (pre-tokenized), or if `tokenize=False` but a column
  contains string/bytes data (raw text).
  """

  def __init__(self, feature_names: list[str], tokenize: bool = True):
    self.feature_names = feature_names
    self.tokenize = tokenize

  def map(self, element: dict[str, Any]) -> dict[str, Any]:
    """Applies the feature filtering to the input element."""
    missing = [n for n in self.feature_names if n not in element]
    if missing:
      raise ValueError(
          f"Column {missing} not found in dataset. Available columns: {sorted(element.keys())}. "
          "Please set train_data_columns or eval_data_columns accordingly."
      )
    filtered = {k: v for k, v in element.items() if k in self.feature_names}
    for col, val in filtered.items():
      if self.tokenize:
        if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.integer):
          raise ValueError(
              f"tokenize_data=True but column '{col}' contains integer (pre-tokenized) data. "
              "Set tokenize_train_data or tokenize_eval_data to False if your dataset is already tokenized."
          )
        if isinstance(val, (list, tuple)) and val and isinstance(val[0], (int, np.integer)):
          raise ValueError(
              f"tokenize_data=True but column '{col}' contains integer (pre-tokenized) data. "
              "Set tokenize_train_data or tokenize_eval_data to False if your dataset is already tokenized."
          )
      else:
        if isinstance(val, (str, bytes)):
          raise ValueError(
              f"tokenize_data=False but column '{col}' contains text data. "
              "Set tokenize_train_data or tokenize_eval_data to True if your dataset needs tokenization."
          )
    return filtered


@dataclasses.dataclass
class Rekey(grain.MapTransform):
  """Rename keys according to a mapping dict"""

  def __init__(self, mapping_dict, keep_old_keys=False):
    self.mapping_dict = mapping_dict
    self.keep_old_keys = keep_old_keys

  def map(self, element):
    old_keys = set()
    for new_key, old_key in self.mapping_dict.items():
      element[new_key] = element[old_key]
      old_keys.add(old_key)
    if not self.keep_old_keys:
      for key in old_keys:
        del element[key]
    return element


@dataclasses.dataclass
class HFNormalizeFeatures(grain.MapTransform):
  """Normalize feature keys for HuggingFace input"""

  def __init__(self, column_name, is_offline_distillation=False):
    self.column_name = column_name
    self.is_offline_distillation = is_offline_distillation

  def map(self, element):
    result = {
        "inputs": np.asarray(element[self.column_name], dtype=np.int32),
        "targets": np.asarray(element[self.column_name], dtype=np.int32),
    }
    if self.is_offline_distillation:
      seq_len = len(result["inputs"])
      result["top_k_logits"] = np.asarray(element["top_k_logits"], dtype=np.float32).reshape(seq_len, -1)
      result["top_k_indices"] = np.asarray(element["top_k_indices"], dtype=np.int32).reshape(seq_len, -1)
    return result
