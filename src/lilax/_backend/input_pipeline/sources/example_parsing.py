"""Parsing of serialized tf.train.Example records into feature arrays."""

import dataclasses

import grain.python as grain
import numpy as np

from lilax._backend.input_pipeline.protos import example_pb2


@dataclasses.dataclass
class ParseFeatures(grain.MapTransform):
  """Parse serialized tf.train.Example protos for arrayrecord/tfrecord datasets.

  Also validates that the stored field type matches `tokenize`: raises
  ValueError if `tokenize=True` but the column contains integers (pre-tokenized)
  or if `tokenize=False` but the column contains bytes (raw text).
  """

  def __init__(self, data_columns, tokenize):
    self.data_columns = list(data_columns)
    self.tokenize = tokenize

  def map(self, element):
    """Parse a serialized tf.train.Example proto and extract features."""
    example = example_pb2.Example()
    example.ParseFromString(element)
    features = example.features.feature

    parsed = {}
    for col in self.data_columns:
      target_col = col
      if col not in features:
        if col == "text" and "messages" in features:
          target_col = "messages"
        elif col == "messages" and "text" in features:
          target_col = "text"
        else:
          raise ValueError(
              f"Column '{col}' not found in dataset. Available columns: {sorted(features.keys())}. "
              "Please set train_data_columns or eval_data_columns accordingly."
          )

      f = features[target_col]
      if len(f.float_list.value) > 0:
        parsed[col] = np.array(f.float_list.value, dtype=np.float32)
      elif len(f.int64_list.value) > 0:
        parsed[col] = np.array(f.int64_list.value, dtype=np.int32)
      elif len(f.bytes_list.value) > 0:
        parsed[col] = np.array(f.bytes_list.value, dtype=object)
      else:
        parsed[col] = np.array([])

    seq_len = len(parsed.get("inputs", []))
    if seq_len > 0:
      if "top_k_logits" in parsed and len(parsed["top_k_logits"]) > 0:
        parsed["top_k_logits"] = parsed["top_k_logits"].reshape(seq_len, -1)
      if "top_k_indices" in parsed and len(parsed["top_k_indices"]) > 0:
        parsed["top_k_indices"] = parsed["top_k_indices"].reshape(seq_len, -1)

    return parsed
