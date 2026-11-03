"""TensorFlow feature operations used by the TFDS input pipelines."""

from typing import Any, Iterable

import numpy as np

from lilax._backend.input_pipeline import tokenizer

Features = dict[str, Any]


def normalize_features(x, column_name):
  return {"inputs": x[column_name], "targets": x[column_name]}


def get_tokenizer(tokenizer_path, tokenizer_type, add_bos, add_eos, hf_access_token=None):
  tokenizer_model = tokenizer.build_tokenizer(tokenizer_path, tokenizer_type, add_bos, add_eos, hf_access_token)
  return tokenizer_model


def truncate_to_max_allowable_length(x, max_length):
  return {k: v[:max_length] for k, v in x.items()}


def shift_data_by_truncation(x):
  x["inputs"] = x["inputs"][:-1]
  x["targets"] = x["targets"][1:]
  return x


def add_segmentation_and_position(x, data_columns, padding_token=0):
  import tensorflow as tf  # pylint: disable=import-outside-toplevel

  for data_column in data_columns:
    x[f"{data_column}_segmentation"] = tf.cast(x[data_column] != padding_token, tf.int32)
    x[f"{data_column}_position"] = tf.broadcast_to(
        tf.range(x[data_column].shape[-1], dtype=np.int32)[None, :],
        x[data_column].shape,
    )
  return x


def TokenizeOp(
    tokenizer_model,
    features: Features,
    data_keys: Iterable[str] = ("inputs", "targets"),
) -> Features:
  """Op for tokenization"""
  import tensorflow as tf  # pylint: disable=import-outside-toplevel


  def _process_string(string_tensor):
    string_value = string_tensor.numpy().decode("utf-8")
    modified_string = tokenizer_model.encode(string_value)
    return [modified_string]

  for k in data_keys:
    features[k] = tf.py_function(_process_string, [features[k]], Tout=[tf.int32])[0]
  return features
