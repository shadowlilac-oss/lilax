"""Train a SentencePiece tokenizer from Grain data."""

import glob
import os
import shutil
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

from absl import logging
import grain.python as grain
import jax
from sentencepiece import SentencePieceTrainer

from lilax._backend.input_pipeline import input_pipeline_utils
from lilax._backend.utils import gcs_utils


def build_grain_iterator(data_file_pattern: str, data_file_type: str, data_keys: tuple[str, ...] = ("text",)) -> Iterator:
  """Build a Grain iterator from a local or GCS file pattern."""
  if data_file_pattern.startswith("gs://"):
    data_files = gcs_utils.gcs_glob_pattern(data_file_pattern)
  else:
    data_files = glob.glob(str(Path(data_file_pattern).expanduser().resolve()))
  if not data_files:
    raise FileNotFoundError(f"No files found matching pattern: {data_file_pattern}")
  logging.info("Found %d files for tokenizer training.", len(data_files))

  if data_file_type == "parquet":
    dataset = grain.MapDataset.source(data_files)
    dataset = dataset.map(grain.experimental.ParquetIterDataset)
    dataset = grain.experimental.InterleaveIterDataset(dataset, cycle_length=len(data_files))
    dataset = dataset.map(input_pipeline_utils.KeepFeatures(feature_names=list(data_keys)))
    return iter(dataset)
  elif data_file_type == "arrayrecord":
    source = grain.ArrayRecordDataSource(data_files)
    dataset = grain.MapDataset.source(source)
    dataset = dataset.map(input_pipeline_utils.ParseFeatures(list(data_keys), tokenize=True))
    dataset = dataset.map(input_pipeline_utils.NormalizeFeatures(list(data_keys), tokenize=True))
    return iter(dataset)
  elif data_file_type == "tfrecord":
    dataset = grain.MapDataset.source(data_files)
    dataset = dataset.map(input_pipeline_utils.make_tfrecord_iter_dataset)
    dataset = grain.experimental.InterleaveIterDataset(dataset, cycle_length=len(data_files))
    dataset = dataset.map(input_pipeline_utils.ParseFeatures(list(data_keys), tokenize=True))
    dataset = dataset.map(input_pipeline_utils.NormalizeFeatures(list(data_keys), tokenize=True))
    return iter(dataset)
  else:
    raise ValueError(f"Unsupported grain_file_type: {data_file_type!r}. Use 'parquet', 'arrayrecord', or 'tfrecord'.")


def _dump_chars_to_textfile(dataset_iter: Iterator, maxchars: int = int(1e7), data_keys=("text",)) -> tuple[str, int]:
  """Write part of a dataset to a text file."""
  char_count = 0
  temp_dir = tempfile.gettempdir()
  with tempfile.NamedTemporaryFile(
      delete=False, prefix=os.path.join(temp_dir, "ds_chars"), mode="w", encoding="utf-8"
  ) as outfp:
    while char_count < maxchars:
      example = next(dataset_iter)
      for key in data_keys:
        value = example[key]
        if isinstance(value, bytes):
          value = value.decode("utf-8")
        line = value + "\n"
        char_count += len(line)
        outfp.write(line)
  return outfp.name, char_count


def _train_sentencepiece(
    dataset_iter: Iterator,
    *,
    vocab_size: int,
    maxchars: int = int(1e7),
    model_path: str,
    model_type: str = "unigram",
    character_coverage: float = 1.0,
    data_keys=("text",),
):
  """Train a SentencePiece vocabulary on a subset of the dataset."""
  if model_path.startswith("gs://"):
    abs_model_path = model_path
  else:
    abs_model_path = os.path.abspath(os.path.expanduser(model_path))
  fname, _ = _dump_chars_to_textfile(dataset_iter, maxchars=maxchars, data_keys=data_keys)
  temp_dir = tempfile.gettempdir()
  with tempfile.NamedTemporaryFile(delete=False, prefix=os.path.join(temp_dir, "sp_tmp")) as model_fp:
    pass
  SentencePieceTrainer.Train(
      input=fname, vocab_size=vocab_size, character_coverage=character_coverage,
      model_prefix=model_fp.name, model_type=model_type,
  )
  if jax.process_index() == 0:
    if abs_model_path.startswith("gs://"):
      gcs_utils.upload_blob(abs_model_path, model_fp.name + ".model")
      logging.info("Uploaded %s to %s", model_fp.name + ".model", abs_model_path)
    else:
      parent = os.path.dirname(abs_model_path)
      if parent:
        os.makedirs(parent, exist_ok=True)
      shutil.copy(model_fp.name + ".model", abs_model_path)
      logging.info("Copied %s to %s", model_fp.name + ".model", abs_model_path)
  else:
    if abs_model_path.startswith("gs://"):
      while not gcs_utils.gcs_path_exists(abs_model_path):
        time.sleep(1)
    else:
      while not os.path.exists(abs_model_path):
        time.sleep(1)
    time.sleep(1)
  return abs_model_path


def train_tokenizer(
    dataset_iter: Iterator, *, vocab_path: str, vocab_size: int, max_corpus_chars: int, data_keys: tuple[str] = ("text",)
):
  """Train and save a SentencePiece tokenizer."""
  logging.info("SentencePiece vocab not found, building one from data.")
  vocab_path = _train_sentencepiece(
      dataset_iter, vocab_size=vocab_size, maxchars=max_corpus_chars, model_path=vocab_path, data_keys=data_keys
  )
  logging.info("Model saved at %s", vocab_path)
