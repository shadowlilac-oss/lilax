"""Grain iterable datasets over local, GCS, and Hugging Face TFRecord and Parquet files."""

import grain.python as grain
from grain._src.python.dataset.sources.tfrecord_dataset import _TFRecordReader, _TFRecordDatasetIterator
from grain.experimental import TFRecordIterDataset

from lilax._backend.utils import gcs_utils


class _GCSTFRecordReader(_TFRecordReader):
  """Extends Grain's _TFRecordReader to open TFRecord files from GCS via streaming BlobReader."""

  def __init__(self, path: str):
    bucket_name, blob_name = gcs_utils.parse_gcs_bucket_and_prefix(path)
    self._reader = gcs_utils.storage.Client().bucket(bucket_name).blob(blob_name).open("rb")


class _GCSTFRecordDatasetIterator(_TFRecordDatasetIterator):
  """Extends Grain's _TFRecordDatasetIterator to use _GCSTFRecordReader for GCS paths."""

  def __init__(self, path: str):
    grain.DatasetIterator.__init__(self)
    self._reader = _GCSTFRecordReader(path)


class GCSTFRecordIterDataset(TFRecordIterDataset):
  """Extends Grain's TFRecordIterDataset to support GCS paths."""

  def __iter__(self) -> grain.DatasetIterator:
    return _GCSTFRecordDatasetIterator(self._path)


def make_tfrecord_iter_dataset(path: str):
  """Returns the appropriate TFRecordIterDataset for local or GCS paths."""
  if path.startswith("gs://"):
    return GCSTFRecordIterDataset(path)
  return TFRecordIterDataset(path)


def make_parquet_iter_dataset(path: str, hf_access_token: str | None = None):
  """Returns the appropriate ParquetIterDataset for local or HF paths."""
  if path.startswith("hf://"):
    from huggingface_hub import HfFileSystem  # pylint: disable=import-outside-toplevel

    return grain.experimental.ParquetIterDataset(path, filesystem=HfFileSystem(token=hf_access_token))
  return grain.experimental.ParquetIterDataset(path)
