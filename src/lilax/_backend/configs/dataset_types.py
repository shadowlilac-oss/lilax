"""Native dataset pipeline configuration types."""

from enum import Enum


class DatasetType(str, Enum):
  """Supported data loading pipelines."""

  SYNTHETIC = "synthetic"
  HF = "hf"
  GRAIN = "grain"
  TFDS = "tfds"
  C4MLPERF = "c4_mlperf"
  OLMO_GRAIN = "olmo_grain"
