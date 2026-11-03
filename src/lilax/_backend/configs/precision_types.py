"""Native numeric precision configuration types."""

from enum import Enum


class DType(str, Enum):
  """Supported data types for weights and activations."""

  BFLOAT16 = "bfloat16"
  FLOAT32 = "float32"
  FLOAT16 = "float16"
  FLOAT8_E4M3FN = "float8_e4m3fn"
  FLOAT8_E5M2 = "float8_e5m2"


class MatmulPrecision(str, Enum):
  """Precision levels for matrix multiplications."""

  DEFAULT = "default"
  HIGH = "high"
  HIGHEST = "highest"
  # same as default
  BFLOAT16 = "bfloat16"
  # same as highest
  FLOAT32 = "float32"
