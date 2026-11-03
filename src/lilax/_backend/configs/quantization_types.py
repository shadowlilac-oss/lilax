"""Native quantization configuration types."""

from enum import Enum


class QuantizationType(str, Enum):
  """Supported quantization schemes."""

  NONE = ""
  FP4 = "fp4"
  INT4 = "int4"
  INT8 = "int8"
  INTMP = "intmp"
  FP8_E5M2 = "fp8_e5m2"
  FP8_E4M3 = "fp8_e4m3"
  FP8 = "fp8"
  NANOO_FP8 = "nanoo_fp8"
  FP8_NANO_V2 = "fp8_nanoo"
  FP8_GPU = "fp8_gpu"
  FP8_FULL = "fp8_full"
  TE_NO_QUANT = "te_no_quant"
  TE_FP8_DS = "te_fp8_delayedscaling"
  TE_FP8_CS = "te_fp8_currentscaling"
  TE_MXFP8 = "te_mxfp8"
  JAX_MXFP8 = "jax_mxfp8"
  JAX_MXFP4 = "jax_mxfp4"
  TE_NVFP4 = "te_nvfp4"
  TE_NVFP4_NO_RHT = "te_nvfp4_no_rht"
  SERVE_FP8_WEIGHT = "serve_fp8_weight"


# The OCP MX recipes that run through jax.nn.scaled_dot_general.
JAX_MX_QUANTIZATIONS = (QuantizationType.JAX_MXFP8.value, QuantizationType.JAX_MXFP4.value)


class TEGroupedGemmQuantizationType(str, Enum):
  """Supported quantization schemes for TE grouped GEMM in MoE layers."""

  EMPTY = ""
  TE_NO_QUANT = "te_no_quant"  # Default precision, e.g. BF16, without quantization
  TE_MXFP8 = "te_mxfp8"


class TeCommGemmOverlapPolicy(str, Enum):
  """Transformer Engine collective GEMM overlap scope."""

  DISABLED = "disabled"
  MLP = "mlp"
  FULL = "full"


class KvQuantAxis(str, Enum):
  """Axes to quantize over for the Key-Value cache."""

  NONE = ""
  DKV = "dkv"
  HEADS_AND_DKV = "heads_and_dkv"
