"""Configuration domain types."""

from enum import Enum, IntEnum
from typing import Literal


class DType(str, Enum):
  BFLOAT16 = "bfloat16"
  FLOAT32 = "float32"
  FLOAT16 = "float16"
  FLOAT8_E4M3FN = "float8_e4m3fn"
  FLOAT8_E5M2 = "float8_e5m2"


class MatmulPrecision(str, Enum):
  DEFAULT = "default"
  HIGH = "high"
  HIGHEST = "highest"
  BFLOAT16 = "bfloat16"
  FLOAT32 = "float32"


class QuantizationType(str, Enum):
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


class TeCommGemmOverlapPolicy(str, Enum):
  DISABLED = "disabled"
  MLP = "mlp"
  FULL = "full"


class RematPolicy(str, Enum):
  FULL = "full"
  MINIMAL = "minimal"
  SAVE_DOT_WITH_CONTEXT_EXCEPT_MLP = "save_dot_with_context_except_mlp"
  SAVE_DOT_EXCEPT_MLPWI = "save_dot_except_mlpwi"
  SAVE_DOT_EXCEPT_MLP = "save_dot_except_mlp"
  SAVE_QKV_PROJ = "save_qkv_proj"
  QKV_PROJ_OFFLOADED = "qkv_proj_offloaded"
  CUSTOM = "custom"
  MINIMAL_OFFLOADED = "minimal_offloaded"
  SAVE_OUT_PROJ = "save_out_proj"


class KvQuantAxis(str, Enum):
  NONE = ""
  DKV = "dkv"
  HEADS_AND_DKV = "heads_and_dkv"


class DecoderBlockType(Enum):
  DEFAULT = "default"
  LLAMA2 = "llama2"
  MISTRAL = "mistral"
  MIXTRAL = "mixtral"
  DEEPSEEK = "deepseek"
  GEMMA = "gemma"
  GEMMA2 = "gemma2"
  GEMMA3 = "gemma3"
  GEMMA4 = "gemma4"
  GEMMA4_SMALL = "gemma4_small"
  QWEN2 = "qwen2"
  QWEN3 = "qwen3"
  QWEN3_MOE = "qwen3_moe"
  QWEN3_CUSTOM_MOE = "qwen3_custom_moe"
  QWEN3_NEXT = "qwen3_next"
  QWEN3_5 = "qwen3_5"
  GPT3 = "gpt3"
  GPT_OSS = "gpt_oss"
  SIMPLE = "simple"
  SIMPLE_MLP = "simple_mlp"
  LLAMA4 = "llama4"
  OLMO3 = "olmo3"
  DEEPSEEK4 = "deepseek4"
  ENVY = "envy"
  WEAVER = "weaver"
  MING = "ming"
  HY4 = "hy4"


ModelName = Literal[
    "default", "llama2-7b", "llama2-13b", "llama2-70b", "llama3-8b", "llama3.1-8b-Instruct",
    "llama3-70b", "llama3.1-70b-Instruct", "llama3.1-8b", "llama3.1-70b", "llama3.1-405b",
    "llama3.3-70b", "mistral-7b", "mixtral-8x7b", "mixtral-8x22b", "deepseek2-16b", "deepseek2-236b",
    "deepseek3-671b", "deepseek3-671b-2dfsdp", "deepseek3-671b-batchsplit", "deepseek3-671b-lineage",
    "deepseek3-test", "deepseek3-tiny", "deepseek3.2-671b", "deepseek4-tiny", "deepseek4-284b",
    "deepseek4-1600b", "deepseek-custom", "kimi-k2-1t", "gemma-7b", "gemma-2b", "gemma2-2b", "gemma2-9b",
    "gemma2-27b", "gemma3-4b", "gemma3-12b", "gemma3-27b", "gemma4-26b", "gemma4-31b", "gemma4-e2b",
    "gemma4-e4b", "maxtext-omni-gemma3-qwen3", "qwen2.5-1.5b", "qwen2.5-7b", "qwen2.5-14b", "qwen3-0.6b",
    "qwen3-1.7b", "qwen3-1.7b-base", "qwen3-4b", "qwen3-4b-base", "qwen3-4b-thinking-2507", "qwen3-8b",
    "qwen3-8b-base", "qwen3-14b", "qwen3-14b-base", "qwen3-32b", "qwen3-235b-a22b", "qwen3-30b-a3b",
    "qwen3-30b-a3b-base", "qwen3-480b-a35b", "qwen3-vl-2b", "qwen3-vl-4b", "qwen3-vl-30b-a3b",
    "weaver-mini", "weaver-max", "weaver-mini-diffuser", "qwen3-next-80b-a3b", "qwen3.8-flash-next",
    "qwen3.8-flash-next-mini", "qwen3-omni-30b-a3b", "qwen3-custom-30b-a3b", "qwen3.5-9b",
    "qwen3.5-35b-a3b", "qwen3.5-35b-a3b-fp8", "qwen3.5-35b-fp8", "qwen3.5-397b-a17b",
    "qwen3.5-397b-a17b-fp8", "qwen3.8-2.4t-a95b", "gpt3-175b", "gpt3-22b", "gpt3-6b", "gpt3-52k",
    "gpt-oss-20b", "gpt-oss-120b", "llama4-17b-16e", "llama4-17b-128e", "olmo3-7b", "olmo3-7b-pt",
    "olmo3-32b", "envy-test", "envy-switch-base", "envy-switch-large", "envy-switch-xxl", "glm5.3-flash",
    "ming-3.0-audio-tiny", "hy4-preview",
]


class OptimizerType(str, Enum):
  ADAMW = "adamw"
  ADAM_PAX = "adam_pax"
  SGD = "sgd"
  MUON = "muon"


class LearningRateScheduleType(str, Enum):
  COSINE = "cosine"
  WSD = "wsd"


class WsdDecayStyle(str, Enum):
  LINEAR = "linear"
  COSINE = "cosine"


class TokenizerType(str, Enum):
  SENTENCEPIECE = "sentencepiece"
  HUGGINGFACE = "huggingface"
  TIKTOKEN = "tiktoken"


class RopeType(str, Enum):
  DEFAULT = "default"
  LLAMA3_1 = "llama3.1"
  YARN = "yarn"


class SamplingStrategy(str, Enum):
  GREEDY = "greedy"
  WEIGHTED = "weighted"
  NUCLEUS = "nucleus"
  TOPK = "topk"
  COMPOSITE = "composite"


class ProfilerType(str, Enum):
  NONE = ""
  XPLANE = "xplane"
  NSYS = "nsys"


class XProfTPUPowerTraceMode(IntEnum):
  POWER_TRACE_NONE = 0
  POWER_TRACE_NORMAL = 1
  POWER_TRACE_SPI = 2


class VisionEncoderBlockType(Enum):
  NONE = "none"
  GEMMA3 = "gemma3"
  LLAMA4 = "llama4"
  QWEN3_OMNI = "qwen3_omni"
  GEMMA4 = "gemma4"
  QWEN3_5 = "qwen3_5"
  QWEN3_VL = "qwen3_vl"
