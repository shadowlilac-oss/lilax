"""Native rematerialization configuration types."""

from enum import Enum


class RematPolicy(str, Enum):
  """Available rematerialization (gradient checkpointing) policies."""

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


class RematLocation(str, Enum):
  """Specifies where to store activations for rematerialization."""

  REMAT = "remat"
  DEVICE = "device"
  OFFLOAD = "offload"
