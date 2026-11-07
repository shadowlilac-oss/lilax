"""Focused configuration owners for the supported model families."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from .domains.data_mhc import ManifoldConstrainedHyperConnections as HyperconnectionConfig
from .domains.execution_loss import LogitsAndLoss as LogitsConfig, MTP as MTPConfig
from .domains.execution_multimodal import (
    AudioEncoder, VisionProjector as VisionProjectorConfig, VisionTower,
)
from .domain_types import VisionEncoderBlockType
from .domains.model_engram import Engram as EngramConfig
from .domains.model_ming import Ming as MingConfig
from .domains.model_moe import DeepSeekMoE as DeepSeekMoEConfig, MoEGeneral
from .domains.model_quantization import Quantization
from .domains.model_qwen import Qwen3Next as Qwen3NextConfig, Qwen3_8FlashNext as Qwen3_8FlashNextConfig


class MoEParameters(MoEGeneral):
  model_config = ConfigDict(extra="forbid")
  base_moe_mlp_dim: ClassVar[int] = -1
  act_quantization_calibration_method: str = Quantization.model_fields["act_quantization_calibration_method"]


class MoEConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  general: MoEParameters | None = None
  deepseek: DeepSeekMoEConfig | None = None


QwenConfig = Qwen3NextConfig | Qwen3_8FlashNextConfig


class VisionConfig(VisionTower):
  model_config = ConfigDict(extra="forbid")
  use_multimodal: bool = False
  attention_for_vit: str = "dot_product"
  vision_encoder_block: VisionEncoderBlockType = VisionEncoderBlockType.NONE
  freeze_vision_encoder_params: bool = True
  image_size_for_vit: int | list[int] | None = 896
  posemb_type_for_vit: str = "learn"


class AudioConfig(AudioEncoder):
  model_config = ConfigDict(extra="forbid")
  use_audio: bool = False
  freeze_audio_encoder_params: bool = True
