from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt


class DPO(BaseModel):
  """Configuration for DPO and ORPO preference optimization algorithms."""

  algo: Literal["dpo", "orpo"] = Field("dpo", description="Alignment algorithm to use.")
  dpo_beta: float = Field(0.1, description="Beta parameter for DPO.")
  orpo_lambda: float = Field(0.1, description="Weight for preference loss in ORPO.")
  dpo_label_smoothing: float = Field(0.0, ge=0.0, le=1.0, description="Label smoothing for DPO.")
  max_prompt_length: int | None = Field(None, gt=0, description="Maximum length for prompt. If None, defaults to half of max_target_length.")


class FineTuning(BaseModel):
  """Configuration for fine-tuning methods like DPO, SFT, and GRPO."""

  use_dpo: bool = Field(False, description="If True, enables Direct Preference Optimization training.")
  use_sft: bool = Field(False, description="If True, enables Supervised Fine-Tuning.")
  sft_train_on_completion_only: bool = Field(False, description="If True, trains only on the completion part of the text.")
  formatting_func_path: str = Field("", description="Path to the custom data formatting function for SFT.")
  formatting_func_kwargs: dict = Field(default_factory=dict, description="Keyword arguments to pass to the custom data formatting function for SFT.")
  use_grpo: bool | None = Field(None, description="If True, enables Group Relative Policy Optimization.")


class LoRA(BaseModel):
  """Configuration for LoRA / QLoRA adapters."""

  model_config = ConfigDict(extra="forbid")

  enable_lora: bool = Field(False, description="If True, enables LoRA/QLoRA during fine-tuning.")
  lora_rank: NonNegativeInt = Field(0, description="LoRA rank. Set >0 when LoRA is enabled.")
  lora_alpha: NonNegativeFloat = Field(0.0, description="LoRA alpha scaling factor.")
  lora_module_path: str = Field("", description=(
      "Regex identifying target NNX modules for LoRA. "
      "Example for standard models: 'decoder/layers/.*(self_attention/(query|out)|mlp/(wi_0|wo))'. "
      "Example for MoE: 'decoder/scanned_blocks/layers.*/.*(MoeBlock_0|shared_experts)/(wi_0|wo)'."
  ))
  lora_weight_qtype: str | None = Field(None, description="Optional quantization type for QLoRA (e.g., 'nf4'). If set, QLoRA is applied.")
  lora_tile_size: NonNegativeInt | None = Field(None, description="Tile size for block-wise quantization. Typically 32 or 64.")
  lora_restore_path: str = Field("", description="Optional path to LoRA weights to load before training. Ignored if the current run is resumed.")
