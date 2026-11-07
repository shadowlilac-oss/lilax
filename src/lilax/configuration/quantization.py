"""Independent quantization recipes and their existing backend dispatch settings."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, PositiveInt

from .domain_types import KvQuantAxis, QuantizationType
from .domains.data_qat import QwixQATConfig


class _Recipe(BaseModel):
  model_config = ConfigDict(extra="forbid")


class ComputeRecipeConfig(_Recipe):
  quantization: Literal[
      QuantizationType.NONE, QuantizationType.FP8, QuantizationType.NANOO_FP8, QuantizationType.TE_NO_QUANT,
      QuantizationType.TE_FP8_DS, QuantizationType.TE_FP8_CS, QuantizationType.TE_MXFP8,
      QuantizationType.TE_NVFP4, QuantizationType.TE_NVFP4_NO_RHT,
  ] = QuantizationType.NONE


class AqtInt8Config(_Recipe):
  quantization_local_shard_count: int = -1


class AqtMixedPrecisionConfig(_Recipe):
  quant_cfg_path: str = ""


class JaxMXConfig(_Recipe):
  quantization: Literal[QuantizationType.JAX_MXFP8, QuantizationType.JAX_MXFP4]
  jax_mx_gradient_mode: Literal["", "mxfp8", "mxfp4"] = ""


class ServeFP8WeightConfig(_Recipe):
  act_quantization_calibration_method: str = "absmax"


class QwixRecipeConfig(_Recipe):
  quantization: Literal[
      QuantizationType.NONE, QuantizationType.INT4, QuantizationType.INT8, QuantizationType.FP4,
      QuantizationType.FP8_E5M2, QuantizationType.FP8_E4M3, QuantizationType.FP8,
      QuantizationType.FP8_GPU, QuantizationType.FP8_NANO_V2,
  ] = QuantizationType.NONE
  quantize_router_proj: bool = True
  qwix_bwd_weight_grad_tile_count: PositiveInt = 1


class QwixFP8FullConfig(_Recipe):
  quantize_mtp: bool = False
  quantize_router_proj: bool = True
  quantize_logits_proj: bool = False
  logits_proj_quant_calibration_method: str = ""
  logits_proj_bwd_quant_calibration_method: str = ""
  sa_qproj_bwd_bf16: bool = False
  weight_quantization_calibration_method: str = "absmax"
  act_quantization_calibration_method: str = "absmax"
  bwd_quantization_calibration_method: str = "absmax"
  drhs_grad_quantization_calibration_method: str | None = None


class BatchSplitQwixConfig(_Recipe):
  weight_quantization_calibration_method: str = "absmax"
  act_quantization_calibration_method: str = "absmax"
  bwd_quantization_calibration_method: str = "absmax"


class BatchSplitManualConfig(_Recipe):
  use_qwix_quantization: bool = False


class KVCacheQuantizationConfig(_Recipe):
  quantize_kvcache: bool = False
  kv_quant_axis: KvQuantAxis = KvQuantAxis.HEADS_AND_DKV
  kv_quant_dtype: Literal["int8", "int4"] = "int8"


class SparsityConfig(_Recipe):
  weight_sparsity_n: int | None = None
  weight_sparsity_m: int | None = None
  weight_sparsity_update_step: int = 10
  weight_sparsity_start_step: int = 50


QuantizationRecipe = (
    ComputeRecipeConfig | AqtInt8Config | AqtMixedPrecisionConfig | JaxMXConfig | ServeFP8WeightConfig
    | QwixRecipeConfig | QwixFP8FullConfig | BatchSplitQwixConfig | BatchSplitManualConfig | QwixQATConfig
)

_DISPATCH = {
    ComputeRecipeConfig: {}, AqtInt8Config: {"quantization": "int8"},
    AqtMixedPrecisionConfig: {"quantization": "intmp"}, JaxMXConfig: {},
    ServeFP8WeightConfig: {"quantization": "serve_fp8_weight"},
    QwixRecipeConfig: {"use_qwix_quantization": True},
    QwixFP8FullConfig: {"quantization": "fp8_full", "use_qwix_quantization": True},
    BatchSplitQwixConfig: {"quantization": "fp8_full", "use_qwix_quantization": True},
    BatchSplitManualConfig: {"quantization": "fp8_full", "use_manual_quantization": True},
}


def recipe_source_values(recipe):
  if isinstance(recipe, QwixQATConfig):
    return {"qwix_qat_config": recipe}
  if type(recipe) not in _DISPATCH:
    raise TypeError("quantization_config must be a supported quantization recipe")
  if isinstance(recipe, AqtMixedPrecisionConfig) and not recipe.quant_cfg_path:
    raise ValueError("AqtMixedPrecisionConfig requires a nonempty quant_cfg_path")
  return {
      "use_qwix_quantization": False, "use_manual_quantization": False,
      **_DISPATCH[type(recipe)], **recipe.model_dump(mode="json"),
  }
