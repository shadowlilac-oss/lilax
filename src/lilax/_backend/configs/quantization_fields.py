"""Native model quantization and sparsity configuration fields."""

from typing import Literal

from lilax._backend.configs.path_types import PathStr
from lilax._backend.configs.quantization_types import KvQuantAxis, QuantizationType, TeCommGemmOverlapPolicy
from pydantic.fields import Field
from pydantic.main import BaseModel
from pydantic.types import PositiveInt


class Quantization(BaseModel):
  """Configuration for model quantization."""

  quantization: None | QuantizationType = Field(
      QuantizationType.NONE,
      description="Activates quantization for transformer layers.",
  )
  jax_mx_gradient_mode: Literal["", "mxfp8", "mxfp4"] = Field(
      "",
      description="Gradient format for jax_mxfp8 / jax_mxfp4: '' (same as the operands), 'mxfp8' or 'mxfp4'.",
  )
  unquantized_modules: list[str] = Field(
      default_factory=list,
      description=(
          "List of submodule names or name patterns to keep unquantized even when weight_dtype is FP8. "
          "Weights for modules specified here will use `dtype` (e.g. bfloat16). "
          "Accepted names include: 'token_embedder', 'logits_dense', 'gate', 'shared_expert_gate', "
          "'conv1d', 'in_proj_ba', 'norm'."
      ),
  )
  weight_block_size: None | int | list[int] = Field(
      None,
      description=(
          "Block size for block-scaled quantized weights (e.g. 128 for symmetric 128x128 block scaling, "
          "or a list/tuple like [128, 64] for asymmetric block scaling). "
          "None for per-tensor scaling."
      ),
  )
  replicate_quant_scale: bool = Field(
      False,
      description="Replicates quantization scale to avoid inefficient XLA fusion.",
  )
  quant_cfg_path: PathStr = Field("", description="Path to the configuration file for 'intmp' quantization.")
  quantize_kvcache: bool = Field(False, description="If True, quantizes the Key-Value cache.")
  quantize_mtp: bool = Field(
      False,
      description=(
          "If True, quantizes the Multi-Token Prediction (MTP) block. Only supported with"
          " `mtp_num_layers > 0` and `quantization=fp8_full`."
      ),
  )
  quantize_router_proj: bool = Field(
      True,
      description=(
          "If True, quantizes the MoE router (gate) projection matmul when quantization is enabled. Targets the"
          " `GateLogit` module (matching path regex r'.*/gate$'). Default is True for backward compatibility."
          " Applicable when use_qwix_quantization=True and quantization is set; ignored otherwise. Distinct from"
          " `float32_gate_logits`, which sets the gate module's compute dtype (operand casts, bias add, score"
          " function) and does not control quantization."
      ),
  )
  quantize_logits_proj: bool = Field(
      False,
      description=(
          "If True, quantizes the output logits (logits_dense) projection when quantization is enabled."
          " Targets the `logits_dense` module (matching path regex r'decoder/logits_dense.*'). Default is False."
          " Only supported with `quantization=fp8_full`, `logits_via_embedding=False`, and `num_vocab_tiling = 1`."
      ),
  )
  logits_proj_quant_calibration_method: str = Field(
      "",
      description=(
          "Calibration method for the output logits (logits_dense) projection when `quantize_logits_proj=True`."
          " If empty (default), inherits `weight_quantization_calibration_method` and"
          " `act_quantization_calibration_method`. Set to e.g. 'absmax' to force absmax calibration."
      ),
  )
  logits_proj_bwd_quant_calibration_method: str = Field(
      "",
      description=(
          "Backward (gradient) calibration method for the output logits (logits_dense) projection when"
          " `quantize_logits_proj=True`. Applies to both the activation-gradient and weight-gradient arms. If empty"
          " (default), inherits `bwd_quantization_calibration_method` and `drhs_grad_quantization_calibration_method`."
          " Set to e.g. 'absmax' to force absmax calibration."
      ),
  )
  sa_qproj_bwd_bf16: bool = Field(
      False,
      description=(
          "If True, keeps the backward pass in bfloat16 (unquantized) for self-attention query projection matmuls"
          " when `quantization=fp8_full`."
      ),
  )
  kv_quant_axis: KvQuantAxis = Field(KvQuantAxis.HEADS_AND_DKV, description="Axes to quantize over for the KV cache.")
  kv_quant_dtype: Literal["int8", "int4"] = Field("int8", description="Data type for KV cache quantization.")
  quantization_local_shard_count: int = Field(
      -1,
      description="AQT int8 only: number of local shards along the contraction axis of weight-gradient matmuls, each"
      " with its own scale. -1 means jax.local_device_count(); 0 leaves the weight-gradient matmul unquantized.",
  )
  qwix_bwd_weight_grad_tile_count: PositiveInt = Field(
      1,
      description="Qwix: split the contraction axis of each weight-gradient matmul into this many tiles, each with"
      " its own quantization scale. 1 disables tiling. Tiles are summed after the matmul, so under FSDP tiling"
      " makes XLA all-reduce the full weight gradient instead of reduce-scattering it. fp8_full ignores it.",
  )
  use_qwix_quantization: bool = Field(False, description="Whether to use qwix for quantization.")
  use_manual_quantization: bool = Field(
      False,
      description="Whether to use manual quantization for batch split. Only used if `use_batch_split_schedule=True`.",
  )
  weight_quantization_calibration_method: str = Field(
      "absmax",
      description="Quantization calibration method used for weights.",
  )
  act_quantization_calibration_method: str = Field(
      "absmax",
      description="Quantization calibration method used for activations.",
  )
  bwd_quantization_calibration_method: str = Field(
      "absmax",
      description="Quantization calibration method used for gradients.",
  )
  drhs_grad_quantization_calibration_method: str | None = Field(
      None,
      description=(
          "Calibration for the cotangent in the weight-gradient matmul only (qwix drhs_grad; dW = X^T dY). None "
          "inherits bwd_quantization_calibration_method. With absmax, this arm's per-channel scale reduces over the "
          "token axis and therefore across every data shard, including DCN slices; a fixed range (e.g. 'fixed,0.01') "
          "removes that reduction. The activation-gradient arm (dX = dY W^T) is unaffected."
      ),
  )
  weight_sparsity_n: int | None = Field(
      None,
      description=("The 'N' in N:M sparsity, representing the maximum number of non-zero" " values in each block."),
  )
  weight_sparsity_m: int | None = Field(
      None,
      description=("The 'M' in N:M sparsity, representing the number of values in each" " block."),
  )
  weight_sparsity_update_step: int = Field(10, description="The step size for updating weight sparsity masks.")
  weight_sparsity_start_step: int = Field(
      50,
      description=("The first number of steps before updating the sparsity masks."),
  )
  te_comm_gemm_overlap: str = Field(
      TeCommGemmOverlapPolicy.DISABLED.value,
      description=(
          "Transformer Engine collective GEMM overlap policy. "
          "'disabled' disables overlap; 'mlp' overlaps MLP up/down projections; "
          "'full' also overlaps attention QKV and output projections."
      ),
  )
