"""Native DeepSeek-style MoE routing and Lineage execution configuration fields."""

from typing import Any, Literal

from pydantic.fields import Field
from pydantic.functional_validators import field_validator
from pydantic.main import BaseModel
from pydantic.types import NonNegativeInt


class DeepSeekMoE(BaseModel):
  """Configuration specific to DeepSeek-style MoE layers."""

  first_num_dense_layers: NonNegativeInt = Field(0, description="Number of initial dense layers in the model.")
  shared_experts: NonNegativeInt = Field(0, description="Number of shared experts.")
  routed_scaling_factor: float = Field(1.0, description="Scaling factor for routing scores.")
  routed_score_func: str = Field("", description="Scoring function for routing (e.g., 'softmax', 'sigmoid').")
  routed_bias: bool = Field(False, description="Whether to add a bias term for routing.")
  routed_bias_update_rate: float = Field(0.0, description="Update rate applied to the router bias term.")
  defer_small_all_reduces: bool = Field(
      False,
      description="Ring-of-experts sparse_matmul only: drop the per-layer all-reduces of the routed-bias expert counts"
      " from the scanned layer. Each MoE layer emits its local partial counts, the scan stacks them, and loss_fn"
      " reduces the stacked array once after the layer loop (one all-reduce). Same sums as the per-layer path. (The"
      " token-overflow flag of moe_dropless_fallback='step' is already reduced once per step without this flag.)",
  )
  log_moe_bias_norms: bool = Field(False, description="Whether to log the norms of MoE router biases.")
  log_step_diagnostics: bool = Field(
      False,
      description="Append to every training step's log line: pre-clip and post-clip global grad norm, param norm,"
      " max |grad|, the routed-bias checksum (sum over MoE layers of sum(bias)), the number of nonzero routed-bias"
      " update entries and the MoE overflow flag.",
  )
  mlp_bias: bool = Field(
      False,
      description="Whether to add a learnable bias for MLP matmul, "
      "and originally implemented to support the GPT-OSS model architecture",
  )
  n_routing_groups: int = Field(-1, description="Number of groups for routing, disabled by default.")
  first_num_hash_layers: int = Field(
      0,
      description="Number of hash routing layers, used in DeepSeek V4 (0 means disabled).",
  )
  topk_routing_group: int = Field(-1, description="Number of top groups to route inputs to.")
  use_batch_split_schedule: bool = Field(
      False,
      description="Whether to split batch into micro-batches to hide communications that yields performance benefits.",
  )
  batch_split_factor: int = Field(
      1,
      description="Factor by which to split the batch into micro-batches. Only used if use_batch_split_schedule is True.",
  )
  use_lineage: bool = Field(
      False,
      description="Whether to use Lineage DeepSeek-V3 execution.",
  )
  lineage_quantization: Literal[
      "none",
      "fp8_full",
      "fp8_full_splash_qk",
      "fp8_full_splash_qk_e3",
      "fp8_full_combine_per_hop",
      "fp8_full_splash_qk_e3_combine_per_hop",
  ] = Field(
      "none",
      description="Quantization of the Lineage sparse-layer routed experts and MLA. Only used if use_lineage is True.",
  )

  @classmethod
  def _lineage_quantization_none(cls, v: Any) -> Any:
    """pyconfig converts the string "none" to None; map it back."""
    return "none" if v is None else v

  # Manually apply the field_validator decorator outside of the class definition to avoid pytype issues
  _validate_lineage_quantization = field_validator("lineage_quantization", mode="before")(_lineage_quantization_none)
  lineage_expert_permutation: None | Literal["lpt"] = Field(
      None,
      description=(
          "Per-layer expert permutation algorithm for the Lineage sparse and MTP layers (see `dsv3.dsv3`'s"
          " `expert_permutation`). None disables expert shuffling. 'lpt' computes each step's permutation on device"
          " from the previous step's per-expert token counts with `dsv3_expert_shuffle.lpt_expert_permutation`; the"
          " first step uses the identity."
      ),
  )
  lineage_diag_band_skip: bool = Field(
      True,
      description="Enable row-band diagonal skip in Lineage Splash attention.",
  )
  lineage_dq_reduction_steps: int = Field(
      1,
      description=("Lineage dkv dq reduction steps: 1 for VMEM reduction, 0 for None."),
  )
