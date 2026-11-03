"""Native DeepSeek Manifold-Constrained Hyper Connections configuration fields."""

from pydantic.fields import Field
from pydantic.functional_validators import model_validator
from pydantic.main import BaseModel
from pydantic.types import PositiveInt


class ManifoldConstrainedHyperConnections(BaseModel):
  """Configuration for DeepSeek Manifold-Constrained Hyper Connections (mHC)."""

  mhc_expansion_rate: PositiveInt = Field(1, description="The number of parallel streams in Hyper Connection.")
  sinkhorn_iterations: PositiveInt = Field(20, description="The number of iterations for the Sinkhorn-Knopp algorithm.")
  enable_mhc_lite: bool = Field(
      False,
      description=(
          "Whether to generate the MHC doubly stochastic matrix via "
          "permutation-based convex combination rather than Sinkhorn-Knopp. "
          "Practical only for a small mhc_expansion_rate (e.g., k=4)."
      ),
  )
  use_mhc_pallas_kernel: bool = Field(
      False,
      description=(
          "Whether to use the Pallas TPU kernel implementation for"
          " mHC-lite when running on TPU. Requires enable_mhc_lite=True."
      ),
  )
  mhc_pallas_kernel_fwd_block_size: int = Field(
      256,
      description="Block size for forward pass of MHC Pallas kernel.",
  )
  mhc_pallas_kernel_bwd_block_size: int = Field(
      128,
      description=(
          "Block size for backward pass of MHC Pallas kernel. Default of 128 is"
          " optimal for TPU v7 memory constraints; 256 is optimal for TPU v6."
      ),
  )
  mhc_pallas_kernel_bwd_feature_block_size: int = Field(
      1024,
      description=("Feature block size for backward pass of MHC Pallas kernel."),
  )
  te_mhc: bool = Field(
      False,
      description=(
          "Use Transformer Engine's fused Triton kernels for Sinkhorn mHC on NVIDIA GPUs."
          " Requires mhc_expansion_rate=4 and enable_mhc_lite=False."
      ),
  )

  @model_validator(mode="after")
  def validate_mhc_kernel(self) -> "ManifoldConstrainedHyperConnections":
    if self.use_mhc_pallas_kernel and not self.enable_mhc_lite:
      raise ValueError("use_mhc_pallas_kernel=True requires enable_mhc_lite=True.")
    if self.te_mhc:
      if self.mhc_expansion_rate != 4:
        raise ValueError(f"`te_mhc=True` requires `mhc_expansion_rate=4`, got {self.mhc_expansion_rate}.")
      if self.enable_mhc_lite or self.use_mhc_pallas_kernel:
        raise ValueError("`te_mhc=True` requires `enable_mhc_lite=False` and `use_mhc_pallas_kernel=False`.")
    return self
