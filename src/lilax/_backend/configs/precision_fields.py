"""Native data type and matmul precision configuration fields."""

from lilax._backend.configs.precision_types import DType, MatmulPrecision
from pydantic.fields import Field
from pydantic.main import BaseModel


class DataTypes(BaseModel):
  """Configuration for data types and precision."""

  dtype: DType = Field(DType.BFLOAT16, description="The data type for activations.")
  grad_dtype: DType = Field(DType.FLOAT32, description="The data type for gradients.")
  weight_dtype: DType = Field(DType.FLOAT32, description="The data type for model weights.")
  matmul_precision: MatmulPrecision = Field(
      MatmulPrecision.DEFAULT,
      description="Precision level for matrix multiplications.",
  )
  activations_in_float32: bool = Field(
      False,
      description="If True, sets activations to float32 before the nonlinearity.",
  )
  dtype_mm: str = Field("float32", description="Data type for multimodal model's vision encoder")
