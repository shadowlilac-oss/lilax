"""Native tensor placement fields derived from the custom rematerialization policy."""

from pydantic.fields import Field
from pydantic.main import BaseModel


class CustomRematTensors(BaseModel):
  """Tensors kept on device or offloaded to host under `remat_policy='custom'`."""

  tensors_on_device: None | list[str] = Field(
      None, description="List of tensors to keep on device memory for custom remat."
  )
  tensors_to_offload: None | list[str] = Field(
      None, description="List of tensors to offload to host memory for custom remat."
  )
