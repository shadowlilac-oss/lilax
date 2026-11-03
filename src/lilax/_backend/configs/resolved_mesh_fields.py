"""Native device count and parallelism fields resolved from the ICI and DCN mesh configuration."""

from pydantic.fields import Field
from pydantic.main import BaseModel


class ResolvedMeshTopology(BaseModel):
  """Target device count and parallelism values resolved from the per-axis ICI and DCN settings."""

  num_diloco_replicas: None | int = Field(
      None,
      description="The number of diloco replicas, derived from ICI and DCN values.",
  )

  ici_parallelism: None | list[int] = Field(
      None,
      description="Aggregated list of all ICI parallelism values for legacy compatibility.",
  )
  dcn_parallelism: None | list[int] = Field(
      None,
      description="Aggregated list of all DCN parallelism values for legacy compatibility.",
  )

  using_pipeline_parallelism: None | bool = Field(
      None,
      description="Boolean flag indicating if pipeline parallelism is active across ICI or DCN.",
  )

  num_target_devices: None | int = Field(
      None,
      description="The number of devices computed from topology in train_compile or jax.devices() in train",
  )
