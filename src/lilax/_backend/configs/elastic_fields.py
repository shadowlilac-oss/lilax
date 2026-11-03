"""Native elastic training and fault tolerance configuration fields."""

from pydantic.fields import Field
from pydantic.main import BaseModel
from pydantic.types import PositiveInt


class ElasticTraining(BaseModel):
  """Configuration for elastic training and fault tolerance.

  Elastic training is Pathways-specific and does not work on McJAX.
  """

  elastic_enabled: bool = Field(False, description="Whether to enable elastic training.")
  elastic_backup_kind: str = Field(
      "snapshot",
      description=(
          "How an attempt after an elastic event gets its train state back. 'snapshot' keeps a copy of the train"
          " state and the data iterator position in pinned host memory every `elastic_snapshot_interval` steps, and"
          " restores that copy when it is at least as recent as the latest checkpoint; it needs grain's"
          " ElasticIterator (or synthetic data) and is not supported with DiLoCo or rampup batch size,"
          " which must set 'checkpoint'. 'checkpoint' only restores the latest persistent"
          " checkpoint. The snapshot only survives the loss of a slice for state that is replicated across slices, so"
          " with `shard_optimizer_over_data` it falls back to the checkpoint."
      ),
  )
  elastic_snapshot_interval: PositiveInt = Field(
      10,
      description="In snapshot mode, how often (in steps) to copy the train state to pinned host memory.",
  )
  elastic_timeout_seconds: int = Field(
      300,
      description=(
          "The maximum number of seconds to wait for `elastic_minimum_slice_count` slices to become active. If this"
          " timeout is reached during any retry attempt, a `TimeoutError` is raised and training fails."
      ),
  )
  elastic_max_retries: int = Field(
      10,
      description="The maximum number of times to retry training when a slice failure occurs or when scaling up.",
  )
  elastic_min_slice_count: int = Field(
      -1,
      description="The minimum number of slices to wait for before starting training. -1 means wait for all slices.",
  )
