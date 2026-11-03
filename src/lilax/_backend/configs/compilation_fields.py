"""Native ahead-of-time compilation configuration fields."""

from lilax._backend.configs.path_types import PathStr
from pydantic.fields import Field
from pydantic.main import BaseModel


class AOT(BaseModel):
  """Ahead of Time (AOT) Compilation settings."""

  compiled_trainstep_file: PathStr = Field("", description="Name of saved serialized compiled train_step.")
  compile_topology: str = Field("", description="Target hardware version, e.g. 'v5e-256'.")
  compile_topology_num_slices: int = Field(-1, description="Number of target slices.")
  write_estimator_result: bool = Field(False, description="Write estimator.py results in a separate file.")
