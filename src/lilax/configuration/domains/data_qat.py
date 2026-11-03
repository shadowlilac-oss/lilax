from typing import Literal

from pydantic import BaseModel


class QwixQATRule(BaseModel):
  module_path: str = ".*"
  op_names: tuple[str, ...] = ()
  weight_qtype: str | None = None
  act_qtype: str | None = None
  act_static_scale: bool | None = None
  weight_calibration_method: str = "absmax"
  act_calibration_method: str | None = None
  act_batch_axes: tuple[int, ...] = (0,)
  tile_size: int | float | None = None


class QwixQATConfig(BaseModel):
  rules: tuple[QwixQATRule, ...]
  provider: Literal["odml", "qt"] = "odml"
  disable_per_channel_weights: bool = False
  fixed_range_for_inputs: tuple[float, float] | None = None
  fixed_range_for_outputs: tuple[float, float] | None = None
  strict: bool = True
