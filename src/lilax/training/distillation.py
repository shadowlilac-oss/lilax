"""Learn-to-initialize options for the existing teacher/student transformation."""

from pydantic import BaseModel, ConfigDict


class LearnToInitConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  enabled: bool = False
  use_general_linear_map: bool = False
  weights_copy_map: dict[str, str] | None = None
  student_weights_share_map: dict[str, str] | None = None
  attention_module_name: str | None = None
  layer_indices: list[int] | None = None
  student_params_to_update: list[str] | None = None

  def _source_values(self):
    return {
        "learn_to_init_mode": self.enabled, "lti_use_general_linear_map": self.use_general_linear_map,
        "distill_weights_copy_map": self.weights_copy_map or {},
        "distill_student_weights_share_map": self.student_weights_share_map or {},
        "attn_module_name": self.attention_module_name, "lti_layer_indices": self.layer_indices,
        "student_params_to_update": self.student_params_to_update,
    }
