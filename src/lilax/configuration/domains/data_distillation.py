from typing import Any, Literal

from pydantic import BaseModel, Field


class Distillation(BaseModel):
  """Configuration for Knowledge Distillation."""

  student_overrides: dict[str, Any] = Field(default_factory=dict, description="Overrides specific to the Student model (e.g., {'num_query_heads': 16}).")
  teacher_overrides: dict[str, Any] = Field(default_factory=dict, description="Overrides specific to the Teacher model (e.g., {'num_query_heads': 64}).")
  offline_data_dir: str | None = Field(None, description="GCS or local path to the pre-generated ArrayRecord teacher data.")
  distill_alpha: float = Field(0.5, description="Weight for the distillation loss component.")
  distill_temperature: float = Field(1.0, description="Temperature for distillation softening.")
  distill_beta: float = Field(0.0, description="Weight for the feature loss component. Use 0.0 to disable")
  distill_feature_loss_type: Literal["cosine", "l2"] = Field("cosine", description="The type of loss to use for feature distillation ('cosine' or 'l2').")
  distill_layer_indices: None | list = Field(None, description="Feature indices for feature loss.")
  distill_alpha_end: float | None = Field(None, description="Target alpha at end of training. None keeps alpha fixed.")
  distill_alpha_schedule: Literal["constant", "linear", "cosine"] = Field("constant", description="Schedule type for alpha annealing ('constant', 'linear', or 'cosine').")
  distill_temperature_end: float | None = Field(None, description="Target temperature at end of training. None keeps temperature fixed.")
  distill_temperature_schedule: Literal["constant", "linear", "cosine"] = Field("constant", description="Schedule type for temperature annealing ('constant', 'linear', or 'cosine').")
  distill_beta_end: float | None = Field(None, description="Target beta_feature at end of training. None keeps beta fixed.")
  distill_beta_schedule: Literal["constant", "linear", "cosine"] = Field("constant", description="Schedule type for beta annealing ('constant', 'linear', or 'cosine').")
  learn_to_init_mode: bool = Field(False, description="Runs in the learn-to-init mode only")
  lti_use_general_linear_map: bool = Field(False, description=(
      "enable general map (i.e. single learnable projection instead of the bi-linear mapping. "
      "Needs much more HBM."
  ))
  distill_weights_copy_map: dict[str, Any] = Field(default_factory=dict, description="Dictionary of copying original teacher weights to the student model.")
  distill_student_weights_share_map: dict[str, Any] = Field(default_factory=dict, description="Experimental weight sharing map inside the student model for learn-to-init phase")
  attn_module_name: str | None = Field(None, description="Attention nnx module attribute name to augment with LTI logic")
  lti_layer_indices: list[int] | None = Field(None, description="List of layer indices to apply LTI modifications. If None, applied to all layers.")
  student_params_to_update: None | list = Field(None, description=(
      "a list of model param name templates to finetune in the student model. "
      "The other parameters will be frozen if this attribute is non empty)"
  ))
