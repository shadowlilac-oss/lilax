"""Native Multi-Token Prediction configuration fields."""

from pydantic.fields import Field
from pydantic.main import BaseModel
from pydantic.types import NonNegativeFloat, NonNegativeInt


class MTP(BaseModel):
  """Multi-Token Prediction Configs."""

  mtp_num_layers: NonNegativeInt = Field(0, description="The number of auxiliary prediction layers to use for MTP.")
  mtp_loss_scaling_factor: NonNegativeFloat = Field(
      0.1,
      description="The scaling factor (lambda) for the MTP auxiliary loss.",
  )
  mtp_eval_target_module: NonNegativeInt = Field(
      0,
      description="Specifies which MTP layer is used to calculate metrics.",
  )
  mtp_reuse_input_embedding: bool = Field(
      False,
      description="Reuse the main decoder's token embeddings instead of a second lookup.",
  )
  mtp_norm_main_hidden_state: bool = Field(
      False,
      description=(
          "Apply the decoder's final norm to the main hidden state before passing it to MTP, as in"
          " Megatron-LM where `final_layernorm` precedes the MTP `hnorm`."
      ),
  )
