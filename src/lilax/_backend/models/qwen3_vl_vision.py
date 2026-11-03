"""Qwen3 VL Vision model tower NNX subclasses."""

from lilax._backend.models.qwen3_omni.vision import Qwen3OmniMoeVisionEncoder, Qwen3OmniMoeVisionProjector


class Qwen3VLVisionEncoder(Qwen3OmniMoeVisionEncoder):
  """Subclass of Qwen3OmniMoeVisionEncoder for Qwen3-VL models.

  Inherits all core vision tower layers (patch embedding, position embedding,
  rotary embeddings, attention, and transformer blocks) without modification.
  """


class Qwen3VLVisionProjector(Qwen3OmniMoeVisionProjector):
  """Subclass of Qwen3OmniMoeVisionProjector for Qwen3-VL models.

  Inherits the final projection/merger layers without modification.
  """
