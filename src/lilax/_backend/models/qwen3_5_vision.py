"""Qwen3.5 Vision model tower NNX subclasses.

These classes subclass the Qwen3-Omni vision tower components to provide
clean class type names (Qwen3_5MoeVision...) in the Flax NNX metadata,
ensuring that the JAX parameter keys stored in checkpoints do not contain
the word 'Omni'.
"""

from lilax._backend.models.qwen3 import Qwen3OmniMoeVisionEncoder, Qwen3OmniMoeVisionProjector


class Qwen3_5MoeVisionEncoder(Qwen3OmniMoeVisionEncoder):
  """Subclass of Qwen3OmniMoeVisionEncoder for Qwen3.5 VL models.

  Inherits all core vision tower layers (patch embedding, position embedding,
  rotary embeddings, attention, and transformer blocks) without modification.
  """


class Qwen3_5MoeVisionProjector(Qwen3OmniMoeVisionProjector):
  """Subclass of Qwen3OmniMoeVisionProjector for Qwen3.5 VL models.

  Inherits the final projection/merger layers without modification.
  """
