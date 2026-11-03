"""Special token IDs for Qwen3-Omni preprocessing."""

QWEN_SPECIAL_TOKEN_CONFIGS = {
    "qwen3-omni-30b-a3b": {
        "vision_start": 151652,  # <|vision_start|>
        "vision_end": 151653,  # <|vision_end|>
        "image_pad": 151655,  # <|image_pad|>
        "video_pad": 151656,  # <|video_pad|>
        "audio_start": 151669,  # <|audio_start|>
        "audio_end": 151670,  # <|audio_end|>
        "audio_pad": 151675,  # <|audio_pad|>
    },
    "qwen3.5": {
        "vision_start": 248053,  # <|vision_start|>
        "vision_end": 248054,  # <|vision_end|>
        "image_pad": 248056,  # <|image_pad|>
        "video_pad": 248057,  # <|video_pad|>
        "audio_start": 248070,  # <|audio_start|>
        "audio_end": 248071,  # <|audio_end|>
        "audio_pad": 248076,  # <|audio_pad|>
    },
}


class QwenTokens:
  """Helper class to manage special token IDs for Qwen models.

  Token attributes are set dynamically from QWEN_SPECIAL_TOKEN_CONFIGS, so adding a new
  token type only requires updating that dict — no changes here needed.
  """

  _DEFAULT_MODEL = "qwen3-omni-30b-a3b"

  vision_start: int | None = None
  vision_end: int | None = None
  image_pad: int | None = None
  video_pad: int | None = None
  audio_start: int | None = None
  audio_end: int | None = None
  audio_pad: int | None = None

  def __init__(self, config=None):
    # If config is None, will fall back to default Qwen3-Omni tokens.
    self.model_name = getattr(config, "model_name", None) or self._DEFAULT_MODEL
    # Match by prefix (e.g. "qwen3.5" covers qwen3.5 family), fall back to default.
    token_config = next(
        (v for k, v in QWEN_SPECIAL_TOKEN_CONFIGS.items() if self.model_name.startswith(k)),
        QWEN_SPECIAL_TOKEN_CONFIGS[self._DEFAULT_MODEL],
    )
    # Sets model-specific token attributes from QWEN_SPECIAL_TOKEN_CONFIGS above.
    self.__dict__.update(token_config)
