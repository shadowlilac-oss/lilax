"""Multimodal processor routing for stitched Omni architectures."""

import numpy as np

from lilax._backend.multimodal.processor_gemma3 import GEMMA_NUM_PLACEHOLDER_TOKENS_PER_IMAGE
from lilax._backend.multimodal.processor_qwen3_omni import QWEN_SPECIAL_TOKEN_CONFIGS
from lilax._backend.utils import max_logging

VISION_TOKENS_PER_IMAGE = {"gemma3": GEMMA_NUM_PLACEHOLDER_TOKENS_PER_IMAGE}
DECODER_SPECIAL_TOKENS = {
    "qwen3": {
        "vision_start": QWEN_SPECIAL_TOKEN_CONFIGS["qwen3-omni-30b-a3b"]["vision_start"],
        "vision_end": QWEN_SPECIAL_TOKEN_CONFIGS["qwen3-omni-30b-a3b"]["vision_end"],
        "image_pad": QWEN_SPECIAL_TOKEN_CONFIGS["qwen3-omni-30b-a3b"]["image_pad"],
    },
}


def get_image_offsets_omni(vision_block, decoder_block, processor_output=None):
  """Calculate the token increase from inserting visual sequences."""
  if vision_block not in VISION_TOKENS_PER_IMAGE or decoder_block not in DECODER_SPECIAL_TOKENS:
    raise ValueError(f"Stitched model not supported for vision='{vision_block}', decoder='{decoder_block}'.")
  num_tokens_per_image = VISION_TOKENS_PER_IMAGE[vision_block]
  has_images = processor_output is not None and processor_output.pixel_values is not None
  num_images = processor_output.pixel_values.shape[0] if has_images else 1
  return (num_tokens_per_image - 1) * num_images


def add_extra_tokens_for_omni(tokens, vision_block, decoder_block, processor_output=None, num_tokens_per_image=None):
  """Expand image placeholders or prepend vision tokens when image data is present."""
  if vision_block not in VISION_TOKENS_PER_IMAGE or decoder_block not in DECODER_SPECIAL_TOKENS:
    raise ValueError(f"Stitched model not supported for vision='{vision_block}', decoder='{decoder_block}'.")
  if num_tokens_per_image is None:
    num_tokens_per_image = VISION_TOKENS_PER_IMAGE[vision_block]
  tokens_config = DECODER_SPECIAL_TOKENS[decoder_block]
  image_pad_id = tokens_config["image_pad"]
  vision_start_id = tokens_config["vision_start"]
  vision_end_id = tokens_config["vision_end"]
  dtype = tokens.dtype if isinstance(tokens, np.ndarray) else np.int32
  token_list = np.asarray(tokens).flatten().tolist()
  if image_pad_id in token_list:
    expanded_tokens = []
    for token in token_list:
      if token == image_pad_id:
        expanded_tokens.extend([image_pad_id] * num_tokens_per_image)
      else:
        expanded_tokens.append(token)
    return np.array(expanded_tokens, dtype=dtype)
  if processor_output is not None and processor_output.pixel_values is not None:
    max_logging.warning(
        "No visual placeholder (<|image_pad|>) found in prompt tokens, but image data is present. "
        "Ensure prompts are formatted via reformat_prompt to avoid token offset discrepancies."
    )
    num_images = processor_output.pixel_values.shape[0]
    vision_block_seq = []
    for _ in range(num_images):
      vision_block_seq.extend([vision_start_id] + [image_pad_id] * num_tokens_per_image + [vision_end_id])
    return np.array(vision_block_seq + token_list, dtype=dtype)
  return np.array(token_list, dtype=dtype)


def get_bidirectional_mask_vision_omni(vision_block, decoder_block, decoder_input_tokens):
  """Generate a bidirectional attention mask for stitched-model vision tokens."""
  if vision_block not in VISION_TOKENS_PER_IMAGE or decoder_block not in DECODER_SPECIAL_TOKENS:
    raise ValueError(f"Stitched model not supported for vision='{vision_block}', decoder='{decoder_block}'.")
  image_pad_id = DECODER_SPECIAL_TOKENS[decoder_block]["image_pad"]
  return decoder_input_tokens == image_pad_id
