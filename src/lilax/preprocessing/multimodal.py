"""Native multimodal preprocessing and prompt preparation."""

from lilax._backend.multimodal.processor import (
    downsample_video_mask_to_tokens,
    get_bidirectional_mask_audio,
    get_bidirectional_mask_vision,
    get_dummy_audio_shape_for_init,
    get_dummy_image_shape_for_init,
    get_image_offsets,
    prepare_text_for_image_fusion,
    preprocess_image_for_training,
    preprocess_mm_data,
    reformat_prompt,
    reformat_response,
)
from lilax._backend.multimodal.processor_qwen3_omni import Qwen3OmniPreprocessorOutput, get_rope_index
from lilax._backend.multimodal.utils import PreprocessorOutput

__all__ = [
    "PreprocessorOutput",
    "Qwen3OmniPreprocessorOutput",
    "downsample_video_mask_to_tokens",
    "get_bidirectional_mask_audio",
    "get_bidirectional_mask_vision",
    "get_dummy_audio_shape_for_init",
    "get_dummy_image_shape_for_init",
    "get_image_offsets",
    "get_rope_index",
    "prepare_text_for_image_fusion",
    "preprocess_image_for_training",
    "preprocess_mm_data",
    "reformat_prompt",
    "reformat_response",
]
