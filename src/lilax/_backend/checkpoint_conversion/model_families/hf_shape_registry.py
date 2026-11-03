"""Registry of model-family Hugging Face weight-shape builders keyed by model name."""

from lilax._backend.checkpoint_conversion.model_families.deepseek.hf_shape import DEEPSEEK_HF_WEIGHTS_TO_SHAPE
from lilax._backend.checkpoint_conversion.model_families.deepseek_v4.hf_shape import DEEPSEEKV4_HF_WEIGHTS_TO_SHAPE
from lilax._backend.checkpoint_conversion.model_families.gemma2.hf_shape import GEMMA2_HF_WEIGHTS_TO_SHAPE
from lilax._backend.checkpoint_conversion.model_families.gemma3.hf_shape import GEMMA3_HF_WEIGHTS_TO_SHAPE
from lilax._backend.checkpoint_conversion.model_families.gemma4.hf_shape import (
    GEMMA4_HF_WEIGHTS_TO_SHAPE,
    GEMMA4_SMALL_HF_WEIGHTS_TO_SHAPE,
)
from lilax._backend.checkpoint_conversion.model_families.gpt_oss.hf_shape import GPT_OSS_HF_WEIGHTS_TO_SHAPE
from lilax._backend.checkpoint_conversion.model_families.hy4.hf_shape import HY4_HF_WEIGHTS_TO_SHAPE
from lilax._backend.checkpoint_conversion.model_families.llama31.hf_shape import LLAMA31_HF_WEIGHTS_TO_SHAPE
from lilax._backend.checkpoint_conversion.model_families.mixtral.hf_shape import MIXTRAL_HF_WEIGHTS_TO_SHAPE
from lilax._backend.checkpoint_conversion.model_families.qwen.hf_shape import QWEN_HF_WEIGHTS_TO_SHAPE
from lilax._backend.checkpoint_conversion.model_families.qwen3_5.hf_shape import QWEN3_5_HF_WEIGHTS_TO_SHAPE
from lilax._backend.checkpoint_conversion.model_families.qwen3_8.hf_shape import QWEN3_8_HF_WEIGHTS_TO_SHAPE
from lilax._backend.checkpoint_conversion.model_families.qwen3_8_flash_next.hf_shape import (
    QWEN3_8_FLASH_NEXT_HF_WEIGHTS_TO_SHAPE,
)
from lilax._backend.checkpoint_conversion.model_families.qwen3_next.hf_shape import QWEN3_NEXT_HF_WEIGHTS_TO_SHAPE
from lilax._backend.checkpoint_conversion.model_families.qwen3_vl.hf_shape import QWEN3_VL_HF_WEIGHTS_TO_SHAPE


HF_SHAPE = {
    "gemma2-2b": GEMMA2_HF_WEIGHTS_TO_SHAPE,
    "gemma2-9b": GEMMA2_HF_WEIGHTS_TO_SHAPE,
    "gemma2-27b": GEMMA2_HF_WEIGHTS_TO_SHAPE,
    "gemma3-4b": GEMMA3_HF_WEIGHTS_TO_SHAPE,
    "gemma3-12b": GEMMA3_HF_WEIGHTS_TO_SHAPE,
    "gemma3-27b": GEMMA3_HF_WEIGHTS_TO_SHAPE,
    "gemma4-26b": GEMMA4_HF_WEIGHTS_TO_SHAPE,
    "gemma4-31b": GEMMA4_HF_WEIGHTS_TO_SHAPE,
    "gemma4-e2b": GEMMA4_SMALL_HF_WEIGHTS_TO_SHAPE,
    "gemma4-e4b": GEMMA4_SMALL_HF_WEIGHTS_TO_SHAPE,
    "qwen2.5-1.5b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen2.5-7b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen2.5-14b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-0.6b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-4b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-4b-thinking-2507": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-8b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-14b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-32b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "llama3.1-8b": LLAMA31_HF_WEIGHTS_TO_SHAPE,
    "llama3.1-8b-Instruct": LLAMA31_HF_WEIGHTS_TO_SHAPE,
    "llama3.1-70b": LLAMA31_HF_WEIGHTS_TO_SHAPE,
    "llama3.1-405b": LLAMA31_HF_WEIGHTS_TO_SHAPE,
    "qwen3-30b-a3b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-30b-a3b-base": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-235b-a22b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-480b-a35b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "deepseek2-16b": DEEPSEEK_HF_WEIGHTS_TO_SHAPE,
    "deepseek3-671b": DEEPSEEK_HF_WEIGHTS_TO_SHAPE,
    "deepseek3.2-671b": DEEPSEEK_HF_WEIGHTS_TO_SHAPE,
    "deepseek4-284b": DEEPSEEKV4_HF_WEIGHTS_TO_SHAPE,
    "deepseek4-1600b": DEEPSEEKV4_HF_WEIGHTS_TO_SHAPE,
    "mixtral-8x7b": MIXTRAL_HF_WEIGHTS_TO_SHAPE,
    "mixtral-8x22b": MIXTRAL_HF_WEIGHTS_TO_SHAPE,
    "gpt-oss-20b": GPT_OSS_HF_WEIGHTS_TO_SHAPE,
    "gpt-oss-120b": GPT_OSS_HF_WEIGHTS_TO_SHAPE,
    "qwen3.5-9b": QWEN3_5_HF_WEIGHTS_TO_SHAPE,
    "qwen3.5-35b-a3b": QWEN3_5_HF_WEIGHTS_TO_SHAPE,
    "qwen3.5-35b-a3b-fp8": QWEN3_5_HF_WEIGHTS_TO_SHAPE,
    "qwen3.5-35b-fp8": QWEN3_5_HF_WEIGHTS_TO_SHAPE,
    "qwen3.5-397b-a17b": QWEN3_5_HF_WEIGHTS_TO_SHAPE,
    "qwen3.5-397b-a17b-fp8": QWEN3_5_HF_WEIGHTS_TO_SHAPE,
    "qwen3.8-2.4t-a95b": QWEN3_8_HF_WEIGHTS_TO_SHAPE,
    "qwen3-next-80b-a3b": QWEN3_NEXT_HF_WEIGHTS_TO_SHAPE,
    "qwen3.8-flash-next": QWEN3_8_FLASH_NEXT_HF_WEIGHTS_TO_SHAPE,
    "qwen3.8-flash-next-mini": QWEN3_8_FLASH_NEXT_HF_WEIGHTS_TO_SHAPE,
    "qwen3-vl-2b": QWEN3_VL_HF_WEIGHTS_TO_SHAPE,
    "qwen3-vl-4b": QWEN3_VL_HF_WEIGHTS_TO_SHAPE,
    "qwen3-vl-30b-a3b": QWEN3_VL_HF_WEIGHTS_TO_SHAPE,
    "hy4-preview": HY4_HF_WEIGHTS_TO_SHAPE,
}
