"""Registry of model-family parameter mappings and transformation hooks keyed by model name."""

from lilax._backend.checkpoint_conversion.model_families.deepseek.param_mapping import (
    DEEPSEEK_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    DEEPSEEK_MAXTEXT_TO_HF_PARAM_MAPPING,
    DEEPSEEK_NNX_TO_VLLM_PARAM_HOOK_FN,
)
from lilax._backend.checkpoint_conversion.model_families.deepseek_v4.param_mapping import (
    DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_MAPPING,
)
from lilax._backend.checkpoint_conversion.model_families.gemma2.param_mapping import (
    GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING,
)
from lilax._backend.checkpoint_conversion.model_families.gemma3.param_mapping import (
    GEMMA3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    GEMMA3_MAXTEXT_TO_HF_PARAM_MAPPING,
)
from lilax._backend.checkpoint_conversion.model_families.gemma4.param_mapping import (
    GEMMA4_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    GEMMA4_MAXTEXT_TO_HF_PARAM_MAPPING,
    GEMMA4_SMALL_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    GEMMA4_SMALL_MAXTEXT_TO_HF_PARAM_MAPPING,
)
from lilax._backend.checkpoint_conversion.model_families.gpt_oss.param_mapping import (
    GPT_OSS_MAXTEXT_TO_HF_PARAM_MAPPING,
    GPT_OSS_TO_HF_PARAM_HOOK_FN,
)
from lilax._backend.checkpoint_conversion.model_families.hy4.param_mapping import (
    HY4_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    HY4_MAXTEXT_TO_HF_PARAM_MAPPING,
)
from lilax._backend.checkpoint_conversion.model_families.llama31.param_mapping import (
    LLAMA31_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    LLAMA31_MAXTEXT_TO_HF_PARAM_MAPPING,
    LLAMA31_NNX_TO_VLLM_PARAM_HOOK_FN,
)
from lilax._backend.checkpoint_conversion.model_families.ming.param_mapping import (
    MING_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    MING_MAXTEXT_TO_HF_PARAM_MAPPING,
)
from lilax._backend.checkpoint_conversion.model_families.mixtral.param_mapping import (
    MIXTRAL_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    MIXTRAL_MAXTEXT_TO_HF_PARAM_MAPPING,
)
from lilax._backend.checkpoint_conversion.model_families.olmo3.param_mapping import (
    OLMO3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    OLMO3_MAXTEXT_TO_HF_PARAM_MAPPING,
)
from lilax._backend.checkpoint_conversion.model_families.qwen.param_mapping import (
    QWEN3_NNX_TO_VLLM_PARAM_HOOK_FN,
    QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
)
from lilax._backend.checkpoint_conversion.model_families.qwen3_5.param_mapping import (
    QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING,
)
from lilax._backend.checkpoint_conversion.model_families.qwen3_8.param_mapping import (
    QWEN3_8_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    QWEN3_8_MAXTEXT_TO_HF_PARAM_MAPPING,
)
from lilax._backend.checkpoint_conversion.model_families.qwen3_8_flash_next.param_mapping import (
    QWEN3_8_FLASH_NEXT_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    QWEN3_8_FLASH_NEXT_MAXTEXT_TO_HF_PARAM_MAPPING,
)
from lilax._backend.checkpoint_conversion.model_families.qwen3_next.param_mapping import (
    QWEN3_NEXT_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    QWEN3_NEXT_MAXTEXT_TO_HF_PARAM_MAPPING,
)
from lilax._backend.checkpoint_conversion.model_families.qwen3_omni.param_mapping import (
    QWEN3_OMNI_MOE_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    QWEN3_OMNI_MOE_MAXTEXT_TO_HF_PARAM_MAPPING,
)
from lilax._backend.checkpoint_conversion.model_families.qwen3_vl.param_mapping import (
    QWEN3_VL_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    QWEN3_VL_MAXTEXT_TO_HF_PARAM_MAPPING,
)
from lilax._backend.checkpoint_conversion.model_families.weaver.param_mapping import (
    WEAVER_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    WEAVER_MAXTEXT_TO_HF_PARAM_MAPPING,
)


PARAM_MAPPING = {
    "gemma2-2b": GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma2-9b": GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma2-27b": GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma3-4b": GEMMA3_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma3-12b": GEMMA3_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma3-27b": GEMMA3_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma4-26b": GEMMA4_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma4-31b": GEMMA4_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma4-e2b": GEMMA4_SMALL_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma4-e4b": GEMMA4_SMALL_MAXTEXT_TO_HF_PARAM_MAPPING,
    "ming-3.0-audio-tiny": MING_MAXTEXT_TO_HF_PARAM_MAPPING,
    "hy4-preview": HY4_MAXTEXT_TO_HF_PARAM_MAPPING,
    "deepseek4-284b": DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_MAPPING,
    "deepseek4-1600b": DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen2.5-1.5b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen2.5-7b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen2.5-14b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-0.6b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-1.7b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-1.7b-base": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-4b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-4b-base": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-4b-thinking-2507": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-8b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-8b-base": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-14b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-14b-base": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-32b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-vl-2b": QWEN3_VL_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-vl-4b": QWEN3_VL_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-vl-30b-a3b": QWEN3_VL_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-omni-30b-a3b": QWEN3_OMNI_MOE_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gpt-oss-20b": GPT_OSS_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gpt-oss-120b": GPT_OSS_MAXTEXT_TO_HF_PARAM_MAPPING,
    "deepseek2-16b": DEEPSEEK_MAXTEXT_TO_HF_PARAM_MAPPING,
    "deepseek3-671b": DEEPSEEK_MAXTEXT_TO_HF_PARAM_MAPPING,
    "deepseek3.2-671b": DEEPSEEK_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-next-80b-a3b": QWEN3_NEXT_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3.8-flash-next": QWEN3_8_FLASH_NEXT_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3.8-flash-next-mini": QWEN3_8_FLASH_NEXT_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3.5-397b-a17b": QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3.5-397b-a17b-fp8": QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3.5-35b-a3b": QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3.5-35b-a3b-fp8": QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3.5-35b-fp8": QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3.5-9b": QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3.8-2.4t-a95b": QWEN3_8_MAXTEXT_TO_HF_PARAM_MAPPING,
    "mixtral-8x7b": MIXTRAL_MAXTEXT_TO_HF_PARAM_MAPPING,
    "mixtral-8x22b": MIXTRAL_MAXTEXT_TO_HF_PARAM_MAPPING,
    "weaver-mini": WEAVER_MAXTEXT_TO_HF_PARAM_MAPPING,
    "weaver-max": WEAVER_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-30b-a3b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-30b-a3b-base": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-235b-a22b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-coder-480b-a35b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "olmo3-7b": OLMO3_MAXTEXT_TO_HF_PARAM_MAPPING,
    "olmo3-7b-pt": OLMO3_MAXTEXT_TO_HF_PARAM_MAPPING,
    "olmo3-32b": OLMO3_MAXTEXT_TO_HF_PARAM_MAPPING,
    "llama3.1-8b": LLAMA31_MAXTEXT_TO_HF_PARAM_MAPPING,
    "llama3.1-8b-Instruct": LLAMA31_MAXTEXT_TO_HF_PARAM_MAPPING,
    "llama3.1-70b": LLAMA31_MAXTEXT_TO_HF_PARAM_MAPPING,
    "llama3.1-405b": LLAMA31_MAXTEXT_TO_HF_PARAM_MAPPING,
}

HOOK_FNS = {
    "gemma2-2b": GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma2-9b": GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma2-27b": GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma3-4b": GEMMA3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma3-12b": GEMMA3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma3-27b": GEMMA3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma4-26b": GEMMA4_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma4-31b": GEMMA4_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma4-e2b": GEMMA4_SMALL_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma4-e4b": GEMMA4_SMALL_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "ming-3.0-audio-tiny": MING_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "hy4-preview": HY4_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "deepseek4-tiny": DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "deepseek4-284b": DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "deepseek4-1600b": DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen2.5-1.5b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen2.5-7b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen2.5-14b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-0.6b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-1.7b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-1.7b-base": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-4b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-4b-base": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-4b-thinking-2507": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-8b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-8b-base": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-14b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-14b-base": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-32b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-vl-2b": QWEN3_VL_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-vl-4b": QWEN3_VL_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-vl-30b-a3b": QWEN3_VL_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-omni-30b-a3b": QWEN3_OMNI_MOE_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gpt-oss-20b": GPT_OSS_TO_HF_PARAM_HOOK_FN,
    "gpt-oss-120b": GPT_OSS_TO_HF_PARAM_HOOK_FN,
    "deepseek2-16b": DEEPSEEK_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "deepseek3-671b": DEEPSEEK_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "deepseek3.2-671b": DEEPSEEK_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-next-80b-a3b": QWEN3_NEXT_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3.8-flash-next": QWEN3_8_FLASH_NEXT_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3.8-flash-next-mini": QWEN3_8_FLASH_NEXT_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3.5-397b-a17b": QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3.5-397b-a17b-fp8": QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3.5-35b-a3b": QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3.5-35b-a3b-fp8": QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3.5-35b-fp8": QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3.5-9b": QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3.8-2.4t-a95b": QWEN3_8_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "mixtral-8x7b": MIXTRAL_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "mixtral-8x22b": MIXTRAL_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "weaver-mini": WEAVER_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "weaver-max": WEAVER_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-30b-a3b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-30b-a3b-base": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-235b-a22b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-coder-480b-a35b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "olmo3-7b": OLMO3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "olmo3-7b-pt": OLMO3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "olmo3-32b": OLMO3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "llama3.1-8b": LLAMA31_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "llama3.1-8b-Instruct": LLAMA31_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "llama3.1-70b": LLAMA31_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "llama3.1-405b": LLAMA31_MAXTEXT_TO_HF_PARAM_HOOK_FN,
}

VLLM_HOOK_FNS = {
    "deepseek3": DEEPSEEK_NNX_TO_VLLM_PARAM_HOOK_FN,
    "qwen3": QWEN3_NNX_TO_VLLM_PARAM_HOOK_FN,
    "llama3.1": LLAMA31_NNX_TO_VLLM_PARAM_HOOK_FN,
}
