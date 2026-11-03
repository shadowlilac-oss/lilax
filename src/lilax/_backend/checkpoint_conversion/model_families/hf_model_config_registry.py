"""Registry of model-family Hugging Face architectural configurations keyed by model name."""

from lilax._backend.checkpoint_conversion.model_families.deepseek.hf_model_configs import (
    deepseek2_16b_config,
    deepseek3_671b_config,
    deepseek32_671b_config,
)
from lilax._backend.checkpoint_conversion.model_families.deepseek_v4.hf_model_configs import (
    deepseek4_284b_config,
    deepseek4_1600b_config,
)
from lilax._backend.checkpoint_conversion.model_families.gemma2.hf_model_configs import (
    gemma2_2b_config,
    gemma2_9b_config,
    gemma2_27b_config,
)
from lilax._backend.checkpoint_conversion.model_families.gemma3.hf_model_configs import (
    gemma3_4b_config,
    gemma3_12b_config,
    gemma3_27b_config,
)
from lilax._backend.checkpoint_conversion.model_families.gemma4.hf_model_configs import (
    gemma4_26b_config,
    gemma4_31b_config,
    gemma4_e2b_config,
    gemma4_e4b_config,
)
from lilax._backend.checkpoint_conversion.model_families.gpt_oss.hf_model_configs import (
    gpt_oss_20b_config,
    gpt_oss_120b_config,
)
from lilax._backend.checkpoint_conversion.model_families.hy4.hf_model_configs import hy4_preview_config
from lilax._backend.checkpoint_conversion.model_families.llama31.hf_model_configs import (
    llama31_8b_config,
    llama31_70b_config,
    llama31_405b_config,
)
from lilax._backend.checkpoint_conversion.model_families.ming.hf_model_configs import ming_3_0_audio_tiny_config
from lilax._backend.checkpoint_conversion.model_families.mixtral.hf_model_configs import (
    mixtral_8x7b_config,
    mixtral_8x22b_config,
)
from lilax._backend.checkpoint_conversion.model_families.olmo3.hf_model_configs import (
    olmo3_7b_config,
    olmo3_32b_config,
)
from lilax._backend.checkpoint_conversion.model_families.qwen.hf_model_configs import (
    qwen25_1_5b_config,
    qwen25_7b_config,
    qwen25_14b_config,
    qwen3_0_6b_config,
    qwen3_1_7b_config,
    qwen3_4b_config,
    qwen3_8b_config,
    qwen3_14b_config,
    qwen3_32b_config,
    qwen3_30b_a3b_thinking_2507_config,
    qwen3_30b_a3b_base_config,
    qwen3_235b_a22b_thinking_2507_config,
    qwen3_coder_480b_a35b_config,
)
from lilax._backend.checkpoint_conversion.model_families.qwen3_5.hf_model_configs import (
    qwen3_5_397b_a17b_config,
    qwen3_5_35b_a3b_config,
    qwen3_5_9b_config,
)
from lilax._backend.checkpoint_conversion.model_families.qwen3_8.hf_model_configs import qwen3_8_2_4t_a95b_config
from lilax._backend.checkpoint_conversion.model_families.qwen3_8_flash_next.hf_model_configs import (
    qwen3_8_flash_next_config,
    qwen3_8_flash_next_mini_config,
)
from lilax._backend.checkpoint_conversion.model_families.qwen3_next.hf_model_configs import qwen3_next_80b_a3b_config
from lilax._backend.checkpoint_conversion.model_families.qwen3_omni.hf_model_configs import qwen3_omni_30b_a3b_config
from lilax._backend.checkpoint_conversion.model_families.qwen3_vl.hf_model_configs import (
    qwen3_vl_2b_config,
    qwen3_vl_4b_config,
    qwen3_vl_30b_a3b_config,
)
from lilax._backend.checkpoint_conversion.model_families.weaver.hf_model_configs import (
    weaver_max_config,
    weaver_mini_config,
)


HF_MODEL_CONFIGS = {
    "gemma2-2b": gemma2_2b_config,
    "gemma2-9b": gemma2_9b_config,
    "gemma2-27b": gemma2_27b_config,
    "gemma3-4b": gemma3_4b_config,
    "gemma3-12b": gemma3_12b_config,
    "gemma3-27b": gemma3_27b_config,
    "gemma4-26b": gemma4_26b_config,
    "gemma4-31b": gemma4_31b_config,
    "gemma4-e2b": gemma4_e2b_config,
    "gemma4-e4b": gemma4_e4b_config,
    "qwen2.5-1.5b": qwen25_1_5b_config,
    "qwen2.5-7b": qwen25_7b_config,
    "qwen2.5-14b": qwen25_14b_config,
    "qwen3-0.6b": qwen3_0_6b_config,
    "qwen3-1.7b": qwen3_1_7b_config,
    "qwen3-1.7b-base": qwen3_1_7b_config,
    "qwen3-4b": qwen3_4b_config,
    "qwen3-4b-base": qwen3_4b_config,
    "qwen3-4b-thinking-2507": qwen3_4b_config,
    "qwen3-8b": qwen3_8b_config,
    "qwen3-8b-base": qwen3_8b_config,
    "qwen3-14b": qwen3_14b_config,
    "qwen3-14b-base": qwen3_14b_config,
    "qwen3-32b": qwen3_32b_config,
    "qwen3-vl-2b": qwen3_vl_2b_config,
    "qwen3-vl-4b": qwen3_vl_4b_config,
    "qwen3-vl-30b-a3b": qwen3_vl_30b_a3b_config,
    "weaver-mini": weaver_mini_config,
    "weaver-max": weaver_max_config,
    "llama3.1-8b": llama31_8b_config,
    "llama3.1-8b-Instruct": llama31_8b_config,
    "llama3.1-70b": llama31_70b_config,
    "llama3.1-405b": llama31_405b_config,
    "qwen3-30b-a3b": qwen3_30b_a3b_thinking_2507_config,
    "qwen3-30b-a3b-base": qwen3_30b_a3b_base_config,
    "qwen3-235b-a22b": qwen3_235b_a22b_thinking_2507_config,
    "qwen3-480b-a35b": qwen3_coder_480b_a35b_config,
    "qwen3.8-2.4t-a95b": qwen3_8_2_4t_a95b_config,
    "deepseek2-16b": deepseek2_16b_config,
    "deepseek3-671b": deepseek3_671b_config,
    "deepseek3.2-671b": deepseek32_671b_config,
    "deepseek4-284b": deepseek4_284b_config,
    "deepseek4-1600b": deepseek4_1600b_config,
    "qwen3-omni-30b-a3b": qwen3_omni_30b_a3b_config,
    "qwen3-next-80b-a3b": qwen3_next_80b_a3b_config,
    "qwen3.8-flash-next": qwen3_8_flash_next_config,
    "qwen3.8-flash-next-mini": qwen3_8_flash_next_mini_config,
    "qwen3.5-397b-a17b": qwen3_5_397b_a17b_config,
    "qwen3.5-397b-a17b-fp8": qwen3_5_397b_a17b_config,
    "qwen3.5-35b-a3b": qwen3_5_35b_a3b_config,
    "qwen3.5-35b-a3b-fp8": qwen3_5_35b_a3b_config,
    "qwen3.5-35b-fp8": qwen3_5_35b_a3b_config,
    "qwen3.5-9b": qwen3_5_9b_config,
    "gpt-oss-20b": gpt_oss_20b_config,
    "gpt-oss-120b": gpt_oss_120b_config,
    "hy4-preview": hy4_preview_config,
    "ming-3.0-audio-tiny": ming_3_0_audio_tiny_config,
    "mixtral-8x7b": mixtral_8x7b_config,
    "mixtral-8x22b": mixtral_8x22b_config,
    "olmo3-7b": olmo3_7b_config,
    "olmo3-7b-pt": olmo3_7b_config,
    "olmo3-32b": olmo3_32b_config,
}
