"""Architectural configurations of the Hugging Face Qwen3-Omni MoE models."""

import transformers


qwen3_omni_30b_a3b_config = transformers.Qwen3OmniMoeConfig(
    architectures=["Qwen3OmniMoeForConditionalGeneration"],
    thinker_config={
        "text_config": {
            "num_hidden_layers": 48,
            "num_experts": 128,
        },
        "audio_config": {
            "encoder_layers": 32,
            "d_model": 1280,
            "encoder_attention_heads": 20,
        },
        "vision_config": {
            "depth": 27,
            "num_heads": 16,
            "hidden_size": 1152,
        },
    },
)
