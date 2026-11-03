"""Architectural configurations of the Hugging Face Qwen3.8 text-only models."""

import transformers


qwen3_8_2_4t_a95b_dict = {
    "architectures": ["Qwen3_5MoeForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_output_gate": True,
    "bos_token_id": 248044,
    "dtype": "bfloat16",
    "eos_token_id": 248044,
    "full_attention_interval": 4,
    "head_dim": 256,
    "hidden_act": "silu",
    "hidden_size": 8192,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 128,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 262144,
    "model_type": "qwen3_5_moe_text",
    "moe_intermediate_size": 2048,
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "num_attention_heads": 64,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "num_hidden_layers": 92,
    "num_key_value_heads": 4,
    "output_gate_type": "swish",
    "partial_rotary_factor": 0.25,
    "rms_norm_eps": 1.0e-6,
    "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
    "shared_expert_intermediate_size": 2048,
    "tie_word_embeddings": False,
    "use_cache": True,
    "vocab_size": 248320,
}
qwen3_8_2_4t_a95b_config = transformers.Qwen3_5MoeTextConfig(**qwen3_8_2_4t_a95b_dict)
qwen3_8_2_4t_a95b_config.full_attention_interval = qwen3_8_2_4t_a95b_dict["full_attention_interval"]
