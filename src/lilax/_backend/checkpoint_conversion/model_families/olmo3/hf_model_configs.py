"""Architectural configurations of the Hugging Face OLMo 3 models."""

import transformers


olmo3_7b_dict = {
    "architectures": ["Olmo3ForCausalLM"],
    "model_type": "olmo3",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "intermediate_size": 11008,
    "vocab_size": 100278,
    "max_position_embeddings": 8192,
    "rope_theta": 500000,
    "sliding_window": 4096,
    "rms_norm_eps": 1.0e-6,
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": False,
    "pad_token_id": 100277,
    "hidden_act": "silu",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "use_cache": True,
}
olmo3_7b_config = transformers.Olmo3Config(**olmo3_7b_dict)

olmo3_32b_dict = {
    **olmo3_7b_dict,
    "hidden_size": 5120,
    "num_hidden_layers": 64,
    "num_attention_heads": 40,
    "num_key_value_heads": 8,
    "intermediate_size": 27648,
}
olmo3_32b_config = transformers.Olmo3Config(**olmo3_32b_dict)
