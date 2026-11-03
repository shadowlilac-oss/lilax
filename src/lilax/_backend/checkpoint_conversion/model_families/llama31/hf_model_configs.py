"""Architectural configurations of the Hugging Face LLaMA 3.1 models."""

import transformers


llama31_8b_config = transformers.LlamaConfig(
    vocab_size=128256,
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    max_position_embeddings=131072,
    head_dim=128,
    rms_norm_eps=1e-5,
    bos_token_id=128000,
    eos_token_id=128001,
    attention_bias=False,
    attention_dropout=0.0,
    hidden_act="silu",
    initializer_range=0.02,
    mlp_bias=False,
    model_type="llama",
    pretraining_tp=1,
    rope_scaling={
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    rope_theta=500000.0,
    tie_word_embeddings=False,
    use_cache=True,
)

llama31_70b_config = transformers.LlamaConfig(
    vocab_size=128256,
    hidden_size=8192,
    intermediate_size=28672,
    num_hidden_layers=80,
    num_attention_heads=64,
    head_dim=128,
    num_key_value_heads=8,
    max_position_embeddings=131072,
    rms_norm_eps=1e-05,
    bos_token_id=128000,
    eos_token_id=[128001, 128008, 128009],
    rope_scaling={
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    rope_theta=500000.0,
    tie_word_embeddings=False,
)

llama31_405b_config = transformers.LlamaConfig(
    vocab_size=128256,
    hidden_size=16384,
    intermediate_size=53248,
    num_hidden_layers=126,
    num_attention_heads=128,
    num_key_value_heads=8,
    head_dim=128,
    max_position_embeddings=131072,
    rms_norm_eps=1e-05,
    bos_token_id=128000,
    eos_token_id=128001,
)
