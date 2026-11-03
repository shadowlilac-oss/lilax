"""Validation of native Engram configuration."""


def validate_engram(config):
  if config.engram_layers:
    if not config.hf_access_token or not config.tokenizer_path:
      raise ValueError(
          "Engram requires both 'hf_access_token' and 'tokenizer_path' " "to load the Hugging Face tokenizer."
      )
    if len(config.engram_vocab_bases) != (config.engram_max_ngram_size - 1):
      raise ValueError(
          f"Engram vocab size mismatch: expected {config.engram_max_ngram_size - 1} (max_ngram_size - 1), "
          f"but got {config.engram_vocab_bases}."
      )
