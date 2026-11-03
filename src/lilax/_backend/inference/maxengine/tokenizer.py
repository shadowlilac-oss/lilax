"""Tokenizer metadata and construction for MaxEngine."""


def get_tokenizer(config, tokenizer_type, tokenizer_parameters):
  try:
    tokenizer_type_val = tokenizer_type.DESCRIPTOR.values_by_name[config.tokenizer_type].number
    return tokenizer_parameters(
        path=config.tokenizer_path,
        tokenizer_type=tokenizer_type_val,
        access_token=config.hf_access_token,
        use_chat_template=config.use_chat_template,
        extra_ids=0,
    )
  except KeyError as _:
    raise KeyError(f"Unsupported tokenizer type: {config.tokenizer_type}") from None


def build_tokenizer(metadata, tokenizer_type, token_utils):
  if metadata.tokenizer_type == tokenizer_type.tiktoken:
    return token_utils.TikToken(metadata)
  elif metadata.tokenizer_type == tokenizer_type.sentencepiece:
    return token_utils.SentencePieceTokenizer(metadata)
  elif metadata.tokenizer_type == tokenizer_type.huggingface:
    tokenizer_model = token_utils.HuggingFaceTokenizer(metadata)
    if tokenizer_model.tokenizer.pad_token_id is None:
      if tokenizer_model.tokenizer.unk_token_id is not None:
        tokenizer_model.tokenizer.pad_token_id = tokenizer_model.tokenizer.unk_token_id
      else:
        print(f"Warning: setting pad_token_id to eos_token_id:{tokenizer_model.tokenizer.eos_token_id}")
        tokenizer_model.tokenizer.pad_token_id = tokenizer_model.tokenizer.eos_token_id
    return tokenizer_model
  else:
    raise ValueError(f"Unsupported tokenizer type: {metadata.tokenizer_type}")
