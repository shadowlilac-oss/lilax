"""Hugging Face tokenizer invocation and token-id extraction."""

INPUT_TOKENS_KEY = "input_ids"


def extract_token_ids(tokens):
  """Extracts token IDs from various tokenizer output formats.

  This helper function standardizes the extraction of tokenized integer IDs
  from common return types of Hugging Face tokenizers, including
  `BatchEncoding` objects, dictionaries, or simple lists.

  Args:
    tokens: The object containing token IDs. Supported types include:
      - A list of integers.
      - A dictionary containing the `INPUT_TOKENS_KEY`.
      - An object (e.g., `BatchEncoding`) with an attribute named `INPUT_TOKENS_KEY`.

  Returns:
    A list of integer token IDs.

  Raises:
    ValueError: If the input type is not supported or does not contain the expected key.
  """
  if hasattr(tokens, INPUT_TOKENS_KEY):
    return getattr(tokens, INPUT_TOKENS_KEY)
  elif isinstance(tokens, dict) and INPUT_TOKENS_KEY in tokens:
    return tokens[INPUT_TOKENS_KEY]
  elif isinstance(tokens, list):
    return tokens
  else:
    raise ValueError(f"Can't extract token_ids from type {type(tokens)}")


def tokenization(example, hf_tokenizer, truncation, max_length, column_names):
  """Tokenize a HuggingFace dataset"""
  for column_name in column_names:
    if isinstance(example[column_name], list):
      example[column_name] = [
          hf_tokenizer(x, truncation=truncation, max_length=max_length)["input_ids"] for x in example[column_name]
      ]
    elif isinstance(example[column_name], str):
      example[column_name] = hf_tokenizer(example[column_name], truncation=truncation, max_length=max_length)["input_ids"]
  return example
