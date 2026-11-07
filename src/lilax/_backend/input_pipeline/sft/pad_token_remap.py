"""Remapping of tokenizer pad tokens that collide with EOS."""

from lilax._backend.observability import max_logging

_DEEPSEEK_PAD_TOKEN = "<｜▁pad▁｜>"


def fix_pad_token_if_needed(tokenizer_model) -> bool:
  """Remaps pad_token to `<｜▁pad▁｜>` when the tokenizer shares one id for pad and EOS.

  DeepSeek-V3, V3.2 and V4 tokenizers set pad_token == eos_token (id 1) while reserving
  `<｜▁pad▁｜>` (id 2). SFT masks prompt targets with pad_id and drops targets equal to pad_id
  from the loss, so a shared id also drops every completion EOS.

  Args:
    tokenizer_model: A Hugging Face tokenizer; modified in place.

  Returns:
    True if pad_token was remapped.
  """
  pad_id = getattr(tokenizer_model, "pad_token_id", None)
  eos_id = getattr(tokenizer_model, "eos_token_id", None)
  if pad_id is None or pad_id != eos_id:
    return False
  try:
    if hasattr(tokenizer_model, "get_vocab"):
      has_token = _DEEPSEEK_PAD_TOKEN in tokenizer_model.get_vocab()
    elif hasattr(tokenizer_model, "convert_tokens_to_ids"):
      token_id = tokenizer_model.convert_tokens_to_ids(_DEEPSEEK_PAD_TOKEN)
      has_token = token_id is not None and token_id != getattr(tokenizer_model, "unk_token_id", None)
    else:
      has_token = False
  except Exception:
    has_token = False
  if not has_token:
    return False
  tokenizer_model.pad_token = _DEEPSEEK_PAD_TOKEN
  max_logging.log(
      f"Tokenizer pad_token_id == eos_token_id ({eos_id}); remapped pad_token to {_DEEPSEEK_PAD_TOKEN} "
      f"(pad_id {tokenizer_model.pad_token_id}) so completion EOS tokens keep their loss weight."
  )
  return True
