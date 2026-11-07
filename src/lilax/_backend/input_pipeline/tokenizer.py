"""Provides tokenizers for input pipelines."""

from pathlib import Path
from typing import Collection, Literal, Sequence

import tiktoken
import transformers
from sentencepiece import SentencePieceProcessor
from tiktoken.load import load_tiktoken_bpe

from lilax._backend.observability import max_logging
from lilax._backend.storage.gcs import read_bytes_from_gcs


class TikTokenTokenizer:
  """Tokenizing and encoding/decoding text using the Tiktoken tokenizer."""

  special_tokens: dict[str, int]
  num_reserved_special_tokens = 256
  pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

  def __init__(self, model_path: str, add_bos: bool, add_eos: bool):
    try:
      mergeable_ranks = load_tiktoken_bpe(model_path)
    except Exception as e:
      raise ValueError(f"Failed to load tiktoken tokenizer from {model_path}: {e}") from e
    num_base_tokens = len(mergeable_ranks)
    special_tokens = [
        "<|begin_of_text|>", "<|end_of_text|>", "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>", "<|reserved_special_token_2|>", "<|reserved_special_token_3|>",
        "<|start_header_id|>", "<|end_header_id|>", "<|reserved_special_token_4|>", "<|eot_id|>",
    ] + [f"<|reserved_special_token_{i}|>" for i in range(5, self.num_reserved_special_tokens - 5)]
    self.special_tokens = {token: num_base_tokens + i for i, token in enumerate(special_tokens)}
    self.model = tiktoken.Encoding(
        name=Path(model_path).name, pat_str=self.pat_str, mergeable_ranks=mergeable_ranks,
        special_tokens=self.special_tokens,
    )
    self.eos = add_eos
    self.bos = add_bos
    max_logging.log(f"Reloaded tiktoken model from {model_path}")
    self.n_words: int = self.model.n_vocab
    self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
    self.eos_id: int = self.special_tokens["<|end_of_text|>"]
    self.pad_id: int = -1
    self.stop_tokens = {self.special_tokens["<|end_of_text|>"], self.special_tokens["<|eot_id|>"]}
    max_logging.log(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")

  def encode(
      self, s: str, *, allowed_special: Literal["all"] | Collection[str] = (),
      disallowed_special: Literal["all"] | Collection[str] = (),
  ) -> list[int]:
    assert isinstance(s, str)
    substrs = (
        substr for i in range(0, len(s), 400_000)
        for substr in self._split_whitespaces_or_nonwhitespaces(s[i : i + 400_000], 25_000)
    )
    tokens: list[int] = []
    for substr in substrs:
      tokens.extend(self.model.encode(substr, allowed_special=set(allowed_special), disallowed_special=disallowed_special))
    if self.bos:
      tokens.insert(0, self.bos_id)
    if self.eos:
      tokens.append(self.eos_id)
    return tokens

  def decode(self, t) -> str:
    return self.model.decode(t)

  @staticmethod
  def _split_whitespaces_or_nonwhitespaces(s: str, max_consecutive_slice_len: int):
    current_slice_len = 0
    current_slice_is_space = s[0].isspace() if len(s) > 0 else False
    slice_start = 0
    for i, _ in enumerate(s):
      is_now_space = s[i].isspace()
      if current_slice_is_space ^ is_now_space:
        current_slice_len = 1
        current_slice_is_space = is_now_space
      else:
        current_slice_len += 1
        if current_slice_len > max_consecutive_slice_len:
          yield s[slice_start:i]
          slice_start = i
          current_slice_len = 1
    yield s[slice_start:]


class SentencePieceTokenizer:
  """Tokenizing and encoding/decoding text using sentencepiece."""

  def __init__(self, model_path: str, add_bos: bool, add_eos: bool):
    max_logging.log(f"Loading sentencepiece tokenizer: {model_path}")
    self._tokenizer_model = SentencePieceProcessor()
    try:
      if model_path.startswith("gs://"):
        model_proto = read_bytes_from_gcs(model_path)
        self._tokenizer_model.LoadFromSerializedProto(model_proto)
      else:
        self._tokenizer_model.Load(model_path)
    except Exception as e:
      raise ValueError(f"Failed to load sentencepiece tokenizer from {model_path}: {e}") from e
    self.pad_id = self._tokenizer_model.pad_id()
    self.unk_id = self._tokenizer_model.unk_id()
    self.bos_id = self._tokenizer_model.bos_id()
    self.eos_id = self._tokenizer_model.eos_id()
    self.add_bos = add_bos
    self.add_eos = add_eos

  def encode(self, s: str) -> list[int]:
    token_ids = self._tokenizer_model.EncodeAsIds(s)
    if self.add_bos:
      token_ids = [self.bos_id] + token_ids
    if self.add_eos:
      token_ids += [self.eos_id]
    return token_ids

  def decode(self, t: Sequence[int]) -> str:
    return self._tokenizer_model.DecodeIds(t)


class HFTokenizer:
  """Tokenizing using a Hugging Face tokenizer."""

  def __init__(self, model_path: str, add_bos: bool, add_eos: bool, hf_access_token: str, *, processing_class=None):
    max_logging.log(f"Loading HF tokenizer: {model_path}")
    if processing_class is not None:
      self.tokenizer = getattr(processing_class, "tokenizer", processing_class)
    else:
      try:
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path, add_bos_token=add_bos, add_eos_token=add_eos, token=hf_access_token,
        )
      except Exception as e:
        raise ValueError(f"Failed to load Hugging Face tokenizer from {model_path}: {e}") from e
    self.pad_id = self.tokenizer.pad_token_id
    self.unk_id = self.tokenizer.unk_token_id
    self.bos_id = self.tokenizer.bos_token_id
    self.eos_id = self.tokenizer.eos_token_id

  def encode(self, s: str) -> list[int]:
    return self.tokenizer.encode(s)

  def decode(self, t: Sequence[int]) -> str:
    return self.tokenizer.decode(t)

  def __len__(self) -> int:
    return len(self.tokenizer)

  def batch_decode(self, sequences, **kwargs):
    return self.tokenizer.batch_decode(sequences, **kwargs)

  def convert_ids_to_tokens(self, ids, **kwargs):
    return self.tokenizer.convert_ids_to_tokens(ids, **kwargs)


def build_tokenizer(tokenizer_path, tokenizer_type, add_bos, add_eos, hf_access_token):
  """Loads the tokenizer at tokenizer_path."""
  max_logging.log(f"Tokenizer path: {tokenizer_path}")
  if tokenizer_type == "tiktoken":
    assert "tiktoken" in tokenizer_path, f"Invalid tokenizer type: {tokenizer_type} chosen for {tokenizer_path}"
    return TikTokenTokenizer(tokenizer_path, add_bos, add_eos)
  elif tokenizer_type == "huggingface":
    return HFTokenizer(tokenizer_path, add_bos, add_eos, hf_access_token)
  elif tokenizer_type == "sentencepiece":
    return SentencePieceTokenizer(tokenizer_path, add_bos, add_eos)
  else:
    raise ValueError(f"Invalid tokenizer_type:{tokenizer_type} chosen in config")
