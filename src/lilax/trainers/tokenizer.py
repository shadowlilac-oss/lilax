"""Public tokenizer training interfaces."""

import os

from lilax._backend.trainers.tokenizer.train_tokenizer import build_grain_iterator, train_tokenizer
from lilax._backend.utils.globals import MAXTEXT_ASSETS_ROOT


def train_tokenizer_from_files(
    grain_train_files: str,
    grain_file_type: str = "parquet",
    data_column: str = "text",
    vocab_size: int = 32_768,
    max_corpus_chars: int = 10_000_000,
    assets_path: str = MAXTEXT_ASSETS_ROOT,
    vocab_model_name: str = "tokenizer",
):
  """Train a SentencePiece tokenizer from files using native MaxText defaults."""
  data_keys = (data_column,)
  dataset_iter = build_grain_iterator(grain_train_files, grain_file_type, data_keys=data_keys)
  train_tokenizer(
      dataset_iter, vocab_path=os.path.join(assets_path, vocab_model_name), vocab_size=vocab_size,
      max_corpus_chars=max_corpus_chars, data_keys=data_keys,
  )
