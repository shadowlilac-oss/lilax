"""Dataset descriptors resolved by a Trainer alongside Hugging Face datasets."""

from dataclasses import dataclass

from datasets import Dataset, IterableDataset, load_dataset
from .rl import RLDataOptions, prepare_rl_dataset
from .tokenizer_training import train_tokenizer_from_files


@dataclass(frozen=True, kw_only=True)
class SyntheticDataset:
  reuse_example_batch: int = 0
  max_segments_per_seq: int = -1


@dataclass(frozen=True)
class TeacherLogitsDataset:
  files: str


__all__ = ["Dataset", "IterableDataset", "load_dataset", "SyntheticDataset", "TeacherLogitsDataset",
           "RLDataOptions", "prepare_rl_dataset", "train_tokenizer_from_files"]
