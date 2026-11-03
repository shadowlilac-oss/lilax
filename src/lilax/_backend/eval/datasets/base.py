"""Abstract base classes for benchmark datasets."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
  import transformers


class SampleRequest(NamedTuple):
  prompt: str
  reference: str
  metadata: dict | None = None


class BenchmarkDataset(abc.ABC):
  name: str

  @abc.abstractmethod
  def sample_requests(
      self, num_samples: int | None, tokenizer: transformers.PreTrainedTokenizerBase
  ) -> list[SampleRequest]:
    """Load formatted inference requests and references."""
