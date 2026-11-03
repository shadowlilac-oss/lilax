"""Training engine interfaces and shared metrics types."""

from __future__ import annotations

import abc
from collections.abc import Callable
import dataclasses
from typing import Any

import flax.struct
import jax
from tunix.dist_rl.common.datatypes import RLTrainerPayload
from tunix.dist_rl.common.datatypes import TrainerPayload
from tunix.sft.utils import LossOutput
from tunix.sft.utils import WeightedMetric

__all__ = [
    "AbstractTrainingEngine",
    "LossOutput",
    "MetricsBuffer",
    "RLTrainerPayload",
    "TrainerPayload",
    "TrainingConfig",
    "WeightedMetric",
]


@flax.struct.dataclass
class MetricsBuffer:
  """Unreduced metrics stored on device."""

  id: Any
  weighted_metrics: dict[str, WeightedMetric] = flax.struct.field(default_factory=dict)
  scalar_metrics: dict[str, jax.Array] = flax.struct.field(default_factory=dict)
  aggregation_fns: dict[str, Callable[[jax.Array], Any]] = flax.struct.field(default_factory=dict, pytree_node=False)
  mode: str = flax.struct.field(default="train", pytree_node=False)


@dataclasses.dataclass
class TrainingConfig:
  """Training hyperparameters and operational settings."""

  eval_every_n_steps: int = 0
  max_steps: int | None = None
  gradient_accumulation_steps: int | None = None
  checkpoint_root_directory: str | None = None
  metrics_prefix: str = ""
  max_inflight_computations: int = 2


class AbstractTrainingEngine(abc.ABC):
  """Interface for model training and checkpointing."""

  @abc.abstractmethod
  def __init__(self, training_config: TrainingConfig) -> None:
    """Initialize the engine."""

  @abc.abstractmethod
  def with_loss_fn(self, customized_fn: Callable[..., Any], has_aux: bool = False) -> "AbstractTrainingEngine":
    """Set the loss function."""

  @abc.abstractmethod
  def with_gen_model_input_fn(self, gen_model_input_fn: Callable[[Any], dict[str, Any]]) -> "AbstractTrainingEngine":
    """Set the payload-to-model-input adapter."""

  @abc.abstractmethod
  def compile(self, dummy_data: TrainerPayload | None = None) -> None:
    """Compile the engine for representative data."""

  @abc.abstractmethod
  def fwd_bwd(self, payload: TrainerPayload) -> None:
    """Run forward and backward passes."""

  @abc.abstractmethod
  def update(self) -> int:
    """Apply accumulated gradients and return the step."""

  @abc.abstractmethod
  def eval_step(self, payload: TrainerPayload) -> None:
    """Run an evaluation step."""

  @abc.abstractmethod
  def fwd_only(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a read-only function on the model."""

  @abc.abstractmethod
  def save_checkpoint(self, metadata: Any = None, *, step: int | None = None, force: bool = False) -> None:
    """Save model and optimizer state."""

  @abc.abstractmethod
  def restore_checkpoint(self, *, step: int | None = None) -> Any:
    """Restore state and return checkpoint metadata."""

  @abc.abstractmethod
  def get_metrics(self, clear_cache: bool = True) -> MetricsBuffer:
    """Return accumulated metrics."""

  @abc.abstractmethod
  def prepare_weight_sync(self, *, sync_request: Any = None) -> Any:
    """Stage weights for transfer."""

  @abc.abstractmethod
  def release_weight_sync(self, *, sync_request: Any = None) -> Any:
    """Release staged weight buffers."""

  @property
  @abc.abstractmethod
  def train_step(self) -> int:
    """Return the current training step."""

  @abc.abstractmethod
  def close(self) -> None:
    """Clean up engine resources."""
