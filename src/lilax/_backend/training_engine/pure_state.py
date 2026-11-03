"""Pure-pytree cache of the NNX train state carried across engine steps."""

from __future__ import annotations

from typing import Any

from absl import logging
from flax import nnx
import jax

MODEL_STATE_KEY = "model"
OPTIMIZER_STATE_KEY = "optimizer"
_PURE_STATE_FALLBACK_WARNING = (
    "Cannot keep the train state as a pure pytree across steps (%s), so every fwd_bwd and "
    "update will re-walk the NNX module graph. That is correct but slow -- the two "
    "`nnx.split` calls cost ~92 ms per step on an unrolled 28-layer qwen3-0.6b, against "
    "~2 ms for the pure-state equivalent. Logged once per engine instance."
)


def with_model_state(state_pure: Any, model_pure: Any) -> Any:
  return nnx.State({**state_pure.raw_mapping, MODEL_STATE_KEY: model_pure.raw_mapping})


def _check_reusable(state_pure: Any, params_pure: Any, rest_pure: Any) -> str | None:
  if not isinstance(state_pure, nnx.State) or MODEL_STATE_KEY not in state_pure:
    return f"the train state's pure form has no {MODEL_STATE_KEY!r} entry"
  if not hasattr(state_pure, "raw_mapping"):
    return "this version of flax.nnx.State does not expose raw_mapping"
  rebuilt = with_model_state(state_pure, nnx.merge_state(params_pure, rest_pure))
  if jax.tree.structure(rebuilt) != jax.tree.structure(state_pure):
    return f"state[{MODEL_STATE_KEY!r}] is not the model's own state"
  return None


class PureStateCache:
  """Holds the split graphdefs and pure `(params, rest, state)` trees, falling back to re-splitting when unsound.

  `model_graphdef` and `state_graphdef` do not exist until the first split assigns them.
  """

  model_graphdef: Any
  state_graphdef: Any

  def __init__(self) -> None:
    self.params: Any = None
    self.rest: Any = None
    self.state: Any = None
    self._warned = False

  def invalidate(self) -> None:
    self.params = None
    self.rest = None
    self.state = None

  def _disable(self, reason: str) -> None:
    self.invalidate()
    if not self._warned:
      self._warned = True
      logging.warning(_PURE_STATE_FALLBACK_WARNING, reason)

  def refresh(self, train_state: Any, model: Any) -> None:
    self.state_graphdef, state_pure = nnx.split(train_state)
    self.model_graphdef, params_pure, rest_pure = nnx.split(model, nnx.Param, ...)
    reason = _check_reusable(state_pure, params_pure, rest_pure)
    if reason is not None:
      self._disable(reason)
      return
    self.params, self.rest, self.state = params_pure, rest_pure, state_pure

  def read_model(self, model: Any) -> tuple[Any, Any]:
    if self.params is not None:
      return self.params, self.rest
    self.model_graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    return params, rest

  def read_state(self, train_state: Any) -> Any:
    if self.state is not None:
      return self.state
    self.state_graphdef, state_pure = nnx.split(train_state)
    return state_pure

  def publish_model_rest(self, new_rest: Any) -> None:
    if self.params is None:
      return
    if jax.tree.structure(new_rest) != jax.tree.structure(self.rest):
      self._disable("fwd_bwd returned a wider non-parameter state than the model was split into")
      return
    self.rest = new_rest
    self.state = with_model_state(self.state, nnx.merge_state(self.params, new_rest))

  def publish_state(self, new_state_pure: Any) -> None:
    if self.params is None:
      return
    if not isinstance(new_state_pure, nnx.State) or MODEL_STATE_KEY not in new_state_pure:
      self._disable("the update kernel returned a state with no model entry")
      return
    params_pure, rest_pure = nnx.split_state(new_state_pure[MODEL_STATE_KEY], nnx.Param, ...)
    if jax.tree.structure(params_pure) != jax.tree.structure(self.params):
      self._disable("the update kernel's output does not partition into the same parameters")
      return
    self.params, self.rest, self.state = params_pure, rest_pure, new_state_pure
