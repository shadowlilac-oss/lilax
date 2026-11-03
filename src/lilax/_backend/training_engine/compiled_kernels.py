"""The engine's compiled kernels and the batch signatures they were compiled against."""

from __future__ import annotations

from typing import Any


class CompiledKernels:
  """Compiled train and eval kernels; `requested` means compile() was called, `compiled` that train kernels are live.

  `fwd_bwd`, `fwd_bwd_accum` and `update` do not exist until a compile or an invalidation assigns them.
  """

  fwd_bwd: Any
  fwd_bwd_accum: Any
  update: Any

  def __init__(self) -> None:
    self.compiled = False
    self.requested = False
    self.signature: Any = None
    self.jitted: dict[str, Any] = {}
    self.eval: Any = None
    self.eval_signature: Any = None

  def invalidate_train(self) -> None:
    self.compiled = False
    self.fwd_bwd = None
    self.fwd_bwd_accum = None
    self.update = None

  def invalidate_eval(self) -> None:
    self.eval = None
    self.eval_signature = None
