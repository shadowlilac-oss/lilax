
"""Registration of the collective-overlap POST_SCHEDULER pass and its PGLE hook with JAX."""

from __future__ import annotations

import os

from lilax._backend.utils.gpu_collective_overlap_pass.logger import _logger
from lilax._backend.utils.gpu_collective_overlap_pass.pgle_hook import _patch_pgle_profiler
from lilax._backend.utils.gpu_collective_overlap_pass.profiles.protos import _XLA_SRC
from lilax._backend.utils.gpu_collective_overlap_pass.xla_callback import _gpu_collective_overlap_pass

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_registered = False


def register() -> None:
  """Register the collective-overlap POST_SCHEDULER pass and PGLE hook.

  Idempotent: elastic retries re-run initialize(), and each extra registration
  would queue another copy of the pass.
  """
  global _registered
  if _registered:
    return
  if os.environ.get("COLLECTIVE_OVERLAP_DISABLE", "1") == "1":
    _logger.info("gpu_collective_overlap_pass: disabled (set COLLECTIVE_OVERLAP_DISABLE=0 to enable)")
    return
  if not _XLA_SRC:
    _logger.warning(
        "gpu_collective_overlap_pass: not registered; the XLA source path is needed to compile "
        "the profile protos. Set COLLECTIVE_OVERLAP_XLA_PATH to an XLA checkout."
    )
    return
  import jax.extend.xla as jex_xla  # pylint: disable=import-outside-toplevel

  _patch_pgle_profiler()
  jex_xla.register_hlo_module_transformation(
      _gpu_collective_overlap_pass,
      name="gpu_collective_overlap_pass",
      stage=jex_xla.PipelineStage.POST_SCHEDULER,
  )
  _registered = True
  _logger.info("gpu_collective_overlap_pass: registered POST_SCHEDULER pass 'gpu_collective_overlap_pass'.")
