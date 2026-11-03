
"""PGLE profiler interception capturing the FDO profile and the raw XSpace behind it."""

from __future__ import annotations

from lilax._backend.utils.gpu_collective_overlap_pass.logger import _logger
from lilax._backend.utils.gpu_collective_overlap_pass.profiles.costs import (
    _COLLECTIVE_COST_PREFIXES, _GEMM_COST_PREFIXES, _TE_EP_PREFIX, _collective_overrides, _gemm_overrides,
    _profile_costs, _te_ep_overrides,
)
from lilax._backend.utils.gpu_collective_overlap_pass.profiles.fdo import _update_profile
from lilax._backend.utils.gpu_collective_overlap_pass.profiles.xspace import _load_costs_from_xspace_bytes

# TODO tixxx remove the monkey patch once fdo profile can be passed to
# custom pass interface.
# ---------------------------------------------------------------------------
# PGLE profile interception
# ---------------------------------------------------------------------------
_patched = False


def _patch_pgle_profiler() -> None:
  """Hook PGLE to capture its profile and the raw XSpace behind it."""
  global _patched
  if _patched:
    return
  _patched = True
  import jax._src.profiler as _jax_profiler  # pylint: disable=import-outside-toplevel

  _original_consume = _jax_profiler.PGLEProfiler.consume_fdo_profile

  def _consume_and_capture(self):
    result = _original_consume(self)
    if result:
      _update_profile(result)
    return result

  _jax_profiler.PGLEProfiler.consume_fdo_profile = _consume_and_capture

  # PGLEProfiler.trace() calls _profiler.get_fdo_profile(xspace) on each
  # retry's raw XSpace and then discards it.
  _original_get_fdo_profile = _jax_profiler._profiler.get_fdo_profile  # pylint: disable=protected-access

  def _get_fdo_profile_and_capture(xspace):
    try:
      all_costs = _load_costs_from_xspace_bytes(
          bytes(xspace),
          (_TE_EP_PREFIX,) + _GEMM_COST_PREFIXES + _COLLECTIVE_COST_PREFIXES,
      )
      te_ep_costs = {k: v for k, v in all_costs.items() if k.startswith(_TE_EP_PREFIX)}
      gemm_costs = {k: v for k, v in all_costs.items() if k.startswith(_GEMM_COST_PREFIXES)}
      coll_costs = {k: v for k, v in all_costs.items() if k.startswith(_COLLECTIVE_COST_PREFIXES)}
      _gemm_overrides.update(gemm_costs)
      _collective_overrides.update(coll_costs)
      _te_ep_overrides.update(te_ep_costs)
      _profile_costs.update(te_ep_costs)
      _logger.info(
          "gpu_collective_overlap_pass: captured live per-invocation corrections: %d te_ep, %d GEMM, %d collective.",
          len(te_ep_costs),
          len(gemm_costs),
          len(coll_costs),
      )
    except Exception as exc:  # pylint: disable=broad-except
      _logger.warning(
          "gpu_collective_overlap_pass: failed to extract per-invocation costs from live PGLE data: %s",
          exc,
      )
    return _original_get_fdo_profile(xspace)

  _jax_profiler._profiler.get_fdo_profile = _get_fdo_profile_and_capture  # pylint: disable=protected-access
