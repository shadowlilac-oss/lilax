
"""Pooled per-instruction PGLE costs and the live per-invocation cost corrections."""

from __future__ import annotations

# Populated once PGLE delivers its FDO profile bytes.
_profile_costs: dict[str, float] = {}

_TE_EP_PREFIX = "te_ep"
# te_ep corrections replace the PGLE values in _profile_costs. The GEMM and
# collective corrections stay separate and are used only by the stream model
# (the rest of the pass keeps PGLE's numbers).
_te_ep_overrides: dict[str, float] = {}
_gemm_overrides: dict[str, float] = {}
_GEMM_COST_PREFIXES = ("te_gemm", "te_grouped_gemm")
_collective_overrides: dict[str, float] = {}
_COLLECTIVE_COST_PREFIXES = ("all-gather.", "reduce-scatter.", "all-reduce.")
