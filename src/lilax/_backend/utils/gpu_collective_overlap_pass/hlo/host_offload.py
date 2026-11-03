
"""Recognition and registry of host-offload copies (async D2H/H2D DMAs)."""

# This pass runs inside the XLA compile callback and must fail open: any error
# means "keep XLA's schedule", so broad excepts are intentional.
# pylint: disable=broad-exception-caught

from __future__ import annotations

import re
from typing import Optional

from lilax._backend.utils.gpu_collective_overlap_pass.hlo.text import _computation_root, _opcode_str
from lilax._backend.utils.gpu_collective_overlap_pass.logger import _logger

# Host-offload copies: XLA lowers a remat offload to a pair of async DMAs,
# D2H as an async dynamic-update-slice into a host-memory-space buffer (S(5)) in
# the forward pass and H2D as an async dynamic-slice out of that buffer in the
# backward pass. They are copies, not collectives, we use special handling for those.
_HOST_MEMORY_SPACE_TAG = ":S(5)"
# Names (instruction names are unique per module) of host-offload copy starts
# and their dones in the module being processed; see _register_host_offload_copies.
_host_offload_copy_names: set[str] = set()


# Unlike _CALLS_RE, allows hyphens: the wrapped computations are named
# e.g. wrapped_dynamic-update-slice_computation.
_HOST_COPY_CALLS_RE = re.compile(r"calls=%([\w.\-]+)")


def _host_offload_copy_kind(inst, comp_by_name: dict) -> Optional[str]:
  """ "d2h" for an async dynamic-update-slice writing a host-memory-space
  buffer, "h2d" for an async dynamic-slice reading one, else None.

  The scheduled module prints these as "fusion-start" with
  calls=%wrapped_dynamic-(update-)slice_computation, so follow the start's
  calls= to the computation whose root is the DUS/DS (a few hops, in case
  of nested wrappers). Parsed from text: async_wrapped_root() is not reliable
  for these.
  """
  if _opcode_str(inst) != "async-start":
    return None
  try:
    text = inst.to_string()
  except Exception:
    return None
  if _HOST_MEMORY_SPACE_TAG not in text:
    return None
  for _ in range(4):
    m = _HOST_COPY_CALLS_RE.search(text)
    comp = comp_by_name.get(m.group(1)) if m else None
    root = _computation_root(comp) if comp is not None else None
    if root is None:
      return None
    kind = {"dynamic-update-slice": "d2h", "dynamic-slice": "h2d"}.get(_opcode_str(root))
    if kind is not None:
      return kind
    try:
      text = root.to_string()
    except Exception:
      return None
  return None


def _register_host_offload_copies(module) -> None:
  """Rebuild _host_offload_copy_names (starts and their dones) from `module`."""
  names: set[str] = set()
  comp_by_name = {c.name: c for c in module.computations()}
  for comp in module.make_nonfusion_computations():
    for inst in comp.instructions():
      if _host_offload_copy_kind(inst, comp_by_name) is not None:
        names.add(inst.name)
        names.update(user.name for user in inst.users())
  _host_offload_copy_names.clear()
  _host_offload_copy_names.update(names)
  if names:
    _logger.info(
        "gpu_collective_overlap_pass [%s]: registered %d host-offload copy instruction(s) (starts and dones): %s",
        module.name,
        len(names),
        ", ".join(sorted(names)),
    )
