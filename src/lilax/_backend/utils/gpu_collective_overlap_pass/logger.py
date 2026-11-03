
"""Logger shared by every module of the collective-overlap pass."""

import logging

_logger = logging.getLogger(__package__)
_logger.setLevel(logging.DEBUG)
if not _logger.handlers:
  _h = logging.StreamHandler()
  _h.setLevel(logging.DEBUG)
  _h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
  _logger.addHandler(_h)
  _logger.propagate = False
