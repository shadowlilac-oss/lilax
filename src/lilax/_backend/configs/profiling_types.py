"""Native profiling configuration types."""

import enum
from enum import Enum


class XProfTPUPowerTraceMode(enum.IntEnum):  # pylint: disable=invalid-name
  """Enum for XProfTPUPowerTraceMode."""

  __module__ = "lilax._backend.configs.types"

  POWER_TRACE_NONE = 0
  POWER_TRACE_NORMAL = 1
  POWER_TRACE_SPI = 2


class ProfilerType(str, Enum):
  """Supported performance profilers."""

  __module__ = "lilax._backend.configs.types"

  NONE = ""
  XPLANE = "xplane"
  NSYS = "nsys"
