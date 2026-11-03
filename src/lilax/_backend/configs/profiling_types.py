"""Native profiling configuration types."""

import enum
from enum import Enum


class XProfTPUPowerTraceMode(enum.IntEnum):  # pylint: disable=invalid-name
  """Enum for XProfTPUPowerTraceMode."""

  POWER_TRACE_NONE = 0
  POWER_TRACE_NORMAL = 1
  POWER_TRACE_SPI = 2


class ProfilerType(str, Enum):
  """Supported performance profilers."""

  NONE = ""
  XPLANE = "xplane"
  NSYS = "nsys"
