"""Logging utilities."""

import logging as std_logging
from absl import logging


def log(user_str):
  """Log a message at the INFO level."""
  logging.info(user_str, stacklevel=2)


def debug(user_str):
  """Log a message at the DEBUG level."""
  logging.debug(user_str, stacklevel=2)


def info(user_str, stacklevel=2):
  """Log a message at the INFO level."""
  logging.info(user_str, stacklevel=stacklevel)


def warning(user_str):
  """Log a message at the WARNING level."""
  logging.warning(user_str, stacklevel=2)


def error(user_str):
  """Log a message at the ERROR level."""
  logging.error(user_str, stacklevel=2)


class NoisyLogFilter(std_logging.Filter):
  """Filter known noisy log messages."""

  def filter(self, record):
    msg = record.getMessage()
    if "Type mismatch on" in msg:
      return False
    if "No mapping for flat state" in msg:
      return False
    return True
