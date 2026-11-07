"""Process-0 tensorboardX SummaryWriter lifecycle and text summaries."""

import os

import jax
from tensorboardX import writer

from lilax._backend.common.cloud import is_decoupled
from lilax._backend.observability import max_logging


def initialize_summary_writer(tensorboard_dir, run_name, enable_tensorboard=True):
  """Return a tensorboardX SummaryWriter, or None when disabled."""
  if jax.process_index() != 0:
    return None
  if not enable_tensorboard:
    max_logging.log("TensorBoard disabled.")
    return None
  if is_decoupled():
    return None
  if not tensorboard_dir or not run_name:
    max_logging.log("tensorboard_dir or run_name missing; using default SummaryWriter directory.")
    return writer.SummaryWriter()
  summary_writer_path = os.path.join(tensorboard_dir, run_name)
  return writer.SummaryWriter(summary_writer_path)


def close_summary_writer(summary_writer):
  if jax.process_index() == 0 and summary_writer is not None:
    summary_writer.close()


def add_text_to_summary_writer(key, value, summary_writer):
  """Writes given key-value pair to tensorboard as text/summary."""
  if jax.process_index() == 0 and summary_writer is not None:
    summary_writer.add_text(key, value)
