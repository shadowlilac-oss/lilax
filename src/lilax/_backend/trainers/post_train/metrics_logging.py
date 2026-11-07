"""Tunix metrics logger options for post-training."""

import dataclasses
import functools

from tunix.sft import metrics_logger


def tensorboard_logging_options(log_dir, **options):
  """Returns Tunix logger options keeping its default TensorBoard backend without its implicit W&B backend.

  Tunix's default backends initialize W&B whenever it is importable. Lilax initializes W&B only on an explicit
  request, so omitting that backend here is intentional rather than lost parity.
  """
  logging_options = metrics_logger.MetricsLoggerOptions(log_dir=log_dir, **options)
  tensorboard = functools.partial(
      metrics_logger.TensorboardBackend, log_dir=log_dir, flush_every_n_steps=logging_options.flush_every_n_steps
  )
  return dataclasses.replace(logging_options, backend_kwargs={"custom_backend": [tensorboard]})
