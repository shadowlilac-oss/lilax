"""Lifetime of the JAX scalar listeners and backends owned by Tunix metrics loggers."""

import jax

from lilax._backend.observability import max_logging


def release_metrics_logger(logger, *, close_backends):
  """Unregisters a Tunix MetricsLogger's own scalar listeners and, after a failure, closes its backends.

  Tunix's MetricsLogger.close relies on jax.monitoring.clear_event_listeners, which leaves scalar listeners
  registered, so closed backends would otherwise receive scalars recorded by later trainers. Tunix closes backends
  only when training completes, so close_backends covers failures without repeating its process-wide listener clear.
  JAX reports an already removed listener through its presence assertion, or list.remove when assertions are
  disabled. Close failures are logged so the original training error propagates.
  """
  for backend in logger._backends:
    try:
      jax.monitoring.unregister_scalar_listener(backend.log_scalar)
    except (AssertionError, ValueError):
      pass
  if close_backends:
    for backend in logger._backends:
      try:
        backend.close()
      except Exception as error:
        max_logging.log(f"Warning: Failed to close Tunix metrics backend {backend!r}: {error}")
