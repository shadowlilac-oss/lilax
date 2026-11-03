"""Cloud integration controls."""

import os


def is_decoupled() -> bool:
  """Return whether Google Cloud integrations are disabled."""
  return os.environ.get("DECOUPLE_GCLOUD", "").upper() == "TRUE"
