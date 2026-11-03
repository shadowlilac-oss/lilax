"""Upload evaluation results to Google Cloud Storage."""

import logging
import os

from lilax._backend.utils.gcs_utils import upload_blob

logger = logging.getLogger(__name__)


def upload_results(local_path: str, gcs_path: str) -> None:
  if gcs_path.endswith("/"):
    gcs_dest = gcs_path + os.path.basename(local_path)
  else:
    gcs_dest = gcs_path
  upload_blob(gcs_dest, local_path)
  logger.info("Uploaded %s to %s", local_path, gcs_dest)
