"""Write evaluation results as JSON."""

import datetime
import json
import logging
import os
import tempfile

from lilax._backend.storage.gcs import upload_blob

logger = logging.getLogger(__name__)


def write_results(
    benchmark: str, model_name: str, scores: dict, generation_stats: dict, config: dict,
    results_path: str = "./eval_results",
) -> dict:
  timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  safe_model = model_name.replace("/", "_").replace(":", "_")
  filename = f"{benchmark}_{safe_model}_{timestamp}.json"
  results = {
      "benchmark": benchmark,
      "model_name": model_name,
      "timestamp_utc": timestamp,
      "scores": scores,
      "generation_stats": generation_stats,
      "config": config,
  }
  if results_path.startswith("gs://"):
    local_path = os.path.join(tempfile.mkdtemp(prefix="eval_results_"), filename)
    with open(local_path, "w", encoding="utf-8") as result_file:
      json.dump(results, result_file, indent=2)
    gcs_dest = f"{results_path.rstrip('/')}/{filename}"
    upload_blob(gcs_dest, local_path)
    logger.info("Results written to %s", gcs_dest)
  else:
    os.makedirs(results_path, exist_ok=True)
    local_path = os.path.join(results_path, filename)
    with open(local_path, "w", encoding="utf-8") as result_file:
      json.dump(results, result_file, indent=2)
    logger.info("Results written to %s", local_path)
  return {"results": results, "local_path": os.path.abspath(local_path)}
