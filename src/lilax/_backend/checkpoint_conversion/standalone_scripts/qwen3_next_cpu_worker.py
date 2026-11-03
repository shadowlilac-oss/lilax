import os
import pickle
import sys

from lilax._backend.checkpoint_conversion.standalone_scripts.convert_qwen3_next_scanned import _convert


def run(result_fd):
  try:
    _convert(*pickle.load(sys.stdin.buffer))
    error = None
  except Exception as exc:
    error = exc
  with os.fdopen(result_fd, "wb") as result_file:
    pickle.dump(error, result_file)
