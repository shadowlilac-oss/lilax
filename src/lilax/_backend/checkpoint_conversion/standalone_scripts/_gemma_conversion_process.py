import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path


def run_conversion(model: str, base_model_path: str, maxtext_model_path: str, model_size: str) -> None:
  worker = Path(__file__).with_name("_gemma_conversion_worker.py")
  env = os.environ.copy()
  env["PYTHONPATH"] = os.pathsep.join(os.path.abspath(path) for path in sys.path if isinstance(path, str))
  with tempfile.TemporaryDirectory() as directory:
    result_path = os.path.join(directory, "result")
    request = (model, base_model_path, maxtext_model_path, model_size, result_path)
    subprocess.run([sys.executable, str(worker)], input=pickle.dumps(request), env=env, check=True)
    with open(result_path, "rb") as result_file:
      error = pickle.load(result_file)
    if error is not None:
      raise error
