import pickle
import sys
import traceback

import jax

jax.config.update("jax_platform_name", "cpu")

from lilax._backend.checkpoint_conversion.standalone_scripts import convert_gemma_chkpt
from lilax._backend.checkpoint_conversion.standalone_scripts import convert_gemma2_chkpt
from lilax._backend.checkpoint_conversion.standalone_scripts import convert_gemma3_chkpt


if __name__ == "__main__":
  model, base_model_path, maxtext_model_path, model_size, result_path = pickle.load(sys.stdin.buffer)
  converters = {
      "gemma": convert_gemma_chkpt._convert_in_worker,
      "gemma2": convert_gemma2_chkpt._convert_in_worker,
      "gemma3": convert_gemma3_chkpt._convert_in_worker,
  }
  error = None
  try:
    converters[model](base_model_path, maxtext_model_path, model_size)
  except BaseException as exc:
    error = exc
    if not isinstance(exc, SystemExit):
      traceback.print_exc()
  with open(result_path, "wb") as result_file:
    pickle.dump(error, result_file)
