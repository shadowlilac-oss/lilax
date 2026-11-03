import os
import pickle
import sys

from lilax._backend.checkpoint_conversion.standalone_scripts.convert_qwen3_moe import (
    MODEL_PARAMS_DICT, convert_hf_to_maxtext,
)
from lilax._backend.checkpoint_conversion.utils.utils import save_weights_to_checkpoint


def run(result_fd):
  try:
    base_model_path, maxtext_model_path, model_size, device_count, use_ocdbt, use_zarr3 = pickle.load(sys.stdin.buffer)
    weights = convert_hf_to_maxtext(base_model_path, MODEL_PARAMS_DICT[model_size])
    save_weights_to_checkpoint(maxtext_model_path, weights, device_count, use_ocdbt, use_zarr3)
    error = None
  except Exception as exc:
    error = exc
  with os.fdopen(result_fd, "wb") as result_file:
    pickle.dump(error, result_file)
