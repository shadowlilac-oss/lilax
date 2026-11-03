"""Compare converted safetensors checkpoints with Hugging Face weights."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import os
import time

import gcsfs
import numpy as np
from safetensors import safe_open
from safetensors.torch import load as load_safetensors
from tqdm import tqdm

from lilax._backend.checkpoint_conversion.checkpoint_cpu_bootstrap import run_on_cpu
from lilax._backend.checkpoint_conversion.utils.utils import load_hf_dict_from_transformers, print_ram_usage
from lilax._backend.utils import max_logging
from lilax._backend.utils.globals import HF_IDS


def _load_gcs_shard(gcs_path: str, fs: gcsfs.GCSFileSystem) -> dict[str, np.ndarray]:
  max_logging.log(f"Processing GCS shard: {gcs_path}")
  with fs.open(gcs_path, "rb") as f:
    loaded_tensors = load_safetensors(f.read())
  return {key: tensor.numpy() for key, tensor in loaded_tensors.items()}


def _load_local_shard(file_path: str) -> dict[str, np.ndarray]:
  max_logging.log(f"Processing local shard: {file_path}")
  shard_dict = {}
  with safe_open(file_path, framework="pt", device="cpu") as f:
    for key in f.keys():
      shard_dict[key] = f.get_tensor(key).numpy()
  return shard_dict


def load_safetensors_generic(path: str, max_workers: int) -> dict[str, np.ndarray]:
  final_tensor_dict = {}
  is_gcs = path.startswith("gs://")
  print_ram_usage(f"Start {'GCS' if is_gcs else 'Local'} Load")
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    if is_gcs:
      fs = gcsfs.GCSFileSystem()
      files = [f"gs://{f}" for f in fs.glob(f"{path.rstrip('/')}/*.safetensors")]
      max_logging.log(f"Found {len(files)} files in GCS. Loading...")
      futures = [executor.submit(_load_gcs_shard, f, fs) for f in files]
    else:
      files = glob.glob(os.path.join(path, "*.safetensors"))
      max_logging.log(f"Found {len(files)} files locally. Loading...")
      futures = [executor.submit(_load_local_shard, f) for f in files]
    for future in as_completed(futures):
      final_tensor_dict.update(future.result())
  print_ram_usage("End Load")
  return final_tensor_dict


def get_hf_model_state_dict(model_id: str, token: str) -> dict[str, np.ndarray]:
  max_logging.log(f"Loading reference model from HuggingFace: {model_id}...")
  return {key: value.numpy() for key, value in load_hf_dict_from_transformers(model_id, token).items()}


def verify_dictionaries(ref_dict: dict[str, np.ndarray], cand_dict: dict[str, np.ndarray], rtol: float, atol: float) -> bool:
  max_logging.log(f"Verifying with rtol={rtol}, atol={atol}")
  ref_keys, cand_keys = set(ref_dict), set(cand_dict)
  if ref_keys != cand_keys:
    max_logging.log("❌ KEYS DO NOT MATCH")
    max_logging.log(f"Missing in Candidate: {ref_keys - cand_keys}")
    max_logging.log(f"Extra in Candidate: {cand_keys - ref_keys}")
    return False
  max_logging.log("✅ Keys match. Verifying tensor values...")
  for key in tqdm(ref_keys, desc="Verifying tensors"):
    arr_ref, arr_cand = ref_dict[key], cand_dict[key]
    if arr_ref.shape != arr_cand.shape:
      max_logging.log(f"❌ SHAPE MISMATCH found for '{key}'")
      max_logging.log(f"Reference: {arr_ref.shape} vs Candidate: {arr_cand.shape}")
      return False
    max_logging.log(f"✅ Key: {key} shape match.")
    if not np.allclose(arr_ref, arr_cand, rtol=rtol, atol=atol):
      max_logging.log(f"❌ VALUE MISMATCH found for '{key}'")
      max_logging.log(f"Max difference: {np.max(np.abs(arr_ref - arr_cand))} (exceeds rtol={rtol}, atol={atol})")
      return False
    max_logging.log(f"✅ Key: {key} value match.")
  max_logging.log("✅ All values match!")
  return True


def compare_hf_ckpt(candidate_path, reference_path="", config=None, max_workers=12, rtol=1e-2, atol=1e-2):
  return run_on_cpu(
      "compare_hf_ckpt", config if not reference_path else None, candidate_path=os.fspath(candidate_path),
      reference_path=os.fspath(reference_path), max_workers=max_workers, rtol=rtol, atol=atol,
  )


def _compare_hf_ckpt(candidate_path, reference_path="", config=None, max_workers=12, rtol=1e-2, atol=1e-2):
  start = time.perf_counter()
  if reference_path:
    reference = load_safetensors_generic(reference_path, max_workers)
  else:
    if config is None:
      raise ValueError("config is required when reference_path is not provided")
    if config.model_name not in HF_IDS:
      raise ValueError(f"Unsupported model name: {config.model_name}. Supported: {list(HF_IDS.keys())}")
    reference = get_hf_model_state_dict(HF_IDS[config.model_name], config.hf_access_token)
  max_logging.log(f"⏱️ HuggingFace model loaded in {(time.perf_counter() - start) / 60:.2f} minutes")
  start = time.perf_counter()
  candidate = load_safetensors_generic(candidate_path, max_workers)
  max_logging.log(f"⏱️ Safetensors checkpoint loaded in {(time.perf_counter() - start) / 60:.2f} minutes")
  return verify_dictionaries(reference, candidate, rtol=rtol, atol=atol)
