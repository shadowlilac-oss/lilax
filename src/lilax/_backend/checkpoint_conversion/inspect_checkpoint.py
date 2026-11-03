"""Inspect Hugging Face, MaxText, and Orbax checkpoint structures."""

import pathlib
import re
import sys

from etils import epath
from flax import nnx
import jax
import orbax.checkpoint as ocp
from safetensors import safe_open
import torch

from lilax._backend.checkpoint_conversion.checkpoint_cpu_bootstrap import run_on_cpu
from lilax._backend.checkpoint_conversion.utils.utils import param_key_parts_from_path
from lilax._backend.utils import device_mesh
from lilax._backend.utils.diagnostics import pytree
from lilax._backend.utils.model_creation.nnx_abstract_model import create_nnx_abstract_model


def natural_sort_key(s: str):
  return [int(text) if text.isdigit() else text for text in re.split(r"(\d+)", str(s))]


def print_structure(data_dict, output_file=""):
  lines = [f"# {' '.join(sys.orig_argv)}", ""] if output_file else []
  for key in sorted(data_dict, key=natural_sort_key):
    line = f"key: {key} | {data_dict[key]}"
    print(line)
    lines.append(line)
  if output_file:
    pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_file).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nStructure saved to {output_file}")


def inspect_hf(path, format="safetensors", check_dtype=False, output_file=""):
  ckpt_paths = sorted(pathlib.Path(path).glob(f"[!.]*.{format}"))
  if not ckpt_paths:
    raise ValueError(f"No files with extension .{format} found in {path}")
  if format not in ("safetensors", "pth"):
    raise ValueError(f"Unsupported format: {format}")
  params = {}
  for i, ckpt_path in enumerate(ckpt_paths):
    print(f"Loading {ckpt_path.name} ({i+1}/{len(ckpt_paths)})...")
    if format == "safetensors":
      with safe_open(ckpt_path, framework="pt") as f:
        for key in f.keys():
          value = f.get_slice(key)
          params[key] = f"shape: {value.get_shape()}"
          if check_dtype:
            params[key] += f" | dtype: {value.get_dtype()}"
    else:
      checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
      if isinstance(checkpoint, dict):
        for key, value in checkpoint.items():
          if hasattr(value, "shape"):
            params[key] = f"shape: {value.shape}"
          if check_dtype and hasattr(value, "dtype"):
            params[key] += f" | dtype: {value.dtype}"
  print_structure(params, output_file)
  return params


def inspect_maxtext(config, check_dtype=False, output_file=""):
  return run_on_cpu("inspect_maxtext", config, check_dtype=check_dtype, output_file=str(output_file))


def _inspect_maxtext(config, check_dtype=False, output_file=""):
  mesh = jax.sharding.Mesh(device_mesh.create_device_mesh(config), config.mesh_axes)
  _, abstract_model = create_nnx_abstract_model(config, mesh=mesh)
  _, abstract_param, _ = nnx.split(abstract_model, nnx.Param, ...)
  num_params = pytree.calculate_num_params_from_pytree(abstract_param)
  print(f"\nTotal Parameters: {num_params} (~{num_params/1e9:.2f} B)")
  flattened, _ = jax.tree_util.tree_flatten_with_path(abstract_param)
  params = {}
  for path, value in flattened:
    key = ".".join(param_key_parts_from_path(path))
    key = ("params." if key.startswith(("params", "Tid2EidVar")) else "params.params.") + key
    params[key] = f"shape: {value.shape}"
    if check_dtype:
      params[key] += f" | dtype: {value.dtype}"
  print_structure(params, output_file)
  return params


def inspect_orbax(path, check_dtype=False, output_file=""):
  metadata = ocp.StandardCheckpointer().metadata(epath.Path(path))
  tree = metadata.item_metadata.tree
  has_params = isinstance(tree, dict) and "params" in tree
  params = {}
  for path, value in ocp.tree.to_flat_dict(tree).items():
    key = ".".join(param_key_parts_from_path(path))
    if has_params and (not path or path[0] != "params"):
      continue
    params[key] = f"shape: {value.shape}"
    if check_dtype:
      params[key] += f" | dtype: {value.dtype}"
  print_structure(params, output_file)
  return params
