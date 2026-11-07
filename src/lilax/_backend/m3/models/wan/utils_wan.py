"""Wan model sharding, hardware, and configuration utilities."""

from collections.abc import Sequence
import dataclasses
from enum import Enum
import json
import os
from typing import Any

import jax
from jax.sharding import NamedSharding, PartitionSpec as P
import requests
import yaml

from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as tokamax_splash
from lilax._backend.m3.core import sharding as m3_sharding
from lilax._backend.m3.core.sharding import TensorType, create_mesh
from lilax._backend.observability import max_logging

BlockSizes = tokamax_splash.SplashConfig
CONTEXT = "context"
SELF_ATTN_HEAD = "activation_self_attn_heads"
SELF_ATTN_Q_LENGTH = "activation_self_attn_q_length"
SELF_ATTN_KV_LENGTH = "activation_self_attn_kv_length"
CROSS_ATTN_HEAD = "activation_cross_attn_heads"
CROSS_ATTN_Q_LENGTH = "activation_cross_attn_q_length"
CROSS_ATTN_KV_LENGTH = "activation_cross_attn_kv_length"
WAN_MESH_AXES = ("data", "fsdp", "context", "tensor")
VAE_MESH_AXES = ("redundant", "vae_spatial")


@dataclasses.dataclass(frozen=True)
class WanSharding(m3_sharding.Sharding):
  rules: tuple[tuple[str, m3_sharding.MeshAxes], ...] = ()
  mesh_axes: tuple[str, ...] = WAN_MESH_AXES
  context_axis: str = CONTEXT

  @classmethod
  def from_rules(cls, rules: Sequence[Sequence[Any]], mesh_axes: Sequence[str] = WAN_MESH_AXES, **kwargs):
    normalized = []
    for logical_axis, mapped in rules:
      if mapped is not None and not isinstance(mapped, str):
        mapped = tuple(mapped)
      normalized.append((logical_axis, mapped))
    names = [name for name, _ in normalized]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
      raise ValueError(f"{cls.__name__}: logical axes listed more than once in the rule table: {duplicates}.")
    return cls(rules=tuple(normalized), mesh_axes=tuple(mesh_axes), **kwargs)

  def map_axis(self, axis: str, tensor_name: str, tensor_type: TensorType) -> m3_sharding.MeshAxes:
    for logical_axis, mapped in self.rules:
      if logical_axis == axis:
        return mapped
    raise KeyError(
        f"{type(self).__name__}: no rule for logical axis {axis!r} ({tensor_type.value}) on tensor {tensor_name!r}; "
        "add it to the yml rule table."
    )

  def __call__(self, tensor_name: str, axes: Sequence[str | None], tensor_type: TensorType = TensorType.WEIGHT) -> P:
    resolved = []
    claimed: dict[str, str] = {}
    for axis in axes:
      if axis is None:
        resolved.append(None)
        continue
      mapped = self.map_axis(axis, tensor_name, tensor_type)
      names = () if mapped is None else ((mapped,) if isinstance(mapped, str) else tuple(mapped))
      unknown = [name for name in names if name not in self.mesh_axes]
      if unknown:
        raise ValueError(
            f"Sharding rule for logical axis {axis!r} of tensor {tensor_name!r} returned unknown mesh axes "
            f"{unknown}; expected names from {self.mesh_axes}."
        )
      for mesh_axis in names:
        if mesh_axis in claimed:
          raise ValueError(
              f"Tensor {tensor_name!r} maps both {claimed[mesh_axis]!r} and {axis!r} onto mesh axis "
              f"{mesh_axis!r}; a mesh axis can shard only one dimension of a tensor."
          )
        claimed[mesh_axis] = axis
      resolved.append(mapped)
    return P(*resolved)


def param_metadata(sharding: m3_sharding.Sharding | None, tensor_name: str, axes: Sequence[str | None]) -> dict:
  return {} if sharding is None else {"out_sharding": sharding(tensor_name, axes)}


def constrain(sharding: m3_sharding.Sharding | None, x: Any, tensor_name: str, axes: Sequence[str | None]) -> Any:
  return x if sharding is None else sharding.constrain(x, tensor_name, axes)


WAN21_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "wan2.1-t2v-14b.yml")


def yml_rules(key: str, path: str = WAN21_CONFIG_PATH) -> tuple:
  with open(path, "rt", encoding="utf-8") as file:
    return tuple(yaml.safe_load(file)[key])


_ACTIVATIONS = {
    "swish": jax.nn.silu, "silu": jax.nn.silu, "relu": jax.nn.relu,
    "gelu": jax.nn.gelu, "gelu_tanh": jax.nn.gelu, "mish": jax.nn.mish,
}


def get_activation(name: str):
  func = _ACTIVATIONS.get(name)
  if func is None:
    raise ValueError(f"Unknown activation function: {name}")
  return func


class TpuType(Enum):
  TPU_V4 = "v4"
  TPU_V5E = "v5e"
  TPU_V5P = "v5p"
  TPU_V6_LITE = "v6e"
  TPU_7X = "v7x"
  UNKNOWN = "unknown"


def get_tpu_type() -> TpuType:
  try:
    device_kind = jax.devices()[0].device_kind
    if "7x" in device_kind:
      return TpuType.TPU_7X
    elif "v6 lite" in device_kind:
      return TpuType.TPU_V6_LITE
    else:
      return TpuType.UNKNOWN
  except Exception:
    return TpuType.UNKNOWN


def safe_getattr(obj, name, default=None):
  try:
    return getattr(obj, name)
  except (AttributeError, KeyError):
    return default


def get_precision(config):
  if not hasattr(config, "precision") or not config.precision:
    return None
  return jax.lax.Precision[config.precision]


def get_flash_block_sizes(config):
  if not hasattr(config, "flash_block_sizes") or not config.flash_block_sizes:
    return None
  if isinstance(config.flash_block_sizes, str):
    flash_block_sizes = json.loads(config.flash_block_sizes)
  else:
    flash_block_sizes = config.flash_block_sizes
  if flash_block_sizes:
    return BlockSizes(
        block_q=flash_block_sizes["block_q"], block_kv_compute=flash_block_sizes["block_kv_compute"],
        block_kv=flash_block_sizes["block_kv"], block_q_dkv=flash_block_sizes["block_q_dkv"],
        block_kv_dkv=flash_block_sizes["block_kv_dkv"],
        block_kv_dkv_compute=flash_block_sizes["block_kv_dkv_compute"], use_fused_bwd_kernel=True,
    )
  return None


def device_put_replicated(x, sharding):
  arr = getattr(x, "value", x)
  shd = getattr(sharding, "value", sharding)
  if hasattr(shd, "devices") and not hasattr(shd, "device_set"):
    shd = NamedSharding(shd, P())
  res = jax.make_array_from_callback(arr.shape, shd, lambda index: arr[index])
  if hasattr(x, "set_value"):
    x.set_value(res)
    return x
  return res


def create_wan_mesh(config) -> jax.sharding.Mesh:
  if getattr(config, "allow_split_physical_axes", False):
    raise NotImplementedError("allow_split_physical_axes=True is not supported by m3's create_mesh.")
  ici = {axis: getattr(config, f"ici_{axis}_parallelism", 1) for axis in config.mesh_axes}
  dcn = {axis: getattr(config, f"dcn_{axis}_parallelism", 1) for axis in config.mesh_axes}
  return create_mesh(axis_names=tuple(config.mesh_axes), ici=ici, dcn=dcn)


def delete_file(file_path: str):
  if os.path.exists(file_path):
    try:
      os.remove(file_path)
      max_logging.log(f"Successfully deleted file: {file_path}")
    except OSError as e:
      max_logging.log(f"Error deleting file '{file_path}': {e}")
  else:
    max_logging.log(f"The file '{file_path}' does not exist.")


def get_gcs_output_path(config):
  output_dir = getattr(config, "output_dir", "")
  if not output_dir or not output_dir.startswith("gs://"):
    return None
  run_name = getattr(config, "run_name", "")
  return os.path.join(output_dir, run_name) if run_name else output_dir


def load_prompts(prompt_file_path: str, default_prompt: str = "") -> list:
  if not prompt_file_path:
    if not default_prompt:
      raise ValueError("Neither prompt_file nor prompt was specified.")
    return [default_prompt]
  if prompt_file_path.startswith("http://") or prompt_file_path.startswith("https://"):
    response = requests.get(prompt_file_path, timeout=60)
    response.raise_for_status()
    raw_lines = response.text.splitlines()
  else:
    if not os.path.isfile(prompt_file_path):
      raise FileNotFoundError(f"Prompt file not found at local path: {prompt_file_path}")
    with open(prompt_file_path, "r", encoding="utf-8") as file:
      raw_lines = file.readlines()
  prompts = [line.strip() for line in raw_lines if line.strip() and not line.strip().startswith("#")]
  if not prompts:
    if default_prompt:
      return [default_prompt]
    raise ValueError(f"Prompt file '{prompt_file_path}' contains no valid non-empty prompts.")
  return prompts


def chunk_and_pad(items: list, batch_size: int):
  if batch_size <= 0:
    raise ValueError(f"batch_size must be positive, got {batch_size}")
  for i in range(0, len(items), batch_size):
    chunk = items[i : i + batch_size]
    actual_len = len(chunk)
    padded_chunk = chunk + [chunk[-1]] * (batch_size - actual_len) if actual_len < batch_size else chunk
    yield i, padded_chunk, actual_len
