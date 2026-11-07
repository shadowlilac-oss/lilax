"""Model-specific helpers shared by MaxText's vLLM adapter."""

import inspect
import math
from typing import Any

import jax.numpy as jnp

from lilax._backend.observability import max_logging


def call_with_supported_kwargs(func: Any, **kwargs) -> Any:
  try:
    parameters = inspect.signature(func).parameters
  except (TypeError, ValueError):
    return func(**kwargs)
  if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
    return func(**kwargs)
  supported = {name: value for name, value in kwargs.items() if name in parameters}
  dropped = sorted(set(kwargs) - set(supported))
  if dropped:
    max_logging.log(f"{getattr(func, '__name__', repr(func))} does not accept {dropped}; omitting them.")
  return func(**supported)


def normalize_vllm_input_positions(input_positions: Any):
  if input_positions is None:
    return None
  if input_positions.ndim == 1:
    return input_positions.reshape((-1, 1))
  if input_positions.ndim == 2 and input_positions.shape[0] == 3:
    return input_positions.T[:, None, :]
  if input_positions.ndim == 3 and input_positions.shape[0] == 3:
    return jnp.transpose(input_positions, (1, 2, 0))
  raise ValueError(
      "vLLM input positions must have shape (num_tokens,), (3, num_tokens), or (3, batch, seq); "
      f"got {input_positions.shape}."
  )


def build_qwen_gdn_cache_layout(cfg: Any, torch_module: Any):
  key_dim = cfg.gdn_key_head_dim * cfg.gdn_num_key_heads
  value_dim = cfg.gdn_value_head_dim * cfg.gdn_num_value_heads
  conv_dim = key_dim * 2 + value_dim
  shapes = (
      (cfg.gdn_conv_kernel_dim - 1, conv_dim),
      (cfg.gdn_num_value_heads, cfg.gdn_key_head_dim, cfg.gdn_value_head_dim),
  )
  dtypes = (torch_module.bfloat16, torch_module.float32)
  page_size_bytes = sum(
      math.prod(shape) * torch_module.empty((), dtype=dtype).element_size()
      for shape, dtype in zip(shapes, dtypes, strict=True)
  )
  return shapes, dtypes, page_size_bytes


_LAYER_NAME_PREFIX = "layer."


def resolve_layer_kv_cache_indices(layer_name_to_kvcache_index: Any, num_kv_caches: int) -> list[int] | None:
  if not layer_name_to_kvcache_index:
    return None
  if not isinstance(layer_name_to_kvcache_index, dict):
    layer_name_to_kvcache_index = dict(layer_name_to_kvcache_index)
  by_layer: dict[int, int] = {}
  for name, physical_index in layer_name_to_kvcache_index.items():
    if not isinstance(name, str) or not name.startswith(_LAYER_NAME_PREFIX):
      continue
    suffix = name[len(_LAYER_NAME_PREFIX) :]
    if not suffix.isdigit():
      continue
    by_layer[int(suffix)] = int(physical_index)
  if not by_layer:
    return None
  num_layers = max(by_layer) + 1
  missing = sorted(set(range(num_layers)) - set(by_layer))
  if missing:
    raise ValueError(
        f"layer_name_to_kvcache_index is missing decoder layers {missing} (expected layer.0..layer.{num_layers - 1})."
    )
  physical_indices = [by_layer[lyr] for lyr in range(num_layers)]
  out_of_range = [idx for idx in physical_indices if not 0 <= idx < num_kv_caches]
  if out_of_range:
    raise ValueError(
        f"layer_name_to_kvcache_index points outside the kv_caches list (len={num_kv_caches}): {sorted(set(out_of_range))}."
    )
  return physical_indices


def gather_layer_kv_caches(kv_caches: list[Any], physical_indices: list[int] | None) -> list[Any]:
  if physical_indices is None:
    return kv_caches
  return [kv_caches[idx] for idx in physical_indices]


def scatter_layer_kv_caches(
    kv_caches: list[Any], layer_kv_caches: list[Any], physical_indices: list[int] | None
) -> list[Any]:
  if physical_indices is None:
    return layer_kv_caches
  if len(layer_kv_caches) != len(physical_indices):
    raise ValueError(
        f"Expected layer_kv_caches to have length {len(physical_indices)}, " f"but got {len(layer_kv_caches)}."
    )
  updated = list(kv_caches)
  for lyr, idx in enumerate(physical_indices):
    updated[idx] = layer_kv_caches[lyr]
  return updated
