"""Restore portable projection layouts into fused native model parameters."""

from collections.abc import Mapping

import jax

from lilax._backend.checkpoint_conversion.model_families.deepseek import fused as fused_deepseek
from lilax._backend.checkpoint_conversion.model_families.gemma4 import fused as fused_gemma
from lilax._backend.checkpoint_conversion.model_families.gpt3 import fused as fused_gpt3
from lilax._backend.checkpoint_conversion.model_families.llama4 import fused as fused_llama4
from lilax._backend.checkpoint_conversion.model_families.qwen3_vl import fused as fused_qwen3_vl
from lilax._backend.checkpoint_conversion.model_families.qwen3_omni import fused as fused_qwen3_omni
from lilax._backend.checkpoint_conversion.model_families.qwen3_5 import fused as fused_qwen3_5
from lilax._backend.checkpoint_conversion.model_families.qwen3_next import fused as fused_qwen3_next
from lilax._backend.checkpoint_conversion.projection_layout import projection_containers, supports_mlp_layout
from lilax._backend.common.common_types import is_fp8_dtype
from lilax._backend.configs import pyconfig
from lilax._backend.integration.vllm.convert_utils import _partition_size
from lilax._backend.models.creation.checkpoint_loading.metadata import _is_orbax_array_metadata


def _tower_projection(adapter, path, config):
  if not path:
    return None
  vision = "vision_encoder" in path
  audio = "audio_encoder" in path
  qwen = adapter in (fused_qwen3_vl, fused_qwen3_omni, fused_qwen3_5)
  attention = (
      qwen and ((vision and path[-2:] == ("attn", "attn"))
                or (audio and adapter is fused_qwen3_omni and path[-1] == "self_attention_audio"))
      or adapter is fused_gemma and vision and (
          "Gemma3VisionEncoderLayer_0" in path and path[-1] == "MultiHeadDotProductAttention_0"
          or "Gemma4VisionEncoderLayer_0" in path and path[-1] == "attention")
      or adapter is fused_llama4 and vision and "Llama4VisionEncoder_0" in path and path[-1] == "self_attention_vision"
  )
  if attention and config.fused_qkv:
    return "qkv_proj", ("query", "key", "value")
  if config.fused_mlp and adapter is fused_gemma and vision and "Gemma4VisionEncoderLayer_0" in path and path[-1] == "mlp":
    return "wi", ("wi_0", "wi_1")
  if config.fused_mlp and adapter is fused_qwen3_omni and audio and path[-1] == "AudioMLP":
    return "wi", ("wi",)
  return None


def adjust_projection_target(target, metadata, config, mesh, adapter, use_value_wrapper, path=()):
  """Request unfused checkpoint leaves where a supported target uses fused projections."""
  if not hasattr(target, "items") or not isinstance(metadata, dict):
    return target
  target = {key: adjust_projection_target(value, metadata.get(key), config, mesh, adapter, use_value_wrapper, path + (key,))
            for key, value in target.items()}
  tower = _tower_projection(adapter, path, config)
  if tower is not None:
    fused, sources = tower
    if fused not in target:
      return target
    if sources != (fused,) and fused in metadata:
      return target
    if not all(name in metadata for name in sources):
      if any(name in metadata for name in sources):
        raise ValueError("Tower projection checkpoint is missing source projections.")
      return target
    projection = target[fused]
    if not isinstance(projection, Mapping):
      return target
    source_targets = {}
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    for name in sources:
      stored = metadata[name]
      if not isinstance(stored, dict) or set(stored) != set(projection):
        raise ValueError("Tower projection checkpoint leaves do not match the target.")
      source_targets[name] = {}
      for kind, value in projection.items():
        leaf = stored[kind]
        wrapped = isinstance(leaf, dict) and leaf.keys() == {"value"}
        if wrapped:
          leaf = leaf["value"]
        if use_value_wrapper:
          value = value["value"]
        if not _is_orbax_array_metadata(leaf):
          raise ValueError("Tower projection checkpoint is missing array metadata.")
        array = jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=replicated)
        source_targets[name][kind] = {"value": array} if use_value_wrapper else array
    if sources == (fused,):
      stored_kernel = source_targets[fused]["kernel"]
      target_kernel = projection["kernel"]
      if use_value_wrapper:
        stored_kernel, target_kernel = stored_kernel["value"], target_kernel["value"]
      if stored_kernel.shape == target_kernel.shape:
        return target
    del target[fused]
    target.update(source_targets)
    return target
  if (path[-1:] == ("GptOssMlp",) and config.prefuse_moe_weights and "wi" in target
      and "wi" not in metadata and ("wi_0" in metadata or "wi_1" in metadata)):
    weight = target["wi"]
    stored = [metadata.get(name) for name in ("wi_0", "wi_1")]
    if use_value_wrapper:
      weight = weight.get("value") if hasattr(weight, "get") else None
    stored = [item.get("value") if isinstance(item, dict) and item.keys() == {"value"} else item for item in stored]
    if weight is None or not all(_is_orbax_array_metadata(item) for item in stored):
      raise ValueError("GPT-OSS prefused restore requires both separate expert weight metadata entries.")
    sharding = getattr(weight, "sharding", None)
    if not isinstance(sharding, jax.sharding.NamedSharding):
      raise ValueError("Cannot determine GPT-OSS prefused target shard count from wi sharding.")
    axis = len(weight.shape) - 1
    partition = sharding.spec[axis] if axis < len(sharding.spec) else None
    shards = _partition_size(partition, sharding.mesh)
    if (stored[0].shape != stored[1].shape or stored[0].shape[:-1] != weight.shape[:-1]
        or weight.shape[-1] % 2 or stored[0].shape[-1] > weight.shape[-1] // 2
        or stored[0].shape[-1] % shards or (weight.shape[-1] // 2) % shards):
      raise ValueError("GPT-OSS prefused restore requires compatible widths and even target shard chunks.")
  if adapter is None or (adapter in (fused_gemma, fused_qwen3_vl, fused_qwen3_omni, fused_qwen3_next, fused_qwen3_5)
                         and "decoder" not in path):
    return target
  hybrid_qwen = adapter in (fused_qwen3_next, fused_qwen3_5)
  deepseek = adapter is fused_deepseek
  replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  for containers, fused, sources, enabled in (
      (("attention",) if hybrid_qwen else projection_containers(adapter, mlp=False), "qkv_proj",
       ("query", "key", "value"), config.fused_qkv and not deepseek and (not hybrid_qwen or path[-1:] == ("attention",))),
      (("mlp", "shared_expert") if hybrid_qwen else ("mlp", "shared_experts") if deepseek
       else projection_containers(adapter, mlp=True), "wi", ("wi_0", "wi_1"),
       config.fused_mlp and supports_mlp_layout(adapter)),
  ):
    for container in containers:
      node = target.get(container)
      stored = metadata.get(container)
      if not enabled or not isinstance(node, dict) or not isinstance(stored, dict):
        continue
      single_mlp = adapter is fused_gpt3 and fused == "wi" and len(config.mlp_activations) == 1
      if single_mlp:
        sources = ("wi",)
      if fused not in node or (fused in stored and not single_mlp) or not all(name in stored for name in sources):
        continue
      projection = node[fused]
      if not isinstance(projection, Mapping):
        continue
      if single_mlp:
        stored_kernel = stored["wi"].get("kernel") if isinstance(stored["wi"], dict) else None
        if isinstance(stored_kernel, dict) and stored_kernel.keys() == {"value"}:
          stored_kernel = stored_kernel.get("value")
        target_kernel = projection.get("kernel")
        if use_value_wrapper and hasattr(target_kernel, "get"):
          target_kernel = target_kernel.get("value")
        if not _is_orbax_array_metadata(stored_kernel) or target_kernel is None or stored_kernel.shape == target_kernel.shape:
          continue
      if hybrid_qwen and fused == "qkv_proj" and is_fp8_dtype(config.weight_dtype):
        if set(projection) != {"kernel", "kernel_scale"} or any(
            not isinstance(stored[name], dict) or set(stored[name]) != {"kernel", "kernel_scale"} for name in sources
        ):
          raise NotImplementedError("Hybrid Qwen fused FP8 QKV requires all three block scale grids.")
      if deepseek and fused == "wi" and is_fp8_dtype(config.weight_dtype):
        if set(projection) != {"kernel", "kernel_scale"} or any(
            not isinstance(stored[name], dict) or set(stored[name]) != {"kernel", "kernel_scale"} for name in sources
        ):
          raise NotImplementedError("DeepSeek fused FP8 MLP requires both kernel and block scale grids.")
      source_targets = {}
      for name in sources:
        source_metadata = stored[name]
        if not isinstance(source_metadata, dict) or set(source_metadata) != set(projection):
          break
        source_target = {}
        for kind, value in projection.items():
          leaf = source_metadata[kind]
          wrapped = isinstance(leaf, dict) and leaf.keys() == {"value"}
          if wrapped:
            leaf = leaf["value"]
          if use_value_wrapper:
            value = value["value"]
          if not _is_orbax_array_metadata(leaf):
            break
          if (hybrid_qwen and fused == "qkv_proj" and is_fp8_dtype(leaf.dtype)
              and set(projection) != {"kernel", "kernel_scale"}):
            raise NotImplementedError("Hybrid Qwen fused FP8 QKV requires all three block scale grids.")
          dtype = leaf.dtype if deepseek and fused == "wi" else value.dtype
          array = jax.ShapeDtypeStruct(leaf.shape, dtype, sharding=replicated)
          source_target[kind] = {"value": array} if use_value_wrapper else array
        if len(source_target) != len(projection):
          break
        source_targets[name] = source_target
      if len(source_targets) == len(sources):
        del node[fused]
        node.update(source_targets)
  return target


def fuse_restored_projections(checkpoint, target, config, adapter, path=()):
  """Fuse only checkpoint projections that correspond to fused target leaves."""
  if adapter is None or not isinstance(checkpoint, dict) or not isinstance(target, dict):
    return checkpoint
  checkpoint = {key: fuse_restored_projections(value, target.get(key), config, adapter, path + (key,))
                for key, value in checkpoint.items()}
  tower = _tower_projection(adapter, path, config)
  if tower is not None:
    fused, sources = tower
    if fused not in target or not all(name in checkpoint for name in sources):
      return checkpoint
    if sources != (fused,) and fused in checkpoint:
      return checkpoint
    if sources == (fused,) and checkpoint[fused]["kernel"].shape == target[fused]["kernel"].shape:
      return checkpoint
    converted = checkpoint
    for key in reversed(path):
      converted = {key: converted}
    converted = adapter.convert_native_state(converted, config, fused_qkv=fused == "qkv_proj", fused_mlp=fused == "wi")
    for key in path:
      converted = converted[key]
    return converted
  if adapter in (fused_gemma, fused_qwen3_vl, fused_qwen3_omni, fused_qwen3_next, fused_qwen3_5) and "decoder" not in path:
    return checkpoint
  hybrid_qwen = adapter in (fused_qwen3_next, fused_qwen3_5)
  deepseek = adapter is fused_deepseek
  for containers, fused, sources, enabled in (
      (("attention",) if hybrid_qwen else projection_containers(adapter, mlp=False), "qkv_proj",
       ("query", "key", "value"), config.fused_qkv and not deepseek and (not hybrid_qwen or path[-1:] == ("attention",))),
      (("mlp", "shared_expert") if hybrid_qwen else ("mlp", "shared_experts") if deepseek
       else projection_containers(adapter, mlp=True), "wi", ("wi_0", "wi_1"),
       config.fused_mlp and supports_mlp_layout(adapter)),
  ):
    for container in containers:
      node = checkpoint.get(container)
      target_node = target.get(container)
      single_mlp = adapter is fused_gpt3 and fused == "wi" and len(config.mlp_activations) == 1
      if single_mlp:
        sources = ("wi",)
      if (not enabled or not isinstance(node, dict) or not isinstance(target_node, dict)
          or fused not in target_node or (fused in node and not single_mlp) or not all(name in node for name in sources)):
        continue
      if single_mlp and node["wi"]["kernel"].shape == target_node["wi"]["kernel"].shape:
        continue
      conversion_path = path + (container,) if adapter in (fused_gemma, fused_qwen3_vl, fused_qwen3_omni) or hybrid_qwen or deepseek else (container,)
      converted = node
      for key in reversed(conversion_path):
        converted = {key: converted}
      source_config = config
      if deepseek or (adapter is fused_gpt3 and fused == "wi"):
        native_config = getattr(config, "_pydantic_config", config)
        source_config = pyconfig.HyperParameters(native_config.model_copy(update={"fused_mlp": False}))
      converted = adapter.convert_native_state(
          converted, source_config, fused_qkv=fused == "qkv_proj", fused_mlp=fused == "wi"
      )
      for key in conversion_path:
        converted = converted[key]
      checkpoint[container] = converted
  return checkpoint
