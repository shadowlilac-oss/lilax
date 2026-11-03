"""vLLM adapter for MaxText models."""

import os

from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
from jax import numpy as jnp
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
import numpy as np
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.jax.utils.multi_modal_utils import merge_multimodal_embeddings
from vllm.config import VllmConfig

from lilax._backend.common.common_types import MODEL_MODE_AUTOREGRESSIVE
from lilax._backend.configs import pyconfig, types
from lilax._backend.integration.vllm.convert_utils import DEFAULT_TPU_NUM_LANES, compute_padded_moe_mlp_dim
from lilax._backend.integration.vllm.hybrid_cache_utils import (
    build_qwen_gdn_cache_layout,
    call_with_supported_kwargs,
    gather_layer_kv_caches,
    normalize_vllm_input_positions,
    resolve_layer_kv_cache_indices,
    scatter_layer_kv_caches,
)
from lilax._backend.utils import lora_utils, max_logging, model_creation_utils
from lilax._backend.utils.globals import MAXTEXT_CONFIGS_DIR
from .multimodal import get_multimodal_handler


_HYBRID_LAYER_IMBALANCE_THRESHOLD = 1.5


def next_power_of_two(x: int) -> int:
  """Finds the smallest power of two greater than or equal to x."""
  assert x > 0
  if x == 1:
    return 1
  return 1 << (x - 1).bit_length()


def generate_maxtext_config(vllm_config: VllmConfig) -> pyconfig.HyperParameters:
  """Translate vLLM runtime settings into a native MaxText configuration."""
  overrides = vllm_config.additional_config.get("maxtext_config", {})
  if vllm_config.load_config.load_format == "dummy" and overrides.get("load_parameters_path") is not None:
    max_logging.log("Warning: load_parameters_path is set when using dummy load format. Checkpoint loading will be skipped.")
    overrides["load_parameters_path"] = None
  if overrides.get("attention") == "vllm_batched_rpa" or overrides.get("use_batched_rpa", False):
    os.environ["USE_BATCHED_RPA_KERNEL"] = "1"

  sharding = vllm_config.sharding_config
  tp, ep, attn_dp = sharding.tp_size, sharding.expert_size, sharding.attn_dp_size
  kv_tp_size, moe_mlp_tp_size = tp * ep, tp * attn_dp
  hf_config = vllm_config.model_config.hf_config
  hf_config = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
  hidden_size = getattr(hf_config, "moe_intermediate_size", None)
  try:
    num_lanes = pltpu.get_tpu_info().num_lanes
  except Exception:  # pylint: disable=broad-exception-caught
    num_lanes = DEFAULT_TPU_NUM_LANES
  num_kv_heads = hf_config.num_key_value_heads
  num_global_kv_heads = getattr(hf_config, "num_global_key_value_heads", None)
  max_logging.log(
      f"vLLM sharding config: hidden_size={hidden_size}, kv_heads={num_kv_heads}, global_kv_heads={num_global_kv_heads}, "
      f"num_lanes={num_lanes}, tp={tp}, attn_dp={attn_dp}, ep={ep}, moe_mlp_tp_size={moe_mlp_tp_size}"
  )
  if vllm_config.parallel_config.enable_expert_parallel and ep * sharding.attn_dp_expert_size == 1:
    max_logging.warning(
        "--enable-expert-parallel was requested but the mesh has no expert shards "
        f"(expert={ep}, attn_dp_expert={sharding.attn_dp_expert_size}), so MaxText will shard the MoE over "
        "the MLP dimension instead of the expert dimension. The vLLM flag does not reach the JAX mesh; set "
        'expert_parallelism in the vLLM additional_config sharding_strategy, e.g. \'{"sharding": '
        '{"sharding_strategy": {"expert_parallelism": <num_devices>}}}\', to actually shard experts.'
    )
  if kv_tp_size % num_kv_heads == 0 and num_kv_heads < kv_tp_size:
    max_logging.log(f"Padding num_kv_heads from {num_kv_heads} to {kv_tp_size} to match the degree of tensor parallelism.")
    overrides["base_num_kv_heads"] = kv_tp_size
  if num_global_kv_heads is not None and kv_tp_size % num_global_kv_heads == 0 and num_global_kv_heads < kv_tp_size:
    max_logging.log(f"Padding num_global_kv_heads from {num_global_kv_heads} to {kv_tp_size} to match tensor parallelism.")
    overrides["global_num_kv_heads"] = kv_tp_size
  padded_hidden_size = compute_padded_moe_mlp_dim(hidden_size, moe_mlp_tp_size, num_lanes)
  if padded_hidden_size is not None and padded_hidden_size != hidden_size:
    max_logging.warning(
        f"Padding moe_intermediate_size from {hidden_size} to {padded_hidden_size} to match MLP MoE requirements "
        f"(moe_mlp_tp_size={moe_mlp_tp_size}, 2*num_lanes={2 * num_lanes}). This multiplies the MoE weights and "
        f"MoE FLOPs by {padded_hidden_size / hidden_size:g}x. Consider sharding the MoE over the expert axis "
        "instead, by setting expert_parallelism in the vLLM additional_config sharding_strategy."
    )
    overrides["padded_base_moe_mlp_dim"] = padded_hidden_size
  if getattr(vllm_config.cache_config, "mamba_cache_mode", "none") == "align":
    overrides["gdn_mamba_block_size"] = vllm_config.cache_config.block_size
  config_path = os.path.join(MAXTEXT_CONFIGS_DIR, "inference", "vllm.yml")
  raw_keys, explicit_keys = pyconfig._load_raw_config(config_path, overrides)
  return pyconfig._resolve_native_config(raw_keys, explicit_keys, types.MaxTextConfig)


class MaxTextForCausalLM(nnx.Module):
  """A vLLM-compatible causal language model wrapper for MaxText."""

  _self_manages_sharding: bool = True
  supports_multimodal: bool = False

  def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array, mesh: Mesh):
    self.vllm_config = vllm_config
    self.cfg = vllm_config.model_config
    self.maxtext_config = generate_maxtext_config(vllm_config)
    self.multimodal_handler = get_multimodal_handler(self.maxtext_config.model_name)
    self.mesh = mesh
    self.model_mode = MODEL_MODE_AUTOREGRESSIVE
    self.is_text_generation_model = True
    self.model: nnx.Module | None = None
    self._self_manages_sharding = True
    if vllm_config.load_config.load_format == "dummy":
      self.load_weights(rng_key)
    elif self.maxtext_config.load_parameters_path is None:
      max_logging.log("Warning: No load_parameters_path provided. The model will be initialized with random weights.")

  def modules(self):
    return []

  def __call__(
      self, kv_caches: list[jax.Array], input_ids: jax.Array, attention_metadata: AttentionMetadata,
      inputs_embeds: jax.Array | None = None, _input_positions=None, _layer_name_to_kvcache_index=None, *args, **kwargs,
  ) -> tuple[list[jax.Array], jax.Array, list[jax.Array], list[jax.Array] | None]:
    if not isinstance(self.model, nnx.Module):
      raise ValueError("Model must be an instance of type nnx.Module.")
    if kv_caches is not None:
      physical_indices = resolve_layer_kv_cache_indices(_layer_name_to_kvcache_index, len(kv_caches))
      layer_kv_caches = gather_layer_kv_caches(kv_caches, physical_indices)
    else:
      physical_indices = None
      layer_kv_caches = None
    if inputs_embeds is not None:
      decoder_input_embeddings = inputs_embeds[:, None, :].astype(self.maxtext_config.dtype)
      input_ids = jnp.zeros((inputs_embeds.shape[0], 1), dtype=jnp.int32)
    else:
      decoder_input_embeddings = None
      input_ids = jnp.expand_dims(input_ids, axis=1)
    if isinstance(attention_metadata, dict):
      positions = getattr(next(iter(attention_metadata.values())), "input_positions", None)
    else:
      positions = getattr(attention_metadata, "input_positions", None)
    if positions is None:
      positions = _input_positions
    input_positions = normalize_vllm_input_positions(positions)
    model_kwargs = dict(kwargs)
    for extra_key in (
        "inputs_embeds", "input_positions", "layer_name_to_kvcache_index", "_layer_name_to_kv_cache",
        "shared_attention_metadata", "intermediate_tensors", "lora_metadata", "is_first_rank", "is_last_rank",
    ):
      model_kwargs.pop(extra_key, None)
    with self.mesh, logical_axis_rules(self.maxtext_config.logical_axis_rules):
      aux_hidden_states = []
      expert_indices = None
      result = self.model(
          decoder_input_tokens=input_ids, decoder_input_embeddings=decoder_input_embeddings,
          decoder_positions=input_positions, kv_caches=layer_kv_caches, attention_metadata=attention_metadata,
          model_mode=self.model_mode, **model_kwargs,
      )
      if isinstance(result, tuple) and len(result) == 3:
        hidden, layer_kv_caches, expert_indices = result
      else:
        hidden, layer_kv_caches = result
      if kv_caches is not None:
        kv_caches = scatter_layer_kv_caches(kv_caches, layer_kv_caches, physical_indices)
      hidden = hidden.reshape((-1, hidden.shape[-1]))
    return kv_caches, hidden, aux_hidden_states, expert_indices

  def forward(self, *args, **kwargs):
    return self(*args, **kwargs)

  def get_input_embeddings(self) -> jax.Array:
    if not isinstance(self.model, nnx.Module):
      raise ValueError("Model is not initialized.")
    with self.mesh, logical_axis_rules(self.maxtext_config.logical_axis_rules):
      return self.model.token_embedder.embedding

  def embed_multimodal(self, **kwargs) -> list[jax.Array]:
    if not isinstance(self.model, nnx.Module):
      raise ValueError("Model is not initialized.")
    if self.multimodal_handler is None:
      raise ValueError(f"No vLLM multimodal handler is registered for {self.maxtext_config.model_name!r}.")
    return self.multimodal_handler.embed_multimodal(self.model, self.maxtext_config, self.mesh, **kwargs)

  def embed_input_ids(
      self, input_ids: jax.Array, multimodal_embeddings: jax.Array | None = None, is_multimodal: jax.Array | None = None,
  ) -> jax.Array:
    if not isinstance(self.model, nnx.Module):
      raise ValueError("Model is not initialized.")
    with self.mesh, logical_axis_rules(self.maxtext_config.logical_axis_rules):
      inputs_embeds = self.model.token_embedder(input_ids)
      if multimodal_embeddings is not None:
        if self.multimodal_handler is None:
          raise ValueError(f"No vLLM multimodal handler is registered for {self.maxtext_config.model_name!r}.")
        placeholder_ids = self.multimodal_handler.placeholder_token_ids(self.cfg.hf_config)
        inputs_embeds = merge_multimodal_embeddings(input_ids, inputs_embeds, multimodal_embeddings, placeholder_ids)
      return inputs_embeds

  def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
    if not isinstance(self.model, nnx.Module):
      raise ValueError("Model is not initialized.")
    with self.mesh, logical_axis_rules(self.maxtext_config.logical_axis_rules):
      hidden = jnp.expand_dims(hidden_states, axis=1)
      logits = self.model.decoder.apply_output_head(self.model.token_embedder, hidden, True, self.model_mode)
      return logits.squeeze(1)

  def load_weights(self, rng_key: jax.Array) -> None:
    if self.model is not None:
      return
    with self.mesh, logical_axis_rules(self.maxtext_config.logical_axis_rules):
      model = model_creation_utils.from_pretrained(
          self.maxtext_config, mesh=self.mesh, model_mode=self.model_mode, rng_key=rng_key
      )
      if self.maxtext_config.lora.enable_lora:
        model = lora_utils.apply_lora_to_model(model, self.mesh, self.maxtext_config)
        if self.maxtext_config.lora.lora_restore_path:
          lora_utils.restore_lora_from_path(model, self.maxtext_config)
      self.model = nnx.data(model)
    patch_raiden_worker_h2d()

  def get_mrope_input_positions(self, input_tokens: list[int], mm_features: list = None) -> tuple[np.ndarray, int]:
    positions = np.arange(len(input_tokens), dtype=np.int32)
    return np.stack([positions, positions, positions], axis=0), 0


def patch_kv_cache_manager():
  """Patch the TPU cache manager for hybrid attention and GDN layers."""
  import torch
  from tpu_inference import utils as common_utils
  from tpu_inference.layers.common.sharding import ShardingAxisName
  from tpu_inference.runner.kv_cache import get_attention_page_size_bytes
  from tpu_inference.runner.kv_cache_manager import KVCacheManager
  from vllm.v1.kv_cache_interface import MambaSpec

  try:
    original_get_kv_cache_spec = KVCacheManager.get_kv_cache_spec
  except AttributeError as error:
    raise RuntimeError(
        "Failed to apply KVCacheManager patch: KVCacheManager.get_kv_cache_spec not found. "
        "This usually indicates a vLLM / tpu-inference API change or version mismatch."
    ) from error

  def patched_get_kv_cache_spec(self):
    runner = self.runner
    if not hasattr(runner, "model") or not hasattr(runner.model, "maxtext_config"):
      return original_get_kv_cache_spec(self)
    cfg = runner.model.maxtext_config
    decoder_block = getattr(cfg, "decoder_block", "")
    decoder_block_str = decoder_block if isinstance(decoder_block, str) else getattr(decoder_block, "value", "")
    if decoder_block_str in ("qwen3_next", "qwen3_5"):
      mamba_cache_mode = getattr(runner.cache_config, "mamba_cache_mode", "none")
      interval = cfg.inhomogeneous_layer_cycle_interval
      mamba_shapes, mamba_dtypes, unpadded_mamba_page_size = build_qwen_gdn_cache_layout(cfg, torch)
      tp_axis_name = getattr(ShardingAxisName, "KV_HEAD", ShardingAxisName.ATTN_HEAD)
      model_cnt = common_utils.get_mesh_shape_product(runner.mesh, tp_axis_name)
      model_config = runner.model_config
      text_config = getattr(model_config, "hf_text_config", getattr(model_config, "hf_config", None))
      num_kv_heads = getattr(text_config, "num_global_key_value_heads", None) or model_config.get_total_num_kv_heads()
      head_size = getattr(text_config, "global_head_dim", None) or model_config.get_head_size()
      num_kv_heads = common_utils.get_padded_num_heads(num_kv_heads, model_cnt)
      head_size = common_utils.get_padded_head_dim(head_size)
      attn_page_size_bytes = get_attention_page_size_bytes(
          runner.mesh, runner.cache_config.block_size, num_kv_heads, head_size, runner.kv_cache_dtype, False
      )
      num_layers = cfg.base_num_decoder_layers
      num_attn = num_layers // interval
      num_mamba = num_layers - num_attn
      min_count = min(num_attn, num_mamba)
      max_count = max(num_attn, num_mamba)
      group_size = max_count if max_count < min_count * _HYBRID_LAYER_IMBALANCE_THRESHOLD else min_count
      num_attn_groups = (num_attn + group_size - 1) // group_size
      num_mamba_groups = (num_mamba + group_size - 1) // group_size
      uniform_page_size_bytes = num_attn_groups * attn_page_size_bytes + num_mamba_groups * unpadded_mamba_page_size
      self._hybrid_uniform_page_size_bytes = int(uniform_page_size_bytes)
      runner.cache_config.mamba_page_size_padded = int(uniform_page_size_bytes)
      call_with_supported_kwargs(
          self._maybe_set_compact_mamba_num_blocks_override,
          attn_page_size_bytes=attn_page_size_bytes,
          unpadded_mamba_page_size_bytes=int(unpadded_mamba_page_size),
          num_attn_groups=num_attn_groups,
          num_mamba_groups=num_mamba_groups,
          num_attn_layers=num_attn,
          num_mamba_layers=num_mamba,
          group_size=group_size,
      )
    kv_cache_spec = original_get_kv_cache_spec(self)
    if decoder_block_str in ("qwen3_next", "qwen3_5"):
      for layer_index in range(cfg.base_num_decoder_layers):
        layer_name = f"layer.{layer_index}"
        if (layer_index + 1) % interval != 0:
          if layer_name in kv_cache_spec:
            kv_cache_spec[layer_name] = MambaSpec(
                block_size=kv_cache_spec[layer_name].block_size,
                shapes=mamba_shapes,
                dtypes=mamba_dtypes,
                page_size_padded=self._hybrid_uniform_page_size_bytes,
                mamba_cache_mode=mamba_cache_mode,
            )
        elif layer_name in kv_cache_spec:
          kv_cache_spec[layer_name] = self._create_attention_spec(
              kv_cache_spec[layer_name].block_size, num_kv_heads, head_size,
              sliding_window=getattr(kv_cache_spec[layer_name], "sliding_window", None),
          )
    return kv_cache_spec

  KVCacheManager.get_kv_cache_spec = patched_get_kv_cache_spec
  max_logging.log("Successfully applied KVCacheManager patch for hybrid GDN models.")


def patch_raiden_worker_h2d():
  """Apply Raiden weight synchronization to the worker."""
  from tunix.dist_rl.weight_sync.raiden_synchronizer import patch_raiden_worker_sync

  try:
    patch_raiden_worker_sync()
  except Exception as error:  # pylint: disable=broad-exception-caught
    max_logging.log(f"Skipping raiden worker sync patch: {error}")
