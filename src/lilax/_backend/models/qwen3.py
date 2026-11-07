"""Qwen3 family of model decoder layers."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

import functools
import math
from typing import Any, cast

from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import xla_metadata
import jax.nn
import jax.numpy as jnp
from jax.sharding import Mesh
from lilax._backend.common.common_types import (
    Array,
    BATCH,
    Config,
    EMBED,
    LENGTH,
    ShardMode,
    get_quant,
    get_weight_dtype,
)
from lilax._backend.layers import attentions
from lilax._backend.layers import initializers as max_initializers
from lilax._backend.layers import moe
from lilax._backend.layers import nnx_scan
from lilax._backend.layers.quantization.configure import configure_kv_quant
from lilax._backend.layers.attentions import Attention
from lilax._backend.layers.linears import DenseGeneral, MlpBlock
from lilax._backend.layers.moe import RoutedMoE
from lilax._backend.layers.normalizations import Qwen3NextRMSNorm, RMSNorm
from lilax._backend.layers.quantization.aqt import AqtQuantization as Quant
from lilax._backend.models.qwen3_next.gated_delta_net import Qwen3NextGatedDeltaNet
from lilax._backend.utils import model_mode_shapes
from lilax._backend.utils import maxtext_utils
from lilax._backend.distributed.sharding import create_sharding, get_logical_axis_rules, maybe_shard_with_logical

# -----------------------------------------
# Qwen3-Next Layer Implementations
# -----------------------------------------


class Qwen3NextFullAttention(nnx.Module):
  """Qwen3-Next Full Attention Layer.

  This module implements the full self-attention mechanism as used in
  Qwen3-Next models for layers that do not use the Gated Delta Network.
  It wraps the main `attentions.Attention` class, which handles the core attention operation,
  including the query, key, value, and output projections.

  Qwen3 Next Attention differs from standard attention by the following features:
    - Query and Gate splitting from a single q projection.
    - Application of a sigmoid gate to the attention output.
    - Usage of `Qwen3NextRMSNorm` for query and key normalization.
    - Usage of `PartialRotaryEmbedding` for partial rotary position embeddings.
      - Partial ROPE is applied to the first 25% of head dimensions

  Attributes:
    config: MaxText configuration object.
    mesh: The device mesh for sharding.
    model_mode: The operational mode (e.g., 'train', 'prefill').
    layer_idx: The index of the current layer.
    quant: Optional quantization configuration.
    attention: An instance of `attentions.Attention` which contains the
      learnable parameters for query, key, value, and output projections
      (e.g., `attention.query`, `attention.key`, etc.), and performs
      the attention calculation.
  """

  def __init__(
      self, config: Config, mesh: Mesh, model_mode: str, layer_idx: int, quant: None | Quant = None, *, rngs: nnx.Rngs
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.layer_idx = layer_idx
    self.quant = quant
    cfg = self.config

    scaling_factor = self.config.head_dim**-0.5
    batch_size, seq_len = model_mode_shapes.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)

    self.attention = attentions.Attention(
        config=cfg,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        attention_kernel=cfg.attention,
        inputs_q_shape=dummy_inputs_shape,
        inputs_kv_shape=dummy_inputs_shape,
        out_axis_names=(BATCH, LENGTH, EMBED),
        mesh=self.mesh,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        dropout_rate=cfg.dropout_rate,
        name="self_attention",
        quant=self.quant,
        kv_quant=configure_kv_quant(cfg),
        use_qk_norm=cfg.use_qk_norm,
        query_pre_attn_scalar=scaling_factor,
        model_mode=model_mode,
        use_mrope=cfg.use_mrope,
        mrope_section=cfg.mrope_section,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      kv_cache: None | jnp.ndarray = None,
      attention_metadata: None | dict[str, Any] = None,
      out_sharding: jax.sharding.NamedSharding | None = None,
  ):
    attention_output, kv_cache = self.attention(
        inputs_q=inputs,
        inputs_kv=inputs,
        inputs_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
        out_sharding=out_sharding,
    )
    return attention_output, kv_cache


class Qwen3NextSparseMoeBlock(nnx.Module):
  """
  This module encapsulates the unique MoE structure of Qwen3-Next, which includes:
  1. A set of routed experts, where each token is sent to a subset of experts.
  2. A single shared expert, which all tokens pass through.
  3. A learnable gate that determines the contribution of the shared expert.

  Attributes:
    config: The model configuration object.
    mesh: The device mesh for sharding.
    quant: Optional quantization configuration.
  """

  def __init__(self, config: Config, mesh: Mesh, quant: None | Quant = None, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    cfg = self.config

    # 1. Instantiate and apply the routed experts block.
    self.routed_experts = moe.RoutedMoE(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=self.mesh,
        kernel_init=max_initializers.nd_dense_init(cfg.dense_init_scale, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        intermediate_dim=cfg.moe_mlp_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        quant=self.quant,
        rngs=rngs,
    )

    # 2. Instantiate and apply the shared expert, if the config asks for one.
    # cfg.shared_experts defaults to 0. Building the expert every time puts an
    # extra full-size MLP in every layer of a dense configuration.
    self.use_shared_expert = cfg.shared_experts > 0
    if self.use_shared_expert:
      self.shared_expert = MlpBlock(
          config=cfg,
          mesh=mesh,
          in_features=cfg.emb_dim,
          intermediate_dim=cfg.moe_mlp_dim,
          activations=cfg.mlp_activations,
          intermediate_dropout_rate=cfg.dropout_rate,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          quant=self.quant,
          model_mode=config.model_call_mode,
          rngs=rngs,
      )

      # 3. Instantiate and apply the gate for the shared expert.
      self.shared_expert_gate = DenseGeneral(
          in_features_shape=cfg.emb_dim,
          out_features_shape=1,
          use_bias=False,  # Qwen3-Next shared_expert_gate does not have a bias
          dtype=cfg.dtype,
          weight_dtype=get_weight_dtype(cfg, "shared_expert_gate"),
          quant=get_quant(cfg, "shared_expert_gate", self.quant),
          kernel_init=max_initializers.nd_dense_init(cfg.dense_init_scale, "fan_in", "truncated_normal"),
          kernel_axes=("embed", None),
          matmul_precision=cfg.matmul_precision,
          rngs=rngs,
      )
    else:
      self.shared_expert = None
      self.shared_expert_gate = None

  def __call__(
      self,
      hidden_states: Array,
      deterministic: bool,
      forced_routed_experts: jnp.ndarray | None = None,
      intermediate_sharding: jax.sharding.NamedSharding | None = None,
      out_sharding: jax.sharding.NamedSharding | None = None,
  ) -> tuple[Array, Array | None]:
    """Applies the sparse MoE block to the input hidden states.

    Args:
      hidden_states: The input array from the previous layer. Shape: (batch,
        seq, embed_dim)
      deterministic: If True, disables dropout.
      intermediate_sharding: Optional layout for the shared expert's
        intermediate activation, honoured only under `ShardMode.EXPLICIT`.
      out_sharding: Optional layout for the block output, honoured only under
        `ShardMode.EXPLICIT`.

    Returns:
      A tuple containing:
        - The output array of the MoE block.
        - The load balancing loss from the routed experts, if applicable during
        training.
    """
    # 1. Apply the routed experts block.
    routed_output, load_balance_loss, _ = self.routed_experts(
        hidden_states,
        out_sharding=out_sharding,
        forced_routed_experts=forced_routed_experts,
    )

    if not self.use_shared_expert:
      return routed_output, load_balance_loss

    # 2. Apply the shared expert.
    shared_expert_output = self.shared_expert(
        hidden_states,
        deterministic=deterministic,
        intermediate_sharding=intermediate_sharding,
        out_sharding=out_sharding,
    )

    # 3. Apply the gate for the shared expert. The output is (batch, seq, 1), so it
    # carries no feature axis to pin and takes the default layout.
    shared_gate_output = self.shared_expert_gate(hidden_states)

    # 4. Combine the outputs.
    final_output = routed_output + jax.nn.sigmoid(shared_gate_output) * shared_expert_output

    return final_output, load_balance_loss


class Qwen3NextScannableBlock(nnx.Module):
  """A scannable block of Qwen3-Next decoder layers with hierarchical nested scans.

  One block covers a single period of the attention pattern defined by
  `config.inhomogeneous_layer_cycle_interval`: several linear-attention
  (GatedDeltaNet) layers plus one full-attention layer. The linear-attention
  layers are homogeneous, so they are stacked and run through
  `nnx_scan.apply_scanned_layers`; the lone full-attention layer runs inside a
  trip-count-one `jax.lax.scan` that acts as an XLA scheduling barrier.

  Nesting the scans this way lets each sub-layer be rematerialized on its own
  (`apply_internal_remat`) instead of rematerializing the whole block, so only
  one sub-layer's activations are live at a time.

  Attributes:
    config: The model configuration object.
    mesh: The device mesh for sharding.
    model_mode: The operational mode (e.g., 'train', 'prefill').
    quant: Optional quantization configuration.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | Quant = None,
      *,
      num_of_layers: int | None = None,
      layer_idx_offset: int = 0,
      remat_policy_fn: Any | None = None,
      apply_internal_remat: bool = False,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.rngs = rngs
    self.remat_policy_fn = remat_policy_fn
    self.apply_internal_remat = apply_internal_remat
    cfg = self.config
    if num_of_layers is None:
      num_of_layers = cfg.inhomogeneous_layer_cycle_interval
    self.num_of_layers = num_of_layers
    self.layer_idx_offset = layer_idx_offset

    cycle_interval = cfg.inhomogeneous_layer_cycle_interval
    # Qwen3-Next puts the full-attention layer last in every cycle, which is what
    # Qwen3NextDecoderLayer derives from layer_idx when it is not told explicitly.
    full_attention_offset = cycle_interval - 1

    positions = [(layer_idx_offset + i) % cycle_interval for i in range(num_of_layers)]
    self.num_local = sum(1 for p in positions if p != full_attention_offset)
    self.num_global = sum(1 for p in positions if p == full_attention_offset)
    if self.num_global > 1:
      raise ValueError(
          f"A Qwen3-Next scannable block spans {num_of_layers} layers starting at offset {layer_idx_offset}, which "
          f"covers {self.num_global} full-attention layers; the block supports at most one."
      )
    # The local scan runs before the global layer, so the block only reproduces the
    # model's layer order when the full-attention layer is last in the period.
    if self.num_global == 1 and positions[-1] != full_attention_offset:
      raise ValueError(
          f"Qwen3-Next scannable block expects the full-attention layer last in the block, but a block of "
          f"{num_of_layers} layers starting at layer_idx_offset={layer_idx_offset} lands it at block position "
          f"{positions.index(full_attention_offset)}. Blocks must start on a cycle boundary."
      )

    if self.num_local > 0:
      self.local_layers = nnx_scan.create_scanned_layers(
          lambda layer_rngs: self._make_decoder_layer(
              layer_idx=0,
              is_full_attention_layer=False,
              rngs=layer_rngs,
          ),
          length=self.num_local,
          param_scan_axis=self.config.param_scan_axis,
          metadata_axis_name="local_layers",
          rngs=self.rngs,
      )
    else:
      self.local_layers = None

    if self.num_global > 0:
      self.global_layer = self._make_decoder_layer(
          layer_idx=full_attention_offset,
          is_full_attention_layer=True,
          rngs=self.rngs,
      )
    else:
      self.global_layer = None

    self.activation_axis_names = (
        "activation_batch",
        "activation_norm_length",
        "activation_embed",
    )
    # Same convention as Qwen3NextDecoderLayer: pin the carry under ShardMode.EXPLICIT,
    # and leave the AUTO path free of sharding constraints so XLA keeps its fusions.
    if cfg.shard_mode == ShardMode.EXPLICIT:
      self._maybe_shard_with_logical = functools.partial(
          maybe_shard_with_logical,
          mesh=mesh,
          shard_mode=cfg.shard_mode,
          debug_sharding=cfg.debug_sharding,
      )
    else:
      self._maybe_shard_with_logical = lambda inputs, *args, **kwargs: inputs

  def _make_decoder_layer(self, *, layer_idx, is_full_attention_layer, rngs):
    """Builds one sub-layer of the block."""
    return Qwen3NextDecoderLayer(
        config=self.config,
        mesh=self.mesh,
        model_mode=self.model_mode,
        quant=self.quant,
        layer_idx=layer_idx,
        is_full_attention_layer=is_full_attention_layer,
        rngs=rngs,
    )

  def _run_layer(self, layer, y, layer_kwargs, kv_cache=None, forced_routed_experts=None):
    """Invokes one Qwen3NextDecoderLayer, returning (output, updated_kv_cache)."""
    out = layer(y, **layer_kwargs, kv_cache=kv_cache, forced_routed_experts=forced_routed_experts)
    return out if isinstance(out, tuple) else (out, None)

  @property
  def _remat_enabled(self):
    """Whether the block rematerializes its own layers."""
    remat_policy = getattr(self, "remat_policy_override", None) or self.config.remat_policy
    return self.apply_internal_remat and bool(remat_policy) and remat_policy != "none"

  def _scan_local_layers(self, y, layer_kwargs, forced_routed_experts=None):
    """Runs the local (linear attention / GatedDeltaNet) layers via a per-layer rematerialized jax.lax.scan."""
    remat = self._remat_enabled
    if forced_routed_experts is None:

      def apply_fn(layer, carry):
        return self._run_layer(layer, carry, layer_kwargs)[0]

    else:

      def apply_fn(layer, carry, routing):
        return self._run_layer(layer, carry, layer_kwargs, forced_routed_experts=routing)[0]

    return nnx_scan.apply_scanned_layers(
        self.local_layers,
        y,
        length=self.num_local,
        param_scan_axis=self.config.param_scan_axis,
        apply_fn=apply_fn,
        xs=forced_routed_experts,
        remat=remat,
        remat_policy=self.remat_policy_fn if remat else None,
        prevent_cse=maxtext_utils.should_prevent_cse_in_remat(self.config) if remat else True,
    )

  def _scan_global_layer(self, y, layer_kwargs, forced_routed_experts=None):
    """Runs the single global-attention layer inside a length-1 jax.lax.scan."""
    cfg = self.config
    graphdef_g, params_g, intermediate_g, other_g = nnx.split(self.global_layer, nnx.Param, nnx.Intermediate, ...)
    # The params ride in as scan xs rather than in the carry: a carried value the body returns
    # unchanged is a loop invariant, so XLA may hoist the layer's FSDP weight all-gathers out of
    # the trip-count-one loop into the enclosing block scan, where they no longer overlap this layer.
    params_xs = jax.tree.map(lambda x: x[None], params_g)
    intermediate_xs = jax.tree.map(lambda x: x[None], intermediate_g)
    carried_param_paths = {path for path, _ in nnx.to_flat_state(params_g)}

    def run_global_layer(carry, xs):
      hidden_states, other = carry
      params_slice, intermediate_slice = xs
      layer = nnx.merge(graphdef_g, params_slice, intermediate_slice, other)
      new_hidden_states = self._run_layer(
          layer, hidden_states, layer_kwargs, forced_routed_experts=forced_routed_experts
      )[0]
      _, new_params, new_intermediate, new_other = nnx.split(layer, nnx.Param, nnx.Intermediate, ...)
      # As in `nnx_scan.apply_scanned_layers`: the params fed in come back unchanged and are dropped,
      # but params created while tracing the layer (e.g. Qwix LoRA adapters) must leave the scan.
      created_params = nnx.from_flat_state(
          [(path, value) for path, value in nnx.to_flat_state(new_params) if path not in carried_param_paths]
      )
      return (new_hidden_states, new_other), (new_intermediate, created_params)

    global_remat_policy = self.remat_policy_fn
    offload_names = maxtext_utils.get_save_and_offload_names(cfg)
    remat_policy = getattr(self, "remat_policy_override", None) or cfg.remat_policy
    if remat_policy == cfg.remat_policy and (offload_names[0] or offload_names[1]):
      save_names, offload_to_device = offload_names
      global_remat_policy = jax.checkpoint_policies.save_only_these_names(*(save_names + offload_to_device))

    if self._remat_enabled:
      prevent_cse = maxtext_utils.should_prevent_cse_in_remat(self.config)
      run_global_layer = jax.checkpoint(
          run_global_layer,
          policy=global_remat_policy,
          prevent_cse=prevent_cse,
      )

    with xla_metadata.set_xla_metadata(**{"skip-simplify-while-loops_trip-count-one": "true"}):
      (y, final_other), (stacked_intermediate, stacked_created_params) = jax.lax.scan(
          run_global_layer,
          (y, other_g),
          (params_xs, intermediate_xs),
          length=1,
      )

    intermediate_state = jax.tree.map(lambda x: x[0], stacked_intermediate)
    created_params = jax.tree.map(lambda x: x[0], stacked_created_params)
    nnx.update(self.global_layer, final_other, intermediate_state, created_params)
    return y

  def _forward_with_external_kv_cache(self, y, kv_cache, layer_kwargs, sub_layer_attention_metadata=None):
    """Runs the block with externally-supplied per-layer kv caches.

    Inference KV caches are a Python list of per-layer entries, so this path
    unrolls the local layers statically rather than scanning them.

    Args:
      y: The block input.
      kv_cache: Per-sub-layer caches, the local layers first and the
        full-attention layer last.
      layer_kwargs: Keyword arguments for every sub-layer call.
      sub_layer_attention_metadata: Optional per-sub-layer attention metadata,
        ordered like `kv_cache`; entry i replaces `layer_kwargs`'s
        `attention_metadata` for sub-layer i.

    Returns:
      The block output and the tuple of updated per-sub-layer caches.
    """

    def sub_layer_kwargs(position):
      if sub_layer_attention_metadata is None:
        return layer_kwargs
      return {**layer_kwargs, "attention_metadata": sub_layer_attention_metadata[position]}

    updated_kvs = []
    if self.local_layers is not None:
      graphdef, params, state = nnx.split(self.local_layers, nnx.Param, ...)
      scan_axis = self.config.param_scan_axis
      if scan_axis != 0:
        params = jax.tree.map(lambda x: jnp.moveaxis(x, scan_axis, 0), params)
      per_layer_states = []
      for i in range(self.num_local):
        current_params = jax.tree.map(lambda x, i=i: x[i], params)
        current_state = jax.tree.map(lambda x, i=i: x[i], state)
        layer = nnx.merge(graphdef, current_params, current_state)
        current_kv = kv_cache[i] if (kv_cache is not None and i < len(kv_cache)) else None
        y, new_kv = self._run_layer(layer, y, sub_layer_kwargs(i), current_kv)
        updated_kvs.append(new_kv)
        # Collect only non-Param state: parameters are read-only here, so stacking
        # them back would allocate a second copy of every layer weight. Non-Param
        # state is stacked on axis 0, matching nnx_scan.create_scanned_layers.
        _, _, updated_state = nnx.split(layer, nnx.Param, ...)
        per_layer_states.append(updated_state)

      nnx.update(self.local_layers, jax.tree.map(lambda *xs: jnp.stack(xs), *per_layer_states))

    if self.global_layer is not None:
      global_kv = kv_cache[self.num_local] if (kv_cache is not None and self.num_local < len(kv_cache)) else None
      y, new_kv = self._run_layer(self.global_layer, y, sub_layer_kwargs(self.num_local), global_kv)
      updated_kvs.append(new_kv)

    return y, tuple(updated_kvs)

  def __call__(
      self,
      carry: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray = None,
      decoder_positions: None | jnp.ndarray = None,
      deterministic: bool = False,
      model_mode: str = "train",
      previous_chunk=None,
      slot: None | int = None,
      kv_cache=None,
      attention_metadata=None,
      forced_routed_experts=None,
  ) -> tuple[Array, None]:
    """Applies the block of decoder layers to the input carry.

    Args:
      carry: The input tensor from the previous scan iteration.
      kv_cache: Optional externally-managed (vLLM) caches, one per sub-layer,
        in block order (the local layers first, the full-attention layer last).
      attention_metadata: Optional attention metadata: either one object shared
        by every sub-layer or, together with `kv_cache`, a tuple holding one
        entry per sub-layer in block order. vLLM's per-layer dict (keyed by
        global layer) has to be resolved into that tuple by the caller, as
        NNXDecoder._apply_qwen3_hybrid_scanned_blocks does, because the
        sub-layers of a scanned block do not know their global layer index.
      forced_routed_experts: Optional router replay indices for this block,
        shaped `[num_of_layers, batch, seq, top_k]` -- one slice per sub-layer,
        in block order (the local layers first, the full-attention layer last).
      # ... other arguments are broadcasted to each iteration.

    Returns:
      A tuple containing the output of the block (the new carry) and an empty
      value for the scan's `y` collection.
    """
    cfg = self.config
    inputs = carry
    inputs = self._maybe_shard_with_logical(inputs, self.activation_axis_names)

    layer_kwargs = {
        "decoder_segment_ids": decoder_segment_ids,
        "decoder_positions": decoder_positions,
        "deterministic": deterministic,
        "model_mode": model_mode,
        "slot": slot,
        "previous_chunk": previous_chunk,
        "attention_metadata": attention_metadata,
    }

    if forced_routed_experts is not None and forced_routed_experts.shape[0] != self.num_of_layers:
      raise ValueError(
          f"forced_routed_experts must carry one slice per sub-layer of the block: expected leading axis "
          f"{self.num_of_layers}, got {forced_routed_experts.shape[0]} (shape {forced_routed_experts.shape})."
      )
    if isinstance(attention_metadata, dict):
      # Such metadata is keyed by global layer index, which the sub-layers of a scanned block do not have (every local
      # layer is built with layer_idx=0), so a `layer.{i}` lookup here would read another layer's entry. With external
      # kv caches the decoder resolves it and hands the block a per-sub-layer tuple instead.
      raise NotImplementedError(
          "Per-layer attention_metadata (a dict keyed by layer) cannot be resolved inside a scanned block; pass a "
          "tuple with one entry per sub-layer together with externally-managed kv caches, or use scan_layers=False."
      )
    sub_layer_attention_metadata = None
    if isinstance(attention_metadata, tuple):
      if kv_cache is None:
        raise NotImplementedError(
            "Per-sub-layer attention_metadata is only supported together with externally-managed kv caches."
        )
      if len(attention_metadata) != self.num_of_layers:
        raise ValueError(
            f"attention_metadata must carry one entry per sub-layer of the block: expected {self.num_of_layers}, "
            f"got {len(attention_metadata)}."
        )
      sub_layer_attention_metadata = attention_metadata

    if kv_cache is not None:
      if forced_routed_experts is not None:
        raise NotImplementedError("Forced routing is not supported together with externally-managed kv caches.")
      return self._forward_with_external_kv_cache(inputs, kv_cache, layer_kwargs, sub_layer_attention_metadata)

    y = inputs
    if self.local_layers is not None:
      local_routing = None if forced_routed_experts is None else forced_routed_experts[: self.num_local]
      y = self._scan_local_layers(y, layer_kwargs, local_routing)
    if self.global_layer is not None:
      global_routing = None if forced_routed_experts is None else forced_routed_experts[self.num_local]
      y = self._scan_global_layer(y, layer_kwargs, global_routing)

    if cfg.scan_layers:
      return y, None
    return y


class Qwen3NextDecoderLayer(nnx.Module):
  """
  This layer is a hybrid, capable of functioning as either:
  1. A standard attention + MoE layer.
  2. A linear attention + MoE layer.

  NOTE: This implementation assumes every layer contains a MoE block, which is true for
  models like Qwen3-Next-80B-A3B where `decoder_sparse_step=1`. For models that
  interleave dense and sparse MLP layers, conditional logic would be needed here.

  Attributes:
    config: The model configuration object.
    mesh: The device mesh for sharding.
    model_mode: The operational mode (e.g., 'train', 'prefill').
    layer_idx: The index of the current layer in the transformer stack.
    quant: Optional quantization configuration.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      layer_idx: int,
      quant: None | Quant = None,
      *,
      is_full_attention_layer: bool | None = None,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.layer_idx = layer_idx
    self.quant = quant
    cfg = self.config
    self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")
    self.mlp_activation_axis_names = (
        "activation_batch",
        "activation_norm_length",
        "activation_mlp",
    )

    # Physical shardings used to pin sublayer outputs under ShardMode.EXPLICIT. In
    # ShardMode.AUTO the callees ignore these and let GSPMD infer the layout.
    if cfg.shard_mode == ShardMode.EXPLICIT:
      self.out_sharding = create_sharding(mesh, self.activation_axis_names, rules=get_logical_axis_rules())
      self.mlp_intermediate_sharding = create_sharding(
          mesh, self.mlp_activation_axis_names, rules=get_logical_axis_rules()
      )
      self._maybe_shard_with_logical = functools.partial(
          maybe_shard_with_logical,
          mesh=mesh,
          shard_mode=cfg.shard_mode,
          debug_sharding=cfg.debug_sharding,
      )
    else:
      self.out_sharding = None
      self.mlp_intermediate_sharding = None
      self._maybe_shard_with_logical = lambda inputs, *args, **kwargs: inputs

    # First LayerNorm, applied before the attention block.
    self.input_layernorm = Qwen3NextRMSNorm(
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=self._norm_weight_dtype(),
        shard_mode=cfg.shard_mode,
        rngs=rngs,
    )

    # Determine the type of attention mechanism for the current layer. A scanned block
    # knows each sub-layer's role up front and passes it explicitly, because inside a
    # scan the layer's position is not recoverable from layer_idx.
    if is_full_attention_layer is None:
      is_full_attention_layer = (self.layer_idx + 1) % cfg.inhomogeneous_layer_cycle_interval == 0
    self.is_full_attention_layer = is_full_attention_layer

    # Conditionally instantiate either the Linear Attention or Full Attention block.
    if is_full_attention_layer:
      self.attention = self._make_full_attention(rngs=rngs)
    else:
      self.attention = self._make_linear_attention(rngs=rngs)

    # Second LayerNorm, applied before the MoE block.
    self.post_attention_layernorm = Qwen3NextRMSNorm(
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=self._norm_weight_dtype(),
        shard_mode=cfg.shard_mode,
        rngs=rngs,
    )

    self.mlp = self._make_mlp(rngs=rngs)

  def _norm_weight_dtype(self):
    """Storage dtype of the two RMSNorm scales."""
    return self.config.weight_dtype

  def _make_full_attention(self, *, rngs: nnx.Rngs):
    """Builds the full-attention sub-block used on the last layer of every cycle."""
    return Qwen3NextFullAttention(
        config=self.config,
        mesh=self.mesh,
        quant=self.quant,
        model_mode=self.model_mode,
        layer_idx=self.layer_idx,
        rngs=rngs,
    )

  def _make_linear_attention(self, *, rngs: nnx.Rngs):
    """Builds the linear-attention (GatedDeltaNet) sub-block used on all other layers."""
    cfg = self.config
    batch_size, seq_len = model_mode_shapes.get_batch_seq_len_for_mode(cfg, self.model_mode)
    dummy_inputs_shape = (batch_size, seq_len, cfg.emb_dim)
    return Qwen3NextGatedDeltaNet(
        config=cfg,
        inputs_shape=dummy_inputs_shape,
        mesh=self.mesh,
        dtype=cfg.dtype,
        model_mode=self.model_mode,
        quant=self.quant,
        rngs=rngs,
    )

  def _make_mlp(self, *, rngs: nnx.Rngs):
    """Builds the sparse MoE block."""
    return Qwen3NextSparseMoeBlock(config=self.config, mesh=self.mesh, quant=self.quant, rngs=rngs)

  def _apply_mlp(self, hidden_states, deterministic, forced_routed_experts):
    """Runs the block built by `_make_mlp`, returning `(output, load_balance_loss)`."""
    return self.mlp(
        hidden_states,
        deterministic=deterministic,
        forced_routed_experts=forced_routed_experts,
        intermediate_sharding=self.mlp_intermediate_sharding,
        out_sharding=self.out_sharding,
    )

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache: None | dict[str, Array] = None,
      attention_metadata: None | dict[str, Any] = None,
      forced_routed_experts: jnp.ndarray | None = None,
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    inputs = self._maybe_shard_with_logical(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    residual = inputs

    if isinstance(attention_metadata, dict):
      attention_metadata = attention_metadata.get(f"layer.{self.layer_idx}", attention_metadata.get(self.layer_idx))

    # First LayerNorm, applied before the attention block.
    hidden_states = self.input_layernorm(inputs, out_sharding=self.out_sharding)
    hidden_states = self._maybe_shard_with_logical(hidden_states, self.activation_axis_names)

    # Conditionally apply either the Linear Attention or Full Attention block.
    if isinstance(self.attention, Qwen3NextFullAttention):
      attention_output, new_kv_cache = cast(Qwen3NextFullAttention, self.attention)(
          hidden_states,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          kv_cache=kv_cache,  # pyrefly: ignore[bad-argument-type]
          attention_metadata=attention_metadata,
          out_sharding=self.out_sharding,
      )
    else:
      attention_output, new_kv_cache = cast(Qwen3NextGatedDeltaNet, self.attention)(
          hidden_states,
          model_mode=model_mode,
          kv_cache=kv_cache,
          decoder_segment_ids=decoder_segment_ids,
          attention_metadata=attention_metadata,
          out_sharding=self.out_sharding,
      )

    # First residual connection after attention
    attention_output = self._maybe_shard_with_logical(attention_output, self.activation_axis_names)
    hidden_states = residual + attention_output
    hidden_states = self._maybe_shard_with_logical(hidden_states, self.activation_axis_names)

    # Prepare for the MoE block by capturing the new residual
    residual = hidden_states

    # Second LayerNorm, applied before the MoE block.
    hidden_states = self.post_attention_layernorm(hidden_states, out_sharding=self.out_sharding)
    hidden_states = self._maybe_shard_with_logical(hidden_states, self.activation_axis_names)

    # Instantiate and call our `Qwen3NextSparseMoeBlock`.
    mlp_output, load_balance_loss = self._apply_mlp(hidden_states, deterministic, forced_routed_experts)

    # Keep the load balancing loss so it can be collected and added to the total loss during training. It is
    # assigned rather than sown: `sow` appends, so Intermediate would grow per call, which a scan body cannot express.
    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.moe_lb_loss = nnx.Intermediate(load_balance_loss)

    # Final residual connection (after the MoE block)
    mlp_output = self._maybe_shard_with_logical(mlp_output, self.activation_axis_names)
    layer_output = residual + mlp_output
    layer_output = self._maybe_shard_with_logical(
        layer_output,
        self.activation_axis_names,
    )
    return layer_output, new_kv_cache


# -----------------------------------------
# The Base Decoder Layer for Qwen3
# -----------------------------------------
class AttentionWithNorm(nnx.Module):
  """Base class with shared common components: self-attention block with normalization."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | Quant,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant

    batch_size, seq_len = model_mode_shapes.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)
    self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")
    self.mlp_activation_axis_names = ("activation_batch", "activation_norm_length", "activation_mlp")

    # Physical shardings used to pin sublayer outputs under ShardMode.EXPLICIT. In
    # ShardMode.AUTO the callees ignore these and let GSPMD infer the layout.
    if config.shard_mode == ShardMode.EXPLICIT:
      self.out_sharding = create_sharding(mesh, self.activation_axis_names, rules=get_logical_axis_rules())
      self.mlp_intermediate_sharding = create_sharding(
          mesh, self.mlp_activation_axis_names, rules=get_logical_axis_rules()
      )
      self._maybe_shard_with_logical = functools.partial(
          maybe_shard_with_logical,
          mesh=mesh,
          shard_mode=config.shard_mode,
          debug_sharding=config.debug_sharding,
      )
    else:
      self.out_sharding = None
      self.mlp_intermediate_sharding = None
      self._maybe_shard_with_logical = lambda inputs, *args, **kwargs: inputs

    # Corresponds to Qwen3's `input_layernorm`
    self.pre_self_attention_layer_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )

    # Self-attention block
    query_pre_attn_scalar = config.head_dim**-0.5  # Qwen3 specific scaling
    self.self_attention = Attention(
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        max_prefill_predict_length=config.max_prefill_predict_length,
        attention_kernel=config.attention,
        inputs_q_shape=dummy_inputs_shape,
        inputs_kv_shape=dummy_inputs_shape,
        mesh=mesh,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        dropout_rate=config.dropout_rate,
        float32_qk_product=config.float32_qk_product,
        float32_logits=config.float32_logits,
        quant=quant,
        kv_quant=configure_kv_quant(config),
        use_ragged_attention=config.use_ragged_attention,
        ragged_block_size=config.ragged_block_size,
        use_qk_norm=config.use_qk_norm,
        query_pre_attn_scalar=query_pre_attn_scalar,
        model_mode=model_mode,
        use_mrope=config.use_mrope,
        mrope_section=config.mrope_section,
        rngs=rngs,
    )

    # Post Attention LayerNorm (corresponds to Qwen3's `post_attention_layernorm`)
    self.post_self_attention_layer_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )

  def apply_attention_with_norm(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      kv_cache: None | jnp.ndarray = None,
      attention_metadata: None | dict[str, Any] = None,
  ):
    """Applies self-attention with pre and post-layer normalization."""
    inputs = self._maybe_shard_with_logical(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    # Pre attention norm
    lnx = self.pre_self_attention_layer_norm(inputs, out_sharding=self.out_sharding)
    lnx = self._maybe_shard_with_logical(lnx, self.activation_axis_names)
    # Self attention
    attention_lnx, kv_cache = self.self_attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        out_sharding=self.out_sharding,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )
    attention_lnx = self._maybe_shard_with_logical(attention_lnx, self.activation_axis_names)
    # Residual connection after attention
    intermediate_inputs = inputs + attention_lnx
    # Post attention norm
    hidden_states = self.post_self_attention_layer_norm(intermediate_inputs, out_sharding=self.out_sharding)
    hidden_states = self._maybe_shard_with_logical(hidden_states, self.activation_axis_names)
    return hidden_states, intermediate_inputs, kv_cache


# -----------------------------------------
# The Dense Decoder Layer for Qwen3
# -----------------------------------------
class Qwen3DecoderLayer(AttentionWithNorm):
  """Qwen3 Transformer decoder layer (dense)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | Quant,
      rngs: nnx.Rngs,
  ):
    super().__init__(config, mesh, model_mode, quant, rngs)
    self.mlp = MlpBlock(
        in_features=config.emb_dim,
        intermediate_dim=config.mlp_dim,
        activations=config.mlp_activations,
        intermediate_dropout_rate=config.dropout_rate,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        config=config,
        mesh=mesh,
        quant=quant,
        model_mode=model_mode,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache: None | jnp.ndarray = None,
      attention_metadata: None | dict[str, Any] = None,
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    hidden_states, intermediate_inputs, kv_cache = self.apply_attention_with_norm(
        inputs,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )

    mlp_lnx = self.mlp(
        hidden_states,
        deterministic=deterministic,
        intermediate_sharding=self.mlp_intermediate_sharding,
        out_sharding=self.out_sharding,
    )
    mlp_lnx = self._maybe_shard_with_logical(mlp_lnx, self.activation_axis_names)

    layer_output = intermediate_inputs + mlp_lnx
    layer_output = self._maybe_shard_with_logical(layer_output, self.activation_axis_names)

    return layer_output, kv_cache


# -----------------------------------------
# The MoE Decoder Layer for Qwen3
# -----------------------------------------
class Qwen3MoeDecoderLayer(AttentionWithNorm):
  """Qwen3 Transformer decoder layer (MoE)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | Quant,
      rngs: nnx.Rngs,
  ):
    super().__init__(config, mesh, model_mode, quant, rngs)
    self.moe_block = RoutedMoE(
        config=config,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        mesh=mesh,
        kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        intermediate_dim=config.moe_mlp_dim,  # same as config.mlp_dim
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        quant=quant,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache: None | jnp.ndarray = None,
      attention_metadata: None | dict[str, Any] = None,
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    is_scan_carry = False
    if isinstance(inputs, tuple) and len(inputs) == 3:
      hidden_states, stacked_kv_cache, layer_idx = inputs
      kv_cache = stacked_kv_cache[layer_idx]
      inputs = hidden_states
      is_scan_carry = True
    elif isinstance(inputs, tuple):
      inputs = inputs[0]
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    hidden_states, intermediate_inputs, kv_cache = self.apply_attention_with_norm(
        inputs,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )

    mlp_lnx, load_balance_loss, _ = self.moe_block(hidden_states, out_sharding=self.out_sharding)
    mlp_lnx = self._maybe_shard_with_logical(mlp_lnx, self.activation_axis_names)
    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.moe_lb_loss = nnx.Intermediate(load_balance_loss)

    layer_output = intermediate_inputs + mlp_lnx
    layer_output = self._maybe_shard_with_logical(layer_output, self.activation_axis_names)

    if is_scan_carry:

      def update_cache(cache, val):
        if jnp.size(val) > 0:
          return cache.at[layer_idx].set(val)
        return cache

      stacked_kv_cache = jax.tree_util.tree_map(update_cache, stacked_kv_cache, kv_cache)  # pyrefly: ignore[unbound-name]
      return (layer_output, stacked_kv_cache, layer_idx + 1), None  # pyrefly: ignore[unbound-name]
    else:
      return layer_output, kv_cache


