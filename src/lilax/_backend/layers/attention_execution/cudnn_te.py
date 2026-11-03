"""Runs cuDNN fused flash attention through Transformer Engine's DotProductAttention."""

from typing import Callable

from flax import nnx
import jax.numpy as jnp
from jax.sharding import Mesh

from lilax._backend.common.common_types import (
    Array,
    AttentionType,
    Config,
    DEFAULT_MASK_VALUE,
    DType,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
)
from lilax._backend.layers import nnx_wrappers


def _sinks_to_te_softmax_offset(sinks: Array, num_query_heads: int) -> Array:
  """Convert MaxText per-head sinks (H,) to TE softmax_offset (1, H, 1, 1) float32."""
  sinks = jnp.asarray(sinks, dtype=jnp.float32)
  if sinks.ndim == 1:
    if sinks.shape[0] != num_query_heads:
      raise ValueError(f"Expected sinks shape ({num_query_heads},), got {sinks.shape}.")
    return sinks.reshape(1, num_query_heads, 1, 1)
  if sinks.shape == (1, num_query_heads, 1, 1):
    return sinks
  raise ValueError(
      f"Unsupported sinks shape {sinks.shape}; expected ({num_query_heads},) or (1, {num_query_heads}, 1, 1)."
  )


def _inject_te_softmax_offset(dpa_layer, softmax_offset: Array) -> None:
  """Overwrites the learnable softmax offset of a ToNNX-wrapped TE DotProductAttention.

  Transformer Engine owns `softmax_offset` as a parameter of its own module (matching the
  PyTorch API), so MaxText grafts its `sinks` parameter into that slot right before the call.
  The grafted array is an ordinary input of the computation, so gradients flow back to `sinks`
  and TE's internal zero-initialized parameter never reaches the optimizer or the checkpoint.

  The parameter lives under a TE-internal submodule whose name depends on whether the fused or
  the unfused backend was selected, hence the lookup by leaf name.
  """
  flat_state = nnx.to_flat_state(nnx.state(dpa_layer))
  offset_path = next((path for path, _ in flat_state if path[-1] == "softmax_offset"), None)
  if offset_path is None:
    raise RuntimeError(
        "Attention sinks require a `softmax_offset` parameter in Transformer Engine's "
        "DotProductAttention, but no such parameter was found after initialization. The "
        "installed Transformer Engine likely renamed it. Available parameters: "
        f"{sorted('/'.join(map(str, path)) for path, _ in flat_state)}."
    )
  nnx.update(dpa_layer, nnx.from_flat_state([(offset_path, nnx.Param(softmax_offset))]))


def cudnn_flash_attention(
    query: Array,
    key: Array,
    value: Array,
    decoder_segment_ids: Array | None,
    segment_positions: Array | None,
    model_mode: str = MODEL_MODE_TRAIN,
    sinks: Array | None = None,
    *,
    config: Config,
    mesh: Mesh,
    attention_type: AttentionType,
    sliding_window_size: int | None,
    num_query_heads: int,
    num_kv_heads: int,
    dropout_rate: float,
    dtype: DType,
    float32_logits: bool,
    max_target_length: int,
    rngs: nnx.Rngs | None,
    generate_attention_mask: Callable,
) -> Array:
  """CUDNN Flash Attention with Transformer Engine.
  1. Stable API, supports MHA, GQA, SWA, Packing and Context Parallelism
  2. Context Parallelism currently only supports causal masking
  3. Only Ring attention has packing support with striped load balancing
    (context_parallel_strategy="ring" and context_parallel_load_balance=true)
  4. Breaks with TE 2.12 and 2.13 (known bug); works with TE stable release <=2.11 or >=2.14.

  `rngs` are the layer's attention-dropout streams, and `generate_attention_mask` is the layer's dense mask builder.
  """
  from transformer_engine.jax.flax.transformer import DotProductAttention  # pytype: disable=import-error
  from transformer_engine.jax.attention import SequenceDescriptor  # pytype: disable=import-error

  _, _, _, head_dim = query.shape
  using_context_parallelism = mesh.shape[config.context_sharding] > 1
  using_load_balanced_ring_cp = (
      using_context_parallelism
      and config.context_parallel_strategy == "ring"
      and config.context_parallel_load_balance
  )
  if using_context_parallelism and attention_type == AttentionType.CHUNK:
    raise ValueError("Chunk attention is not supported with CUDNN context parallelism.")

  te_sliding_window_size = None
  mask_type = "padding_causal"
  qkv_layout = "BSHD_BSHD_BSHD"
  max_segments_per_seq = 1
  if attention_type == AttentionType.LOCAL_SLIDING:
    te_sliding_window_size = [sliding_window_size - 1, 0]

  if sinks is not None:
    if using_context_parallelism:
      raise ValueError(
          "Attention sinks are not supported with context parallelism by Transformer Engine "
          "fused attention, which requires a vanilla softmax when context parallelism is "
          f"active (mesh axis {config.context_sharding!r} has size "
          f"{mesh.shape[config.context_sharding]}). Disable context parallelism or "
          "use attention=dot_product."
      )
  dummy_attn_mask = None
  if config.packing and config.dataset_type != "synthetic":
    if using_context_parallelism and not using_load_balanced_ring_cp:
      raise ValueError("Packing is only supported for load balanced ring attention with context parallelism.")
    qkv_layout = "THD_THD_THD"
    if decoder_segment_ids is None:
      decoder_segment_ids = jnp.ones(shape=query.shape[:2], dtype=jnp.int32)

    def _sequence_descriptor(segment_ids):
      try:
        return SequenceDescriptor.from_segment_ids_and_pos(
            segment_ids=segment_ids,
            segment_pos=segment_positions,
            is_thd=True,
            is_segment_ids_reordered=False,
        )
      except TypeError:
        return SequenceDescriptor.from_segment_ids_and_pos(segment_ids=segment_ids, segment_pos=segment_positions)

    attn_mask = _sequence_descriptor(decoder_segment_ids)
    if sinks is not None:
      dummy_attn_mask = _sequence_descriptor(jnp.ones(shape=query.shape[:2], dtype=jnp.int32))
    max_segments_per_seq = config.max_segments_per_seq
  elif using_context_parallelism:
    if attention_type == AttentionType.LOCAL_SLIDING:
      raise AssertionError(
          "Sliding window attention requires context parallelism with load-balanced ring strategy "
          "and packing enabled."
      )
    attn_mask = None
    mask_type = "causal"
  elif model_mode == MODEL_MODE_PREFILL and config.attention_kernel == "cudnn":
    attn_mask = None
    mask_type = "causal"
  else:
    if decoder_segment_ids is None:
      decoder_segment_ids = jnp.ones(shape=query.shape[:2], dtype=jnp.int32)
    if sinks is not None:
      dummy_attn_mask = jnp.zeros(
          (1, 1, 1, max_target_length, max_target_length),
          dtype=jnp.uint8,
      )
    mask_attention_type = attention_type
    if mask_attention_type == AttentionType.LOCAL_SLIDING:
      mask_attention_type = AttentionType.FULL
    attn_mask = generate_attention_mask(
        query,
        key,
        decoder_segment_ids,
        model_mode,
        attention_type=mask_attention_type,
    )
    attn_mask = jnp.where((attn_mask >= DEFAULT_MASK_VALUE * 0.5), 0, 1).astype(jnp.uint8)

  dpa_layer = DotProductAttention(
      head_dim=head_dim,
      num_attention_heads=num_query_heads,
      num_gqa_groups=num_kv_heads,
      attn_mask_type=mask_type,
      attn_bias_type="no_bias",
      attention_dropout=dropout_rate,
      dropout_rng_name="aqt",
      dtype=dtype,
      float32_logits=float32_logits,
      qkv_layout=qkv_layout,
      scale_factor=1.0,
      transpose_batch_sequence=False,
      window_size=te_sliding_window_size,
      context_parallel_causal_load_balanced=config.context_parallel_load_balance,
      context_parallel_axis=config.context_sharding,
      context_parallel_strategy=config.context_parallel_strategy,
      max_segments_per_seq=max_segments_per_seq,
      softmax_type="learnable" if sinks is not None else "vanilla",
  )

  bridge_rngs = rngs
  if sinks is not None and bridge_rngs is None:
    bridge_rngs = nnx.Rngs(params=0)
  dpa_layer = nnx_wrappers.ToNNX(dpa_layer, rngs=bridge_rngs)
  if sinks is not None:
    dummy_query_prefill = jnp.zeros(
        (1, max_target_length, num_query_heads, config.head_dim),
        dtype=dtype,
    )
    dummy_key_prefill = jnp.zeros(
        (1, max_target_length, num_kv_heads, config.head_dim),
        dtype=dtype,
    )
    dummy_value_prefill = jnp.zeros(
        (1, max_target_length, num_kv_heads, config.head_dim),
        dtype=dtype,
    )
    dpa_layer.lazy_init(
        dummy_query_prefill,
        dummy_key_prefill,
        dummy_value_prefill,
        sequence_descriptor=dummy_attn_mask,
    )
    _inject_te_softmax_offset(dpa_layer, _sinks_to_te_softmax_offset(sinks, num_query_heads))
  return dpa_layer(query, key, value, sequence_descriptor=attn_mask)
