"""Qwen3-Next Gated Delta Network linear-attention layer."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

import functools
import os
from typing import Any

from flax import nnx
import jax
import jax.nn
import jax.numpy as jnp
from lilax._backend.common.common_types import (
    Array,
    Config,
    DType,
    HEAD,
    KV_BATCH,
    KV_HEAD,
    LENGTH,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_TRAIN,
    ShardMode,
    get_quant,
    get_weight_dtype,
)
from lilax._backend.inference import kvcache
from lilax._backend.layers.linears import DenseGeneral
from lilax._backend.layers.normalizations import Qwen3NextRMSNormGated
from lilax._backend.layers.quantization.aqt import AqtQuantization as Quant
from lilax._backend.models.qwen3_next.gated_delta_rule import jax_ar_gated_delta_rule, jax_chunk_gated_delta_rule
from lilax._backend.distributed.sharding import (
    get_logical_axis_rules,
    logical_to_mesh_axes,
    maybe_shard_with_name,
    remove_incompatible_mesh_axes_from_partition_spec,
)


def gdn_context_axes(cfg) -> tuple[str, ...]:
  """Mesh axes carrying the GatedDeltaNet sequence, empty when it is replicated.

  Either context knob can carry it, and both can be live at once, so this
  returns a tuple that `lax` collectives accept directly.
  """
  return tuple(
      name
      for name, size in (
          ("context", cfg.ici_context_parallelism),
          ("context_usp_ulysses", getattr(cfg, "ici_context_usp_ulysses_parallelism", 1)),
      )
      if size > 1
  )


class Qwen3NextGatedDeltaNet(nnx.Module):
  """
  This module implements the full end-to-end logic of a Gated Delta Network layer.

  End-to-End Equations Implemented:
  Let `x` be the input `hidden_states`.

  Step A: Input Projections
  1. (q_raw, k_raw, v_raw, z) = Linear_qkvz(x)
  2. (b, a) = Linear_ba(x)

  Step B: 1D Convolution
  1. qkv_conv = silu(Conv1D(concatenate(q_raw, k_raw, v_raw)))
  2. (q, k, v) = split(qkv_conv)

  Step C: Gated Delta Rule (Recurrent Core)
  1. Gates: β=sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)
  2. Core Calculation: core_attn_out = jax_chunk_gated_delta_rule(q, k, v, g, β)

  Step D: Final Output Stage
  1. y = RMSNorm(core_attn_out) * silu(z)
  2. output = Linear_out(y)
  """

  def __init__(
      self,
      config: Config,
      inputs_shape: tuple | None = None,
      mesh=None,
      dtype: DType = jnp.float32,
      model_mode: str = MODEL_MODE_TRAIN,
      quant: None | Quant = None,
      *,
      rngs: nnx.Rngs,
  ):
    """
    Args:
      config: MaxText configuration object.
      mesh: Optional JAX device mesh (required for vLLM paged-state path).
      quant: Optional quantization configuration.
      rngs: The random number generators for initialization, passed by the nnx.to_linen wrapper.
    """
    self.config = config
    self.mesh = mesh
    self.quant = quant

    self._gdn_replicate_expert = os.environ.get("MAXTEXT_GDN_REPLICATE_EXPERT", "False").lower() == "true"
    cfg = self.config

    in_features = cfg.emb_dim
    self.num_v_heads = cfg.gdn_num_value_heads
    self.num_k_heads = cfg.gdn_num_key_heads
    self.head_k_dim = cfg.gdn_key_head_dim
    self.head_v_dim = cfg.gdn_value_head_dim
    self.key_dim = self.head_k_dim * self.num_k_heads
    self.value_dim = self.head_v_dim * self.num_v_heads
    conv_dim = self.key_dim * 2 + self.value_dim
    conv_kernel_size = cfg.gdn_conv_kernel_dim
    self.v_heads_per_k_head = self.num_v_heads // self.num_k_heads

    if model_mode != MODEL_MODE_TRAIN and inputs_shape is not None:
      runtime_batch_size = inputs_shape[0]

      self.cache = kvcache.KVCache(
          max_prefill_length=cfg.max_prefill_predict_length,
          max_target_length=cfg.max_target_length,
          batch=runtime_batch_size,
          key_seq_len=1,
          value_seq_len=1,
          key_heads=self.num_v_heads,
          value_heads=self.num_v_heads,
          key_head_size=self.head_k_dim,
          value_head_size=self.head_v_dim,
          dtype=dtype,
          is_gdn=True,
          conv_kernel_size=conv_kernel_size,
          conv_dim=conv_dim,
          model_mode=model_mode,
          rngs=rngs,
      )
    else:
      self.cache = None  # No cache for train mode or when inputs_shape not provided

    block_size = getattr(cfg, "weight_block_size", None)

    # Submodule instantiations
    self.in_proj_qkvz = DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=(self.key_dim * 2 + self.value_dim * 2),
        dtype=cfg.dtype,
        weight_dtype=get_weight_dtype(cfg, "in_proj_qkvz"),
        kernel_axes=("embed_attn", "gdn_head"),
        matmul_precision=cfg.matmul_precision,
        shard_mode=cfg.shard_mode,
        block_size=block_size,
        quant=get_quant(cfg, "in_proj_qkvz", self.quant),
        rngs=rngs,
    )
    self.in_proj_ba = DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=(self.num_v_heads * 2),
        dtype=cfg.dtype,
        weight_dtype=get_weight_dtype(cfg, "in_proj_ba"),
        kernel_axes=("embed_attn", "gdn_head"),
        matmul_precision=cfg.matmul_precision,
        shard_mode=cfg.shard_mode,
        quant=get_quant(cfg, "in_proj_ba", self.quant),
        rngs=rngs,
    )

    def conv_kernel_init(key, shape, dtype=jnp.float32):
      sample_dtype = jnp.float32 if jnp.dtype(dtype).itemsize < 2 else dtype
      return jax.nn.initializers.lecun_normal()(key, shape, sample_dtype).astype(dtype)

    self.conv1d = nnx.Conv(
        in_features=conv_dim,
        out_features=conv_dim,
        kernel_size=(conv_kernel_size,),
        feature_group_count=conv_dim,  # Depthwise
        padding="CAUSAL",
        use_bias=False,
        dtype=cfg.dtype,
        param_dtype=get_weight_dtype(cfg, "conv1d"),
        kernel_init=conv_kernel_init,
        precision=cfg.matmul_precision,
        rngs=rngs,
    )

    # Initialize A_log to match torch.log(torch.uniform(0, 16))
    def a_log_init(key, shape, dtype=jnp.float32):
      # Sample from Uniform(epsilon, 16) to avoid log(0)
      a_vals = jax.random.uniform(key, shape=shape, dtype=jnp.float32, minval=1e-9, maxval=16.0)
      return jnp.log(a_vals).astype(dtype)

    self.A_log = nnx.Param(a_log_init(rngs.params(), (self.num_v_heads,), dtype=get_weight_dtype(cfg, "A_log")))
    self.dt_bias = nnx.Param(
        nnx.initializers.ones(rngs.params(), (self.num_v_heads,), dtype=get_weight_dtype(cfg, "dt_bias"))
    )

    self.norm = Qwen3NextRMSNormGated(
        num_features=self.head_v_dim,  # Normalize over the head dimension (D_v)
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=get_weight_dtype(cfg, "norm"),
        shard_mode=cfg.shard_mode,
        rngs=rngs,
    )
    self.out_proj = DenseGeneral(
        in_features_shape=self.value_dim,
        out_features_shape=(in_features,),
        dtype=cfg.dtype,
        weight_dtype=get_weight_dtype(cfg, "out_proj"),
        kernel_axes=("gdn_head", "embed_attn"),
        matmul_precision=cfg.matmul_precision,
        shard_mode=cfg.shard_mode,
        block_size=block_size,
        quant=get_quant(cfg, "out_proj", self.quant),
        rngs=rngs,
    )

  def _explicit_activation_shardings(self, batch: int):
    """Physical layouts for the GatedDeltaNet activations under explicit sharding.

    Explicit sharding cannot infer a layout across the reshapes, the head repeat
    or the
    shard_map boundary in `__call__`, so every intermediate is pinned to the
    logical
    axes the auto path already asks GSPMD for.

    Returns:
      `(flat, head, state)` shardings for `(B, S, H*D)`, `(B, S, H, D)` and
      `(B, H, D_k, D_v)` shaped activations, all `None` under `ShardMode.AUTO`.
    """
    if self.config.shard_mode != ShardMode.EXPLICIT or self.mesh is None:
      return None, None, None

    logical_rules = get_logical_axis_rules()
    # Same sequence axis the shard_map specs below use: replicated unless a
    # context axis carries it.
    cp_len = LENGTH if gdn_context_axes(self.config) else None

    def _sharding(logical_axes):
      pspec = logical_to_mesh_axes(logical_axes, mesh=self.mesh, rules=logical_rules)
      # Training microbatches can be smaller than the physical batch partition.
      # Only dim 0 is inspected, so the trailing sizes are placeholders.
      shape = (batch,) + (1,) * (len(logical_axes) - 1)
      pspec = remove_incompatible_mesh_axes_from_partition_spec(
          pspec, shape, self.mesh, dims=(0,), allow_remove_axes=True
      )
      return jax.sharding.NamedSharding(self.mesh, pspec)

    gdn_head_axis = HEAD if self.config.attention in ("vllm_rpa", "vllm_batched_rpa") else KV_HEAD
    return (
        _sharding((KV_BATCH, cp_len, gdn_head_axis)),
        _sharding((KV_BATCH, cp_len, gdn_head_axis, None)),
        _sharding((KV_BATCH, gdn_head_axis, None, None)),
    )

  @staticmethod
  def _mamba_block_table_slots(
      attention_metadata,
      seq_lens,
      query_start_loc,
      mamba_block_size: int,
      padded_num_reqs_per_dp: int,
      dp_size: int,
      local_rows: int,
  ):
    """Resolves the write and read mamba slots from the mamba block table.

    Used under mamba prefix caching, where vLLM checkpoints the recurrent state
    per block rather than keeping one resident slot per request: a request
    resumes from the block covering its last already-computed token and writes
    to the block covering its last token, so the two differ whenever a step
    crosses a block boundary.

    Args:
      attention_metadata: vLLM attention metadata for this step.
      seq_lens: Per-request total sequence lengths, truncated to the active
        request bucket.
      query_start_loc: Per-request token offsets, truncated to the active
        request bucket.
      mamba_block_size: Tokens per mamba block.
      padded_num_reqs_per_dp: Active request bucket size per DP rank.
      dp_size: Number of DP ranks.
      local_rows: Rows of mamba state resident on one DP rank.

    Returns:
      A `(write_state_indices, read_state_indices)` pair, both rank-local.
    """
    block_tables = attention_metadata.block_tables
    num_reqs_total = attention_metadata.seq_lens.shape[0]
    block_tables = jnp.reshape(block_tables, (num_reqs_total, -1))
    max_num_reqs_per_dp = num_reqs_total // dp_size
    block_tables_sliced = jnp.reshape(
        jnp.reshape(block_tables, (dp_size, max_num_reqs_per_dp, -1))[:, :padded_num_reqs_per_dp, :],
        (-1, block_tables.shape[-1]),
    )

    query_start_loc_reshaped = jnp.reshape(query_start_loc, (dp_size, padded_num_reqs_per_dp + 1))
    query_lens = jnp.reshape(query_start_loc_reshaped[:, 1:] - query_start_loc_reshaped[:, :-1], (-1,))
    num_computed = seq_lens - query_lens

    read_col = jnp.maximum(num_computed - 1, 0) // mamba_block_size
    write_col = jnp.maximum(seq_lens - 1, 0) // mamba_block_size
    batch_idx = jnp.arange(seq_lens.shape[0])

    # The kernel DMAs these ids with bounds checks disabled, so clamp them here
    # to keep an out-of-range id from halting the core.
    write_state_indices = jnp.clip(block_tables_sliced[batch_idx, write_col], 0, local_rows - 1)
    read_state_indices = jnp.clip(block_tables_sliced[batch_idx, read_col], 0, local_rows - 1)
    return write_state_indices.astype(jnp.int32), read_state_indices.astype(jnp.int32)

  def __call__(
      self,
      hidden_states: Array,
      model_mode: str = MODEL_MODE_TRAIN,
      kv_cache=None,
      decoder_segment_ids: None | Array = None,
      attention_metadata=None,
      out_sharding: jax.sharding.NamedSharding | None = None,
      **kwargs,
  ) -> tuple[Array, Any | None]:
    # hidden_states: (B, S, E)
    cfg = self.config
    batch, seq_len, _ = hidden_states.shape
    flat_sharding, head_sharding, state_sharding = self._explicit_activation_shardings(batch)
    gdn_head_axis = HEAD if cfg.attention in ("vllm_rpa", "vllm_batched_rpa") else KV_HEAD

    active_cache = kv_cache if kv_cache is not None else self.cache

    # When kv_cache is a 2-tuple of paged mamba state arrays from vLLM, use
    # run_jax_gdn_attention from tpu_inference for correct sequential token processing.
    # vLLM addresses the recurrent state one of two ways, and only one of them
    # is present at a time: a resident per-request slot in
    # `mamba_state_indices`, or -- under mamba prefix caching, where
    # `mamba_state_indices` is None -- a block id looked up in the mamba block
    # table, which needs `gdn_mamba_block_size`.
    mamba_block_size = cfg.gdn_mamba_block_size
    use_paged_state = (
        kv_cache is not None
        and isinstance(kv_cache, tuple)
        and len(kv_cache) == 2
        and attention_metadata is not None
        and (
            getattr(attention_metadata, "mamba_state_indices", None) is not None
            or (mamba_block_size and getattr(attention_metadata, "block_tables", None) is not None)
        )
        and self.mesh is not None
    )

    # =========================================================================
    # STEP A: Input Projections
    # =========================================================================
    # qkvz: (B, S, 2 * K_dim + 2 * V_dim)
    qkvz = self.in_proj_qkvz(hidden_states, out_sharding=flat_sharding)
    # ba: (B, S, 2 * H_v)
    ba = self.in_proj_ba(hidden_states, out_sharding=flat_sharding)

    # =========================================================================
    # QKVZ and BA Reshaping and Splitting (shared by both paths)
    # =========================================================================
    # Per-K_head group dim: 2 * D_k + 2 * D_v * V_per_K
    new_shape_qkvz = (
        batch,
        seq_len,
        self.num_k_heads,  # H_k
        2 * self.head_k_dim + 2 * self.head_v_dim * self.v_heads_per_k_head,
    )
    # mixed_qkvz: (B, S, H_k, 2*D_k + 2*D_v*V_per_K)
    mixed_qkvz = jnp.reshape(qkvz, new_shape_qkvz, out_sharding=head_sharding)
    if self.mesh is not None:
      logical_rules = get_logical_axis_rules()
      # LENGTH, not None. This with_sharding_constraint told XLA to gather the
      # full sequence onto every device for the qkvz projection. With ctx=2 at
      # seq 32,768 that is six live bf16[2, 32768, 16, 512] buffers of 1.07 GB
      # each -- found by diffing XLA buffer assignment between ctx=1 and ctx=2,
      # and the reason the first version of this patch cut GDN memory but still
      # lost overall.
      cp_len = LENGTH if gdn_context_axes(cfg) else None
      qkvz_pspec = logical_to_mesh_axes((KV_BATCH, cp_len, gdn_head_axis, None), mesh=self.mesh, rules=logical_rules)
      # Training microbatches can be smaller than the physical KV_BATCH mesh partition.
      qkvz_pspec = remove_incompatible_mesh_axes_from_partition_spec(
          qkvz_pspec,
          mixed_qkvz.shape,
          self.mesh,
          dims=(0,),
          allow_remove_axes=True,
      )
      qkvz_sharding = jax.sharding.NamedSharding(self.mesh, qkvz_pspec)
      # Under ShardMode.EXPLICIT `with_sharding_constraint` is an assertion rather than
      # a hint and rejects any layout it has to change, so reshard through the helper.
      mixed_qkvz = maybe_shard_with_name(
          mixed_qkvz,
          qkvz_sharding,
          shard_mode=cfg.shard_mode,
          debug_sharding=cfg.debug_sharding,
      )

    split_indices_qkvz = [
        self.head_k_dim,  # D_k
        2 * self.head_k_dim,  # 2 * D_k
        2 * self.head_k_dim + (self.v_heads_per_k_head * self.head_v_dim),  # 2 * D_k + V_per_K * D_v
    ]
    # query: (B, S, H_k, D_k)
    # key: (B, S, H_k, D_k)
    # value_raw: (B, S, H_k, V_per_K * D_v)
    # z_raw: (B, S, H_k, V_per_K * D_v)
    query, key, value_raw, z_raw = jnp.split(mixed_qkvz, split_indices_qkvz, axis=3)

    # value: (B, S, H_v, D_v)
    value = jnp.reshape(
        value_raw,
        (batch, seq_len, self.num_v_heads, self.head_v_dim),
        out_sharding=head_sharding,
    )
    # z: (B, S, H_v, D_v)
    z = jnp.reshape(
        z_raw,
        (batch, seq_len, self.num_v_heads, self.head_v_dim),
        out_sharding=head_sharding,
    )

    # BA Reshaping and Splitting
    new_shape_ba = (
        batch,
        seq_len,
        self.num_k_heads,  # H_k
        2 * self.v_heads_per_k_head,
    )
    # mixed_ba: (B, S, H_k, 2 * V_per_K)
    mixed_ba = jnp.reshape(ba, new_shape_ba, out_sharding=head_sharding)

    split_indices_ba = [self.v_heads_per_k_head]
    # b_raw: (B, S, H_k, V_per_K)
    # a_raw: (B, S, H_k, V_per_K)
    b_raw, a_raw = jnp.split(mixed_ba, split_indices_ba, axis=3)

    # b: (B, S, H_v)
    b = jnp.reshape(b_raw, (batch, seq_len, self.num_v_heads), out_sharding=flat_sharding)
    # a: (B, S, H_v)
    a = jnp.reshape(a_raw, (batch, seq_len, self.num_v_heads), out_sharding=flat_sharding)

    if use_paged_state:
      # =========================================================================
      # vLLM PAGED STATE PATH: use tpu_inference fused conv + ragged delta-rule.
      # =========================================================================
      try:
        from tpu_inference.layers.common.gdn_attention import run_jax_gdn_attention  # pylint: disable=import-outside-toplevel # pytype: disable=import-error
        from tpu_inference.layers.common.sharding import ShardingAxisName  # pylint: disable=import-outside-toplevel # pytype: disable=import-error
        from tpu_inference.layers.common.utils import (  # pylint: disable=import-outside-toplevel # pytype: disable=import-error
            reorder_concatenated_tensor_for_sharding,
            truncate_sharded_tensor,
        )
        from tpu_inference.utils import get_mesh_shape_product  # pylint: disable=import-outside-toplevel # pytype: disable=import-error
        from jax.sharding import PartitionSpec as P_spec  # pylint: disable=import-outside-toplevel # pytype: disable=import-error
      except ImportError as e:
        raise ImportError(
            "GDN attention kernel require the vllm-tpu package. Please install it with `pip install vllm-tpu`."
        ) from e

      attn_data = ShardingAxisName.ATTN_DATA
      # Head axis for the GDN kernel + the producer-side reshapes. Default ATTN_HEAD
      # (model*expert); the experimental MAXTEXT_GDN_REPLICATE_EXPERT path uses 'model' only
      # so GDN replicates over the expert axis (no expert-axis transpose all-to-all).
      attn_head = ShardingAxisName.MODEL if self._gdn_replicate_expert else ShardingAxisName.ATTN_HEAD
      tp_size = get_mesh_shape_product(self.mesh, attn_head)
      num_tokens = batch * seq_len

      # Build mixed_qkv in the kernel's per-shard layout via shard_map concatenation.
      # Each TP shard already holds its local q/k/v head slices → concatenate locally
      # to get [q_local | k_local | v_local] with no cross-device communication.
      q_flat = query.reshape(num_tokens, self.key_dim)  # (T, key_dim) sharded on ATTN_HEAD
      k_flat = key.reshape(num_tokens, self.key_dim)
      v_flat = value_raw.reshape(num_tokens, self.value_dim)  # (T, value_dim) sharded on ATTN_HEAD
      mixed_qkv = jax.shard_map(
          lambda q, k, v: jnp.concatenate([q, k, v], axis=-1),
          mesh=self.mesh,
          in_specs=(P_spec(attn_data, attn_head),) * 3,
          out_specs=P_spec(attn_data, attn_head),
          check_vma=False,
      )(q_flat, k_flat, v_flat)

      b_flat = b.reshape(num_tokens, self.num_v_heads)
      a_flat = a.reshape(num_tokens, self.num_v_heads)

      # Conv weight: transpose from (kernel_size, 1, conv_dim) → (conv_dim, 1, kernel_size),
      # then reorder so each TP shard gets its local [q_local | k_local | v_local] channels.
      conv_weight = jnp.transpose(self.conv1d.kernel.value, (2, 1, 0))
      conv_weight = reorder_concatenated_tensor_for_sharding(
          conv_weight, [self.key_dim, self.key_dim, self.value_dim], tp_size, 0
      )

      conv_state_paged, recurrent_state_paged = kv_cache

      # Compile against the active request bucket rather than the runner's
      # maximum-size metadata buffers.
      dp_size = get_mesh_shape_product(self.mesh, attn_data)
      padded_num_reqs_per_dp = attention_metadata.padded_num_reqs // dp_size  # pyrefly: ignore[missing-attribute]
      query_start_loc = truncate_sharded_tensor(
          attention_metadata.query_start_loc,  # pyrefly: ignore[missing-attribute]
          padded_num_reqs_per_dp + 1,
          dp_size,
      )
      seq_lens = truncate_sharded_tensor(
          attention_metadata.seq_lens,  # pyrefly: ignore[missing-attribute]
          padded_num_reqs_per_dp,
          dp_size,
      )

      if getattr(attention_metadata, "mamba_state_indices", None) is not None:
        # Resident per-request slot: the same slot is read and written.
        state_indices = truncate_sharded_tensor(
            attention_metadata.mamba_state_indices.astype(jnp.int32),  # pyrefly: ignore[missing-attribute]
            padded_num_reqs_per_dp,
            dp_size,
        )
        read_state_indices = state_indices
      else:
        state_indices, read_state_indices = self._mamba_block_table_slots(
            attention_metadata,
            seq_lens,
            query_start_loc,
            mamba_block_size,
            padded_num_reqs_per_dp,
            dp_size,
            local_rows=max(conv_state_paged.shape[0] // dp_size, 1),
        )

      (new_conv_state_paged, new_recurrent_state_paged), gdn_output = run_jax_gdn_attention(
          mixed_qkv,
          b_flat,
          a_flat,
          conv_state_paged,
          recurrent_state_paged,
          conv_weight,
          None,  # conv_bias: MaxText conv1d uses use_bias=False.
          jnp.asarray(self.A_log[...], dtype=cfg.dtype),
          jnp.asarray(self.dt_bias[...], dtype=cfg.dtype),
          state_indices,
          query_start_loc,
          attention_metadata.request_distribution,  # pyrefly: ignore[missing-attribute]
          seq_lens,
          self.num_k_heads,
          self.num_v_heads,
          self.head_k_dim,
          self.head_v_dim,
          cfg.gdn_conv_kernel_dim,
          mesh=self.mesh,
          read_state_indices=read_state_indices,
      )

      # Reshape GDN output and apply gated norm + out projection.
      gdn_output = gdn_output.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)
      gated_output = self.norm(gdn_output, z)
      gated_output = gated_output.reshape(batch, seq_len, -1)
      output = self.out_proj(gated_output)

      return output, (new_conv_state_paged, new_recurrent_state_paged)

    # Flatten head dimensions for concatenation before conv
    # q: (B, S, K_dim)
    q = jnp.reshape(query, (batch, seq_len, -1), out_sharding=flat_sharding)
    # k: (B, S, K_dim)
    k = jnp.reshape(key, (batch, seq_len, -1), out_sharding=flat_sharding)
    # v: (B, S, V_dim)
    v = jnp.reshape(value, (batch, seq_len, -1), out_sharding=flat_sharding)

    # =========================================================================
    # STEP B: 1D Convolution
    # =========================================================================
    qkv = jnp.concatenate([q, k, v], axis=-1)
    batch, seq_len, _ = qkv.shape
    conv_kernel_size = self.config.gdn_conv_kernel_dim

    conv_state = None
    recurrent_state = None
    next_conv_state = None
    if model_mode != MODEL_MODE_TRAIN and active_cache is not None:
      recurrent_state, conv_state = active_cache.get_gdn_states()
      orig_cache_batch = conv_state.shape[0]

      # 1. Safely shrink/expand conv_state to match incoming qkv (e.g. 16 -> 1)
      if conv_state.shape[0] != batch:
        if conv_state.shape[0] == 1:
          conv_state = jnp.broadcast_to(conv_state, (batch,) + conv_state.shape[1:])
        elif conv_state.shape[0] < batch:
          pad_amt = batch - conv_state.shape[0]
          conv_state = jnp.pad(conv_state, ((0, pad_amt), (0, 0), (0, 0)))
        else:
          conv_state = conv_state[:batch]

      # 2. Safely shrink/expand recurrent_state to match incoming qkv
      if recurrent_state.shape[0] != batch:
        if recurrent_state.shape[0] == 1:
          recurrent_state = jnp.broadcast_to(recurrent_state, (batch,) + recurrent_state.shape[1:])
        elif recurrent_state.shape[0] < batch:
          pad_amt = batch - recurrent_state.shape[0]
          recurrent_state = jnp.pad(recurrent_state, ((0, pad_amt), (0, 0), (0, 0), (0, 0)))
        else:
          recurrent_state = recurrent_state[:batch]

      conv_input = jnp.concatenate([conv_state, qkv], axis=1)

      if decoder_segment_ids is not None:
        valid_lens = jnp.sum(decoder_segment_ids != 0, axis=1)

        def extract_state(c_in, v_len):
          return jax.lax.dynamic_slice_in_dim(c_in, v_len, conv_kernel_size - 1, axis=0)

        next_conv_state = jax.vmap(extract_state)(conv_input, valid_lens)
      else:
        next_conv_state = conv_input[:, -(conv_kernel_size - 1) :, :]
    else:
      conv_input = jnp.pad(qkv, ((0, 0), (conv_kernel_size - 1, 0), (0, 0)))

    # Perform the convolution.
    conv_out = self.conv1d(conv_input, out_sharding=flat_sharding)
    # Slice the output to match the original input sequence length.
    conv_out = conv_out[:, -seq_len:, :]
    qkv_conv = jax.nn.silu(conv_out.astype(jnp.float32)).astype(cfg.dtype)
    # q_conv shape: (B, S, key_dim), k_conv shape: (B, S, key_dim), v_conv shape: (B, S, value_dim)
    q_conv, k_conv, v_conv = jnp.split(qkv_conv, [self.key_dim, 2 * self.key_dim], axis=-1)

    # Reshape for multi-head processing
    # query shape: (B, S, H_k, D_k)
    query = jnp.reshape(
        q_conv,
        (batch, seq_len, self.num_k_heads, self.head_k_dim),
        out_sharding=head_sharding,
    )
    # key shape: (B, S, H_k, D_k)
    key = jnp.reshape(
        k_conv,
        (batch, seq_len, self.num_k_heads, self.head_k_dim),
        out_sharding=head_sharding,
    )
    # value shape: (B, S, H_v, D_v)
    value = jnp.reshape(
        v_conv,
        (batch, seq_len, self.num_v_heads, self.head_v_dim),
        out_sharding=head_sharding,
    )

    # =========================================================================
    # STEP C: Gated Delta Rule Recurrence
    # =========================================================================
    A_log = jnp.asarray(self.A_log[...], dtype=cfg.dtype)
    dt_bias = jnp.asarray(self.dt_bias[...], dtype=cfg.dtype)
    if cfg.shard_mode == ShardMode.EXPLICIT:
      # Both are stored replicated but broadcast against (B, S, H_v) activations whose
      # head axis is sharded, and explicit sharding requires broadcast operands to
      # agree -- the same fix `align_scale_with_normalized_axis` applies to the norm
      # scales.
      head_spec = jax.sharding.PartitionSpec(jax.typeof(a).sharding.spec[-1])
      A_log = jax.sharding.reshard(A_log, head_spec)
      dt_bias = jax.sharding.reshard(dt_bias, head_spec)
    # beta shape: (B, S, H_v)
    beta = jax.nn.sigmoid(b)
    # g shape: (B, S, H_v)
    g = -jnp.exp(A_log) * jax.nn.softplus(a + dt_bias)

    if decoder_segment_ids is not None:
      mask = decoder_segment_ids != 0
      # Apply mask by broadcasting to respective shapes
      key = jnp.where(mask[..., None, None], key, 0.0)
      value = jnp.where(mask[..., None, None], value, 0.0)
      g = jnp.where(mask[..., None], g, 0.0)

    if self.num_v_heads > self.num_k_heads and self.num_v_heads % self.num_k_heads == 0:
      repeats = self.num_v_heads // self.num_k_heads
      # query shape after repeat: (B, S, H_v, D_k)
      query = jnp.repeat(query, repeats, axis=2, out_sharding=head_sharding)
      # key shape after repeat: (B, S, H_v, D_k)
      key = jnp.repeat(key, repeats, axis=2, out_sharding=head_sharding)

    if seq_len == 1 and model_mode == MODEL_MODE_AUTOREGRESSIVE:
      core_attn_out, next_recurrent_state = jax_ar_gated_delta_rule(
          query,
          key,
          value,
          g,
          beta,
          initial_state=recurrent_state,  # pyrefly: ignore[bad-argument-type]
          use_qk_norm_in_gdn=cfg.use_qk_norm_in_gdn,
          compute_dtype=cfg.dtype,
      )
    elif self.mesh is not None:
      logical_rules = get_logical_axis_rules()
      recurrent_state_arg = (
          recurrent_state
          if recurrent_state is not None
          else jnp.zeros(
              (batch, self.num_v_heads, self.head_k_dim, self.head_v_dim),
              dtype=cfg.dtype,
              out_sharding=state_sharding,
          )
      )
      # LENGTH, not None. The sequence axis was hardcoded to replicated, so
      # ici_context_parallelism could never shard the GDN sequence while still
      # consuming the context axis from the mesh -- which is why raising ctx
      # made memory worse instead of better. The scan handles a sharded
      # sequence via the two-pass affine composition in kernels/attention/gdn_cp.py.
      # Either context axis can carry the sequence. LENGTH maps to both in the
      # logical rules, so this is correct whichever one is configured.
      cp_axes = gdn_context_axes(cfg)
      cp_len = LENGTH if cp_axes else None
      qkv_pspec = logical_to_mesh_axes((KV_BATCH, cp_len, gdn_head_axis, None), mesh=self.mesh, rules=logical_rules)
      g_beta_pspec = logical_to_mesh_axes((KV_BATCH, cp_len, gdn_head_axis), mesh=self.mesh, rules=logical_rules)
      state_pspec = logical_to_mesh_axes((KV_BATCH, gdn_head_axis, None, None), mesh=self.mesh, rules=logical_rules)
      # Keep every shard_map input/output batch spec consistent when replication is required.
      qkv_pspec = remove_incompatible_mesh_axes_from_partition_spec(
          qkv_pspec,
          query.shape,
          self.mesh,
          dims=(0,),
          allow_remove_axes=True,
      )
      g_beta_pspec = remove_incompatible_mesh_axes_from_partition_spec(
          g_beta_pspec,
          g.shape,
          self.mesh,
          dims=(0,),
          allow_remove_axes=True,
      )
      state_pspec = remove_incompatible_mesh_axes_from_partition_spec(
          state_pspec,
          recurrent_state_arg.shape,
          self.mesh,
          dims=(0,),
          allow_remove_axes=True,
      )

      if cfg.shard_mode == ShardMode.EXPLICIT:
        # shard_map manualises the mesh axes it is given and will not insert a reshard
        # for an operand whose layout differs from `in_specs`, so hand it arrays that
        # already match.
        query = jax.sharding.reshard(query, qkv_pspec)
        key = jax.sharding.reshard(key, qkv_pspec)
        value = jax.sharding.reshard(value, qkv_pspec)
        g = jax.sharding.reshard(g, g_beta_pspec)
        beta = jax.sharding.reshard(beta, g_beta_pspec)
        recurrent_state_arg = jax.sharding.reshard(recurrent_state_arg, state_pspec)

      @functools.partial(
          jax.shard_map,
          mesh=self.mesh,
          in_specs=(
              qkv_pspec,  # query
              qkv_pspec,  # key
              qkv_pspec,  # value
              g_beta_pspec,  # g
              g_beta_pspec,  # beta
              state_pspec,  # initial_state
          ),
          out_specs=(
              qkv_pspec,  # core_attn_out
              state_pspec,  # final_state
          ),
          check_vma=False,
      )
      def shard_mapped_delta_rule(q, k, v, g_val, beta_val, init_h):
        return jax_chunk_gated_delta_rule(
            query=q,
            key=k,
            value=v,
            g=g_val,
            beta=beta_val,
            chunk_size=cfg.gdn_chunk_size,
            initial_state=init_h,
            use_qk_norm_in_gdn=cfg.use_qk_norm_in_gdn,
            compute_dtype=cfg.dtype,
            cp_axis=cp_axes or None,
        )

      core_attn_out, next_recurrent_state = shard_mapped_delta_rule(query, key, value, g, beta, recurrent_state_arg)
    else:
      core_attn_out, next_recurrent_state = jax_chunk_gated_delta_rule(
          query,
          key,
          value,
          g,
          beta,
          chunk_size=cfg.gdn_chunk_size,
          initial_state=recurrent_state,
          use_qk_norm_in_gdn=cfg.use_qk_norm_in_gdn,
          compute_dtype=cfg.dtype,
      )

    if model_mode != MODEL_MODE_TRAIN and active_cache is not None:
      assert next_conv_state is not None
      assert next_recurrent_state is not None
      if next_conv_state.shape[0] != orig_cache_batch:  # pyrefly: ignore[unbound-name]
        if next_conv_state.shape[0] == 1:
          next_conv_state = jnp.broadcast_to(next_conv_state, (orig_cache_batch,) + next_conv_state.shape[1:])
          next_recurrent_state = jnp.broadcast_to(
              next_recurrent_state, (orig_cache_batch,) + next_recurrent_state.shape[1:]
          )
        elif next_conv_state.shape[0] < orig_cache_batch:
          pad_amt = orig_cache_batch - next_conv_state.shape[0]
          next_conv_state = jnp.pad(next_conv_state, ((0, pad_amt), (0, 0), (0, 0)))
          next_recurrent_state = jnp.pad(next_recurrent_state, ((0, pad_amt), (0, 0), (0, 0), (0, 0)))
        else:
          next_conv_state = next_conv_state[:orig_cache_batch]
          next_recurrent_state = next_recurrent_state[:orig_cache_batch]

    next_recurrent_state = next_recurrent_state.astype(cfg.dtype)
    if model_mode != MODEL_MODE_TRAIN and active_cache is not None:
      active_cache.update_gdn_states(next_recurrent_state, next_conv_state)  # pyrefly: ignore[bad-argument-type]

    # =========================================================================
    # STEP D: Final Output Stage
    # =========================================================================

    # The normalization and gating is applied per-head on the value dimension.

    # Apply the norm and gate. Output shape: (B, S, H_v, D_v)
    gated_output_reshaped = self.norm(core_attn_out, z, out_sharding=head_sharding)

    # Reshape back to a single feature dimension for the final projection.
    # Shape from (B, S, H_v, D_v) -> (B, S, value_dim)
    gated_output = jnp.reshape(gated_output_reshaped, (batch, seq_len, -1), out_sharding=flat_sharding)

    # Final output shape: (B, S, E)
    output = self.out_proj(gated_output, out_sharding=out_sharding)

    return output, active_cache

  def init_kv_caches(self, batch_size: int):
    """Initializes KVCache dynamically using the traced runtime batch size."""
    cfg = self.config
    conv_dim = self.key_dim * 2 + self.value_dim
    conv_kernel_size = cfg.gdn_conv_kernel_dim

    return kvcache.KVCache(
        max_prefill_length=cfg.max_prefill_predict_length,
        max_target_length=cfg.max_target_length,
        batch=batch_size,
        key_seq_len=1,
        value_seq_len=1,
        key_heads=self.num_v_heads,
        value_heads=self.num_v_heads,
        key_head_size=self.head_k_dim,
        value_head_size=self.head_v_dim,
        dtype=self.dtype,  # pyrefly: ignore[missing-attribute]
        is_gdn=True,
        conv_kernel_size=conv_kernel_size,
        conv_dim=conv_dim,
        model_mode=self.model_mode,  # pyrefly: ignore[missing-attribute]
        rngs=self.rngs,  # pyrefly: ignore[missing-attribute]
    )
