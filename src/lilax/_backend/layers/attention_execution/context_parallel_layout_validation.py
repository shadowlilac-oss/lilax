"""Validates the layer configuration and mesh layout of TPU context-parallel flash attention strategies."""

from typing import Callable

from jax.sharding import Mesh

from lilax._backend.common.common_types import AttentionType, AxisNames, Config
from lilax._backend.kernels.attention import tokamax_ring_attention
from lilax._backend.kernels.attention import ulysses_attention
from lilax._backend.kernels.attention import usp_attention


def validate_tokamax_ring_layout(
    config: Config,
    mesh: Mesh,
    attention_type: AttentionType,
    num_query_heads: int,
    num_kv_heads: int,
    flash_axis_names_q: AxisNames,
    flash_axis_names_kv: AxisNames,
    logical_to_mesh_axes: Callable,
) -> None:
  """Validates the Tokamax ring strategy, which only constrains TPU targets."""
  target_hardware = mesh.devices[(0,) * mesh.devices.ndim].platform
  if target_hardware == "tpu":
    if not config.use_tokamax_splash:
      raise ValueError("TPU Tokamax ring attention requires use_tokamax_splash=True.")
    if config.use_jax_splash:
      raise ValueError("TPU Tokamax ring attention requires use_jax_splash=False.")
    if attention_type not in (AttentionType.GLOBAL, AttentionType.MLA):
      raise ValueError("TPU Tokamax ring attention supports only attention_type='global' or 'mla'.")
    if attention_type == AttentionType.MLA:
      if config.packing:
        raise ValueError("TPU Tokamax ring attention with MLA does not support packing yet.")
      if config.use_batch_split_schedule:
        raise ValueError("TPU Tokamax ring attention with MLA does not support the DeepSeek batch-split schedule.")

    context_axis = config.context_sharding
    axis_names_q = logical_to_mesh_axes(flash_axis_names_q)
    axis_names_kv = logical_to_mesh_axes(flash_axis_names_kv)
    axis_names_kv = tokamax_ring_attention.with_sequence_axis(
        axis_names_kv,
        context_axis,
        sequence_dim=2,
    )
    tokamax_ring_attention.validate_ring_mesh_axis(
        axis_names_q=axis_names_q,
        axis_names_kv=axis_names_kv,
        sequence_dim_q=2,
        sequence_dim_kv=2,
        mesh=mesh,
        ring_axis=context_axis,
    )
    tokamax_ring_attention.validate_head_sharding(
        axis_names_q=axis_names_q,
        axis_names_kv=axis_names_kv,
        mesh=mesh,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim_q=1,
        head_dim_kv=1,
    )
    tokamax_ring_attention.validate_dkv_sharding(
        axis_names_q=axis_names_q,
        axis_names_kv=axis_names_kv,
        dkv_dim_q=3,
        dkv_dim_kv=3,
    )


def validate_ulysses_layout(
    config: Config,
    mesh: Mesh,
    attention_type: AttentionType,
    num_query_heads: int,
    num_kv_heads: int,
    dropout_rate: float,
    use_ragged_attention: bool,
    flash_axis_names_q: AxisNames,
    flash_axis_names_kv: AxisNames,
    logical_to_mesh_axes: Callable,
) -> None:
  """Validates the Ulysses strategy, which is TPU-only."""
  target_hardware = mesh.devices[(0,) * mesh.devices.ndim].platform
  if target_hardware != "tpu":
    raise ValueError("Ulysses context parallelism (context_parallel_strategy='ulysses') is only supported on TPU.")
  if not config.use_tokamax_splash:
    raise ValueError("TPU Ulysses attention requires use_tokamax_splash=True.")
  if config.use_jax_splash:
    raise ValueError("TPU Ulysses attention requires use_jax_splash=False.")
  if attention_type != AttentionType.GLOBAL:
    raise ValueError("TPU Ulysses attention is initially supported only for global causal attention.")
  if config.enable_dropout and dropout_rate > 0.0:
    raise ValueError("TPU Ulysses attention does not support dropout yet.")
  if use_ragged_attention:
    raise ValueError("TPU Ulysses attention does not support ragged attention.")

  context_axis = config.context_sharding
  axis_names_q = logical_to_mesh_axes(flash_axis_names_q)
  axis_names_kv = logical_to_mesh_axes(flash_axis_names_kv)
  axis_names_kv = ulysses_attention.with_sequence_axis(
      axis_names_kv,
      context_axis,
      sequence_dim=2,
  )
  ulysses_attention.validate_ulysses_mesh_axis(
      axis_names_q=axis_names_q,
      axis_names_kv=axis_names_kv,
      sequence_dim_q=2,
      sequence_dim_kv=2,
      mesh=mesh,
      ulysses_axis=context_axis,
  )
  if mesh.shape[context_axis] <= 1:
    raise ValueError("TPU Ulysses attention requires a context parallel mesh axis larger than one.")
  ulysses_attention.validate_head_sharding(
      axis_names_q=axis_names_q,
      axis_names_kv=axis_names_kv,
      mesh=mesh,
      num_query_heads=num_query_heads,
      num_kv_heads=num_kv_heads,
      head_dim_q=1,
      head_dim_kv=1,
      ulysses_size=mesh.shape[context_axis],
      attention_label="TPU Ulysses attention",
  )
  ulysses_attention.validate_dkv_sharding(
      axis_names_q=axis_names_q,
      axis_names_kv=axis_names_kv,
      dkv_dim_q=3,
      dkv_dim_kv=3,
      attention_label="TPU Ulysses attention",
  )


def validate_usp_layout(
    config: Config,
    mesh: Mesh,
    attention_type: AttentionType,
    num_query_heads: int,
    num_kv_heads: int,
    dropout_rate: float,
    use_ragged_attention: bool,
    flash_axis_names_q: AxisNames,
    flash_axis_names_kv: AxisNames,
    logical_to_mesh_axes: Callable,
) -> None:
  """Validates the USP (ring x Ulysses) strategy, which is TPU-only."""
  target_hardware = mesh.devices[(0,) * mesh.devices.ndim].platform
  if target_hardware != "tpu":
    raise ValueError("USP context parallelism (context_parallel_strategy='usp') is only supported on TPU.")
  if not config.use_tokamax_splash:
    raise ValueError("TPU USP attention requires use_tokamax_splash=True.")
  if config.use_jax_splash:
    raise ValueError("TPU USP attention requires use_jax_splash=False.")
  if attention_type != AttentionType.GLOBAL:
    raise ValueError("TPU USP attention is initially supported only for global causal attention.")
  if config.enable_dropout and dropout_rate > 0.0:
    raise ValueError("TPU USP attention does not support dropout yet.")
  if use_ragged_attention:
    raise ValueError("TPU USP attention does not support ragged attention.")

  ring_axis = config.context_sharding
  ulysses_axis = config.ulysses_context_sharding
  axis_names_q = logical_to_mesh_axes(flash_axis_names_q)
  axis_names_kv = logical_to_mesh_axes(flash_axis_names_kv)
  axis_names_kv = usp_attention.with_usp_sequence_axes(
      axis_names_kv,
      ring_axis,
      ulysses_axis,
      sequence_dim=2,
  )
  usp_attention.validate_usp_mesh_axes(
      axis_names_q=axis_names_q,
      axis_names_kv=axis_names_kv,
      sequence_dim_q=2,
      sequence_dim_kv=2,
      mesh=mesh,
      ring_axis=ring_axis,
      ulysses_axis=ulysses_axis,
  )
  if mesh.shape[ring_axis] <= 1:
    raise ValueError("TPU USP attention requires a context parallel mesh axis larger than one.")
  if mesh.shape[ulysses_axis] <= 1:
    raise ValueError("TPU USP attention requires a context ulysses parallel mesh axis larger than one.")
  ulysses_attention.validate_head_sharding(
      axis_names_q=axis_names_q,
      axis_names_kv=axis_names_kv,
      mesh=mesh,
      num_query_heads=num_query_heads,
      num_kv_heads=num_kv_heads,
      head_dim_q=1,
      head_dim_kv=1,
      ulysses_size=mesh.shape[ulysses_axis],
      attention_label="TPU USP attention",
  )
  ulysses_attention.validate_dkv_sharding(
      axis_names_q=axis_names_q,
      axis_names_kv=axis_names_kv,
      dkv_dim_q=3,
      dkv_dim_kv=3,
      attention_label="TPU USP attention",
  )
