"""Single DeepSeekV3 dense layer."""

from collections.abc import Mapping
import functools
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

from lilax._backend.experimental.lineage import dsv3_mla
from lilax._backend.experimental.lineage import dsv3_types
from lilax._backend.experimental.lineage import quantization
from lilax._backend.experimental.lineage.sharding import axis_collectives


class NormFn(Protocol):

  def __call__(
      self, x: jt.Num[jax.Array, "*input_dims"], scale: jt.Num[jax.Array, "..."] | None = None,
  ) -> jt.Num[jax.Array, "*input_dims"]:
    ...


class RopeFn(Protocol):

  def __call__(
      self, x: jt.Num[jax.Array, "*BT N D"],
      freqs: tuple[jt.Num[jax.Array, "*BT 1 D"], jt.Num[jax.Array, "*BT 1 D"]],
  ) -> jt.Num[jax.Array, "*BT N D"]:
    ...


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_mlp(x: jt.Num[jax.Array, "*BT D"], w: dsv3_types.DSv3MLPWeightsPytree) -> jt.Num[jax.Array, "*BT D"]:
  """Performs computation for an MLP."""
  assert w.gate_0 is not None
  assert w.gate_1 is not None
  assert w.linear is not None
  dot = functools.partial(jnp.tensordot, axes=1)
  with jax.named_scope("gate_0"):
    g0 = dot(x, w.gate_0)
  with jax.named_scope("gate_1"):
    g1 = dot(x, w.gate_1)
  with jax.named_scope("silu"):
    act = jax.nn.silu(g0) * g1
  with jax.named_scope("linear"):
    return dot(act, w.linear)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_dense_layer(
    x: jt.Num[jax.Array, "B T D"],
    w: dsv3_types.DSv3DenseLayerWeightsPytree,
    yarn_freqs: tuple[jt.Num[jax.Array, "B T 1 R"], jt.Num[jax.Array, "B T 1 R"]],
    splash_kernel: Any,
    *,
    qk_head_dim: int,
    rope_head_dim: int,
    num_query_heads: int,
    mscale: float,
    kv_lora_rank: int,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    rope_factor: int,
    mesh: jax.sharding.Mesh,
    norm_fn: NormFn,
    rope_fn: RopeFn,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    segment_ids: jt.Num[jax.Array, "B T"] | None = None,
    quant: quantization.QuantConfig | None = None,
) -> jt.Num[jax.Array, "B T D"]:
  """Executes a single DSv3 dense layer."""
  orig_x = x
  with jax.named_scope("pre_attn_norm"):
    x = norm_fn(x, w.pre_attn_norm_scale)
  x = dsv3_mla.dsv3_mla(
      x, yarn_freqs, splash_kernel, w.mla, segment_ids=segment_ids, kv_lora_rank=kv_lora_rank,
      qk_head_dim=qk_head_dim, rope_head_dim=rope_head_dim, num_query_heads=num_query_heads,
      max_position_embeddings=max_position_embeddings, original_max_position_embeddings=original_max_position_embeddings,
      rope_factor=rope_factor, mscale=mscale, norm_fn=norm_fn, rope_fn=rope_fn, mesh=mesh,
      axis_mapping=axis_mapping, quant_rule=quant.mla if quant else None,
  )
  attn_out = x + orig_x
  with jax.named_scope("post_attn_norm"):
    x = norm_fn(attn_out, w.post_attn_norm_scale)
  mlp_out = dsv3_mlp(x, w.mlp)
  return attn_out + mlp_out


@jt.jaxtyped(typechecker=typeguard.typechecked)
def collect_w_dcn(
    w: dsv3_types.DSv3DenseLayerWeightsPytree, *, axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> dsv3_types.DSv3DenseLayerWeightsPytree:
  """Collects dense layer weights across the DCN mesh axis."""
  return axis_collectives.collect_along_axis(w, "dcn", axis_mapping)


def _w_ici_axes() -> dsv3_types.DSv3DenseLayerWeightsPytree:
  """Returns the ICI axes each dense layer weight is collected along."""
  return dsv3_types.DSv3DenseLayerWeightsPytree(
      pre_attn_norm_scale=("fsdp_attention", "attention"),
      mla=dsv3_types.DSv3MLAWeightsPytree(
          q_down=("fsdp_attention", "attention"), q_up=("fsdp_attention",), q_norm_scale=("fsdp_attention",),
          kv_down=("fsdp_attention", "attention"), k_up=("fsdp_attention",), v_up=("fsdp_attention",),
          kv_norm_scale=("fsdp_attention",), out=("fsdp_attention",),
      ),
      post_attn_norm_scale=("fsdp_attention", "attention"),
      mlp=dsv3_types.DSv3MLPWeightsPytree(
          gate_0=("fsdp_attention", "attention"), gate_1=("fsdp_attention", "attention"),
          linear=("fsdp_attention", "attention"),
      ),
  )


@jt.jaxtyped(typechecker=typeguard.typechecked)
def collect_w_ici(
    w: dsv3_types.DSv3DenseLayerWeightsPytree, *, axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> dsv3_types.DSv3DenseLayerWeightsPytree:
  """Collects dense layer weights across ICI mesh axes on SparseCore 1."""
  return axis_collectives.collect_along_axis(
      w, _w_ici_axes(), axis_mapping, compute_type="tpu_sparsecore",
      compiler_options={"sparse_core_config": {"core_ids": [1]}},
  )


@jt.jaxtyped(typechecker=typeguard.typechecked)
def reduce_w_ici(
    grad_w: dsv3_types.DSv3DenseLayerWeightsPytree,
    like: dsv3_types.DSv3DenseLayerWeightsPytree,
    *,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> dsv3_types.DSv3DenseLayerWeightsPytree:
  """Reduces dense layer weight gradients across ICI mesh axes on SparseCore 0."""
  return axis_collectives.reduce_along_axis(
      grad_w, _w_ici_axes(), axis_mapping, like, compute_type="tpu_sparsecore",
      compiler_options={"sparse_core_config": {"core_ids": [0]}},
  )
