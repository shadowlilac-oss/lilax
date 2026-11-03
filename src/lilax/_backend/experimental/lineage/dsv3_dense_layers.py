"""DeepSeekV3 scan over multiple dense layers."""

from collections.abc import Mapping
import functools
from typing import Any, Protocol

import jax
import jaxtyping as jt
import typeguard

from lilax._backend.experimental.lineage import dsv3_dense_layer
from lilax._backend.experimental.lineage import dsv3_types
from lilax._backend.experimental.lineage import quantization
from lilax._backend.experimental.lineage.scan import weight_scan


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


@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_dense_layers(
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
  """Executes dense layers with pipelined weight collection."""
  _collect_w_dcn = functools.partial(dsv3_dense_layer.collect_w_dcn, axis_mapping=axis_mapping)
  _collect_w_ici = functools.partial(dsv3_dense_layer.collect_w_ici, axis_mapping=axis_mapping)
  _reduce_w_ici = functools.partial(dsv3_dense_layer.reduce_w_ici, axis_mapping=axis_mapping)
  _dsv3_dense_layer = functools.partial(
      dsv3_dense_layer.dsv3_dense_layer, yarn_freqs=yarn_freqs, splash_kernel=splash_kernel,
      qk_head_dim=qk_head_dim, rope_head_dim=rope_head_dim, num_query_heads=num_query_heads, mscale=mscale,
      kv_lora_rank=kv_lora_rank, max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings, rope_factor=rope_factor,
      mesh=mesh, norm_fn=norm_fn, rope_fn=rope_fn, axis_mapping=axis_mapping, segment_ids=segment_ids, quant=quant,
  )

  def _scan_fn(x, w):
    return _dsv3_dense_layer(x, w), ()

  x, _ = weight_scan.w_collect_pipelined_scan(_scan_fn, x, w, _collect_w_dcn, _collect_w_ici, _reduce_w_ici)
  return x
