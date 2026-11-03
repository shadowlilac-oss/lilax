"""DeepSeekV3 scan over multiple sparse layers."""

from collections.abc import Callable, Mapping
import functools
from typing import Any, Protocol

import jax
import jaxtyping as jt
import typeguard

from lilax._backend.experimental.lineage import dsv3_expert_shuffle
from lilax._backend.experimental.lineage import dsv3_sparse_layer
from lilax._backend.experimental.lineage import dsv3_types
from lilax._backend.experimental.lineage import ops
from lilax._backend.experimental.lineage import quantization
from lilax._backend.experimental.lineage.scan import weight_scan_prologue_epilogue


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


GmmFn = dsv3_sparse_layer.GmmFn


def _with_expert_permutations(fn: Callable[..., Any], *, accepts: bool = True) -> Callable[..., Any]:
  """Adapts a collect or layer function to the scan's per-layer w_aux."""
  @functools.wraps(fn)
  def wrapped(*args, w_aux=None, **kwargs):
    if accepts:
      kwargs["expert_permutations"] = w_aux
    return fn(*args, **kwargs)
  return wrapped


@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_sparse_layers(
    x: jt.Num[jax.Array, "B T D"],
    w: dsv3_types.DSv3SparseLayerWeightsPytree,
    yarn_freqs: tuple[jt.Num[jax.Array, "B T 1 R"], jt.Num[jax.Array, "B T 1 R"]],
    splash_kernel: Any,
    *,
    num_experts: int,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    n_routing_groups: int,
    topk_routing_group: int,
    topk_in_group: int,
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
    gmm_fn: GmmFn,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    expert_axis_name: str = "expert",
    capacity_factor: float = 2.0,
    router_dtype: jax.typing.DTypeLike | None = None,
    segment_ids: jt.Num[jax.Array, "B T"] | None = None,
    ragged_buffer_factor: float,
    quant: quantization.QuantConfig | None = None,
    expert_permutations: jt.Int[jax.Array, "L E"] | None = None,
) -> tuple[jt.Num[jax.Array, "B T D"], dsv3_types.DSv3RouterAux[jax.Array]]:
  """Executes sparse layers with pipelined microbatches and weight collection."""
  num_layers = jax.tree.leaves(w)[0].shape[0]
  if expert_permutations is not None:
    dsv3_expert_shuffle.validate_expert_permutations(expert_permutations, num_experts=num_experts, num_layers=num_layers)
    if expert_permutations.shape[0] != num_layers:
      raise ValueError(f"expert_permutations has {expert_permutations.shape[0]} rows," f" expected {num_layers}")
  mb0, mb1 = ops.split_microbatches(x, mesh=mesh)
  yarn_freqs_0, yarn_freqs_1 = ops.split_microbatches(yarn_freqs, mesh=mesh)
  segment_ids_0, segment_ids_1 = ops.split_microbatches(segment_ids, mesh=mesh)
  _collect_w_dcn = functools.partial(dsv3_sparse_layer.collect_w_dcn, axis_mapping=axis_mapping)
  _collect_w_ici_fwd = functools.partial(
      dsv3_sparse_layer.collect_w_ici, axis_mapping=axis_mapping, split=True, expert_axis_name=expert_axis_name,
  )
  _collect_w_ici_bwd = functools.partial(
      dsv3_sparse_layer.collect_w_ici, axis_mapping=axis_mapping, split=False, expert_axis_name=expert_axis_name,
  )
  w_first = _collect_w_ici_fwd(_collect_w_dcn(jax.tree.map(lambda _w: _w[0], w)))
  (bank_0, bank_offset_0), (bank_1, bank_offset_1) = dsv3_sparse_layer.init_activation_bank(
      mb0, w_first, num_layers=num_layers, num_experts_per_tok=num_experts_per_tok,
      ragged_buffer_factor=ragged_buffer_factor, capacity_factor=capacity_factor,
      expert_axis_name=expert_axis_name, axis_mapping=axis_mapping, mesh=mesh,
  )
  attention_kwargs = dict(
      yarn_freqs=(yarn_freqs_0, yarn_freqs_1), splash_kernel=splash_kernel,
      qk_head_dim=qk_head_dim, rope_head_dim=rope_head_dim, num_query_heads=num_query_heads, mscale=mscale,
      kv_lora_rank=kv_lora_rank, max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings, rope_factor=rope_factor,
      mesh=mesh, norm_fn=norm_fn, rope_fn=rope_fn, axis_mapping=axis_mapping,
      segment_ids=(segment_ids_0, segment_ids_1), quant=quant,
  )
  _prologue = weight_scan_prologue_epilogue.PipelinedScanLayer(
      fwd=functools.partial(
          dsv3_sparse_layer.dsv3_sparse_layer_prologue_fwd,
          banks=(bank_0, bank_1), bank_offsets=(bank_offset_0, bank_offset_1), **attention_kwargs,
      ),
      bwd=functools.partial(dsv3_sparse_layer.dsv3_sparse_layer_prologue_bwd, **attention_kwargs),
  )

  def _scan_body_fwd(*args, **kwargs):
    out, res = dsv3_sparse_layer.dsv3_sparse_layer_scan_body_fwd(*args, **kwargs)
    assert res.mb0.moe is not None
    moe_0 = res.mb0.moe._replace(out_proj_out=res.mb0.moe.out_proj_out + res.mb0.moe.orig_x, orig_x=None)
    return out, res._replace(mb0=res.mb0._replace(moe=moe_0))

  sparse_kwargs = dict(
      **attention_kwargs, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok,
      routed_scaling_factor=routed_scaling_factor, n_routing_groups=n_routing_groups,
      topk_routing_group=topk_routing_group, topk_in_group=topk_in_group, gmm_fn=gmm_fn,
      expert_axis_name=expert_axis_name, capacity_factor=capacity_factor, router_dtype=router_dtype,
  )
  _scan_body = weight_scan_prologue_epilogue.PipelinedScanLayer(
      fwd=functools.partial(_scan_body_fwd, **sparse_kwargs),
      bwd=functools.partial(dsv3_sparse_layer.dsv3_sparse_layer_scan_body_bwd, **sparse_kwargs),
  )
  _epilogue = weight_scan_prologue_epilogue.PipelinedScanLayer(
      fwd=functools.partial(dsv3_sparse_layer.dsv3_sparse_layer_epilogue_fwd, **sparse_kwargs),
      bwd=functools.partial(dsv3_sparse_layer.dsv3_sparse_layer_epilogue_bwd, **sparse_kwargs),
  )
  _reduce_w_ici_sc1 = functools.partial(dsv3_sparse_layer.reduce_w_ici_sc1, axis_mapping=axis_mapping)
  _reduce_w_ici_sc0 = functools.partial(dsv3_sparse_layer.reduce_w_ici_sc0, axis_mapping=axis_mapping)
  collect_fn_pre = None
  reduce_fn_pre = None
  if expert_permutations is not None:
    collect_fn_pre = _with_expert_permutations(
        functools.partial(dsv3_sparse_layer.shuffle_w_routed, axis_mapping=axis_mapping, expert_axis_name=expert_axis_name)
    )
    reduce_fn_pre = _with_expert_permutations(
        functools.partial(dsv3_sparse_layer.unshuffle_w_routed, axis_mapping=axis_mapping, expert_axis_name=expert_axis_name)
    )
    _collect_w_ici_fwd = _with_expert_permutations(_collect_w_ici_fwd, accepts=False)
    _collect_w_ici_bwd = _with_expert_permutations(_collect_w_ici_bwd, accepts=False)
    _reduce_w_ici_sc1 = _with_expert_permutations(_reduce_w_ici_sc1, accepts=False)
    _reduce_w_ici_sc0 = _with_expert_permutations(_reduce_w_ici_sc0, accepts=False)
    _prologue = weight_scan_prologue_epilogue.PipelinedScanLayer(
        fwd=_with_expert_permutations(_prologue.fwd, accepts=False),
        bwd=_with_expert_permutations(_prologue.bwd, accepts=False),
    )
    _scan_body = weight_scan_prologue_epilogue.PipelinedScanLayer(
        fwd=_with_expert_permutations(_scan_body.fwd), bwd=_with_expert_permutations(_scan_body.bwd, accepts=False),
    )
    _epilogue = weight_scan_prologue_epilogue.PipelinedScanLayer(
        fwd=_with_expert_permutations(_epilogue.fwd), bwd=_with_expert_permutations(_epilogue.bwd, accepts=False),
    )
  (out_mb0, out_mb1), aux = weight_scan_prologue_epilogue.w_collect_pipelined_scan_prologue_epilogue(
      _prologue, _scan_body, _epilogue, (mb0, mb1), w, _collect_w_dcn, _collect_w_ici_fwd,
      functools.partial(dsv3_sparse_layer.reduce_w_dcn, axis_mapping=axis_mapping),
      _reduce_w_ici_sc1, _reduce_w_ici_sc0, _collect_w_ici_bwd,
      quantize_fn=(functools.partial(dsv3_sparse_layer.quantize_weights, quant=quant)
                   if quant and (quant.routed_experts or quant.mla) else None),
      offload_fn=dsv3_sparse_layer.offload_residuals, load_fn=dsv3_sparse_layer.load_residuals,
      w_aux=expert_permutations, collect_fn_pre=collect_fn_pre, reduce_fn_pre=reduce_fn_pre,
  )
  out = ops.merge_microbatches(out_mb0, out_mb1, mesh=mesh)
  return out, aux
