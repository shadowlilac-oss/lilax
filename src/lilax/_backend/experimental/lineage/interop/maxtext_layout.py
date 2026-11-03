"""Conversion between MaxText and Lineage DSv3 parameter layouts."""

from collections.abc import Mapping
import contextlib
import dataclasses
import functools
import math
from typing import Any, cast

import jax
import jax.numpy as jnp

from lilax._backend.experimental.lineage import dsv3_config
from lilax._backend.experimental.lineage import dsv3_embed
from lilax._backend.experimental.lineage import dsv3_model
from lilax._backend.experimental.lineage import dsv3_mtp_block
from lilax._backend.experimental.lineage import dsv3_types

P = jax.sharding.PartitionSpec
PARAM_SCAN_AXIS = 1
Tree = dict[str, Any]


def _unbox(x: Any) -> Any:
  while hasattr(x, "value"):
    x = x.value
  return x


def _arr(x: jax.Array | None) -> jax.Array:
  if x is None:
    raise ValueError("Expected an array leaf, got None.")
  return x


def _deinterleave_rope(w: jax.Array, rope_head_dim: int) -> jax.Array:
  if rope_head_dim <= 0:
    return w
  if w.shape[-1] < rope_head_dim:
    raise ValueError(f"Cannot split {rope_head_dim} RoPE channels off a trailing axis of" f" size {w.shape[-1]}.")
  nope = w[..., :-rope_head_dim]
  rope = w[..., -rope_head_dim:]
  return jnp.concatenate([nope, rope[..., 0::2], rope[..., 1::2]], axis=-1)


def _interleave_rope(w: jax.Array, rope_head_dim: int) -> jax.Array:
  if rope_head_dim <= 0:
    return w
  nope = w[..., :-rope_head_dim]
  rope = w[..., -rope_head_dim:]
  half = rope_head_dim // 2
  interleaved = jnp.stack([rope[..., :half], rope[..., half:]], axis=-1)
  return jnp.concatenate([nope, interleaved.reshape(*rope.shape[:-1], rope_head_dim)], axis=-1)


def _to_lineage_leaf(x: Any, *, param_scan_axis: int, dtype: Any) -> jax.Array:
  x = jnp.asarray(_unbox(x), dtype)
  if 0 < param_scan_axis < x.ndim:
    x = jnp.moveaxis(x, param_scan_axis, 0)
  return x


def _to_maxtext_leaf(x: Any, *, param_scan_axis: int) -> jax.Array:
  x = jnp.asarray(x)
  if 0 < param_scan_axis < x.ndim:
    x = jnp.moveaxis(x, 0, param_scan_axis)
  return x


def _mla_from_maxtext(
    attn: Mapping[str, Any], leaf, *, qk_head_dim: int, rope_head_dim: int,
) -> dsv3_types.DSv3MLAWeightsPytree:
  kv_up = leaf(attn["wkv_b"]["kernel"])
  return dsv3_types.DSv3MLAWeightsPytree(
      q_down=leaf(attn["wq_a"]["kernel"]), q_up=_deinterleave_rope(leaf(attn["wq_b"]["kernel"]), rope_head_dim),
      q_norm_scale=leaf(attn["q_norm"]["scale"]),
      kv_down=_deinterleave_rope(leaf(attn["wkv_a"]["kernel"]), rope_head_dim),
      k_up=kv_up[..., :qk_head_dim], v_up=kv_up[..., qk_head_dim:],
      kv_norm_scale=leaf(attn["kv_norm"]["scale"]), out=leaf(attn["out"]["kernel"]),
  )


def _mla_to_maxtext(mla: dsv3_types.DSv3MLAWeightsPytree, leaf, *, rope_head_dim: int) -> Tree:
  return {
      "wq_a": {"kernel": leaf(mla.q_down)},
      "wq_b": {"kernel": leaf(_interleave_rope(_arr(mla.q_up), rope_head_dim))},
      "q_norm": {"scale": leaf(mla.q_norm_scale)},
      "wkv_a": {"kernel": leaf(_interleave_rope(_arr(mla.kv_down), rope_head_dim))},
      "kv_norm": {"scale": leaf(mla.kv_norm_scale)},
      "wkv_b": {"kernel": leaf(jnp.concatenate([_arr(mla.k_up), _arr(mla.v_up)], axis=-1))},
      "out": {"kernel": leaf(mla.out)},
  }


def _dense_layer_from_maxtext(
    layer: Mapping[str, Any], leaf, *, qk_head_dim: int, rope_head_dim: int,
) -> dsv3_types.DSv3DenseLayerWeightsPytree:
  mlp = layer["mlp"]
  return dsv3_types.DSv3DenseLayerWeightsPytree(
      pre_attn_norm_scale=leaf(layer["pre_self_attention_layer_norm"]["scale"]),
      mla=_mla_from_maxtext(layer["self_attention"], leaf, qk_head_dim=qk_head_dim, rope_head_dim=rope_head_dim),
      post_attn_norm_scale=leaf(layer["post_self_attention_layer_norm"]["scale"]),
      mlp=dsv3_types.DSv3MLPWeightsPytree(
          gate_0=leaf(mlp["wi_0"]["kernel"]), gate_1=leaf(mlp["wi_1"]["kernel"]), linear=leaf(mlp["wo"]["kernel"]),
      ),
  )


def _dense_layer_to_maxtext(w: dsv3_types.DSv3DenseLayerWeightsPytree, leaf, *, rope_head_dim: int) -> Tree:
  return {
      "pre_self_attention_layer_norm": {"scale": leaf(w.pre_attn_norm_scale)},
      "self_attention": _mla_to_maxtext(w.mla, leaf, rope_head_dim=rope_head_dim),
      "post_self_attention_layer_norm": {"scale": leaf(w.post_attn_norm_scale)},
      "mlp": {"wi_0": {"kernel": leaf(w.mlp.gate_0)}, "wi_1": {"kernel": leaf(w.mlp.gate_1)},
              "wo": {"kernel": leaf(w.mlp.linear)}},
  }


def _sparse_layer_from_maxtext(
    layer: Mapping[str, Any], leaf, bias_leaf, *, qk_head_dim: int, rope_head_dim: int, flat_routed: bool = False,
) -> tuple[dsv3_types.DSv3SparseLayerWeightsPytree, jax.Array]:
  ds_moe = layer["DeepSeekMoeBlock_0"]
  moe_block = ds_moe["MoeBlock_0"]
  shared = ds_moe["shared_experts"]
  if "wi" in moe_block:
    gate = leaf(moe_block["wi"])
  else:
    gate = jnp.concatenate([leaf(moe_block["wi_0"]), leaf(moe_block["wi_1"])], axis=-1)
  wo = leaf(moe_block["wo"])
  if flat_routed:
    gate = gate.reshape(*gate.shape[:-3], -1, gate.shape[-1])
    wo = wo.reshape(*wo.shape[:-3], -1, wo.shape[-1])
  weights = dsv3_types.DSv3SparseLayerWeightsPytree(
      pre_attn_norm_scale=leaf(layer["pre_self_attention_layer_norm"]["scale"]),
      mla=_mla_from_maxtext(layer["self_attention"], leaf, qk_head_dim=qk_head_dim, rope_head_dim=rope_head_dim),
      post_attn_norm_scale=leaf(layer["post_self_attention_layer_norm"]["scale"]),
      moe=dsv3_types.DSv3MoEWeightsPytree(
          router=dsv3_types.DSv3MoERouterWeightsPytree(kernel=leaf(moe_block["gate"]["kernel"]), bias=None),
          routed=dsv3_types.DSv3MoERoutedExpertWeightsPytree(gate=gate, linear=wo),
          shared=dsv3_types.DSv3MoESharedExpertWeightsPytree(
              gate_0=leaf(shared["wi_0"]["kernel"]), gate_1=leaf(shared["wi_1"]["kernel"]),
              linear=leaf(shared["wo"]["kernel"]),
          ),
      ),
  )
  return weights, bias_leaf(moe_block["gate"]["bias"])


def _sparse_layer_to_maxtext(
    w: dsv3_types.DSv3SparseLayerWeightsPytree, bias: jax.Array, leaf, *, rope_head_dim: int,
) -> Tree:
  gate = _arr(w.moe.routed.gate)
  wo = _arr(w.moe.routed.linear)
  router_kernel = _arr(w.moe.router.kernel)
  num_experts = router_kernel.shape[-1]
  if gate.ndim == 2:
    gate = gate.reshape(num_experts, -1, gate.shape[-1])
    wo = wo.reshape(num_experts, -1, wo.shape[-1])
  elif gate.ndim == 3 and router_kernel.ndim == 3:
    gate = gate.reshape(gate.shape[0], num_experts, -1, gate.shape[-1])
    wo = wo.reshape(wo.shape[0], num_experts, -1, wo.shape[-1])
  hidden = gate.shape[-1] // 2
  return {
      "pre_self_attention_layer_norm": {"scale": leaf(w.pre_attn_norm_scale)},
      "self_attention": _mla_to_maxtext(w.mla, leaf, rope_head_dim=rope_head_dim),
      "post_self_attention_layer_norm": {"scale": leaf(w.post_attn_norm_scale)},
      "DeepSeekMoeBlock_0": {
          "MoeBlock_0": {"gate": {"kernel": leaf(w.moe.router.kernel), "bias": _arr(bias)},
                         "wi_0": leaf(gate[..., :hidden]), "wi_1": leaf(gate[..., hidden:]), "wo": leaf(wo)},
          "shared_experts": {"wi_0": {"kernel": leaf(w.moe.shared.gate_0)},
                             "wi_1": {"kernel": leaf(w.moe.shared.gate_1)}, "wo": {"kernel": leaf(w.moe.shared.linear)}},
      },
  }


def maxtext_to_lineage_params(
    tree: Mapping[str, Any], cfg: dsv3_config.DSv3Config, *, param_scan_axis: int = PARAM_SCAN_AXIS,
    flat_routed: bool = False,
) -> tuple[dsv3_model.DSv3Params[dsv3_types.ArrayType], dsv3_model.DSv3MutableState[dsv3_types.ArrayType]]:
  m = cfg.model
  scanned = functools.partial(_to_lineage_leaf, param_scan_axis=param_scan_axis, dtype=m.weight_dtype)
  unscanned = functools.partial(_to_lineage_leaf, param_scan_axis=0, dtype=m.weight_dtype)
  unscanned_bias = functools.partial(_to_lineage_leaf, param_scan_axis=0, dtype=m.router_bias_dtype)
  dims = dict(qk_head_dim=m.qk_head_dim, rope_head_dim=m.rope_head_dim)
  decoder = tree["decoder"]
  sparse, router_bias = _sparse_layer_from_maxtext(decoder["moe_layers"], scanned, unscanned_bias, flat_routed=flat_routed, **dims)
  mtp = []
  mtp_router_bias = []
  for k in range(1, m.num_mtp_layers + 1):
    layer = tree["mtp_block"][f"mtp_layer_{k}"]
    prefix = f"mtp_{k}"
    mtp_sparse, bias = _sparse_layer_from_maxtext(
        layer[f"{prefix}_transformer_layer"], unscanned, unscanned_bias, flat_routed=flat_routed, **dims,
    )
    mtp.append(dsv3_mtp_block.DSv3MTPDepthWeightsPytree(
        layer=dsv3_types.DSv3MTPWeightsPytree(
            ehproj=dsv3_types.DSv3EHProjWeightsPytree(
                enorm_scale=unscanned(layer[f"{prefix}_embedding_norm"]["scale"]),
                hnorm_scale=unscanned(layer[f"{prefix}_hidden_state_norm"]["scale"]),
                eh_proj=unscanned(layer[f"{prefix}_projection"]["kernel"]),
            ), sparse=mtp_sparse,
        ), final_norm_scale=unscanned(layer[f"{prefix}_final_norm"]["scale"]),
    ))
    mtp_router_bias.append(bias)
  params = dsv3_model.DSv3Params[dsv3_types.ArrayType](
      embed=dsv3_embed.DSv3EmbedWeightsPytree(table=unscanned(tree["token_embedder"]["embedding"])),
      decoder=dsv3_types.DSv3WeightsPytree(dense=_dense_layer_from_maxtext(decoder["dense_layers"], scanned, **dims), sparse=sparse),
      mtp=tuple(mtp), head=dsv3_embed.DSv3HeadWeightsPytree(
          final_norm_scale=unscanned(decoder["decoder_norm"]["scale"]),
          kernel=None if m.tied_head else unscanned(decoder["logits_dense"]["kernel"]),
      ),
  )
  expert_permutations, mtp_expert_permutations = dsv3_model.identity_expert_permutations(cfg)
  state = dsv3_model.DSv3MutableState[dsv3_types.ArrayType](
      router_bias=router_bias, mtp_router_bias=tuple(mtp_router_bias), expert_permutations=expert_permutations,
      mtp_expert_permutations=mtp_expert_permutations,
  )
  return params, state


def lineage_to_maxtext_params(
    params: dsv3_model.DSv3Params, state: dsv3_model.DSv3MutableState, cfg: dsv3_config.DSv3Config,
    *, param_scan_axis: int = PARAM_SCAN_AXIS,
) -> Tree:
  m = cfg.model
  scanned = functools.partial(_to_maxtext_leaf, param_scan_axis=param_scan_axis)
  unscanned = functools.partial(_to_maxtext_leaf, param_scan_axis=0)
  rope = dict(rope_head_dim=m.rope_head_dim)
  assert state.router_bias is not None
  if len(params.mtp) != len(state.mtp_router_bias):
    raise ValueError(f"Got {len(state.mtp_router_bias)} MTP router biases for" f" {len(params.mtp)} MTP depths.")
  decoder = {
      "decoder_norm": {"scale": unscanned(params.head.final_norm_scale)},
      "dense_layers": _dense_layer_to_maxtext(params.decoder.dense, scanned, **rope),
      "moe_layers": _sparse_layer_to_maxtext(params.decoder.sparse, state.router_bias, scanned, **rope),
  }
  if params.head.kernel is not None:
    decoder["logits_dense"] = {"kernel": unscanned(params.head.kernel)}
  tree: Tree = {"token_embedder": {"embedding": unscanned(params.embed.table)}, "decoder": decoder}
  if params.mtp:
    mtp_block = {}
    for k, (depth, bias) in enumerate(zip(params.mtp, state.mtp_router_bias), start=1):
      prefix = f"mtp_{k}"
      ehproj = depth.layer.ehproj
      mtp_block[f"mtp_layer_{k}"] = {
          f"{prefix}_embedding_norm": {"scale": unscanned(ehproj.enorm_scale)},
          f"{prefix}_hidden_state_norm": {"scale": unscanned(ehproj.hnorm_scale)},
          f"{prefix}_projection": {"kernel": unscanned(ehproj.eh_proj)},
          f"{prefix}_transformer_layer": _sparse_layer_to_maxtext(depth.layer.sparse, bias, unscanned, **rope),
          f"{prefix}_final_norm": {"scale": unscanned(depth.final_norm_scale)},
      }
    tree["mtp_block"] = mtp_block
  return tree


def restore_sharding(
    shape: tuple[int, ...], mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
) -> jax.sharding.NamedSharding:
  remaining = list(mesh.axis_names)
  spec: list[str | tuple[str, ...] | None] = [None] * len(shape)
  for i, dim in enumerate(shape):
    assigned: list[str] = []
    rem_dim = dim
    still_remaining: list[str] = []
    for axis in remaining:
      axis_size = mesh.shape[axis]
      if rem_dim % axis_size == 0:
        assigned.append(axis)
        rem_dim //= axis_size
      else:
        still_remaining.append(axis)
    remaining = still_remaining
    if len(assigned) == 1:
      spec[i] = assigned[0]
    elif len(assigned) > 1:
      spec[i] = tuple(assigned)
  if not any(spec):
    return jax.sharding.NamedSharding(mesh, P())
  return jax.sharding.NamedSharding(mesh, P(*spec))


@contextlib.contextmanager
def _mesh_context(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh):
  if isinstance(mesh, jax.sharding.AbstractMesh):
    with jax.sharding.use_abstract_mesh(jax.sharding.AbstractMesh((), ())):
      with jax.sharding.use_abstract_mesh(mesh):
        yield
  else:
    with jax.set_mesh(mesh):
      yield


def _rebind_mesh(tree: Any, mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> Any:
  def _rebind(a):
    if not isinstance(getattr(a, "sharding", None), jax.sharding.NamedSharding):
      return a
    sharding = jax.sharding.NamedSharding(mesh, a.sharding.spec)
    return jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=sharding)
  return jax.tree.map(_rebind, tree)


def _move_mla_trailing_sharding_to_lora(a: Any) -> Any:
  if a is None:
    return None
  sharding = a.sharding
  if not isinstance(sharding, jax.sharding.NamedSharding):
    return a
  mesh = sharding.mesh
  spec = list(sharding.spec) + [None] * (a.ndim - len(sharding.spec))
  trailing = spec[-1]
  if trailing is None:
    return a
  spec[-1] = None
  axes = (trailing,) if isinstance(trailing, str) else tuple(trailing)
  factor = math.prod(mesh.shape[axis] for axis in axes)
  if spec[-3] is None and a.shape[-3] % factor == 0:
    spec[-3] = trailing
  return jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=jax.sharding.NamedSharding(mesh, P(*spec)))


def _prepare_mla_for_maxtext_restore(node: Any) -> Any:
  if not isinstance(node, dsv3_types.DSv3MLAWeightsPytree):
    return node
  return dataclasses.replace(
      node, q_up=_move_mla_trailing_sharding_to_lora(node.q_up),
      k_up=_move_mla_trailing_sharding_to_lora(node.k_up), v_up=_move_mla_trailing_sharding_to_lora(node.v_up),
  )


def lineage_abstract_params(
    cfg: dsv3_config.DSv3Config, mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
) -> tuple[Any, Any]:
  with _mesh_context(mesh):
    out = jax.eval_shape(lambda: dsv3_model.init_dsv3_params(jax.random.key(0), cfg, cast(jax.sharding.Mesh, mesh)))
  shardings = dsv3_model.dsv3_param_shardings(cfg, cast(jax.sharding.Mesh, mesh))
  return jax.tree.map(lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s), out, shardings)


def _unmerge_routed_abstract_leaf(a: Any, num_experts: int) -> Any:
  if a is None:
    return None
  sharding = a.sharding
  prefix_shape = a.shape[:-2]
  rows = a.shape[-2] // num_experts
  cols = a.shape[-1]
  unmerged_shape = (*prefix_shape, num_experts, rows, cols)
  if not isinstance(sharding, jax.sharding.NamedSharding):
    return jax.ShapeDtypeStruct(unmerged_shape, a.dtype)
  mesh = sharding.mesh
  spec = list(sharding.spec) + [None] * (a.ndim - len(sharding.spec))
  merged = spec[-2]
  axes = (merged,) if isinstance(merged, str) else tuple(merged or ())
  ep_axes = axes[:-2] or None
  row_ax = axes[-2] if len(axes) >= 2 else None
  col_ax = axes[-1] if len(axes) >= 1 else None
  unmerged_spec = (*spec[:-2], ep_axes, row_ax, col_ax)
  return jax.ShapeDtypeStruct(unmerged_shape, a.dtype, sharding=jax.sharding.NamedSharding(mesh, P(*unmerged_spec)))


def _prepare_sparse_for_maxtext_restore(node: Any) -> Any:
  if not isinstance(node, dsv3_types.DSv3SparseLayerWeightsPytree):
    return node
  mla = _prepare_mla_for_maxtext_restore(node.mla)
  gate = node.moe.routed.gate
  router_k = node.moe.router.kernel
  if gate is None or router_k is None:
    return dataclasses.replace(node, mla=mla)
  is_flat = gate.ndim == 2 or (gate.ndim == 3 and router_k.ndim == 3)
  if not is_flat:
    return dataclasses.replace(node, mla=mla)
  num_experts = router_k.shape[-1]
  routed = dsv3_types.DSv3MoERoutedExpertWeightsPytree(
      gate=_unmerge_routed_abstract_leaf(gate, num_experts),
      linear=_unmerge_routed_abstract_leaf(node.moe.routed.linear, num_experts),
  )
  return dataclasses.replace(node, mla=mla, moe=dataclasses.replace(node.moe, routed=routed))


def maxtext_abstract_param_tree(
    cfg: dsv3_config.DSv3Config, mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    *, param_scan_axis: int = PARAM_SCAN_AXIS,
) -> Tree:
  """Builds a MaxText-layout abstract parameter tree for checkpoint restore."""
  params, state = lineage_abstract_params(cfg, mesh)
  params = jax.tree.map(
      _prepare_mla_for_maxtext_restore, params, is_leaf=lambda x: isinstance(x, dsv3_types.DSv3MLAWeightsPytree),
  )
  params = jax.tree.map(
      _prepare_sparse_for_maxtext_restore, params,
      is_leaf=lambda x: isinstance(x, dsv3_types.DSv3SparseLayerWeightsPytree),
  )
  with _mesh_context(mesh):
    tree = jax.eval_shape(
        functools.partial(lineage_to_maxtext_params, cfg=cfg, param_scan_axis=param_scan_axis), params, state,
    )
  return _rebind_mesh(tree, mesh)


def convert_maxtext_params(
    tree: Mapping[str, Any], cfg: dsv3_config.DSv3Config, mesh: jax.sharding.Mesh,
    *, param_scan_axis: int = PARAM_SCAN_AXIS,
) -> tuple[dsv3_model.DSv3Params[dsv3_types.ArrayType], dsv3_model.DSv3MutableState[dsv3_types.ArrayType]]:
  """Converts a restored tree into the configured Lineage shardings."""
  out_shardings = dsv3_model.dsv3_param_shardings(cfg, mesh)
  gate_sharding = out_shardings[0].decoder.sparse.moe.routed.gate
  gate_spec = gate_sharding.spec if isinstance(gate_sharding, jax.sharding.NamedSharding) else gate_sharding
  flat_routed = gate_spec is not None and len(gate_spec) == 3
  convert = jax.sharding.auto_axes(
      functools.partial(maxtext_to_lineage_params, cfg=cfg, param_scan_axis=param_scan_axis, flat_routed=flat_routed),
      out_sharding=out_shardings,
  )
  with jax.set_mesh(mesh):
    return jax.jit(convert)(jax.tree.map(_unbox, tree))
