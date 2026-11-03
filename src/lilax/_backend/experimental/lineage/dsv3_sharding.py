"""Sharding specs, axis mapping and physical mesh construction for DSv3."""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import math
from typing import Any

import jax
from jax.experimental import mesh_utils
import numpy as np

from lilax._backend.experimental.lineage import dsv3_types
from lilax._backend.experimental.lineage import ops

P = jax.sharding.PartitionSpec
AxisMapping = Mapping[str, str | tuple[str, ...]]
DCN_AXIS = "dcn"
LOGICAL_AXES = ("dcn", "attention", "fsdp_attention", "expert", "fsdp_moe")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DSv3WeightShardings:
  decoder: dsv3_types.DSv3WeightsPytree[Any]
  mtp: dsv3_types.DSv3MTPWeightsPytree[Any]
  embed_table: P
  final_norm_scale: P
  head_kernel: P


@dataclasses.dataclass(frozen=True)
class DSv3ShardingConfig:
  axis_mapping: AxisMapping
  weights: DSv3WeightShardings
  activation: P
  segment_ids: P
  token: P
  yarn_freqs: P


@dataclasses.dataclass(frozen=True)
class TopologySharding:
  axis_mapping: AxisMapping
  weights: DSv3WeightShardings


def drop_layer_axis(specs: Any) -> Any:
  return jax.tree.map(lambda spec: P(*spec[1:]), specs, is_leaf=lambda x: isinstance(x, P))


_MLA_8X8X8 = dsv3_types.DSv3MLAWeightsPytree(
    q_down=P(None, ("y", "z"), None), q_up=P(None, ("y", "z"), "core", None),
    q_norm_scale=P(None, None), kv_down=P(None, ("y", "z"), None),
    k_up=P(None, None, "core", ("y", "z")), v_up=P(None, None, "core", ("y", "z")),
    kv_norm_scale=P(None, None), out=P(None, "core", None, ("y", "z")),
)
_SPARSE_8X8X8 = dsv3_types.DSv3SparseLayerWeightsPytree(
    pre_attn_norm_scale=P(None, None), mla=_MLA_8X8X8, post_attn_norm_scale=P(None, None),
    moe=dsv3_types.DSv3MoEWeightsPytree(
        router=dsv3_types.DSv3MoERouterWeightsPytree(kernel=P(), bias=P()),
        routed=dsv3_types.DSv3MoERoutedExpertWeightsPytree(
            gate=P(None, ("x", "core"), ("y", "z"), None),
            linear=P(None, ("x", "core"), ("y", "z"), None),
        ),
        shared=dsv3_types.DSv3MoESharedExpertWeightsPytree(
            gate_0=P(None, None, ("y", "z")), gate_1=P(None, None, ("y", "z")), linear=P(None, ("y", "z"), None),
        ),
    ),
)
_MLA_8X16X16 = dsv3_types.DSv3MLAWeightsPytree(
    q_down=P(None, "z", None), q_up=P(None, ("y", "z"), "core", None), q_norm_scale=P(None, None),
    kv_down=P(None, "z", None), k_up=P(None, ("y", "z"), "core", None),
    v_up=P(None, ("y", "z"), "core", None), kv_norm_scale=P(None, None), out=P(None, "core", None, ("y", "z")),
)
_SPARSE_8X16X16 = dsv3_types.DSv3SparseLayerWeightsPytree(
    pre_attn_norm_scale=P(None, None), mla=_MLA_8X16X16, post_attn_norm_scale=P(None, None),
    moe=dsv3_types.DSv3MoEWeightsPytree(
        router=dsv3_types.DSv3MoERouterWeightsPytree(kernel=P(), bias=P()),
        routed=dsv3_types.DSv3MoERoutedExpertWeightsPytree(
            gate=P(None, ("x", "core", "y"), "z", None), linear=P(None, ("x", "core", "y"), "z", None),
        ),
        shared=dsv3_types.DSv3MoESharedExpertWeightsPytree(
            gate_0=P(None, None, ("y", "z")), gate_1=P(None, None, ("y", "z")), linear=P(None, ("y", "z"), None),
        ),
    ),
)
_SPARSE_8X16X32 = dsv3_types.DSv3SparseLayerWeightsPytree(
    pre_attn_norm_scale=P(None, None), mla=_MLA_8X16X16, post_attn_norm_scale=P(None, None),
    moe=dsv3_types.DSv3MoEWeightsPytree(
        router=dsv3_types.DSv3MoERouterWeightsPytree(kernel=P(), bias=P()),
        routed=dsv3_types.DSv3MoERoutedExpertWeightsPytree(
            gate=P(None, ("x", "core", "y"), "z", None), linear=P(None, ("x", "core", "y"), "z", None),
        ),
        shared=dsv3_types.DSv3MoESharedExpertWeightsPytree(
            gate_0=P(None, None, "z"), gate_1=P(None, None, "z"), linear=P(None, "z", None),
        ),
    ),
)
_MLA_8X32X32 = dsv3_types.DSv3MLAWeightsPytree(
    q_down=P(None, "z", None), q_up=P(None, "z", "core", None), q_norm_scale=P(None, None),
    kv_down=P(None, "z", None), k_up=P(None, "z", "core", None), v_up=P(None, "z", "core", None),
    kv_norm_scale=P(None, None), out=P(None, "core", None, "z"),
)
_SPARSE_8X32X32 = dsv3_types.DSv3SparseLayerWeightsPytree(
    pre_attn_norm_scale=P(None, None), mla=_MLA_8X32X32, post_attn_norm_scale=P(None, None),
    moe=dsv3_types.DSv3MoEWeightsPytree(
        router=dsv3_types.DSv3MoERouterWeightsPytree(kernel=P(), bias=P()),
        routed=dsv3_types.DSv3MoERoutedExpertWeightsPytree(
            gate=P(None, ("x", "core", "y", "z"), None), linear=P(None, ("x", "core", "y", "z"), None),
        ),
        shared=dsv3_types.DSv3MoESharedExpertWeightsPytree(
            gate_0=P(None, "z", None), gate_1=P(None, "z", None), linear=P(None, "z", None),
        ),
    ),
)
_MLA_4X8X8 = dsv3_types.DSv3MLAWeightsPytree(
    q_down=P(None, ("x", "y"), None), q_up=P(None, ("x", "y"), "core", None), q_norm_scale=P(None, None),
    kv_down=P(None, ("x", "y"), None), k_up=P(None, None, "core", ("x", "y")),
    v_up=P(None, None, "core", ("x", "y")), kv_norm_scale=P(None, None), out=P(None, "core", None, ("x", "y")),
)
_SPARSE_4X8X8 = dsv3_types.DSv3SparseLayerWeightsPytree(
    pre_attn_norm_scale=P(None, None), mla=_MLA_4X8X8, post_attn_norm_scale=P(None, None),
    moe=dsv3_types.DSv3MoEWeightsPytree(
        router=dsv3_types.DSv3MoERouterWeightsPytree(kernel=P(), bias=P()),
        routed=dsv3_types.DSv3MoERoutedExpertWeightsPytree(
            gate=P(None, ("z", "core"), ("x", "y"), None), linear=P(None, ("z", "core"), ("x", "y"), None),
        ),
        shared=dsv3_types.DSv3MoESharedExpertWeightsPytree(
            gate_0=P(None, None, ("x", "y")), gate_1=P(None, None, ("x", "y")), linear=P(None, ("x", "y"), None),
        ),
    ),
)
_MLA_4X4X8 = dataclasses.replace(_MLA_4X8X8, q_up=P(None, None, "core", ("x", "y")))
_SPARSE_4X4X8 = dataclasses.replace(_SPARSE_4X8X8, mla=_MLA_4X4X8)


def _topology_sharding(
    mla: dsv3_types.DSv3MLAWeightsPytree, sparse: dsv3_types.DSv3SparseLayerWeightsPytree,
    *, expert: tuple[str, ...], fsdp: tuple[str, ...], mlp_axis: str | tuple[str, ...],
    mlp_shard_rows: bool, eh_axis: str | tuple[str, ...], embed_axes: tuple[str, ...],
) -> TopologySharding:
  mlp = dsv3_types.DSv3MLPWeightsPytree(
      gate_0=P(None, mlp_axis, None) if mlp_shard_rows else P(None, None, mlp_axis),
      gate_1=P(None, mlp_axis, None) if mlp_shard_rows else P(None, None, mlp_axis),
      linear=P(None, mlp_axis, None),
  )
  return TopologySharding(
      axis_mapping={"attention": "core", "fsdp_attention": ("x", "y", "z"), "expert": expert, "fsdp_moe": fsdp},
      weights=DSv3WeightShardings(
          decoder=dsv3_types.DSv3WeightsPytree(
              dense=dsv3_types.DSv3DenseLayerWeightsPytree(
                  pre_attn_norm_scale=P(None, None), mla=mla, post_attn_norm_scale=P(None, None), mlp=mlp,
              ), sparse=sparse,
          ),
          mtp=dsv3_types.DSv3MTPWeightsPytree(
              ehproj=dsv3_types.DSv3EHProjWeightsPytree(enorm_scale=P(None), hnorm_scale=P(None), eh_proj=P(eh_axis, None)),
              sparse=drop_layer_axis(sparse),
          ),
          embed_table=P(None, embed_axes), final_norm_scale=P(None), head_kernel=P(embed_axes, None),
      ),
  )


TOPOLOGY_SHARDINGS: dict[tuple[int, int, int], TopologySharding] = {
    (4, 4, 8): _topology_sharding(_MLA_4X4X8, _SPARSE_4X4X8, expert=("z", "core"), fsdp=("x", "y"),
                                  mlp_axis=("x", "y"), mlp_shard_rows=False, eh_axis=("x", "y"),
                                  embed_axes=("x", "y", "z", "core")),
    (4, 8, 8): _topology_sharding(_MLA_4X8X8, _SPARSE_4X8X8, expert=("z", "core"), fsdp=("x", "y"),
                                  mlp_axis=("x", "y"), mlp_shard_rows=False, eh_axis=("x", "y"),
                                  embed_axes=("x", "y", "z", "core")),
    (8, 8, 8): _topology_sharding(_MLA_8X8X8, _SPARSE_8X8X8, expert=("x", "core"), fsdp=("y", "z"),
                                  mlp_axis=("y", "z"), mlp_shard_rows=False, eh_axis=("y", "z"),
                                  embed_axes=("x", "y", "z", "core")),
    (8, 16, 16): _topology_sharding(_MLA_8X16X16, _SPARSE_8X16X16, expert=("x", "core"), fsdp=("y", "z"),
                                    mlp_axis=("y", "z"), mlp_shard_rows=False, eh_axis=("y", "z"),
                                    embed_axes=("y", "z", "core")),
    (8, 16, 32): _topology_sharding(_MLA_8X16X16, _SPARSE_8X16X32, expert=("x", "core"), fsdp=("y", "z"),
                                    mlp_axis="z", mlp_shard_rows=False, eh_axis=("y", "z"),
                                    embed_axes=("y", "z", "core")),
    (8, 32, 32): _topology_sharding(_MLA_8X32X32, _SPARSE_8X32X32, expert=("x", "core"), fsdp=("y", "z"),
                                    mlp_axis="z", mlp_shard_rows=True, eh_axis="z", embed_axes=("z", "core")),
}


def dcn_batch_axis(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh, axis: str | tuple[str, ...]) -> str | tuple[str, ...]:
  if DCN_AXIS not in mesh.axis_names:
    return axis
  axes = (axis,) if isinstance(axis, str) else tuple(axis)
  return (DCN_AXIS,) + axes


def activation_sharding(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> P:
  return P(dcn_batch_axis(mesh, "fsdp_attention"), "attention", None)


def segment_ids_sharding(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> P:
  return P(dcn_batch_axis(mesh, "fsdp_attention"), None)


def token_sharding(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> P:
  return P(dcn_batch_axis(mesh, "fsdp_attention"), "attention")


def yarn_freqs_sharding(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> P:
  return P(dcn_batch_axis(mesh, "fsdp_attention"), None, None)


def sharding_config(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> DSv3ShardingConfig:
  """Returns the configured shardings for the mesh's per-slice topology."""
  topology = (mesh.shape["x"], mesh.shape["y"], mesh.shape["z"])
  entry = TOPOLOGY_SHARDINGS.get(topology)
  if entry is None:
    raise ValueError(f"No DSv3 shardings for per-slice topology {topology}; supported:" f" {sorted(TOPOLOGY_SHARDINGS)}.")
  return DSv3ShardingConfig(
      axis_mapping={"dcn": DCN_AXIS if DCN_AXIS in mesh.axis_names else (), **entry.axis_mapping},
      weights=entry.weights, activation=activation_sharding(mesh), segment_ids=segment_ids_sharding(mesh),
      token=token_sharding(mesh), yarn_freqs=yarn_freqs_sharding(mesh),
  )


def _axis_names(v: str | Sequence[str]) -> tuple[str, ...]:
  return (v,) if isinstance(v, str) else tuple(v)


def _to_axis_tuple(v: str | Sequence[str]) -> str | tuple[str, ...]:
  if isinstance(v, str):
    return v
  return v[0] if len(v) == 1 else tuple(v)


_MLA_HEAD_DIMS = (("q_up", -2), ("k_up", -2), ("v_up", -2), ("out", -3))


def _is_mla_or_spec(x: Any) -> bool:
  return isinstance(x, (dsv3_types.DSv3MLAWeightsPytree, P))


def _reshard_mla_heads(node: Any, head_axis: str | tuple[str, ...]) -> Any:
  if not isinstance(node, dsv3_types.DSv3MLAWeightsPytree):
    return node
  head_axes = _axis_names(head_axis)
  updates = {}
  for field, dim in _MLA_HEAD_DIMS:
    spec = getattr(node, field)
    if spec is None:
      continue
    entries = list(spec)
    idx = len(entries) + dim
    other_axes = set()
    for i, entry in enumerate(entries):
      if i != idx and entry is not None:
        other_axes.update(_axis_names(entry))
    overlap = [ax for ax in head_axes if ax in other_axes]
    if overlap:
      raise ValueError(
          f"The activation layout's `attention` axes {head_axis} overlap with"
          f" the MLA `{field}` FSDP axes {entries}: {overlap}."
      )
    entries[idx] = head_axis
    updates[field] = P(*entries)
  return dataclasses.replace(node, **updates)


def activation_resharded_config(base: DSv3ShardingConfig, activation_axis_mapping: AxisMapping) -> DSv3ShardingConfig:
  """Changes activation axes and reslices MLA heads to match."""
  missing = [axis for axis in LOGICAL_AXES if axis not in activation_axis_mapping]
  if missing:
    raise ValueError(f"activation_axis_mapping lacks logical axes {missing}.")
  for axis in ("dcn", "expert", "fsdp_moe"):
    if _to_axis_tuple(_axis_names(activation_axis_mapping[axis])) != _to_axis_tuple(_axis_names(base.axis_mapping[axis])):
      raise ValueError(
          f"The activation layout must keep `{axis}`:" f" {activation_axis_mapping[axis]} != {base.axis_mapping[axis]}."
      )
  base_attention = _axis_names(base.axis_mapping["attention"])
  new_attention = _axis_names(activation_axis_mapping["attention"])
  if new_attention[: len(base_attention)] != base_attention:
    raise ValueError(
        "The activation layout must shard `attention` over the training axes"
        f" {base.axis_mapping['attention']} as a prefix, got"
        f" {activation_axis_mapping['attention']}."
    )
  head_axis = _to_axis_tuple(new_attention)
  weights = dataclasses.replace(
      base.weights,
      decoder=jax.tree.map(lambda node: _reshard_mla_heads(node, head_axis), base.weights.decoder, is_leaf=_is_mla_or_spec),
      mtp=jax.tree.map(lambda node: _reshard_mla_heads(node, head_axis), base.weights.mtp, is_leaf=_is_mla_or_spec),
  )
  return dataclasses.replace(
      base, axis_mapping={axis: _to_axis_tuple(_axis_names(activation_axis_mapping[axis])) for axis in LOGICAL_AXES},
      weights=weights,
  )


def expert_axes_as_cp_axis_mapping(axis_mapping: AxisMapping) -> AxisMapping:
  attention = _axis_names(axis_mapping["attention"])
  fsdp_attention = _axis_names(axis_mapping["fsdp_attention"])
  moved = tuple(ax for ax in _axis_names(axis_mapping["expert"]) if ax not in attention)
  missing = [ax for ax in moved if ax not in fsdp_attention]
  if missing:
    raise ValueError(
        f"Expert axes {missing} shard neither `attention` {attention} nor"
        f" `fsdp_attention` {fsdp_attention}; they cannot become context parallelism."
    )
  return {
      **axis_mapping, "attention": _to_axis_tuple(attention + moved),
      "fsdp_attention": _to_axis_tuple(tuple(ax for ax in fsdp_attention if ax not in moved)),
  }


def expert_axes_as_cp_config(base: DSv3ShardingConfig) -> DSv3ShardingConfig:
  return activation_resharded_config(base, expert_axes_as_cp_axis_mapping(base.axis_mapping))


def named_shardings(specs: Any, mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh, axis_mapping: AxisMapping) -> Any:
  return jax.tree.map(
      lambda spec: jax.sharding.NamedSharding(mesh, ops.physical_pspec(spec, axis_mapping)),
      specs, is_leaf=lambda x: isinstance(x, P),
  )


def reshard(tree: Any, specs: Any, mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh, axis_mapping: AxisMapping) -> Any:
  return jax.tree.map(jax.reshard, tree, named_shardings(specs, mesh, axis_mapping))


def _physical_coords(device: Any) -> tuple[int, ...]:
  return (*getattr(device, "coords", ()), getattr(device, "core_on_chip", 0))


def devices_by_slice(devices: Sequence[Any]) -> list[list[Any]]:
  by_slice: dict[int, list[Any]] = {}
  for device in devices:
    by_slice.setdefault(getattr(device, "slice_index", 0), []).append(device)
  return [by_slice[slice_index] for slice_index in sorted(by_slice)]


def _physical_slice_grid(slice_devices: Sequence[Any]) -> np.ndarray:
  coords_list = [_physical_coords(device) for device in slice_devices]
  if not coords_list or any(len(coords) != 4 for coords in coords_list):
    raise ValueError("Devices do not expose 3D TPU physical coordinates (x, y, z, core).")
  physical_shape = tuple(max(coords[axis] for coords in coords_list) + 1 for axis in range(4))
  if math.prod(physical_shape) != len(slice_devices):
    raise ValueError(f"Device count {len(slice_devices)} does not fill physical shape" f" {list(physical_shape)}.")
  return np.array(sorted(slice_devices, key=_physical_coords), dtype=object).reshape(physical_shape)


def create_physical_device_mesh(ici_parallelism: Sequence[int], devices: Sequence[Any]) -> np.ndarray:
  """Maps logical mesh axes one-to-one onto the physical TPU torus."""
  ici = tuple(ici_parallelism)
  has_dcn_axis = len(ici) == 5
  if has_dcn_axis:
    if ici[0] != 1:
      raise ValueError(
          "create_physical_device_mesh requires the leading (DCN) entry of"
          f" ici_parallelism {list(ici)} to be 1; the DCN axis is sized by dcn_parallelism."
      )
    ici = ici[1:]
  elif len(ici) != 4:
    raise ValueError(
        "create_physical_device_mesh requires ici_parallelism ordered"
        f" (x, y, z, core) or (dcn, x, y, z, core), got {list(ici)}."
    )
  slices = devices_by_slice(devices)
  if len(slices) > 1 and not has_dcn_axis:
    raise ValueError(
        f"create_physical_device_mesh got devices from {len(slices)} slices"
        " but the mesh has no DCN axis; use mesh_axes ['dcn', 'x', 'y', 'z',"
        " 'core'] with ici_parallelism [1, x, y, z, core]."
    )
  grids = [_physical_slice_grid(s) for s in slices]
  for grid in grids:
    if ici != grid.shape:
      raise ValueError(
          "create_physical_device_mesh requires ici_parallelism"
          f" {list(ici)} to equal the physical (x, y, z, core) topology {list(grid.shape)}."
      )
  if not has_dcn_axis:
    return grids[0]
  return np.stack(grids, axis=0)


def create_device_mesh(
    ici_parallelism: Sequence[int], devices: Sequence[Any], dcn_parallelism: Sequence[int] | None = None,
    allow_split_physical_axes: bool | None = None, *, log_fn: Callable[[str], Any] | None = None,
) -> np.ndarray:
  """Maps physical axes onto the mesh, using JAX mesh assignment when needed."""
  if dcn_parallelism is not None:
    expected = [len(devices_by_slice(devices))] + [1] * (len(dcn_parallelism) - 1)
    if len(ici_parallelism) != 5 or list(dcn_parallelism) != expected:
      raise ValueError(
          "Lineage across several slices requires mesh_axes"
          " ['dcn', 'x', 'y', 'z', 'core'] with ici_parallelism"
          f" [1, x, y, z, core] and dcn_parallelism {expected}; got"
          f" ici_parallelism={list(ici_parallelism)} and dcn_parallelism={list(dcn_parallelism)}."
      )
  try:
    return create_physical_device_mesh(ici_parallelism, devices)
  except ValueError as e:
    if log_fn is not None:
      log_fn(f"Lineage: {e} Falling back to jax mesh_utils device assignment.")
    if dcn_parallelism is not None and len(devices_by_slice(devices)) > 1:
      return mesh_utils.create_hybrid_device_mesh(
          ici_parallelism, dcn_parallelism, devices, allow_split_physical_axes=bool(allow_split_physical_axes),
      )
    if allow_split_physical_axes is not None:
      return mesh_utils.create_device_mesh(ici_parallelism, devices, allow_split_physical_axes=allow_split_physical_axes)
    return mesh_utils.create_device_mesh(ici_parallelism, devices)
