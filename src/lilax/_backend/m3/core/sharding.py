"""Sharding rules and mesh creation for m3 models."""

import abc
import enum
import math
from collections.abc import Mapping, Sequence

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

MESH_AXIS_NAMES = ("data", "fsdp", "tensor", "expert")
MeshAxes = str | tuple[str, ...] | None


class TensorType(enum.Enum):
  WEIGHT = "weight"
  ACTIVATION = "activation"


def _ambient_mesh_is_explicit() -> bool:
  axis_types = tuple(jax.sharding.get_abstract_mesh().axis_types)
  explicit = sum(1 for t in axis_types if t == jax.sharding.AxisType.Explicit)
  if 0 < explicit < len(axis_types):
    raise ValueError(f"Mesh mixes explicit and non-explicit axis types ({axis_types}), which m3 does not support.")
  return explicit == len(axis_types) and explicit > 0


def _validate_mesh_axes(mapped: MeshAxes, axis: str, tensor_name: str) -> None:
  if mapped is None:
    return
  names = (mapped,) if isinstance(mapped, str) else tuple(mapped)
  unknown = [name for name in names if name not in MESH_AXIS_NAMES]
  if unknown:
    raise ValueError(
        f"Sharding rule for logical axis {axis!r} of tensor {tensor_name!r} returned unknown mesh "
        f"axes {unknown}; expected names from {MESH_AXIS_NAMES}."
    )


class Sharding(abc.ABC):
  @abc.abstractmethod
  def map_axis(self, axis: str, tensor_name: str, tensor_type: TensorType) -> MeshAxes:
    """Maps a logical axis to mesh axes, or None for replication."""

  def __call__(
      self,
      tensor_name: str,
      axes: Sequence[str | None],
      tensor_type: TensorType = TensorType.WEIGHT,
  ) -> P:
    resolved = []
    claimed: dict[str, str] = {}
    for axis in axes:
      if axis is None:
        resolved.append(None)
        continue
      mapped = self.map_axis(axis, tensor_name, tensor_type)
      _validate_mesh_axes(mapped, axis, tensor_name)
      for mesh_axis in (mapped,) if isinstance(mapped, str) else tuple(mapped or ()):
        if mesh_axis in claimed:
          raise ValueError(
              f"Tensor {tensor_name!r} maps both {claimed[mesh_axis]!r} and {axis!r} onto mesh axis "
              f"{mesh_axis!r}; a mesh axis can shard only one dimension of a tensor."
          )
        claimed[mesh_axis] = axis
      resolved.append(mapped)
    return P(*resolved)

  def constrain(self, x: jax.Array, tensor_name: str, axes: Sequence[str | None]) -> jax.Array:
    spec = self(tensor_name, axes, TensorType.ACTIVATION)
    if _ambient_mesh_is_explicit():
      return jax.sharding.reshard(x, spec)
    return jax.lax.with_sharding_constraint(x, spec)


class DefaultSharding(Sharding):
  def map_axis(self, axis: str, tensor_name: str, tensor_type: TensorType) -> MeshAxes:
    match axis, tensor_type:
      case "batch", TensorType.ACTIVATION:
        return ("data", "fsdp", "expert")
      case "embed", TensorType.WEIGHT:
        return ("fsdp", "expert")
      case "embed", TensorType.ACTIVATION:
        return None
      case ("heads" | "kv_heads" | "mlp" | "vocab"), _:
        return "tensor"
      case ("length" | "head_dim" | "norm"), _:
        return None
      case _:
        raise KeyError(f"No sharding rule for logical axis {axis!r} ({tensor_type.value}) on tensor {tensor_name!r}.")


def _resolve_axis_sizes(
    sizes: Sequence[int],
    target_product: int,
    kind: str,
    target_desc: str,
    axis_names: Sequence[str] = MESH_AXIS_NAMES,
) -> tuple[int, ...]:
  resolved = list(sizes)
  if resolved.count(-1) > 1:
    raise ValueError(f"At most one {kind} mesh axis may be -1 (auto); got {dict(zip(axis_names, resolved))}.")
  invalid = {name: size for name, size in zip(axis_names, resolved) if size < 1 and size != -1}
  if invalid:
    raise ValueError(f"{kind} mesh axis sizes must be >= 1, or -1 for auto; got {invalid}.")
  if -1 in resolved:
    specified = math.prod(size for size in resolved if size != -1)
    if target_product % specified:
      raise ValueError(
          f"Cannot auto-size the -1 {kind} mesh axis: {target_desc} ({target_product}) is not divisible by "
          f"{specified}, the product of the specified axes {dict(zip(axis_names, resolved))}."
      )
    resolved[resolved.index(-1)] = target_product // specified
  product = math.prod(resolved)
  if product != target_product:
    raise ValueError(
        f"{kind} mesh axes {dict(zip(axis_names, resolved))} have product {product}, "
        f"which does not match the {target_desc} ({target_product})."
    )
  return tuple(resolved)


def _count_slices(devices: Sequence[jax.Device]) -> int:
  return len({getattr(device, "slice_index", 0) for device in devices})


def _shape_from_mapping(
    sizes: Mapping[str, int] | None, kind: str, axis_names: Sequence[str] = MESH_AXIS_NAMES
) -> tuple[int, ...]:
  sizes = dict(sizes or {})
  unknown = sorted(set(sizes) - set(axis_names))
  if unknown:
    raise ValueError(f"Unknown {kind} mesh axes {unknown}; expected names from {tuple(axis_names)}.")
  return tuple(sizes.get(name, 1) for name in axis_names)


def _default_sizes(axis: str, kind: str, axis_names: Sequence[str]) -> dict[str, int]:
  if axis not in axis_names:
    raise ValueError(
        f"The default {kind} layout puts every device on {axis!r}, which is not one of the mesh axes "
        f"{tuple(axis_names)}; pass `{kind.lower()}=` explicitly."
    )
  return {axis: -1}


def create_mesh(
    *,
    ici: Mapping[str, int] | None = None,
    dcn: Mapping[str, int] | None = None,
    devices: Sequence[jax.Device] | None = None,
    axis_names: Sequence[str] = MESH_AXIS_NAMES,
) -> Mesh:
  axis_names = tuple(axis_names)
  ici_sizes = _shape_from_mapping(_default_sizes("fsdp", "ICI", axis_names) if ici is None else ici, "ICI", axis_names)
  dcn_sizes = _shape_from_mapping(_default_sizes("data", "DCN", axis_names) if dcn is None else dcn, "DCN", axis_names)
  devices = list(jax.devices() if devices is None else devices)
  if not devices:
    raise ValueError("Cannot build a mesh over zero devices.")
  num_slices = _count_slices(devices)
  if len(devices) % num_slices:
    raise ValueError(f"{len(devices)} devices do not divide evenly into {num_slices} slices.")
  ici_shape = _resolve_axis_sizes(ici_sizes, len(devices) // num_slices, "ICI", "per-slice device count", axis_names)
  dcn_shape = _resolve_axis_sizes(dcn_sizes, num_slices, "DCN", "slice count", axis_names)
  if num_slices > 1:
    device_array = mesh_utils.create_hybrid_device_mesh(ici_shape, dcn_shape, devices)
  else:
    device_array = mesh_utils.create_device_mesh(ici_shape, devices)
  return Mesh(device_array, axis_names)
