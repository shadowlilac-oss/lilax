"""Token embedding, final norm and LM head of DSv3."""

from collections.abc import Mapping
import dataclasses
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

from lilax._backend.experimental.lineage import dsv3_types, ops
from lilax._backend.experimental.lineage.sharding import partition_specs

ArrayType = dsv3_types.ArrayType
ShardingType = dsv3_types.ShardingType
Initializer = dsv3_types.Initializer
T = TypeVar("T", ArrayType, ShardingType, dsv3_types.AxisNameType)
default_embedding_init = jax.nn.initializers.normal(stddev=1.0)
default_head_kernel_init = dsv3_types.default_kernel_init


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3EmbedWeightsPytree(Generic[T]):
  table: T | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3HeadWeightsPytree(Generic[T]):
  final_norm_scale: T | None = None
  kernel: T | None = None


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def embed_tokens(tokens: jt.Int[jax.Array, "*batch"], table: jt.Num[jax.Array, "V D"], *,
                 dtype: jax.typing.DTypeLike, iota_embed: bool = False,
                 out_sharding: jax.sharding.NamedSharding | None = None) -> jt.Num[jax.Array, "*batch D"]:
  embedding = jnp.asarray(table, dtype)
  if iota_embed:
    iota = jax.lax.iota(jnp.int32, embedding.shape[0])
    one_hot = jnp.array(tokens[..., jnp.newaxis] == iota, dtype=dtype)
    return jnp.dot(one_hot, embedding, out_sharding=out_sharding)
  return embedding.at[tokens].get(out_sharding=out_sharding)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def embed_tokens_collected(tokens: jt.Int[jax.Array, "*batch"], table: jt.Num[jax.Array, "V D"], *,
                           dtype: jax.typing.DTypeLike, iota_embed: bool = False) -> jt.Num[jax.Array, "*batch D"]:
  table_spec = jax.typeof(table).sharding.spec
  if table_spec and table_spec[0] is not None:
    raise ValueError(f"The vocab dim of the table must be replicated: {table}")
  emb_axes = table_spec[1] if len(table_spec) > 1 else None
  tokens_spec = tuple(jax.typeof(tokens).sharding.spec)
  tokens_spec += (None,) * (tokens.ndim - len(tokens_spec))

  def _lookup(tokens_local, table_local):
    table_local = jnp.asarray(table_local, dtype)
    if emb_axes is not None:
      shards = jax.lax.all_gather(table_local.T, emb_axes, axis=0)
      table_local = shards.reshape(-1, shards.shape[-1]).T
    return embed_tokens(tokens_local, table_local, dtype=dtype, iota_embed=iota_embed)

  return jax.shard_map(
      _lookup, mesh=jax.typeof(table).sharding.mesh,
      in_specs=(jax.sharding.PartitionSpec(*tokens_spec), table_spec),
      out_specs=jax.sharding.PartitionSpec(*tokens_spec, None),
  )(tokens, table)


@jt.jaxtyped(typechecker=typeguard.typechecked)
def embed_single_token(table: jt.Num[jax.Array, "V D"], token_id: int, *, dtype: jax.typing.DTypeLike,
                       iota_embed: bool = False) -> jt.Num[jax.Array, "D"]:
  row = jnp.asarray(table, dtype)[token_id]
  if iota_embed:
    row = jnp.where(row == 0, jnp.zeros_like(row), row)
  return row


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def final_norm(x: jt.Num[jax.Array, "*batch D"], scale: jt.Num[jax.Array, "D"], *,
               epsilon: float) -> jt.Num[jax.Array, "*batch D"]:
  return ops.rms_norm(x, jnp.asarray(scale, x.dtype), epsilon=epsilon)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def lm_head(x: jt.Num[jax.Array, "*batch D"], kernel: jt.Num[jax.Array, "D V"], *,
            dtype: jax.typing.DTypeLike, dot_in_fp32: bool = False, cast_logits_to_fp32: bool = True,
            precision: str = "default", out_sharding: jax.sharding.NamedSharding | None = None) -> jt.Num[jax.Array, "*batch V"]:
  dot_dtype = jnp.float32 if dot_in_fp32 else dtype
  inputs = jnp.asarray(x, dot_dtype)
  weights = jnp.asarray(kernel, dot_dtype)
  logits = jax.lax.dot_general(inputs, weights, (((inputs.ndim - 1,), (0,)), ((), ())),
                               precision=jax.lax.Precision(precision), out_sharding=out_sharding)
  if cast_logits_to_fp32:
    logits = logits.astype(jnp.float32)
  return logits


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def collect_head_kernel(kernel: jt.Num[jax.Array, "D V"], *, dtype: jax.typing.DTypeLike) -> jt.Num[jax.Array, "D V"]:
  spec = jax.typeof(kernel).sharding.spec
  if len(spec) > 1 and spec[1] is not None:
    raise ValueError(f"The vocab dim of the kernel must be replicated: {kernel}")
  emb_axes = spec[0] if spec else None
  emb_axes = (emb_axes,) if isinstance(emb_axes, str) else tuple(emb_axes or ())
  mesh = jax.typeof(kernel).sharding.mesh
  if not emb_axes and "dcn" not in mesh.axis_names:
    return jnp.asarray(kernel, dtype)
  replicated_axes = tuple(ax for ax in mesh.axis_names if ax not in emb_axes)

  def _gather(kernel_local):
    kernel_local = jnp.asarray(kernel_local, dtype)
    for ax in replicated_axes:
      kernel_local = jax.lax.pcast(kernel_local, ax, to="reduced")
    if not emb_axes:
      return kernel_local
    return jax.lax.all_gather(kernel_local, emb_axes, axis=0, to="reduced")

  out_ndim = 2 if not emb_axes else 3
  collected = jax.shard_map(
      _gather, mesh=mesh, in_specs=(spec,),
      out_specs=jax.sharding.PartitionSpec(*(None,) * out_ndim, reduced=frozenset(emb_axes + replicated_axes)),
  )(kernel)
  return collected.reshape(kernel.shape)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def tied_lm_head(x: jt.Num[jax.Array, "*batch D"], table: jt.Num[jax.Array, "V D"], *,
                 dtype: jax.typing.DTypeLike, dot_in_fp32: bool = False, cast_logits_to_fp32: bool = True,
                 normalize_logits: bool = True, soft_cap: float | None = None,
                 out_sharding: jax.sharding.NamedSharding | None = None) -> jt.Num[jax.Array, "*batch V"]:
  attend_dtype = jnp.float32 if dot_in_fp32 else dtype
  logits = jnp.dot(x, jnp.asarray(table, jnp.bfloat16).T, preferred_element_type=attend_dtype, out_sharding=out_sharding)
  if normalize_logits:
    logits = logits / jnp.sqrt(x.shape[-1])
  if soft_cap:
    logits = jnp.tanh(logits / soft_cap) * soft_cap
  if cast_logits_to_fp32:
    logits = logits.astype(jnp.float32)
  return logits


def init_dsv3_embed_weights(
    rng: jax.Array, vocab_size: int, emb_dim: int, mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]], *, weight_dtype: jax.typing.DTypeLike = jnp.float32,
    out_shardings: DSv3EmbedWeightsPytree[ShardingType] | None = None,
    embedding_init: Initializer = default_embedding_init,
) -> DSv3EmbedWeightsPytree[ArrayType]:
  shardings = _resolve(out_shardings, mesh, axis_mapping)
  shardings = shardings or DSv3EmbedWeightsPytree[ShardingType]()
  return DSv3EmbedWeightsPytree[ArrayType](
      table=embedding_init(rng, (vocab_size, emb_dim), weight_dtype, out_sharding=shardings.table)
  )


def init_dsv3_head_weights(
    rng: jax.Array, vocab_size: int, emb_dim: int, mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]], *, weight_dtype: jax.typing.DTypeLike = jnp.float32,
    tied_head: bool = False, out_shardings: DSv3HeadWeightsPytree[ShardingType] | None = None,
    kernel_init: Initializer = default_head_kernel_init, scale_init: Initializer = dsv3_types.default_scale_init,
) -> DSv3HeadWeightsPytree[ArrayType]:
  shardings = _resolve(out_shardings, mesh, axis_mapping)
  shardings = shardings or DSv3HeadWeightsPytree[ShardingType]()
  scale_key, kernel_key = jax.random.split(rng)
  kernel = None
  if not tied_head:
    kernel = kernel_init(kernel_key, (emb_dim, vocab_size), weight_dtype, out_sharding=shardings.kernel)
  return DSv3HeadWeightsPytree[ArrayType](
      final_norm_scale=scale_init(scale_key, (emb_dim,), weight_dtype, out_sharding=shardings.final_norm_scale),
      kernel=kernel,
  )


def _resolve(out_shardings: Any, mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
             axis_mapping: Mapping[str, str | tuple[str, ...]]) -> Any:
  if out_shardings is None:
    return None

  def _named(sharding):
    if not isinstance(sharding, jax.sharding.PartitionSpec):
      return sharding
    return jax.sharding.NamedSharding(mesh, partition_specs.physical_pspec(sharding, axis_mapping))

  return jax.tree.map(_named, out_shardings, is_leaf=lambda sharding: isinstance(sharding, jax.sharding.PartitionSpec))
