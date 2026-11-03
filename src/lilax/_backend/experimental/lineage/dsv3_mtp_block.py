"""Multi-token prediction training and evaluation block of DSv3."""

from collections.abc import Callable, Sequence
import dataclasses
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

from lilax._backend.experimental.lineage import dsv3_embed
from lilax._backend.experimental.lineage import dsv3_loss
from lilax._backend.experimental.lineage import dsv3_types

T = TypeVar("T", dsv3_types.ArrayType, dsv3_types.ShardingType, dsv3_types.AxisNameType)
EmbedFn = Callable[[jax.Array], jax.Array]
LayerFn = Callable[
    [jax.Array, jax.Array, dsv3_types.DSv3MTPWeightsPytree, jax.Array | None],
    tuple[jax.Array, dsv3_types.DSv3RouterAux[Any]],
]
HeadFn = Callable[[jax.Array], jax.Array]


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3MTPDepthWeightsPytree(Generic[T]):
  """Weights of one MTP depth."""

  layer: dsv3_types.DSv3MTPWeightsPytree[T] = dataclasses.field(default_factory=dsv3_types.DSv3MTPWeightsPytree[T])
  final_norm_scale: T | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3MTPBlockOutputs:
  """Loss sums, valid counts, predictions, masks and router statistics."""

  losses: jax.Array | None
  weights: jax.Array | None
  preds: jax.Array | None
  mask: jax.Array | None
  aux: list[dsv3_types.DSv3RouterAux[Any]]


def _shift_left_one(x: jax.Array) -> jax.Array:
  """Left-shifts along the sequence axis and zeroes the last position."""
  sharding = jax.typeof(x).sharding
  spec = tuple(sharding.spec)
  if len(spec) > 1 and spec[1] is not None:
    gathered = jax.sharding.NamedSharding(sharding.mesh, jax.sharding.PartitionSpec(spec[0], None, *spec[2:]))
    return jax.reshard(_shift_left_one(jax.reshard(x, gathered)), sharding)
  local_rolled = jnp.roll(x, -1, axis=1)
  last_mask = jnp.arange(local_rolled.shape[1]) == local_rolled.shape[1] - 1
  last_mask = last_mask.reshape((1, -1) + (1,) * (local_rolled.ndim - 2))
  return jnp.where(last_mask, 0, local_rolled)


def _replicated(x: jax.Array) -> jax.Array:
  """Gathers an explicitly sharded array onto every device."""
  sharding = jax.typeof(x).sharding
  if all(axis is None for axis in sharding.spec):
    return x
  return jax.reshard(x, jax.sharding.NamedSharding(sharding.mesh, jax.sharding.PartitionSpec()))


def roll_and_mask(x: jax.Array, *, shift: int = -1) -> jax.Array:
  """Rolls left along the sequence axis and zeroes vacated positions."""
  if shift == 0:
    return x
  if shift == -1:
    return _shift_left_one(x)
  return jnp.roll(x, shift, axis=1).at[:, shift:, ...].set(0)


def roll_and_mask_by_segment(x: jax.Array, segment_ids: jax.Array | None) -> jax.Array:
  """Rolls one position left within packed documents."""
  if segment_ids is None:
    return roll_and_mask(x)
  rolled = _shift_left_one(x)
  seg_next = _shift_left_one(segment_ids)
  is_boundary = (segment_ids != seg_next) | (segment_ids == 0)
  mask = is_boundary[(...,) + (None,) * (x.ndim - 2)]
  return jnp.where(mask, 0, rolled)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_mtp_block(
    main_hidden_state: jt.Num[jax.Array, "B T D"],
    input_ids: jt.Int[jax.Array, "B T"],
    target_ids: jt.Int[jax.Array, "B T"],
    target_mask: jt.Int[jax.Array, "B T"],
    segment_ids: jt.Int[jax.Array, "B T"] | None,
    depth_weights: Sequence[DSv3MTPDepthWeightsPytree],
    *,
    embed_fn: EmbedFn,
    layer_fn: LayerFn,
    head_fn: HeadFn,
    dtype: jax.typing.DTypeLike,
    norm_epsilon: float,
    compute_losses: bool = True,
    eval_target_module: int = 0,
    expert_permutations: Sequence[jax.Array | None] | None = None,
    main_token_embeddings: jt.Num[jax.Array, "B T D"] | None = None,
    token_zero_embedding: jt.Num[jax.Array, "D"] | None = None,
) -> DSv3MTPBlockOutputs:
  """Runs the MTP depths and returns loss and evaluation data."""
  num_depths = len(depth_weights)
  if expert_permutations is None:
    expert_permutations = [None] * num_depths
  if len(expert_permutations) != num_depths:
    raise ValueError(f"Got {len(expert_permutations)} expert permutations for" f" {num_depths} MTP depths.")
  if (main_token_embeddings is None) != (token_zero_embedding is None):
    raise ValueError("main_token_embeddings and token_zero_embedding must be given together.")

  target_mask = (jnp.asarray(target_mask) != 0).astype(jnp.int32)
  mtp_hidden_state = main_hidden_state
  rolled_input_ids = input_ids
  rolled_target_ids = target_ids
  rolled_target_mask = target_mask != 0
  rolled_segment_ids = segment_ids
  losses = []
  weights = []
  preds = []
  masks = []
  aux_per_depth = []
  rolled_token_embeddings = main_token_embeddings
  not_filled = jnp.ones_like(input_ids, dtype=jnp.int32)
  if token_zero_embedding is not None:
    token_zero_embedding = _replicated(token_zero_embedding)

  for k in range(1, num_depths + 1):
    with jax.named_scope(f"mtp_depth_{k}"):
      if rolled_token_embeddings is not None and token_zero_embedding is not None:
        is_filled = roll_and_mask_by_segment(not_filled, rolled_segment_ids) == 0
        rolled_token_embeddings = jnp.where(
            is_filled[..., None], token_zero_embedding,
            roll_and_mask_by_segment(rolled_token_embeddings, rolled_segment_ids),
        )
      rolled_input_ids = roll_and_mask_by_segment(rolled_input_ids, rolled_segment_ids)
      rolled_target_ids = roll_and_mask_by_segment(rolled_target_ids, rolled_segment_ids)
      rolled_target_mask = roll_and_mask_by_segment(rolled_target_mask, rolled_segment_ids)
      if rolled_segment_ids is not None:
        rolled_segment_ids = roll_and_mask(rolled_segment_ids)

      if rolled_token_embeddings is not None:
        target_token_embedding = rolled_token_embeddings
      else:
        target_token_embedding = embed_fn(rolled_input_ids)
      target_token_embedding = target_token_embedding.astype(dtype)
      w = depth_weights[k - 1]
      mtp_hidden_state, aux = layer_fn(mtp_hidden_state, target_token_embedding, w.layer, expert_permutations[k - 1])
      aux_per_depth.append(aux)
      assert w.final_norm_scale is not None
      normed = dsv3_embed.final_norm(mtp_hidden_state, w.final_norm_scale, epsilon=norm_epsilon)
      mtp_logits = head_fn(normed)
      mtp_xent = dsv3_loss.cross_entropy_with_integer_labels(mtp_logits, rolled_target_ids)
      mtp_xent_masked = mtp_xent * rolled_target_mask
      if compute_losses:
        losses.append(jnp.sum(mtp_xent_masked))
        weights.append(jnp.sum(rolled_target_mask).astype(jnp.float32))
      if eval_target_module == k:
        preds.append(jnp.argmax(mtp_logits, axis=-1).astype(jnp.float32))
        masks.append(rolled_target_mask)

  return DSv3MTPBlockOutputs(
      losses=jnp.stack(losses) if losses else None,
      weights=jnp.stack(weights) if weights else None,
      preds=jnp.stack(preds) if preds else None,
      mask=jnp.stack(masks) if masks else None,
      aux=aux_per_depth,
  )


def mtp_loss(outputs: DSv3MTPBlockOutputs, *, scaling_factor: float) -> jax.Array | float:
  """Pools the depth losses as MaxText's calculate_mtp_loss."""
  if outputs.losses is None or outputs.weights is None:
    return 0.0
  avg = jnp.sum(outputs.losses) / (jnp.sum(outputs.weights) + dsv3_loss.EPS)
  return avg * scaling_factor


def mtp_acceptance_rate(
    outputs: DSv3MTPBlockOutputs, main_logits: jax.Array, *, eval_target_module: int,
) -> jax.Array | float:
  """Compares the target depth's predictions with the main predictions."""
  if outputs.preds is None or outputs.mask is None:
    return 0.0
  mtp_preds = outputs.preds.astype(jnp.int32)
  rolled_main_preds = jnp.argmax(main_logits, axis=-1)
  for _ in range(eval_target_module):
    rolled_main_preds = roll_and_mask(rolled_main_preds)
  correct = jnp.sum((mtp_preds == rolled_main_preds) * outputs.mask)
  total_valid = jnp.sum(outputs.mask)
  return (correct / (total_valid + dsv3_loss.EPS)) * 100
