"""Loss functions of the DSv3 training objective."""

import dataclasses
import functools
import operator
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

from lilax._backend.experimental.lineage import dsv3_types

T = TypeVar("T", dsv3_types.ArrayType, dsv3_types.ShardingType, dsv3_types.AxisNameType)
EPS = 1e-8


@jax.custom_vjp
def cross_entropy_with_logits(logits: jnp.ndarray, targets: jnp.ndarray, z_loss: float = 0.0):
  logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
  log_softmax = logits - logits_sum
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  log_z = jnp.squeeze(logits_sum, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss += total_z_loss
  return loss, total_z_loss


def _cross_entropy_with_logits_fwd(logits, targets, z_loss=0.0):
  max_logit = logits.max(axis=-1, keepdims=True)
  shifted = logits - max_logit
  exp_shifted = jnp.exp(shifted)
  sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
  log_softmax = shifted - jnp.log(sum_exp)
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  log_z = jnp.squeeze(jnp.log(sum_exp) + max_logit, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss += total_z_loss
  return (loss, total_z_loss), (logits, targets, z_loss, exp_shifted, sum_exp, log_z)


def _cross_entropy_with_logits_bwd(res, cotangents):
  g = cotangents[0]
  logits, targets, z_loss, exp_shifted, sum_exp, log_z = res
  deriv = jnp.expand_dims(1 + 2 * z_loss * log_z, -1) * exp_shifted / sum_exp - targets
  g_logits = jnp.expand_dims(g, axis=-1) * deriv
  return jnp.asarray(g_logits, logits.dtype), None, None


cross_entropy_with_logits.defvjp(_cross_entropy_with_logits_fwd, _cross_entropy_with_logits_bwd)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def cross_entropy_with_integer_labels(
    logits: jt.Float[jax.Array, "*batch V"], labels: jt.Int[jax.Array, "*batch"],
) -> jt.Float[jax.Array, "*batch"]:
  lse = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
  target = jnp.take_along_axis(logits, labels[..., None], axis=-1)
  return jnp.squeeze(lse - target, axis=-1)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3TokenLossSums(Generic[T]):
  xent: T | None = None
  z_loss: T | None = None
  weights: T | None = None


def _constrain(x: jax.Array, sharding: jax.sharding.NamedSharding) -> jax.Array:
  if sharding.mesh.abstract_mesh.are_all_axes_explicit:
    return jax.reshard(x, sharding)
  return jax.lax.with_sharding_constraint(x, sharding)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def masked_token_loss(
    logits: jt.Float[jax.Array, "B T V"], targets: jt.Int[jax.Array, "B T"],
    target_segmentation: jt.Int[jax.Array, "B T"], *, z_loss_multiplier: float = 0.0,
    token_sharding: jax.sharding.NamedSharding | None = None,
) -> DSv3TokenLossSums[jax.Array]:
  one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])
  xent, z_loss = cross_entropy_with_logits(logits, one_hot_targets, z_loss=z_loss_multiplier)
  if token_sharding is not None:
    xent = _constrain(xent, token_sharding)
    z_loss = _constrain(z_loss, token_sharding)
  valid = target_segmentation != 0
  xent = xent * valid
  z_loss = z_loss * valid
  return DSv3TokenLossSums[jax.Array](xent=jnp.sum(xent), z_loss=jnp.sum(z_loss), weights=jnp.sum(valid))


def _with_layer_axis(x: jax.Array, ndim_per_layer: int) -> tuple[jax.Array, bool]:
  scanned = x.ndim == ndim_per_layer + 1
  return (x if scanned else x[None]), scanned


@jax.named_call
def moe_switch_load_balance_loss(aux: dsv3_types.DSv3RouterAux[Any], *, num_experts: int,
                                 num_experts_per_tok: int, weight: float) -> jax.Array:
  selected, _ = _with_layer_axis(aux.selected_experts, 3)
  gates, _ = _with_layer_axis(aux.logits, 3)

  def layer_loss(selected_l, gates_l):
    probs = jax.nn.softmax(gates_l.astype(jnp.float32), axis=-1)
    expert_mask = jax.nn.one_hot(selected_l, num_classes=num_experts, dtype=jnp.float32)
    density = jnp.mean(jnp.sum(expert_mask, axis=2), axis=1) / num_experts_per_tok
    density_prob = jnp.mean(probs, axis=1)
    return jnp.mean(density * density_prob)

  num_layers = selected.shape[0]
  total = functools.reduce(operator.add, (layer_loss(selected[index], gates[index]) for index in range(num_layers)))
  return total / num_layers * (num_experts**2) * weight


@jax.named_call
def moe_megatron_seq_aux_loss(aux: dsv3_types.DSv3RouterAux[Any], *, num_experts: int,
                              num_experts_per_tok: int, weight: float) -> jax.Array:
  logits, scanned = _with_layer_axis(aux.logits, 3)

  def layer_loss(logits_l):
    scores = jnp.maximum(logits_l.astype(jnp.float32), 1e-6)
    probs = scores / (jnp.sum(scores, axis=-1, keepdims=True) + 1e-6)
    _, top_k_indices = jax.lax.top_k(scores, k=num_experts_per_tok)
    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=num_experts, dtype=jnp.int32)
    density = jnp.mean(jnp.sum(expert_mask, axis=2), axis=1) / num_experts_per_tok
    density_prob = jnp.mean(probs, axis=1)
    return jnp.mean(density * density_prob) * (num_experts**2) * weight

  losses = jax.vmap(layer_loss)(logits)
  return losses if scanned else losses[0]


def expert_counts(aux: dsv3_types.DSv3RouterAux[Any], *, num_experts: int) -> jax.Array:
  if aux.group_sizes is not None:
    return jnp.sum(aux.group_sizes, axis=-2, dtype=jnp.int32)
  selected, scanned = _with_layer_axis(aux.selected_experts, 3)
  counts = jnp.stack([
      jnp.sum(jax.nn.one_hot(selected[index], num_experts, dtype=jnp.int32), axis=(0, 1, 2))
      for index in range(selected.shape[0])
  ])
  return counts if scanned else counts[0]


def load_balance_updates_from_counts(counts: jax.Array, *, num_experts: int, rate: float) -> jax.Array:
  total_tokens = jnp.sum(counts, axis=-1, keepdims=True)
  average_load = total_tokens / num_experts
  return jnp.sign(average_load - counts) * rate


@jax.named_call
def routed_bias_updates(aux: dsv3_types.DSv3RouterAux[Any], *, num_experts: int, rate: float) -> jax.Array:
  counts = expert_counts(aux, num_experts=num_experts)
  return load_balance_updates_from_counts(counts, num_experts=num_experts, rate=rate)


def moe_router_aux_outputs(
    aux: dsv3_types.DSv3RouterAux[Any], *, num_experts: int, num_experts_per_tok: int,
    load_balance_loss_weight: float, megatron_seq_aux_loss: bool, routed_bias_update_rate: float,
) -> tuple[jax.Array | None, jax.Array | None]:
  lb_loss = None
  if load_balance_loss_weight > 0.0:
    loss_fn = moe_megatron_seq_aux_loss if megatron_seq_aux_loss else moe_switch_load_balance_loss
    lb_loss = loss_fn(aux, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok,
                      weight=load_balance_loss_weight)
  updates = None
  if routed_bias_update_rate > 0.0:
    updates = routed_bias_updates(aux, num_experts=num_experts, rate=routed_bias_update_rate)
  return lb_loss, updates
