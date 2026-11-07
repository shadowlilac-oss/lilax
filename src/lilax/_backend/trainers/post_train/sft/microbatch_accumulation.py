"""MaxText gradient accumulation inside one Tunix SFT train step."""

import jax
import jax.numpy as jnp

MICROBATCH_MEAN_AUX_KEYS = ("moe_lb_loss", "indexer_loss", "mtp_loss")


def accumulate_microbatch_gradients(grad_fn, diff_params, rest, inputs, accumulation_steps):
  """Accumulates summed-loss gradients over microbatches and normalizes them by the total target-token count."""
  def to_microbatches(value):
    microbatch_shape = (value.shape[0] // accumulation_steps, accumulation_steps) + value.shape[1:]
    return jnp.swapaxes(jnp.reshape(value, microbatch_shape), 0, 1)

  def accumulate(carry, microbatch):
    (loss, (aux, rest)), grads = grad_fn(diff_params, carry["rest"], **microbatch)
    return {
        "loss": carry["loss"] + loss,
        "total_weights": carry["total_weights"] + aux["total_weights"],
        "grads": jax.tree.map(lambda total, grad: total + grad.astype(jnp.float32), carry["grads"], grads),
        "rest": rest,
    }, aux

  init = {
      "loss": 0.0, "total_weights": 0, "rest": rest,
      "grads": jax.tree.map(lambda param: jnp.zeros_like(param, dtype=jnp.float32), diff_params),
  }
  carry, aux = jax.lax.scan(accumulate, init, jax.tree.map(to_microbatches, inputs), length=accumulation_steps)
  has_weights = carry["total_weights"] > 0
  denominator = jnp.maximum(carry["total_weights"], 1)
  grads = jax.tree.map(
      lambda total, param: jnp.where(has_weights, total / denominator, 0.0).astype(param.dtype),
      carry["grads"], diff_params,
  )
  aux = jax.tree.map(lambda value: jnp.sum(value, axis=0), aux)
  for key in MICROBATCH_MEAN_AUX_KEYS:
    if key in aux:
      aux[key] = aux[key] / accumulation_steps
  return carry["loss"], aux, carry["rest"], grads
