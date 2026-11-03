"""Sparse MoE execution over sequence chunks of the shard-local tokens, with the in-layer dropless fallback."""

import functools

import jax
import jax.numpy as jnp


def _per_device_flag(has_overflow):
  return jnp.reshape(jnp.asarray(has_overflow, dtype=jnp.bool_), (1,))


class TokenChunkExecution:
  """Runs the sparse expert pass on the shard-local inputs, split into `n_chunks` sequence chunks."""

  def __init__(
      self,
      expert_pass,
      x,
      logits,
      pre_bias_logits,
      w0,
      w1,
      wo,
      w0_bias,
      w1_bias,
      wo_bias,
      sharded_input_ids,
      rngs,
      forced_routed_experts,
      per_expert_scale,
      *,
      config,
      n_chunks,
      barrier_enabled,
      defer_reductions,
      log_max_load_ratio,
      finalize_bias_updates,
  ):
    self.expert_pass = expert_pass
    self.x = x
    self.logits = logits
    self.pre_bias_logits = pre_bias_logits
    self.w0 = w0
    self.w1 = w1
    self.wo = wo
    self.w0_bias = w0_bias
    self.w1_bias = w1_bias
    self.wo_bias = wo_bias
    self.sharded_input_ids = sharded_input_ids
    self.rngs = rngs
    self.forced_routed_experts = forced_routed_experts
    self.per_expert_scale = per_expert_scale
    self.config = config
    self.n_chunks = n_chunks
    self.barrier_enabled = barrier_enabled
    self.defer_reductions = defer_reductions
    self.log_max_load_ratio = log_max_load_ratio
    self.finalize_bias_updates = finalize_bias_updates

  def _route_and_compute(self, force_dropless):
    """Runs route+compute once; force_dropless=True redoes all n_chunks, not just the overflowing one(s)."""
    n_chunks, barrier_enabled = self.n_chunks, self.barrier_enabled
    x, logits, pre_bias_logits = self.x, self.logits, self.pre_bias_logits
    w0_bias, w1_bias, wo_bias = self.w0_bias, self.w1_bias, self.wo_bias
    sharded_input_ids, forced_routed_experts = self.sharded_input_ids, self.forced_routed_experts
    if n_chunks <= 1 or not self.config().use_ring_of_experts:
      out, lb_loss, bias_updates, has_overflow, required_rbf, max_load_ratio, _ = self.expert_pass(
          x,
          logits,
          pre_bias_logits,
          self.w0,
          self.w1,
          self.wo,
          w0_bias,
          w1_bias,
          wo_bias,
          sharded_input_ids,
          self.rngs,
          forced_routed_experts,
          force_dropless=force_dropless,
          per_expert_scale=self.per_expert_scale,
      )
      if not self.defer_reductions:
        bias_updates = self.finalize_bias_updates(bias_updates)
      return out, lb_loss, bias_updates, has_overflow, required_rbf, max_load_ratio

    seq_len = x.shape[1]
    chunk = seq_len // n_chunks
    outs, lb_losses, bias_updates_list, has_overflows, required_rbfs, max_load_ratios = [], [], [], [], [], []
    _prev = None
    accum_chunk_wgrad = self.config().moe_accumulate_chunk_wgrad
    cur_w0, cur_w1, cur_wo = self.w0, self.w1, self.wo
    for c in range(n_chunks):
      sl = slice(c * chunk, (c + 1) * chunk)
      x_c = x[:, sl, :]
      if barrier_enabled and _prev is not None:
        x_c, _prev = jax.lax.optimization_barrier((x_c, _prev))
      out_c, lb_c, bu_c, ov_c, req_c, ratio_c, chunk_weights = self.expert_pass(
          x_c,
          logits[:, sl, :],
          None if pre_bias_logits is None else pre_bias_logits[:, sl, :],
          cur_w0,
          cur_w1,
          cur_wo,
          w0_bias,
          w1_bias,
          wo_bias,
          None if sharded_input_ids is None else sharded_input_ids[:, sl],
          self.rngs,
          None if forced_routed_experts is None else forced_routed_experts[:, sl, :],
          force_dropless=force_dropless,
          gather_weights=not accum_chunk_wgrad or c == 0,
          return_weights=accum_chunk_wgrad and c < n_chunks - 1,
          per_expert_scale=self.per_expert_scale,
      )
      if accum_chunk_wgrad:
        cur_w0, cur_w1, cur_wo = chunk_weights
      if barrier_enabled:
        _prev = out_c
      outs.append(out_c)
      lb_losses.append(lb_c)
      bias_updates_list.append(bu_c)
      has_overflows.append(ov_c)
      required_rbfs.append(req_c)
      max_load_ratios.append(ratio_c)
    output = jnp.concatenate(outs, axis=1)
    lb_loss = None if lb_losses[0] is None else sum(lb_losses) / n_chunks
    if bias_updates_list[0] is None:
      bias_updates = None
    elif self.defer_reductions:
      bias_updates = jnp.stack(bias_updates_list)
    else:
      bias_updates = self.finalize_bias_updates(sum(bias_updates_list))
    has_overflow = jnp.any(jnp.stack(has_overflows))
    required_rbf = None if required_rbfs[0] is None else jnp.max(jnp.stack(required_rbfs))
    max_load_ratio = None if max_load_ratios[0] is None else jnp.max(jnp.stack(max_load_ratios))
    return output, lb_loss, bias_updates, has_overflow, required_rbf, max_load_ratio

  def _dropless_chunks(self, n_chunks):
    x = self.x
    batch, seq_len = x.shape[0], x.shape[1]
    chunk = seq_len // n_chunks

    def split(a):
      if a is None:
        return None
      return jnp.moveaxis(a.reshape((batch, n_chunks, chunk) + a.shape[2:]), 1, 0)

    def body(carry, xs):
      x_c, logits_c, pre_bias_logits_c, input_ids_c, forced_c = xs
      out_c, lb_c, bu_c, _, req_c, ratio_c, _ = self.expert_pass(
          x_c,
          logits_c,
          pre_bias_logits_c,
          self.w0,
          self.w1,
          self.wo,
          self.w0_bias,
          self.w1_bias,
          self.wo_bias,
          input_ids_c,
          self.rngs,
          forced_c,
          force_dropless=True,
          per_expert_scale=self.per_expert_scale,
      )
      return carry, (out_c, lb_c, bu_c, req_c, ratio_c)

    xs = tuple(
        split(a) for a in (x, self.logits, self.pre_bias_logits, self.sharded_input_ids, self.forced_routed_experts)
    )
    body = jax.checkpoint(body, policy=jax.checkpoint_policies.nothing_saveable)
    _, (outs, lb_losses, bias_updates, required_rbfs, max_load_ratios) = jax.lax.scan(body, None, xs)
    output = jnp.moveaxis(outs, 0, 1).reshape((batch, seq_len) + outs.shape[3:])
    lb_loss = None if lb_losses is None else jnp.mean(lb_losses, axis=0)
    bias_updates = None if bias_updates is None else self.finalize_bias_updates(jnp.sum(bias_updates, axis=0))
    required_rbf = None if required_rbfs is None else jnp.max(required_rbfs)
    max_load_ratio = None if max_load_ratios is None else jnp.max(max_load_ratios)
    return output, lb_loss, bias_updates, jnp.bool_(False), required_rbf, max_load_ratio

  def without_fallback(self, force_dropless):
    """Runs the chunks once, capped by the ragged buffer unless `force_dropless`."""
    out, lb_loss, bias_updates, has_overflow, required_rbf, max_load_ratio = self._route_and_compute(force_dropless)
    has_overflow = _per_device_flag(has_overflow)
    if self.defer_reductions and bias_updates is not None:
      if bias_updates.ndim == 1:
        bias_updates = bias_updates[None, :]
      bias_updates = bias_updates[None]
    if self.log_max_load_ratio:
      return out, lb_loss, bias_updates, has_overflow, jnp.bool_(False), required_rbf, max_load_ratio
    return out, lb_loss, bias_updates, has_overflow, jnp.bool_(False), required_rbf

  def with_dropless_fallback(self, took_fallback, fallback_chunks):
    """Reruns the batch dropless in `fallback_chunks` chunks when the capped ragged buffer would overflow."""

    def _capped_branch():
      out, lb_loss, bias_updates, has_overflow, required_rbf, max_load_ratio = self._route_and_compute(False)
      return out, lb_loss, bias_updates, jnp.asarray(has_overflow, dtype=jnp.bool_), required_rbf, max_load_ratio

    @functools.partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
    def _dropless_branch():
      return self._dropless_chunks(fallback_chunks)

    out, lb_loss, bias_updates, has_overflow, required_rbf, max_load_ratio = jax.lax.cond(
        took_fallback, _dropless_branch, _capped_branch
    )
    has_overflow = _per_device_flag(has_overflow)
    if self.log_max_load_ratio:
      return out, lb_loss, bias_updates, has_overflow, took_fallback, required_rbf, max_load_ratio
    return out, lb_loss, bias_updates, has_overflow, took_fallback, required_rbf
