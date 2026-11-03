"""FSDP weight all-gather for the DeepSeek FP8 batch-split schedule."""

import jax


def gather_weights(weights, mesh):
  """all-gathers FSDP sharded weights."""

  def fn(weights):
    (
        (pre_attn_norm, post_attn_norm),
        (wq_a, wq_b, q_norm, wkv_a, wkv_b, kv_norm, out),
    ), (
        (gate, bias),
        (routed_wi_0, routed_wi_1, routed_wo),
        (shared_wi_0, shared_wi_1, shared_wo),
    ) = weights
    # All-gather across FSDP axis. Expert axis is used for FSDP in attention.
    wq_a = jax.lax.all_gather(wq_a, axis_name="expert", tiled=True, axis=1)
    wq_a = jax.lax.all_gather(wq_a, axis_name="fsdp", tiled=True)
    wq_b = jax.lax.all_gather(wq_b, axis_name="expert", tiled=True, axis=1)
    wq_b = jax.lax.all_gather(wq_b, axis_name="fsdp", tiled=True)
    wkv_a = jax.lax.all_gather(wkv_a, axis_name="expert", tiled=True, axis=1)
    wkv_a = jax.lax.all_gather(wkv_a, axis_name="fsdp", tiled=True)
    wkv_b = jax.lax.all_gather(wkv_b, axis_name="expert", tiled=True, axis=1)
    wkv_b = jax.lax.all_gather(wkv_b, axis_name="fsdp", tiled=True)
    out = jax.lax.all_gather(out, axis_name="expert", tiled=True)
    out = jax.lax.all_gather(out, axis_name="fsdp", tiled=True, axis=2)
    gate = jax.lax.all_gather(gate, axis_name="fsdp", tiled=True)
    routed_wi_0 = jax.lax.all_gather(routed_wi_0, axis_name="fsdp", tiled=True)
    routed_wi_1 = jax.lax.all_gather(routed_wi_1, axis_name="fsdp", tiled=True)
    routed_wo = jax.lax.all_gather(routed_wo, axis_name="fsdp", tiled=True)
    shared_wi_0 = jax.lax.all_gather(shared_wi_0, axis_name="expert", tiled=True, axis=1)
    shared_wi_0 = jax.lax.all_gather(shared_wi_0, axis_name="fsdp", tiled=True)
    shared_wi_1 = jax.lax.all_gather(shared_wi_1, axis_name="expert", tiled=True, axis=1)
    shared_wi_1 = jax.lax.all_gather(shared_wi_1, axis_name="fsdp", tiled=True)
    shared_wo = jax.lax.all_gather(shared_wo, axis_name="expert", tiled=True)
    shared_wo = jax.lax.all_gather(shared_wo, axis_name="fsdp", tiled=True, axis=1)
    return (
        (
            (pre_attn_norm, post_attn_norm),
            (wq_a, wq_b, q_norm, wkv_a, wkv_b, kv_norm, out),
        ),
        (
            (gate, bias),
            (routed_wi_0, routed_wi_1, routed_wo),
            (shared_wi_0, shared_wi_1, shared_wo),
        ),
    )

  return jax.shard_map(
      fn,
      mesh=mesh,
      in_specs=(
          (
              (
                  (
                      jax.sharding.PartitionSpec(None),
                      jax.sharding.PartitionSpec(None),
                  ),
                  (
                      jax.sharding.PartitionSpec("fsdp", "expert"),
                      jax.sharding.PartitionSpec("fsdp", "expert", None),
                      jax.sharding.PartitionSpec(None),
                      jax.sharding.PartitionSpec("fsdp", "expert"),
                      jax.sharding.PartitionSpec("fsdp", "expert", None),
                      jax.sharding.PartitionSpec(None),
                      jax.sharding.PartitionSpec("expert", None, "fsdp"),
                  ),
              ),
              (
                  (
                      jax.sharding.PartitionSpec("fsdp", None),
                      jax.sharding.PartitionSpec(None),
                  ),
                  (
                      jax.sharding.PartitionSpec("fsdp", None, "expert"),
                      jax.sharding.PartitionSpec("fsdp", None, "expert"),
                      jax.sharding.PartitionSpec("fsdp", "expert", None),
                  ),
                  (
                      jax.sharding.PartitionSpec("fsdp", "expert"),
                      jax.sharding.PartitionSpec("fsdp", "expert"),
                      jax.sharding.PartitionSpec("expert", "fsdp"),
                  ),
              ),
          ),
      ),
      out_specs=(
          (
              (
                  jax.sharding.PartitionSpec(None),
                  jax.sharding.PartitionSpec(None),
              ),
              (
                  jax.sharding.PartitionSpec(None, None),
                  jax.sharding.PartitionSpec(None, None, None),
                  jax.sharding.PartitionSpec(None),
                  jax.sharding.PartitionSpec(None, None),
                  jax.sharding.PartitionSpec(None, None, None),
                  jax.sharding.PartitionSpec(None),
                  jax.sharding.PartitionSpec(None, None, None),
              ),
          ),
          (
              (
                  jax.sharding.PartitionSpec(None, None),
                  jax.sharding.PartitionSpec(None),
              ),
              (
                  jax.sharding.PartitionSpec(None, None, "expert"),
                  jax.sharding.PartitionSpec(None, None, "expert"),
                  jax.sharding.PartitionSpec(None, "expert", None),
              ),
              (
                  jax.sharding.PartitionSpec(None, None),
                  jax.sharding.PartitionSpec(None, None),
                  jax.sharding.PartitionSpec(None, None),
              ),
          ),
      ),
      check_vma=False,
  )(weights)
