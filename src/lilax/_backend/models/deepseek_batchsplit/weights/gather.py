"""FSDP weight all-gather for the DeepSeek BF16 batch-split schedule."""

import jax


def gather_weights(weights, mesh):
  """all-gathers FSDP sharded weights."""

  def fn(weights):
    (
        pre_attn_norm,
        (wq_a, wq_b, q_norm, wkv_a, wkv_b, kv_norm, out),
    ), (
        post_attn_norm,
        (gate, bias),
        (routed_wi_0, routed_wi_1, routed_wo),
        (shared_wi_0, shared_wi_1, shared_wo),
    ) = weights
    wq_a = jax.lax.pcast(wq_a, axis_name="data", to="reduced")
    wq_b = jax.lax.pcast(wq_b, axis_name="data", to="reduced")
    wkv_a = jax.lax.pcast(wkv_a, axis_name="data", to="reduced")
    wkv_b = jax.lax.pcast(wkv_b, axis_name="data", to="reduced")
    out = jax.lax.pcast(out, axis_name="data", to="reduced")
    gate = jax.lax.pcast(gate, axis_name="data", to="reduced")
    routed_wi_0 = jax.lax.pcast(routed_wi_0, axis_name="data", to="reduced")
    routed_wi_1 = jax.lax.pcast(routed_wi_1, axis_name="data", to="reduced")
    routed_wo = jax.lax.pcast(routed_wo, axis_name="data", to="reduced")
    shared_wi_0 = jax.lax.pcast(shared_wi_0, axis_name="data", to="reduced")
    shared_wi_1 = jax.lax.pcast(shared_wi_1, axis_name="data", to="reduced")
    shared_wo = jax.lax.pcast(shared_wo, axis_name="data", to="reduced")
    # Cast to reduced across expert axis for all weights that are replicated
    # across the expert axis. This transposes to an all-reduce across the expert
    # axis in the backward pass.
    wq_a = jax.lax.pcast(wq_a, axis_name="expert", to="reduced")
    wq_b = jax.lax.pcast(wq_b, axis_name="expert", to="reduced")
    wkv_a = jax.lax.pcast(wkv_a, axis_name="expert", to="reduced")
    wkv_b = jax.lax.pcast(wkv_b, axis_name="expert", to="reduced")
    out = jax.lax.pcast(out, axis_name="expert", to="reduced")
    gate = jax.lax.pcast(gate, axis_name="expert", to="reduced")
    shared_wi_0 = jax.lax.pcast(shared_wi_0, axis_name="expert", to="reduced")
    shared_wi_1 = jax.lax.pcast(shared_wi_1, axis_name="expert", to="reduced")
    shared_wo = jax.lax.pcast(shared_wo, axis_name="expert", to="reduced")
    # All-gather across FSDP axis. Setting to="reduced" transposes to a
    # reduce-scatter across the FSDP axis in the backward pass.
    wq_a = jax.lax.all_gather(wq_a, axis_name="fsdp", tiled=True, to="reduced")
    wq_b = jax.lax.all_gather(wq_b, axis_name="fsdp", tiled=True, to="reduced")
    wkv_a = jax.lax.all_gather(wkv_a, axis_name="fsdp", tiled=True, to="reduced")
    wkv_b = jax.lax.all_gather(wkv_b, axis_name="fsdp", tiled=True, to="reduced")
    out = jax.lax.all_gather(out, axis_name="fsdp", tiled=True, axis=2, to="reduced")
    gate = jax.lax.all_gather(gate, axis_name="fsdp", tiled=True, to="reduced")
    routed_wi_0 = jax.lax.all_gather(routed_wi_0, axis_name="fsdp", tiled=True, axis=1, to="reduced")
    routed_wi_1 = jax.lax.all_gather(routed_wi_1, axis_name="fsdp", tiled=True, axis=1, to="reduced")
    routed_wo = jax.lax.all_gather(routed_wo, axis_name="fsdp", tiled=True, axis=2, to="reduced")
    shared_wi_0 = jax.lax.all_gather(shared_wi_0, axis_name="fsdp", tiled=True, to="reduced")
    shared_wi_1 = jax.lax.all_gather(shared_wi_1, axis_name="fsdp", tiled=True, to="reduced")
    shared_wo = jax.lax.all_gather(shared_wo, axis_name="fsdp", tiled=True, axis=1, to="reduced")
    return (
        (
            pre_attn_norm,
            (wq_a, wq_b, q_norm, wkv_a, wkv_b, kv_norm, out),
        ),
        (
            post_attn_norm,
            (gate, bias),
            (routed_wi_0, routed_wi_1, routed_wo),
            (shared_wi_0, shared_wi_1, shared_wo),
        ),
    )

  # Ensure weight AGs aren't fused with ops that use the weights.
  return jax.lax.optimization_barrier(
      jax.shard_map(
          fn,
          mesh=mesh,
          in_specs=(
              (
                  (
                      jax.sharding.PartitionSpec(None),
                      (
                          jax.sharding.PartitionSpec("fsdp", None),
                          jax.sharding.PartitionSpec("fsdp", None, None),
                          jax.sharding.PartitionSpec(None),
                          jax.sharding.PartitionSpec("fsdp", None),
                          jax.sharding.PartitionSpec("fsdp", None, None),
                          jax.sharding.PartitionSpec(None),
                          jax.sharding.PartitionSpec(None, None, "fsdp"),
                      ),
                  ),
                  (
                      jax.sharding.PartitionSpec(None),
                      (
                          jax.sharding.PartitionSpec("fsdp", None),
                          jax.sharding.PartitionSpec(None),
                      ),
                      (
                          jax.sharding.PartitionSpec("expert", "fsdp", None),
                          jax.sharding.PartitionSpec("expert", "fsdp", None),
                          jax.sharding.PartitionSpec("expert", None, "fsdp"),
                      ),
                      (
                          jax.sharding.PartitionSpec("fsdp", None),
                          jax.sharding.PartitionSpec("fsdp", None),
                          jax.sharding.PartitionSpec(None, "fsdp"),
                      ),
                  ),
              ),
          ),
          out_specs=(
              (
                  jax.sharding.PartitionSpec(None),
                  (
                      jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None, None, None, reduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None),
                      jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None, None, None, reduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None),
                      jax.sharding.PartitionSpec(None, None, None, reduced={"data", "fsdp", "expert"}),
                  ),
              ),
              (
                  jax.sharding.PartitionSpec(None),
                  (
                      jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None),
                  ),
                  (
                      jax.sharding.PartitionSpec("expert", None, None, reduced={"data", "fsdp"}),
                      jax.sharding.PartitionSpec("expert", None, None, reduced={"data", "fsdp"}),
                      jax.sharding.PartitionSpec("expert", None, None, reduced={"data", "fsdp"}),
                  ),
                  (
                      jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                  ),
              ),
          ),
          check_vma=True,
      )(weights)
  )
