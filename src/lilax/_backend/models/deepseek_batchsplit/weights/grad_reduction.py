"""Weight gradient reduction collectives for the DeepSeek BF16 batch-split schedule."""

import jax


def reduce_scatter_ws_grad(ws_grad, mesh):
  """reduce-scatters weight gradients to FSDP sharding."""

  # Ensure grad RS/ARs aren't fused with ops that generated them.
  ws_grad = jax.lax.optimization_barrier(ws_grad)

  def fn(ws_grad):
    (
        pre_attn_norm,
        (wq_a, wq_b, q_norm, wkv_a, wkv_b, kv_norm, out),
    ), (
        post_attn_norm,
        (gate, bias),
        (routed_wi_0, routed_wi_1, routed_wo),
        (shared_wi_0, shared_wi_1, shared_wo),
    ) = ws_grad
    # Reduce-scatter across FSDP axis.
    wq_a = jax.lax.psum_scatter(wq_a, axis_name="fsdp", tiled=True)
    wq_b = jax.lax.psum_scatter(wq_b, axis_name="fsdp", tiled=True)
    wkv_a = jax.lax.psum_scatter(wkv_a, axis_name="fsdp", tiled=True)
    wkv_b = jax.lax.psum_scatter(wkv_b, axis_name="fsdp", tiled=True)
    out = jax.lax.psum_scatter(out, axis_name="fsdp", tiled=True, scatter_dimension=2)
    gate = jax.lax.psum_scatter(gate, axis_name="fsdp", tiled=True)
    routed_wi_0 = jax.lax.psum_scatter(routed_wi_0, axis_name="fsdp", tiled=True, scatter_dimension=1)
    routed_wi_1 = jax.lax.psum_scatter(routed_wi_1, axis_name="fsdp", tiled=True, scatter_dimension=1)
    routed_wo = jax.lax.psum_scatter(routed_wo, axis_name="fsdp", tiled=True, scatter_dimension=2)
    shared_wi_0 = jax.lax.psum_scatter(shared_wi_0, axis_name="fsdp", tiled=True)
    shared_wi_1 = jax.lax.psum_scatter(shared_wi_1, axis_name="fsdp", tiled=True)
    shared_wo = jax.lax.psum_scatter(shared_wo, axis_name="fsdp", tiled=True, scatter_dimension=1)
    # All-reduce across expert axis.
    wq_a = jax.lax.psum(wq_a, axis_name="expert")
    wq_b = jax.lax.psum(wq_b, axis_name="expert")
    wkv_a = jax.lax.psum(wkv_a, axis_name="expert")
    wkv_b = jax.lax.psum(wkv_b, axis_name="expert")
    out = jax.lax.psum(out, axis_name="expert")
    gate = jax.lax.psum(gate, axis_name="expert")
    shared_wi_0 = jax.lax.psum(shared_wi_0, axis_name="expert")
    shared_wi_1 = jax.lax.psum(shared_wi_1, axis_name="expert")
    shared_wo = jax.lax.psum(shared_wo, axis_name="expert")
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

  return jax.shard_map(
      fn,
      mesh=mesh,
      in_specs=(
          (
              (
                  jax.sharding.PartitionSpec(None),
                  (
                      jax.sharding.PartitionSpec(None, None, unreduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None, None, None, unreduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None),
                      jax.sharding.PartitionSpec(None, None, unreduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None, None, None, unreduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None),
                      jax.sharding.PartitionSpec(None, None, None, unreduced={"data", "fsdp", "expert"}),
                  ),
              ),
              (
                  jax.sharding.PartitionSpec(None),
                  (
                      jax.sharding.PartitionSpec(None, None, unreduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None),
                  ),
                  (
                      jax.sharding.PartitionSpec("expert", None, None, unreduced={"data", "fsdp"}),
                      jax.sharding.PartitionSpec("expert", None, None, unreduced={"data", "fsdp"}),
                      jax.sharding.PartitionSpec("expert", None, None, unreduced={"data", "fsdp"}),
                  ),
                  (
                      jax.sharding.PartitionSpec(None, None, unreduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None, None, unreduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None, None, unreduced={"data", "fsdp", "expert"}),
                  ),
              ),
          ),
      ),
      out_specs=(
          (
              jax.sharding.PartitionSpec(None),
              (
                  jax.sharding.PartitionSpec("fsdp", None, unreduced={"data"}),
                  jax.sharding.PartitionSpec("fsdp", None, None, unreduced={"data"}),
                  jax.sharding.PartitionSpec(None),
                  jax.sharding.PartitionSpec("fsdp", None, unreduced={"data"}),
                  jax.sharding.PartitionSpec("fsdp", None, None, unreduced={"data"}),
                  jax.sharding.PartitionSpec(None),
                  jax.sharding.PartitionSpec(None, None, "fsdp", unreduced={"data"}),
              ),
          ),
          (
              jax.sharding.PartitionSpec(None),
              (
                  jax.sharding.PartitionSpec("fsdp", None, unreduced={"data"}),
                  jax.sharding.PartitionSpec(None),
              ),
              (
                  jax.sharding.PartitionSpec("expert", "fsdp", None, unreduced={"data"}),
                  jax.sharding.PartitionSpec("expert", "fsdp", None, unreduced={"data"}),
                  jax.sharding.PartitionSpec("expert", None, "fsdp", unreduced={"data"}),
              ),
              (
                  jax.sharding.PartitionSpec("fsdp", None, unreduced={"data"}),
                  jax.sharding.PartitionSpec("fsdp", None, unreduced={"data"}),
                  jax.sharding.PartitionSpec(None, "fsdp", unreduced={"data"}),
              ),
          ),
      ),
      check_vma=True,
  )(ws_grad)


def all_reduce_ws_grad_dcn(ws_grad, mesh):
  """all-reduces weight gradients across DCN axes."""

  # Ensure grad RS/ARs aren't fused with ops that generated them.
  ws_grad = jax.lax.optimization_barrier(ws_grad)

  def fn(ws_grad):
    (
        pre_attn_norm,
        (wq_a, wq_b, q_norm, wkv_a, wkv_b, kv_norm, out),
    ), (
        post_attn_norm,
        (gate, bias),
        (routed_wi_0, routed_wi_1, routed_wo),
        (shared_wi_0, shared_wi_1, shared_wo),
    ) = ws_grad
    # All-reduce across data axis.
    wq_a = jax.lax.psum(wq_a, axis_name="data")
    wq_b = jax.lax.psum(wq_b, axis_name="data")
    wkv_a = jax.lax.psum(wkv_a, axis_name="data")
    wkv_b = jax.lax.psum(wkv_b, axis_name="data")
    out = jax.lax.psum(out, axis_name="data")
    gate = jax.lax.psum(gate, axis_name="data")
    routed_wi_0 = jax.lax.psum(routed_wi_0, axis_name="data")
    routed_wi_1 = jax.lax.psum(routed_wi_1, axis_name="data")
    routed_wo = jax.lax.psum(routed_wo, axis_name="data")
    shared_wi_0 = jax.lax.psum(shared_wi_0, axis_name="data")
    shared_wi_1 = jax.lax.psum(shared_wi_1, axis_name="data")
    shared_wo = jax.lax.psum(shared_wo, axis_name="data")
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

  return jax.shard_map(
      fn,
      mesh=mesh,
      in_specs=(
          (
              (
                  jax.sharding.PartitionSpec(None),
                  (
                      jax.sharding.PartitionSpec("fsdp", None, unreduced={"data"}),
                      jax.sharding.PartitionSpec("fsdp", None, None, unreduced={"data"}),
                      jax.sharding.PartitionSpec(None),
                      jax.sharding.PartitionSpec("fsdp", None, unreduced={"data"}),
                      jax.sharding.PartitionSpec("fsdp", None, None, unreduced={"data"}),
                      jax.sharding.PartitionSpec(None),
                      jax.sharding.PartitionSpec(None, None, "fsdp", unreduced={"data"}),
                  ),
              ),
              (
                  jax.sharding.PartitionSpec(None),
                  (
                      jax.sharding.PartitionSpec("fsdp", None, unreduced={"data"}),
                      jax.sharding.PartitionSpec(None),
                  ),
                  (
                      jax.sharding.PartitionSpec("expert", "fsdp", None, unreduced={"data"}),
                      jax.sharding.PartitionSpec("expert", "fsdp", None, unreduced={"data"}),
                      jax.sharding.PartitionSpec("expert", None, "fsdp", unreduced={"data"}),
                  ),
                  (
                      jax.sharding.PartitionSpec("fsdp", None, unreduced={"data"}),
                      jax.sharding.PartitionSpec("fsdp", None, unreduced={"data"}),
                      jax.sharding.PartitionSpec(None, "fsdp", unreduced={"data"}),
                  ),
              ),
          ),
      ),
      out_specs=(
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
      check_vma=True,
  )(ws_grad)
