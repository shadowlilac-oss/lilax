"""Activation sharding constraint for the DeepSeek FP8 batch-split schedule."""

import jax


def with_data_parallel_constraint(x, mesh):
  activation_pspec = jax.sharding.PartitionSpec(
      ("data", "fsdp", "fsdp_transpose", "expert", "context"),
      None,
      None,
  )
  return jax.lax.with_sharding_constraint(x, jax.NamedSharding(mesh, activation_pspec))
