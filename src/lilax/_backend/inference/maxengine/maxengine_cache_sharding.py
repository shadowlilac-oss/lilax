"""Initialize MaxEngine prefill and autoregressive KV-cache shardings."""

import jax

def named_cache_shardings(annotations, mesh):
  return jax.tree.map(lambda x: jax.sharding.NamedSharding(mesh, x), annotations)


def add_prefill_stack_axis(shardings, mesh):
  return jax.tree.map(
      lambda x: jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, *x.spec)),
      shardings,
  )
