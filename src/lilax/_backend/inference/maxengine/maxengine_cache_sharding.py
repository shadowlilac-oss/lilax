"""Initialize MaxEngine prefill and autoregressive KV-cache shardings."""

import jax

from lilax._backend.inference.maxengine import maxengine_prefill_cache
from lilax._backend.utils import maxtext_utils


def initialize_cache_shardings(engine):
  engine.prefill_kv_cache_annotations = maxtext_utils.get_prefill_kv_cache_annotations_nnx(
      engine.model, engine.config, engine._mesh
  )
  engine.prefill_kv_cache_shardings = jax.tree.map(
      lambda x: jax.sharding.NamedSharding(engine._mesh, x),
      engine.prefill_kv_cache_annotations,
  )
  if engine.config.stack_prefill_result_cache and not engine.config.scan_layers:
    engine.prefill_kv_cache_shardings = jax.tree.map(
        lambda x: jax.sharding.NamedSharding(engine._mesh, jax.sharding.PartitionSpec(None, *x.spec)),
        engine.prefill_kv_cache_shardings,
    )
    engine.prefill_kv_cache_shardings = maxengine_prefill_cache.stacked_prefill_cache_shardings(
        engine.prefill_kv_cache_shardings
    )
  engine.kv_cache_annotations = maxtext_utils.get_kv_cache_annotations_nnx(engine.model_ar, engine.config, engine._mesh)
  engine.kv_cache_shardings = jax.tree.map(
      lambda x: jax.sharding.NamedSharding(engine._mesh, x),
      engine.kv_cache_annotations,
  )
  engine.state_mesh_annotations = None
