"""Transformer Engine mesh resource context for MaxText mesh axes."""

from contextlib import contextmanager, nullcontext


@contextmanager
def maybe_get_transformer_engine_context(config):
  """Enter the Transformer Engine mesh context on GPUs."""
  if config.hardware in ["gpu", "gpu_multiprocess"]:
    with transformer_engine_context():
      yield
  else:
    with nullcontext():
      yield


@contextmanager
def transformer_engine_context():
  """Provide MaxText mesh resources to Transformer Engine."""
  from transformer_engine.jax.sharding import global_shard_guard, MeshResource

  mesh_resource = MeshResource(
      dp_resource="data",
      tp_resource="tensor",
      tpsp_resource="tensor_sequence",
      fsdp_resource="fsdp",
      pp_resource=None,
      cp_resource="context",
      ep_resource="expert",
  )
  with global_shard_guard(mesh_resource):
    yield
