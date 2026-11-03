"""Transformer Engine collective GEMM communicator bootstrap."""

import jax


def bootstrap_transformer_engine_cgemm(config):
  """Initialize communicators for Transformer Engine collective GEMM."""
  import transformer_engine.jax.cpp_extensions as tex

  tsp_size = config.ici_tensor_sequence_parallelism * config.dcn_tensor_sequence_parallelism
  tex.collective_gemm_bootstrap(jax.device_count(), jax.local_device_count(), jax.process_index(), tsp_size)
