"""MoE graphdef variants for dropless replay, evaluation, and first-phase training programs."""

import gc

from flax import nnx

from lilax._backend.observability import max_logging


def build_dropless_graphdef(config, jit_model, state):
  reconstructed = nnx.merge(jit_model, state)
  for _, module in nnx.iter_graph(reconstructed):
    if type(module).__name__ == "RoutedMoE":
      module.force_dropless = True
      module.num_moe_token_chunks = getattr(config, "retry_num_moe_token_chunks", 2)
      module.moe_chunk_barrier = True
    elif hasattr(module, "get_remat_policy"):
      module.remat_policy_override = "full"
      policy = module.get_remat_policy()
      for _, block in nnx.iter_graph(module):
        if hasattr(block, "remat_policy_fn"):
          block.remat_policy_override = module.remat_policy_override
          block.remat_policy_fn = policy
  jit_model_dropless, _ = nnx.split(reconstructed)
  del reconstructed, _
  gc.collect()
  return jit_model_dropless


def build_eval_graphdef(config, jit_model, state):
  if config.eval_ragged_buffer_factor <= 0:
    return None
  reconstructed = nnx.merge(jit_model, state)
  n_moe = 0
  for _, module in nnx.iter_graph(reconstructed):
    if type(module).__name__ == "RoutedMoE":
      module.ragged_buffer_factor_override = float(config.eval_ragged_buffer_factor)
      n_moe += 1
  max_logging.log(f"eval graphdef: ragged_buffer_factor_override={config.eval_ragged_buffer_factor} on {n_moe} RoutedMoE")
  jit_model_eval, _ = nnx.split(reconstructed)
  del reconstructed, _
  gc.collect()
  return jit_model_eval


def build_first_phase_graphdef(config, jit_model, state):
  if getattr(config, "first_phase_ragged_buffer_factor", 0.0) <= 0:
    return None
  reconstructed = nnx.merge(jit_model, state)
  n_moe = 0
  for _, module in nnx.iter_graph(reconstructed):
    if type(module).__name__ == "RoutedMoE":
      module.ragged_buffer_factor_override = float(config.first_phase_ragged_buffer_factor)
      n_moe += 1
  max_logging.log(
      f"first-phase graphdef: ragged_buffer_factor_override={config.first_phase_ragged_buffer_factor} on {n_moe} RoutedMoE"
  )
  jit_model_first_phase, _ = nnx.split(reconstructed)
  del reconstructed, _
  gc.collect()
  return jit_model_first_phase
