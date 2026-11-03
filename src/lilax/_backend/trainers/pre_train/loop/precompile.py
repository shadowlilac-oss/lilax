"""Ahead-of-time lowering and compilation of the native pretraining loop's programs before the first step."""

from flax.core.spmd import logical_axis_rules
import jax

from lilax._backend.utils import maxtext_utils, sharding
from lilax._backend.utils.diagnostics import jaxpr_dump, pytree
from lilax._backend.utils.xla_options import libtpu_flags


def aot_compile_step(p_step, lower_args, compiler_options, prefix):
  compiled = p_step.lower(*lower_args).compile(compiler_options=compiler_options)
  pytree.print_compiled_memory_stats(compiled.memory_analysis(), prefix=prefix)
  return compiled


def precompile_train_programs(config, mesh, state, state_mesh_shardings, init_rng, programs):
  precompiled_train = False
  with jax.set_mesh(mesh), logical_axis_rules(config.logical_axis_rules):
    data_sharding = sharding.get_input_data_sharding(config, mesh)
    shaped_batch = maxtext_utils.get_shaped_batch(config, batch_sharding=data_sharding)
    if config.shard_optimizer_over_data:
      state = sharding.reshard_train_state(state, state_mesh_shardings, config.enable_single_controller)
    if config.enable_diloco:
      lower_args = (state, shaped_batch, init_rng)
    else:
      lower_args = (state, shaped_batch)
    jaxpr_dump.maybe_dump_jaxpr(config, programs["p_train_step"], lower_args)
    if config.compiled_trainstep_file == "" and not jax.config.jax_enable_pgle:
      compiler_options = libtpu_flags.parse_libtpu_flags_to_dict(config.compile_xla_flags)
      aot_compile_step(programs["p_train_step"], lower_args, compiler_options, prefix="train")
      precompiled_train = True
      if programs["p_train_step_dropless"] is not None:
        aot_compile_step(programs["p_train_step_dropless"], lower_args, compiler_options, prefix="train_dropless")
      if programs["p_train_step_first_phase"] is not None:
        aot_compile_step(programs["p_train_step_first_phase"], lower_args, compiler_options, prefix="train_first_phase")
  return state, data_sharding, shaped_batch, precompiled_train


def build_shaped_eval_batch(config, mesh, p_eval_step):
  if p_eval_step is None:
    return None
  with jax.set_mesh(mesh), logical_axis_rules(config.logical_axis_rules_for_eval):
    data_sharding_eval = sharding.get_input_data_sharding(config, mesh, rules=config.logical_axis_rules_for_eval)
    return maxtext_utils.get_shaped_batch(config, batch_sharding=data_sharding_eval, is_eval=True)


def precompile_eval_programs(config, mesh, state, init_rng, shaped_eval_batch, programs):
  if programs["p_eval_step"] is None or config.compiled_trainstep_file != "" or jax.config.jax_enable_pgle:
    return False
  with jax.set_mesh(mesh), logical_axis_rules(config.logical_axis_rules_for_eval):
    compiler_options = libtpu_flags.parse_libtpu_flags_to_dict(config.compile_xla_flags)
    if config.enable_diloco:
      eval_lower_args = (state, shaped_eval_batch, init_rng)
    else:
      eval_lower_args = (state, shaped_eval_batch)
    aot_compile_step(programs["p_eval_step"], eval_lower_args, compiler_options, prefix="eval")
    if programs["p_eval_step_dropless"] is not None:
      aot_compile_step(programs["p_eval_step_dropless"], eval_lower_args, compiler_options, prefix="eval_dropless")
  return True
