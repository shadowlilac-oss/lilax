"""Synthetic-batch warmup of precompiled pretraining programs before the run clock starts."""

import gc
import time

from flax.core.spmd import logical_axis_rules
import jax
import jax.numpy as jnp

from lilax._backend.observability import max_logging


def make_synthetic_batch(config, shaped_batch, seed=0):
  names = sorted(shaped_batch)
  seq_len = int(getattr(config, "max_target_length", 0))

  def _seq_axis(shape):
    axes = [index for index, size in enumerate(shape) if size == seq_len and index > 0]
    return axes[0] if axes else len(shape) - 1

  def _gen(key):
    out = {}
    keys = jax.random.split(key, len(names))
    for sub_key, name in zip(keys, names):
      shape, dtype = shaped_batch[name].shape, shaped_batch[name].dtype
      if name in ("inputs", "targets"):
        out[name] = jax.random.randint(sub_key, shape, 0, max(int(config.vocab_size), 1), dtype=dtype)
      elif name.endswith("_position"):
        out[name] = jax.lax.broadcasted_iota(dtype, shape, _seq_axis(shape))
      elif jnp.issubdtype(dtype, jnp.floating):
        out[name] = jnp.zeros(shape, dtype)
      else:
        out[name] = jnp.ones(shape, dtype)
    return out

  out_shardings = {name: getattr(shaped_batch[name], "sharding", None) for name in names}
  return jax.jit(_gen, out_shardings=out_shardings)(jax.random.PRNGKey(seed))


def _array_leaf_ids(tree):
  return {id(leaf) for leaf in jax.tree_util.tree_leaves(tree) if isinstance(leaf, jax.Array)}


def _delete_outputs(outputs, keep_ids):
  for leaf in jax.tree_util.tree_leaves(outputs):
    if isinstance(leaf, jax.Array) and id(leaf) not in keep_ids and not leaf.is_deleted():
      leaf.delete()


def warmup_programs(
    programs, state, mesh, train_rules, eval_rules, train_batch, eval_batch, rng_args=(), copy_train_state=False
):
  keep_ids = _array_leaf_ids(state)
  timings = {}
  for name, kind, p_step in programs:
    if p_step is None:
      continue
    rules, batch = (train_rules, train_batch) if kind == "train" else (eval_rules, eval_batch)
    if batch is None:
      max_logging.log(f"warmup_programs_in_init: skipping {name} (no synthetic {kind} batch)")
      continue
    start = time.perf_counter()
    with jax.set_mesh(mesh), logical_axis_rules(rules):
      arg_state = state
      if kind == "train" and copy_train_state:
        arg_state = jax.tree_util.tree_map(lambda value: value.copy() if isinstance(value, jax.Array) else value, state)
      outputs = p_step(arg_state, batch, *rng_args)
      jax.block_until_ready(outputs)
    timings[name] = time.perf_counter() - start
    _delete_outputs(outputs, keep_ids)
    if arg_state is not state:
      _delete_outputs(arg_state, keep_ids)
    del outputs, arg_state
    max_logging.log(f"warmup_programs_in_init: {name} ({kind}) {timings[name]:.3f} s")
  gc.collect()
  return timings


def _warmup_program_list(config, programs, precompiled_train, precompiled_eval):
  warmup_list = []
  if precompiled_train:
    warmup_list += [
        ("train", "train", programs["p_train_step"]),
        ("train_first_phase", "train", programs["p_train_step_first_phase"]),
    ]
    if config.warmup_skip_train_dropless:
      if programs["p_train_step_dropless"] is not None:
        max_logging.log("warmup_programs_in_init: skipping train_dropless (warmup_skip_train_dropless=True)")
    else:
      warmup_list.append(("train_dropless", "train", programs["p_train_step_dropless"]))
  if precompiled_eval:
    warmup_list += [("eval", "eval", programs["p_eval_step"]), ("eval_dropless", "eval", programs["p_eval_step_dropless"])]
  return warmup_list


def build_warmup_fn(
    config, programs, precompiled_train, precompiled_eval, state, mesh, init_rng, start_step,
    data_loader, data_sharding, shaped_batch, shaped_eval_batch,
):
  if not getattr(config, "warmup_programs_in_init", False):
    return None
  if not (precompiled_train or precompiled_eval):
    max_logging.log("warmup_programs_in_init: no precompiled program (compiled_trainstep_file or AutoPGLE); skipped")
    return None
  warmup_list = _warmup_program_list(config, programs, precompiled_train, precompiled_eval)
  train_batch_sharding = getattr(data_loader, "input_data_shardings", None) or data_sharding
  train_shaped = {
      key: jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=train_batch_sharding)
      for key, value in shaped_batch.items()
  }
  warmup_train_batch = make_synthetic_batch(config, train_shaped, seed=0) if precompiled_train else None
  warmup_eval_batch = make_synthetic_batch(config, shaped_eval_batch, seed=1) if precompiled_eval else None
  copy_train_state = True
  warmup_rng_args = (jax.jit(jax.random.fold_in)(init_rng, start_step),) if config.enable_diloco else ()

  def _run_warmup():
    warmup_programs(
        warmup_list, state, mesh, config.logical_axis_rules, config.logical_axis_rules_for_eval,
        warmup_train_batch, warmup_eval_batch, rng_args=warmup_rng_args, copy_train_state=copy_train_state,
    )

  return _run_warmup
