"""Dump the traced train-step JAXPR locally and optionally to GCS."""

import os

import jax

from lilax._backend.utils import gcs_utils, max_logging


def maybe_dump_jaxpr(config, p_train_step, train_step_inputs):
  if not config.dump_jaxpr:
    return
  max_logging.log("Tracing train_step to jaxpr...")
  unwrapped_step = getattr(p_train_step, "__wrapped__", p_train_step)

  def to_abstract(x):
    if hasattr(x, "shape") and hasattr(x, "dtype"):
      return jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
    return x

  abstract_inputs = jax.tree.map(to_abstract, train_step_inputs)
  p_train_jaxpr = jax.make_jaxpr(unwrapped_step)(*abstract_inputs)
  local_filename = "train_step.jaxpr"
  local_path = os.path.join(config.dump_jaxpr_local_dir, local_filename)
  os.makedirs(config.dump_jaxpr_local_dir, exist_ok=True)
  with open(local_path, "w") as f:
    f.write(str(p_train_jaxpr))
  max_logging.log(f"Jaxpr dumped locally to {local_path}")
  if config.dump_jaxpr_gcs_dir:
    gcs_utils.upload_dump(
        config.dump_jaxpr_local_dir,
        config.dump_jaxpr_gcs_dir,
        module_name=local_filename,
        delete_local_after=config.dump_jaxpr_delete_local_after,
        all_host_upload=False,
    )
