import gc
import os
import sys

from flax import nnx
import jax
from jax import random
from jax.sharding import Mesh
import numpy as np
from psutil import Process
import tensorstore as ts

from lilax._backend.common.checkpoints import checkpointing
from lilax._backend.common import train_state_nnx
from lilax._backend.configs import pyconfig, types
from lilax._backend.distributed import device_mesh
from lilax._backend.utils import max_logging, maxtext_utils_nnx
from lilax._backend.observability.diagnostics import pytree
from lilax._backend.utils.training import state_setup
from lilax._backend.utils import train_utils
from lilax._backend.utils.globals import MAXTEXT_CONFIGS_DIR
from lilax._backend.utils.model_creation import model_factory, nnx_abstract_model


def fmt_size(num_bytes: int) -> str:
  assert num_bytes > 0
  for unit in ("B", "KiB", "MiB", "GiB"):
    if num_bytes < 1024.0:
      break
    num_bytes /= 1024.0
  return f"{num_bytes:.2f} {unit}"


def convert_from_paxml(paxml_ckpt_path, maxtext_model_name, base_output_directory, run_name):
  overrides = {
      "model_name": maxtext_model_name,
      "base_output_directory": base_output_directory,
      "run_name": run_name,
      "per_device_batch_size": 1,
      "ici_fsdp_parallelism": -1,
      "ici_tensor_parallelism": 1,
      "async_checkpointing": False,
  }
  values, explicit_keys = pyconfig._load_raw_config(os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml"), overrides)
  values["checkpoint_period"] = 1
  explicit_keys.add("checkpoint_period")
  config = pyconfig._resolve_native_config(values, explicit_keys, types.MaxTextConfig)
  return convert(paxml_ckpt_path, config)


def convert(paxml_ckpt_path, config):
  if not paxml_ckpt_path.startswith("gs://"):
    raise ValueError("paxml_ckpt_path should be a gcs path starting with gs://")
  cfg = config
  init_rng, _ = random.split(random.PRNGKey(cfg.init_weights_seed), 2)
  mesh = Mesh(device_mesh.create_device_mesh(cfg), cfg.mesh_axes)
  rngs = maxtext_utils_nnx.create_nnx_rngs(cfg, rng_key=init_rng)
  model = model_factory.from_config(cfg, mesh=mesh, rngs=rngs)
  _, tx = train_utils.create_training_optimizer(cfg, model)
  create_model, _ = nnx_abstract_model.create_nnx_abstract_model(cfg, mesh)

  def init_state_fn():
    nnx_model = create_model()
    optimizer = nnx.Optimizer(nnx_model, tx, wrt=nnx.Param)
    return train_state_nnx.TrainStateNNX(nnx_model, optimizer)

  manager = checkpointing.create_orbax_checkpoint_manager(
      cfg.checkpoint_dir, cfg.enable_checkpointing, cfg.async_checkpointing, cfg.checkpoint_period,
  )
  state, _, _, _, _ = state_setup.setup_training_state(None, cfg, mesh, manager, init_state_fn)
  state = train_state_nnx.to_checkpoint_dict(state)
  state.pop("nnx_aux", None)

  max_logging.log("start")
  pytree.print_mem_stats("After params initialized")

  axis = cfg.param_scan_axis
  move = lambda x: np.moveaxis(x, 0, axis)
  qkv = ".params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv"
  prefix = ".params.lm.transformer.repeat.sub.x_layers_0."
  mapping = {
      "['token_embedder']['embedding']": (".params.lm.softmax.logits_ffn.linear.w", lambda x: x.T),
      "['decoder']['position_embedder']['embedding']": (".params.lm.position_emb.emb_var", None),
      "['decoder']['decoder_norm']['scale']": (".params.lm.final_ln.scale", lambda x: x.T),
      "['decoder']['decoder_norm']['bias']": (".params.lm.final_ln.bias", None),
  }
  for name, source in (("pre_self_attention_norm", "layer_norm"), ("mlp']['mlp_layer_norm", "ff_layer.layer_norm")):
    for part in ("scale", "bias"):
      mapping[f"['decoder']['layers']['{name}']['{part}']"] = (prefix + source + "." + part, move)
  for index, name in enumerate(("query", "key", "value")):
    for part, suffix in (("kernel", "w"), ("bias", "b")):
      mapping[f"['decoder']['layers']['self_attention']['{name}']['{part}']"] = (
          qkv + "." + suffix, lambda x, i=index: np.moveaxis(x[:, i], 0, axis),
      )
  mapping["['decoder']['layers']['self_attention']['qkv_proj']['kernel']"] = (
      qkv + ".w", lambda x: np.moveaxis(x, [2, 0], [0, axis]),
  )
  mapping["['decoder']['layers']['self_attention']['qkv_proj']['bias']"] = (qkv + ".b", move)
  mapping["['decoder']['layers']['self_attention']['out']['kernel']"] = (
      prefix + "self_attention.post.w", lambda x: np.moveaxis(x, [0, 1], [axis, -1]),
  )
  mapping["['decoder']['layers']['self_attention']['out']['bias']"] = (prefix + "self_attention.post.b", move)
  for name, source in (("wi", "ffn_layer1"), ("wo", "ffn_layer2")):
    for part, suffix in (("kernel", "linear.w"), ("bias", "bias.b")):
      mapping[f"['decoder']['layers']['mlp']['{name}']['{part}']"] = (
          prefix + "ff_layer." + source + "." + suffix, move,
      )

  state_map = {"['step']": ("step", None), "['opt_state']['count']": ("opt_states_0.no_prefix_0.count", None)}
  for key, (source, transform) in mapping.items():
    opt_prefix = f"p#{cfg.base_num_decoder_layers}#i-1_2" if "x_layers_0" in source else "no_prefix_2"
    state_map[f"['params']['params']{key}"] = ("mdl_vars" + source, transform)
    for moment, pax_moment in (("mu", "m"), ("nu", "v")):
      state_map[f"['opt_state']['{moment}']['params']{key}"] = (
          f"opt_states_0.{opt_prefix}.{pax_moment}{source}", transform,
      )

  def verify(key_path, _):
    key = jax.tree_util.keystr(key_path)
    assert key in state_map, f"{key} not found"

  jax.tree_util.tree_map_with_path(verify, state)
  bucket, checkpoint_prefix = paxml_ckpt_path[len("gs://"):].split("/", 1)
  memory_metrics = {"max_cpu_bytes": 0}

  def load(key_path, value):
    key = jax.tree_util.keystr(key_path)
    path, transform = state_map[key]
    spec = {"driver": "zarr", "metadata_key": ".zarray", "kvstore": {
        "bucket": bucket, "driver": "gcs", "path": os.path.join(checkpoint_prefix, path),
    }}
    arr = ts.open(ts.Spec(spec), open=True).result().read().result()
    if transform is not None:
      arr = transform(arr)
    assert value.shape == arr.shape, f"{key_path}, {value.shape}, {arr.shape}"
    shape, sharding = value.shape, value.sharding
    result = jax.make_array_from_single_device_arrays(
        shape, sharding,
        [jax.device_put(np.array(arr[index]), device)
         for device, index in sharding.addressable_devices_indices_map(shape).items()],
    )
    memory_metrics["max_cpu_bytes"] = max(Process().memory_info().rss, memory_metrics["max_cpu_bytes"])
    del arr
    gc.collect()
    max_logging.log(f"{key} finished")
    pytree.print_mem_stats("After params conversion")
    return result

  converted = jax.tree_util.tree_map_with_path(load, state)
  max_logging.log("converted state finished")
  pytree.print_mem_stats("converted state finished")
  step = int(converted["step"]) if isinstance(converted, dict) else int(converted.step)
  if checkpointing.save_checkpoint(manager, step, converted, config=cfg):
    max_logging.log(f"Saved a checkpoint at step {step}.")
  if checkpointing.reached_preemption(manager, step):
    checkpointing.wait_until_finished(manager)
    sys.exit()

  max_logging.log(f"Peak cpu memory in a single process: {fmt_size(memory_metrics['max_cpu_bytes'])}")
  max_logging.log("checkpoint converted and saved successfully.")
