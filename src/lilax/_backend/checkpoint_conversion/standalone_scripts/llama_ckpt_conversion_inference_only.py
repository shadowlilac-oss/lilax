import os
import pathlib
import pickle
import subprocess
import sys

import jax
import numpy as np
import torch
from flax.training import train_state

from lilax._backend.common.checkpoints import checkpointing


MODEL_PARAMS_DICT = {
    "llama2-7b": {"num_layers": 32, "num_heads": 32, "num_kv_heads": 32, "dims_per_head": 128, "vocab": 32000},
    "llama2-13b": {"num_layers": 40, "num_heads": 40, "num_kv_heads": 40, "dims_per_head": 128, "vocab": 32000},
    "llama2-70b": {"num_layers": 80, "num_heads": 64, "num_kv_heads": 8, "dims_per_head": 128, "vocab": 32000},
    "llama3-8b": {"num_layers": 32, "num_heads": 32, "num_kv_heads": 8, "dims_per_head": 128, "vocab": 128256},
    "llama3-70b": {"num_layers": 80, "num_heads": 64, "num_kv_heads": 8, "dims_per_head": 128, "vocab": 128256},
    "llama3.1-8b": {"num_layers": 32, "num_heads": 32, "num_kv_heads": 8, "dims_per_head": 128, "vocab": 128256},
    "llama3.1-70b": {"num_layers": 80, "num_heads": 64, "num_kv_heads": 8, "dims_per_head": 128, "vocab": 128256},
    "llama3.1-405b": {"num_layers": 126, "num_heads": 128, "num_kv_heads": 8, "dims_per_head": 128, "vocab": 128256},
    "llama3.3-70b": {"num_layers": 80, "num_heads": 64, "num_kv_heads": 8, "dims_per_head": 128, "vocab": 128256},
}

SIMULATED_CPU_DEVICES_COUNT = 64


def permute_to_match_maxtext_rope(arr):
  return jax.numpy.concatenate((arr[..., ::2], arr[..., 1::2]), axis=arr.ndim - 1)


def convert(base_model_path, maxtext_model_path, model_size):
  environment = os.environ.copy()
  environment["JAX_PLATFORMS"] = "cpu"
  environment["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={SIMULATED_CPU_DEVICES_COUNT}"
  arguments = (os.fspath(base_model_path), os.fspath(maxtext_model_path), model_size)
  child = subprocess.run(
      [sys.executable, "-c", (
          "import os; result_fd = os.dup(1); os.dup2(2, 1); "
          "from lilax._backend.checkpoint_conversion.standalone_scripts "
          "import llama_ckpt_conversion_inference_only as converter; converter._run_cpu_worker(result_fd)"
      )],
      input=pickle.dumps(arguments), stdout=subprocess.PIPE, env=environment, check=True,
  )
  error = pickle.loads(child.stdout)
  if error is not None:
    raise error


def _run_cpu_worker(result_fd):
  try:
    _convert(*pickle.load(sys.stdin.buffer))
    error = None
  except Exception as exc:
    error = exc
  with os.fdopen(result_fd, "wb") as result_file:
    pickle.dump(error, result_file)


def _convert(base_model_path, maxtext_model_path, model_size):
  params = MODEL_PARAMS_DICT[model_size]
  heads, kv_heads, head_dim, vocab = (
      params["num_heads"], params["num_kv_heads"], params["dims_per_head"], params["vocab"]
  )
  checkpoints = {}
  for path in sorted(pathlib.Path(base_model_path).glob("[!.]*.pth")):
    checkpoints[int(path.name.split(".", maxsplit=2)[1])] = torch.load(path, map_location="cpu", weights_only=True)
  shards = [checkpoints[i] for i in sorted(checkpoints)]

  def concat(key, axis, selected=None):
    return np.concatenate([shard[key].type(torch.float16).numpy() for shard in selected or shards], axis=axis)

  embedding = concat("tok_embeddings.weight", 0 if model_size.startswith("llama3") else 1)
  if not model_size.startswith("llama3"):
    embedding = embedding[:vocab, :]
  for shard in shards:
    del shard["tok_embeddings.weight"]
  weights = {
      "token_embedder": {"embedding": embedding},
      "decoder": {
          "decoder_norm": {"scale": shards[0]["norm.weight"].type(torch.float16).numpy()},
          "logits_dense": {"kernel": concat("output.weight", 0).transpose()[:, :vocab]},
      },
  }
  kv_shards = shards[::2] if model_size == "llama3.1-405b" else shards
  for layer in range(params["num_layers"]):
    prefix = f"layers.{layer}"
    query = concat(f"{prefix}.attention.wq.weight", 0).transpose().reshape(heads * head_dim, heads, head_dim)
    key = concat(f"{prefix}.attention.wk.weight", 0, kv_shards).transpose().reshape(heads * head_dim, kv_heads, head_dim)
    value = concat(f"{prefix}.attention.wv.weight", 0, kv_shards).transpose().reshape(heads * head_dim, kv_heads, head_dim)
    out = concat(f"{prefix}.attention.wo.weight", 1).reshape(heads * head_dim, heads, head_dim).transpose(1, 2, 0)
    weights["decoder"][f"layers_{layer}"] = {
        "self_attention": {
            "query": {"kernel": permute_to_match_maxtext_rope(query) / np.sqrt(head_dim)},
            "key": {"kernel": permute_to_match_maxtext_rope(key)},
            "value": {"kernel": value}, "out": {"kernel": out},
        },
        "pre_self_attention_layer_norm": {"scale": shards[0][f"{prefix}.attention_norm.weight"].type(torch.float16).numpy()},
        "post_self_attention_layer_norm": {"scale": shards[0][f"{prefix}.ffn_norm.weight"].type(torch.float16).numpy()},
        "mlp": {
            "wi_0": {"kernel": concat(f"{prefix}.feed_forward.w1.weight", 0).transpose()},
            "wi_1": {"kernel": concat(f"{prefix}.feed_forward.w3.weight", 0).transpose()},
            "wo": {"kernel": concat(f"{prefix}.feed_forward.w2.weight", 1).transpose()},
        },
    }
    for shard in shards:
      for name in ("attention.wq", "attention.wk", "attention.wv", "attention.wo",
                   "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"):
        del shard[f"{prefix}.{name}.weight"]

  mesh = jax.sharding.Mesh(jax.devices(), "checkpoint_sharding_axis")
  first = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("checkpoint_sharding_axis"))
  second = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "checkpoint_sharding_axis"))
  replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None))

  def checkpoint_device_put(arr):
    if arr.shape[0] % SIMULATED_CPU_DEVICES_COUNT == 0:
      return jax.device_put(arr, device=first)
    if len(arr.shape) > 1 and arr.shape[1] % SIMULATED_CPU_DEVICES_COUNT == 0:
      return jax.device_put(arr, device=second)
    return jax.device_put(arr, device=replicated)

  weights = jax.tree_util.tree_map(checkpoint_device_put, weights)
  manager = checkpointing.create_orbax_checkpoint_manager(maxtext_model_path, True, False, 1)
  state = train_state.TrainState(step=0, apply_fn=None, params={"params": weights}, tx=None, opt_state={})
  if manager is not None:
    checkpointing.save_checkpoint(manager, 0, state)
    if checkpointing.reached_preemption(manager, 0):
      checkpointing.wait_until_finished(manager)
