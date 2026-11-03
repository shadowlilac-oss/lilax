"""Search batch sizes and rematerialization policies using ahead-of-time compilation."""

import os
import json
import sys
import time
from enum import IntEnum

import jax

from lilax._backend.configs.runtime import resolve_runtime_config
from lilax._backend.trainers.pre_train import train_compile
from lilax.configuration import serialize_config


class Action(IntEnum):
  REMAT = 0
  OFFLOAD = 1
  DEVICE = 2


class RematPolicy:
  def __init__(self, tensor_names: list[str], tensors: dict | None = None, initial_level: Action = Action.REMAT):
    self.tensors = {name: initial_level for name in tensor_names} if tensors is None else tensors
    self.tensor_order = tensor_names

  @property
  def to_dict(self) -> dict[str, str]:
    mapping = {0: "remat", 1: "offload", 2: "device"}
    return {key: mapping[value.value] if isinstance(value, Action) else mapping[value] for key, value in self.tensors.items()}

  def next_policy(self):
    policy = RematPolicy(self.tensor_order, self.tensors.copy())
    for action in [Action.OFFLOAD, Action.DEVICE]:
      for name in reversed(self.tensor_order):
        if policy.tensors[name] < action:
          policy.tensors[name] = action
          return policy
    return None

  def previous_policy(self):
    policy = RematPolicy(self.tensor_order, self.tensors.copy())
    for action in [Action.OFFLOAD, Action.REMAT]:
      for name in self.tensor_order:
        if policy.tensors[name] > action:
          policy.tensors[name] = action
          return policy
    return None

  def __repr__(self):
    return str(self.to_dict)


def tensor_score(tensor_name: str, config) -> tuple:
  scores = {
      "context": (-config.max_target_length, -config.num_query_heads * config.head_dim),
      "mlpwi_0": (-config.emb_dim, -config.mlp_dim),
      "mlpwi_1": (-config.emb_dim, -config.mlp_dim),
      "mlpwo": (-config.mlp_dim, -config.emb_dim),
      "query_proj": (-config.emb_dim, -config.num_query_heads * config.head_dim),
      "key_proj": (-config.emb_dim, -config.num_kv_heads * config.head_dim),
      "kv_proj": (-config.emb_dim, -config.num_kv_heads * config.head_dim),
      "value_proj": (-config.emb_dim, -config.num_kv_heads * config.head_dim),
      "out_proj": (-config.num_query_heads * config.head_dim, -config.emb_dim),
      "qkv_proj": (-config.emb_dim, -(config.num_query_heads + 2 * config.num_kv_heads) * config.head_dim),
      "mlpwi": (-config.emb_dim, -config.mlp_dim),
  }
  return scores[tensor_name]


def generate_priority_list(config, provided_tensor_names):
  keys = {
      (True, 1): ["context", "qkv_proj", "mlpwi", "mlpwo", "out_proj"],
      (True, 2): ["context", "qkv_proj", "mlpwi_0", "mlpwi_1", "mlpwo", "out_proj"],
      (False, 1): ["context", "query_proj", "key_proj", "value_proj", "kv_proj", "mlpwi", "mlpwo", "out_proj"],
      (False, 2): ["context", "query_proj", "key_proj", "value_proj", "kv_proj", "mlpwi_0", "mlpwi_1", "mlpwo", "out_proj"],
  }
  names = sorted(keys[config.fused_mlp, len(config.mlp_activations)], key=lambda name: tensor_score(name, config))
  return [name for name in names if name not in provided_tensor_names]


def find_pdb_scalar(config):
  scalar = 1.0
  for axis in config.mesh_axes:
    if axis not in ("data", "fsdp", "fsdp_transpose", "expert", "stage"):
      scalar *= getattr(config, f"ici_{axis}_parallelism")
  return scalar


def candidate_config(config, policy, pdb):
  """Return the public training configuration for an estimated candidate."""
  policy_dict = policy.to_dict if isinstance(policy, RematPolicy) else policy
  overrides = {"per_device_batch_size": pdb, "remat_policy": "custom", **policy_dict}
  if "decoder_layer_input" not in config._explicit_keys or config.decoder_layer_input not in ("device", "offload"):
    overrides["decoder_layer_input"] = "device"
  values = config.get_source_values()
  values.update(overrides)
  return type(config)._from_resolved(values, config._explicit_keys | overrides.keys())


def is_oom(config, policy: RematPolicy, pdb: float) -> bool:
  """Compile a candidate configuration and report whether it runs out of memory."""
  candidate = candidate_config(config, policy, pdb)
  print(f"Checking whether batch_size={pdb} and policy={policy} is OOM")
  stdout_fd = os.dup(sys.stdout.fileno())
  stderr_fd = os.dup(sys.stderr.fileno())
  try:
    with open(os.devnull, "w") as devnull:
      os.dup2(devnull.fileno(), sys.stdout.fileno())
      os.dup2(devnull.fileno(), sys.stderr.fileno())
      result = train_compile.is_oom(resolve_runtime_config(candidate))
  finally:
    os.dup2(stdout_fd, sys.stdout.fileno())
    os.dup2(stderr_fd, sys.stderr.fileno())
    os.close(stdout_fd)
    os.close(stderr_fd)
  print(f"Is OOM: {result}")
  return result


def largest_batch_size(config, policy, min_pdb=None, max_pdb=32.0, pdb_scalar=1.0) -> float:
  if pdb_scalar == 0.0:
    raise ValueError("pdb_scalar cannot be value zero.")
  min_pdb = 1.0 / pdb_scalar if min_pdb is None else min_pdb
  if min_pdb <= 0.0 or max_pdb < min_pdb:
    raise ValueError(f"Abnormal {min_pdb=} or {max_pdb=} values.")
  print(f"Starting binary search for the largest batch size between {min_pdb} and {max_pdb}.")
  if min_pdb == max_pdb:
    return min_pdb if not is_oom(config, policy, min_pdb) else min_pdb - 1 / pdb_scalar
  if is_oom(config, policy, min_pdb):
    print(f"OOM at minimum batch size {min_pdb}.")
    return min_pdb - 1 / pdb_scalar
  if not is_oom(config, policy, max_pdb):
    print(f"No OOM at maximum batch size {max_pdb}.")
    return max_pdb
  low, high, result = int(min_pdb * pdb_scalar), int(max_pdb * pdb_scalar), int(min_pdb * pdb_scalar)
  while low <= high:
    mid = (low + high) // 2
    if mid < min_pdb:
      low = mid + 1
      continue
    if not is_oom(config, policy, mid / pdb_scalar):
      result = mid
      low = mid + 1
    else:
      high = mid - 1
  return result / pdb_scalar


def search_policy_only(tensor_names, config, pdb, init_policy: RematPolicy = None) -> RematPolicy:
  full_remat = RematPolicy(tensor_names, initial_level=Action.REMAT)
  if is_oom(config, full_remat, pdb):
    raise ValueError(f"Given batch size {pdb} leads to OOM even with full remat.")
  policy = RematPolicy(tensor_names, initial_level=Action.DEVICE) if init_policy is None else init_policy
  while is_oom(config, policy, pdb):
    policy = policy.previous_policy()
  return policy


def search(tensor_names, config, *, min_pdb=None, max_pdb=64.0, init_policy=None, pdb_scalar=1.0):
  results = []
  policy = RematPolicy(tensor_names, initial_level=Action.REMAT) if init_policy is None else init_policy
  min_pdb = 1 / pdb_scalar if min_pdb is None else min_pdb
  pdb = max_pdb
  while policy is not None:
    pdb = largest_batch_size(config, policy, min_pdb=min_pdb, max_pdb=pdb, pdb_scalar=pdb_scalar)
    if pdb > 0:
      results.append((pdb, policy.to_dict))
    policy = policy.next_policy()
    if pdb < min_pdb:
      print(f"The value of {pdb=} hits {min_pdb=}. Stop iterating policies.")
      break
  return results


def estimate(config):
  """Return public training configurations for the estimated batch sizes and policies."""
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["LIBTPU_INIT_ARGS"] = (
      os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  )
  runtime_config = resolve_runtime_config(config)
  train_compile.validate_config(runtime_config)
  scalar = find_pdb_scalar(runtime_config)
  names = generate_priority_list(runtime_config, config._explicit_keys)
  start_time = time.time()
  if "per_device_batch_size" in config._explicit_keys:
    pdb = config.per_device_batch_size
    print(f"Batch size provided ({pdb}). Searching for best policy...")
    results = [(pdb, search_policy_only(names, config, pdb).to_dict)]
  else:
    print("No batch size provided. Searching for max batch size and policies...")
    full_remat = RematPolicy(names, initial_level=Action.REMAT)
    max_pdb = largest_batch_size(config, full_remat, min_pdb=1.0 / scalar, pdb_scalar=scalar)
    results = search(names, config, min_pdb=1.0 / scalar, max_pdb=max_pdb, init_policy=full_remat, pdb_scalar=scalar)

  print(f"\nSearch completed in {time.time() - start_time:.2f} seconds.")
  configs = [candidate_config(config, policy, pdb) for pdb, policy in results]
  if config.write_estimator_result:
    output_filename = "remat_configs_from_estimator.py"
    print(f"Writing {len(configs)} suggested configuration(s) to {output_filename}...")
    serialized_configs = [serialize_config(candidate) for candidate in configs]
    for candidate in serialized_configs:
      candidate["values"].pop("hf_access_token", None)
      candidate["explicit_keys"] = [key for key in candidate["explicit_keys"] if key != "hf_access_token"]
    payload = json.dumps(serialized_configs)
    with open(output_filename, "w", encoding="utf-8") as output:
      output.write("import json\n\nfrom lilax.configuration import deserialize_config\n")
      output.write("from lilax.trainers import train_pretraining\n\n")
      output.write(f"configs = [deserialize_config(item) for item in json.loads({payload!r})]\n\n")
      output.write("def train(index):\n  return train_pretraining(configs[index])\n")
  for pdb, policy in results:
    print(f"  - Found valid combo: pdb={pdb}, policy={policy}")
  print("Done.")
  return configs
