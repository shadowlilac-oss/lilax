"""Collect MaxText and Hugging Face parameter names and shapes."""

import json
import os
from pathlib import Path

from flax import nnx
import jax
from transformers import AutoConfig, AutoModelForCausalLM

from lilax._backend.inference.maxengine import maxengine
from lilax._backend.utils import max_logging
from lilax._backend.utils.diagnostics import system_info


def save_parameter_context(config, hf_model_config, agent_directory):
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  assert config.load_full_state_path == "", (
      "This script expects parameters, not a full state. Use generate_param_only_checkpoint first if needed."
  )
  system_info.print_system_information()
  engine = maxengine.MaxEngine(config)
  rng, rng_load_params = jax.random.split(jax.random.PRNGKey(1234))
  params = nnx.to_pure_dict(engine.load_params(rng=rng_load_params))
  maxtext_params = {}
  for path, value in jax.tree_util.tree_leaves_with_path(params):
    parts = []
    for entry in path:
      if isinstance(entry, jax.tree_util.DictKey):
        parts.append(str(entry.key))
      elif isinstance(entry, jax.tree_util.SequenceKey):
        parts.append(str(entry.idx))
      else:
        max_logging.log(f"Warning: Path tuple {path} contains unhandled key type '{type(entry)}'.")
        parts.append(f"__unhandled_key_type_{type(entry).__name__}__")
    maxtext_params["params-" + "-".join(parts)] = list(value.shape)

  hf_config = AutoConfig.from_pretrained(hf_model_config, token=config.hf_access_token)
  hf_model = AutoModelForCausalLM.from_config(hf_config)
  hf_params = {name: list(tensor.shape) for name, tensor in hf_model.state_dict().items()}
  output = Path(config.base_output_directory) if config.base_output_directory else Path(agent_directory) / "context" / config.model_name
  output.mkdir(parents=True, exist_ok=True)
  if jax.process_index() == 0:
    for name, params_info in (("maxtext_params.json", maxtext_params), ("hf_params.json", hf_params)):
      with (output / name).open("wt", encoding="utf8") as file:
        json.dump(params_info, file, indent=2)
      max_logging.log(f"Parameters and shapes saved to {output / name}")
  return output
