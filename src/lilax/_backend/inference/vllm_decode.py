"""Native vLLM and Tunix decoding operations."""

import os
from typing import Any

from flax.core.spmd import logical_axis_rules
import jax
import transformers
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout.vllm_rollout import VllmRollout
from vllm import LLM
from vllm.config import ModelConfig
from vllm.sampling_params import SamplingParams

from lilax._backend.common import profiler
from lilax._backend.common.common_types import Config
from lilax._backend.configs.pyconfig import _lists_to_tuples
from lilax._backend.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from lilax._backend.multimodal import utils as mm_utils
from lilax._backend.utils import max_logging
from lilax._backend.utils.globals import MAXTEXT_CONFIGS_DIR
from lilax.configuration.loading import load_config_values

_USE_MROPE = False
_ORIGINAL_USES_MROPE = ModelConfig.uses_mrope.fget


def _uses_mrope(self):
  return _ORIGINAL_USES_MROPE(self) if _USE_MROPE else False


ModelConfig.uses_mrope = property(_uses_mrope)


def build_chat_messages(config: Config) -> list[dict[str, Any]]:
  """Build the chat messages, prepending a system prompt when set."""
  messages = []
  if config.system_prompt:
    messages.append({"role": "system", "content": config.system_prompt})
  if config.use_multimodal and config.image_path:
    content = [{"type": "image"} for _ in config.image_path.split(",")]
    content.append({"type": "text", "text": config.prompt})
    messages.append({"role": "user", "content": content})
  else:
    messages.append({"role": "user", "content": config.prompt})
  return messages


def create_vllm_llm(config: Config, **overrides: Any) -> LLM:
  """Create a vLLM engine; the MaxText adapter must already be registered."""
  vllm_args = {
      "model": config.tokenizer_path,
      "max_model_len": config.max_target_length,
      "tensor_parallel_size": config.ici_tensor_parallelism,
      "data_parallel_size": config.ici_data_parallelism,
      "hf_config_path": config.vllm_hf_config_path,
      "hf_overrides": config.vllm_hf_overrides,
      "gpu_memory_utilization": config.hbm_utilization_vllm,
      "async_scheduling": config.async_scheduling,
      "additional_config": {
          "maxtext_config": {
              "model_name": config.model_name,
              "use_multimodal": config.use_multimodal,
              "weight_dtype": config.weight_dtype.name,
              "allow_split_physical_axes": True,
              "debug_sharding": config.debug_sharding,
              "prefuse_moe_weights": config.prefuse_moe_weights,
              "scan_layers": config.scan_layers,
          },
          "sharding": {"sharding_strategy": {"enable_dp_attention": config.enable_dp_attention}},
      },
  }
  if config.load_parameters_path:
    vllm_args["additional_config"]["maxtext_config"]["load_parameters_path"] = config.load_parameters_path
  else:
    vllm_args["load_format"] = "dummy"

  enable_expert_parallel = config.ici_expert_parallelism > 1
  if enable_expert_parallel:
    vllm_args["additional_config"]["sharding"]["sharding_strategy"]["expert_parallelism"] = config.ici_expert_parallelism
    vllm_args["enable_expert_parallel"] = enable_expert_parallel
  if config.max_num_batched_tokens is not None:
    vllm_args["max_num_batched_tokens"] = config.max_num_batched_tokens
  if config.max_num_seqs is not None:
    vllm_args["max_num_seqs"] = config.max_num_seqs
  vllm_args.update(overrides)

  global _USE_MROPE
  _USE_MROPE = config.use_multimodal and config.use_mrope
  max_logging.log(
      f"Initializing LLM with DP={config.ici_data_parallelism}, TP={config.ici_tensor_parallelism} "
      f"and EP={config.ici_expert_parallelism if enable_expert_parallel else 1}..."
  )
  vllm_config_path = os.path.join(MAXTEXT_CONFIGS_DIR, "inference", "vllm.yml")
  vllm_config = load_config_values(base_config=vllm_config_path, log_config=False)
  with logical_axis_rules(_lists_to_tuples(vllm_config["logical_axis_rules"])):
    return LLM(**vllm_args)


def decode_with_vllm(config: Config) -> None:
  """Decode using vLLM with a MaxText model implementation."""
  llm = create_vllm_llm(config)
  max_logging.log("Generating output...")
  tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer_path, token=config.hf_access_token)
  prompts = [config.prompt]
  if config.use_chat_template:
    input_with_chat_template = tokenizer.apply_chat_template(
        build_chat_messages(config), tokenize=False, add_generation_prompt=True, add_special_tokens=False
    )
    prompts = [input_with_chat_template]
  max_prompt_length = max(len(tokenizer.encode(prompt)) for prompt in prompts)
  if config.use_multimodal and config.image_path:
    images = [mm_utils.load_image_from_path(path.strip()) for path in config.image_path.split(",")]
    image_data = images[0] if len(images) == 1 else images
    prompts = [{"prompt": prompts[0], "multi_modal_data": {"image": image_data}}]
  max_tokens_to_generate = config.max_target_length - max_prompt_length
  if max_tokens_to_generate <= 0:
    raise ValueError(
        f"max_target_length ({config.max_target_length}) must be greater than max_prompt_length ({max_prompt_length})"
    )
  top_p = config.decode_sampling_nucleus_p if config.decode_sampling_nucleus_p > 0 else 1.0
  top_k = config.decode_sampling_top_k if config.decode_sampling_top_k > 0 else -1
  sampling_params = SamplingParams(
      temperature=config.decode_sampling_temperature, max_tokens=max_tokens_to_generate, top_k=top_k, top_p=top_p
  )
  prof = profiler.Profiler(config)
  prof.activate()
  outputs = llm.generate(prompts, sampling_params)
  prof.deactivate()
  for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    max_logging.log(f"Prompt: {prompt}, Generated text: {generated_text}")


def decode_with_tunix(config: Config, model: Any, mesh: jax.sharding.Mesh) -> None:
  """Decode using vLLM with a MaxText model via Tunix adapter."""
  use_no_op_mappings = False
  if hasattr(config, "vllm_hf_overrides") and config.vllm_hf_overrides:
    overrides = config.vllm_hf_overrides
    if isinstance(overrides, str) and "MaxTextForCausalLM" in overrides:
      use_no_op_mappings = True
    elif isinstance(overrides, dict) and "MaxTextForCausalLM" in overrides.get("architectures", []):
      use_no_op_mappings = True
  tunix_model = TunixMaxTextAdapter(base_model=model, use_no_op_mappings=use_no_op_mappings)
  tokenizer = transformers.AutoTokenizer.from_pretrained(
      config.tokenizer_path, token=config.hf_access_token, model_max_length=config.max_target_length
  )
  tokenizer.bos_token = None
  prompts = [config.prompt]
  if config.use_chat_template:
    input_with_chat_template = tokenizer.apply_chat_template(
        build_chat_messages(config), tokenize=False, add_generation_prompt=True, add_special_tokens=False
    )
    prompts = [input_with_chat_template]
  max_prompt_length = max(len(tokenizer.encode(prompt)) for prompt in prompts)
  max_tokens_to_generate = config.max_target_length - max_prompt_length
  if max_tokens_to_generate <= 0:
    raise ValueError(
        f"max_target_length ({config.max_target_length}) must be greater than max_prompt_length ({max_prompt_length})"
    )
  top_p = config.decode_sampling_nucleus_p if config.decode_sampling_nucleus_p > 0 else 1.0
  top_k = config.decode_sampling_top_k if config.decode_sampling_top_k > 0 else -1
  rollout_vllm_additional_config = {
      "maxtext_config": {
          "model_name": config.model_name,
          "weight_dtype": config.weight_dtype.name,
          "allow_split_physical_axes": True,
          "debug_sharding": config.debug_sharding,
          "prefuse_moe_weights": config.prefuse_moe_weights,
          "scan_layers": config.scan_layers,
      }
  }
  if config.lora.enable_lora:
    rollout_vllm_additional_config["maxtext_config"]["lora"] = {
        "enable_lora": config.lora.enable_lora,
        "lora_restore_path": config.lora.lora_restore_path,
        "lora_rank": config.lora.lora_rank,
        "lora_alpha": config.lora.lora_alpha,
        "lora_module_path": config.lora.lora_module_path,
    }
  rollout_config = base_rollout.RolloutConfig(
      max_tokens_to_generate=max_tokens_to_generate,
      max_prompt_length=max_prompt_length,
      temperature=config.decode_sampling_temperature,
      top_p=top_p,
      top_k=top_k,
      rollout_vllm_model_version=config.tokenizer_path,
      rollout_vllm_hbm_utilization=config.hbm_utilization_vllm,
      rollout_vllm_init_with_random_weights=True,
      rollout_vllm_tpu_backend_type="jax",
      tensor_parallel_size=(config.ici_tensor_parallelism if config.ici_tensor_parallelism > 0 else 1),
      data_parallel_size=jax.device_count() // (config.ici_tensor_parallelism if config.ici_tensor_parallelism > 0 else 1),
      rollout_vllm_additional_config=rollout_vllm_additional_config,
      rollout_vllm_kwargs={"hf_overrides": config.vllm_hf_overrides},
  )
  vllm_rollout = VllmRollout(
      model=tunix_model,
      tokenizer=tokenizer,
      cache_config_or_size=max_prompt_length + max_tokens_to_generate + 256,
      mesh=mesh,
      rollout_config=rollout_config,
  )
  output = vllm_rollout.generate(prompts, rollout_config)
  max_logging.log(f"Prompt: {config.prompt}")
  max_logging.log(f"Output: {output.text[0]}")
