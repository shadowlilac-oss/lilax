"""Single-sample Omni decoding and ChartQA evaluation."""

import functools
import os
import random

import datasets
from flax.core.spmd import logical_axis_rules
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from lilax._backend.common.checkpoints import checkpointing
from lilax._backend.multimodal import processor as mm_processor
from lilax._backend.utils import device_mesh, max_logging
from lilax._backend.utils.globals import MAXTEXT_PKG_DIR
from lilax._backend.utils.model_creation import model_factory
from lilax._backend.utils.tree_ops import partitioning
from lilax.configuration import MaxTextConfig
from lilax.runtime import Runtime


def load_omni_config(checkpoint_path, config_path=None, *, devices=None):
  """Resolve the original single-sample Omni decoding configuration."""
  config_path = config_path or os.path.join(
      MAXTEXT_PKG_DIR, "experimental", "omni_pipeline", "maxtext-omni-gemma3-qwen3.yml"
  )
  custom = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
  skip_keys = {
      "vision_load_path", "llm_load_path", "stitched_output_path", "vision_model_name", "llm_model_name",
      "base_config", "per_device_batch_size", "eval_per_device_batch_size", "ici_fsdp_parallelism",
      "ici_data_parallelism", "ici_tensor_parallelism", "ici_autoregressive_parallelism", "ici_expert_parallelism",
  }
  options = {key: value for key, value in custom.items() if key not in skip_keys}
  options.setdefault("max_prefill_predict_length", 1024)
  options.setdefault("max_target_length", 2048)
  selected_devices = jax.devices() if devices is None else devices
  per_device_batch_size = 1.0 / len(selected_devices)
  options.update(
      override_model_config=True, skip_jax_distributed_system=True, ici_fsdp_parallelism=1,
      ici_data_parallelism=1, ici_autoregressive_parallelism=1, ici_tensor_parallelism=-1,
      per_device_batch_size=per_device_batch_size, async_checkpointing=False,
      load_parameters_path=checkpoint_path,
  )
  runtime = Runtime(MaxTextConfig(**options), devices=devices).resolve_config()
  object.__setattr__(runtime.config, "micro_batch_size_to_train_on", 1)
  object.__setattr__(runtime.config, "global_batch_size_to_train_on", 1)
  object.__setattr__(runtime.config, "per_device_batch_size", per_device_batch_size)
  return runtime


def initialize_model_and_weights(config, checkpoint_path, mesh):
  """Instantiate an Omni model and restore its checkpoint parameters."""
  with jax.set_mesh(mesh):
    model = model_factory.from_config(config, mesh=mesh)
    with model.mesh, logical_axis_rules(config.logical_axis_rules):
      key = jax.random.PRNGKey(0)
      input_shape = (config.micro_batch_size_to_train_on, config.max_target_length)
    image_shape = mm_processor.get_dummy_image_shape_for_init(
        config.model_name, batch_size=config.micro_batch_size_to_train_on
    ) if config.use_multimodal else None
    audio_shape = mm_processor.get_dummy_audio_shape_for_init(config)
    abstract_vars = jax.eval_shape(
        model.init, {"params": key, "dropout": key, "aqt": key},
        np.ones(input_shape, dtype=jnp.int32), np.ones(input_shape, dtype=jnp.int32),
        encoder_images=np.ones(image_shape, dtype=jnp.int32) if config.use_multimodal else None,
        encoder_audios=np.ones(audio_shape, dtype=jnp.float32) if config.use_audio else None,
    )
    target_params = partitioning.unbox_logicallypartioned(abstract_vars["params"])
    max_logging.log(f"Checkpoint parameters path: {checkpoint_path}")
    restored = checkpointing.load_params_from_path(
        checkpoint_path, {"params": target_params}, config.checkpoint_storage_concurrent_gb,
        use_ocdbt=config.checkpoint_storage_use_ocdbt, use_zarr3=config.checkpoint_storage_use_zarr3,
    )
  return model, restored.get("params", restored)


@functools.partial(jax.jit, static_argnums=(0,))
def _prefill_step(model, params, tokens, positions, segment_ids, images):
  logits, mutated = model.apply(
      {"params": params}, decoder_input_tokens=tokens, decoder_positions=positions,
      decoder_segment_ids=segment_ids, encoder_images=images, enable_dropout=False,
      model_mode="prefill", mutable=["cache"],
  )
  return logits, mutated["cache"]


@functools.partial(jax.jit, static_argnums=(0,))
def _ar_step(model, params, cache, token, position):
  logits, mutated = model.apply(
      {"params": params, "cache": cache}, decoder_input_tokens=token, decoder_positions=position,
      decoder_segment_ids=None, encoder_images=None, enable_dropout=False,
      model_mode="autoregressive", mutable=["cache"],
  )
  return logits, mutated["cache"]


@functools.partial(jax.jit, static_argnums=(0, 5))
def _generate_loop(model, params, cache, first_token, start_pos, num_steps):
  def step(carry, _):
    current_cache, token, position = carry
    logits, mutated = model.apply(
        {"params": params, "cache": current_cache}, decoder_input_tokens=token, decoder_positions=position,
        decoder_segment_ids=None, encoder_images=None, enable_dropout=False,
        model_mode="autoregressive", mutable=["cache"],
    )
    next_token = jnp.argmax(logits[:, 0:1, :], axis=-1).astype(jnp.int32)
    return (mutated["cache"], next_token, position + 1), next_token

  _, generated = jax.lax.scan(step, (cache, first_token, start_pos), None, length=num_steps)
  return generated


def decode_omni_sample(model, params, config, mesh, tokenizer, prompt_str, pil_image, max_new_tokens=128):
  """Decode one image and prompt using the model's KV cache."""
  image = np.array(pil_image.convert("RGB"), dtype=np.uint8)
  processed = mm_processor.preprocess_image_for_training(image, config)
  pixels = processed.pixel_values if hasattr(processed, "pixel_values") and processed.pixel_values is not None else processed
  pixels = np.asarray(pixels)
  if pixels.ndim == 4:
    pixels = np.expand_dims(pixels, axis=0)
  images = jnp.array(pixels, dtype=jnp.bfloat16 if config.dtype == "bfloat16" else jnp.float32)
  prompt = mm_processor.reformat_prompt(
      prompt=prompt_str, image_placeholder=config.image_placeholder, model_name=config, num_images=1,
  )
  initial_tokens = tokenizer.encode(prompt, add_special_tokens=False)
  combined = mm_processor.prepare_text_for_image_fusion(
      tokens=initial_tokens, config=config, processor_output=processed,
  ).tolist()
  true_length = len(combined)
  prefill_length = config.max_prefill_predict_length
  if true_length > prefill_length:
    raise ValueError(
        f"The combined length of expanded prompt and vision tokens ({true_length}) "
        f"exceeds config.max_prefill_predict_length ({prefill_length}). "
        "Please increase max_prefill_predict_length in your model config."
    )
  pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
  tokens = np.array([combined + [pad_token] * (prefill_length - true_length)], dtype=np.int32)
  positions = np.tile(np.arange(prefill_length, dtype=np.int32), (1, 1))
  segment_ids = np.zeros((1, prefill_length), dtype=np.int32)
  segment_ids[:, :true_length] = 1
  eos_token_id = tokenizer.eos_token_id
  with jax.set_mesh(mesh):
    logits, cache = _prefill_step(model, params, tokens, positions, segment_ids, images)
    first_token = int(jnp.argmax(logits[0, true_length - 1, :]))
    generated = [first_token]
    if max_new_tokens > 1 and first_token != eos_token_id:
      logits, cache = _ar_step(
          model, params, cache, jnp.array([[first_token]], dtype=np.int32),
          jnp.array([[true_length]], dtype=np.int32),
      )
      second_token = int(jnp.argmax(logits[0, 0, :]))
      generated.append(second_token)
      num_steps = max(0, min(max_new_tokens - 2, config.max_target_length - prefill_length - 2))
      if num_steps > 0 and second_token != eos_token_id:
        sequence = _generate_loop(
            model, params, cache, jnp.array([[second_token]], dtype=np.int32),
            jnp.array([[true_length + 1]], dtype=np.int32), num_steps,
        )
        for token in np.array(sequence).reshape(-1).tolist():
          if token == eos_token_id:
            break
          generated.append(token)
  return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_evaluation(checkpoint_path, runtime, num_samples=5, description="Omni Model", max_new_tokens=128):
  """Log greedy predictions for random ChartQA validation samples."""
  max_logging.log("=" * 60)
  max_logging.log(f"Running Evaluation for {description}...")
  max_logging.log(f"Loading checkpoint from: {checkpoint_path}")
  max_logging.log("=" * 60)
  try:
    dataset = datasets.load_dataset("HuggingFaceM4/ChartQA", split="val")
  except Exception as error:
    max_logging.log(f"Error loading ChartQA evaluation dataset: {error}")
    return
  runtime.resolve_config()
  config = runtime.config
  devices = jax.devices() if runtime.devices is None else runtime.devices
  if devices[0].platform == "cpu":
    mesh = Mesh(np.array([devices[0]]), axis_names=("data",))
  else:
    mesh = device_mesh.get_mesh_from_config(config, devices)
  model, params = initialize_model_and_weights(config, checkpoint_path, mesh)
  tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
  random.seed(42)
  indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
  for index, sample_index in enumerate(indices):
    sample = dataset[sample_index]
    response = decode_omni_sample(
        model, params, config, mesh, tokenizer, f"<image> {sample['query']}", sample["image"], max_new_tokens,
    )
    max_logging.log(
        f"\nSample {index + 1}/{len(indices)} (Index {sample_index}):\n"
        f"  Question: {sample['query']}\n  Ground Truth: {sample['label']}\n  Model Response: {response}\n"
    )
