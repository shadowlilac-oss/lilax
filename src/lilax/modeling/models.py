"""Public native model construction APIs."""

try:
  import triton  # pylint: disable=unused-import
except ImportError:
  pass

import json
import copy
from pathlib import Path
from tempfile import TemporaryDirectory

from etils import epath
from flax import nnx
import jax.numpy as jnp
import orbax.checkpoint as ocp

from lilax._backend.configs.pyconfig import HyperParameters
from lilax._backend.checkpoint_conversion.cpu_bootstrap import convert_on_cpu
from lilax._backend.checkpoint_conversion.to_huggingface import convert_to_huggingface
from lilax._backend.utils import maxtext_utils_nnx
from lilax._backend.models.creation import model_factory, pretrained, scan_layers_sync
from lilax.configuration import AutoConfig, QwixQATConfig
from lilax.configuration.model_mapping import model_config_from_source
from lilax.generation.configuration import ContinuousBatchingConfig, GenerationConfig
from lilax.generation.sequences import generate_sequences
from lilax.generation.batching import generate_batch
from lilax.modeling.hub import get_model_capabilities, resolve_model_name, resolve_model_snapshot
from lilax.runtime import Runtime
from lilax.modeling import persistence as model_persistence
from lilax.modeling import construction as model_construction
from lilax.modeling import qat_application as model_qat_application
from lilax.modeling.sources import resolve_checkpoint_path, resolve_pretrained_source
from lilax.checkpoints.exporting import HuggingFaceExportConfig
from lilax._processing.media import fold_media


class MaxTextModel:
  """A native MaxText model bound to its initialized runtime."""

  def __init__(self, native_model, runtime, *, construction_config=None, checkpoint=None, model_options=None,
               quantization_config=None, config=None):
    self._native_model = native_model
    self.runtime = runtime
    self.construction_config = runtime.config if construction_config is None else construction_config
    self._checkpoint = checkpoint
    self._model_options = model_options
    self._quantization_config = quantization_config
    self.config = model_config_from_source(runtime.public_config) if config is None else config
    self.generation_config = GenerationConfig(eos_token_id=getattr(self.config, "eos_token_id", None))

  @property
  def is_materialized(self):
    return self._native_model is not None

  @property
  def existing_graph_config(self):
    """Return the construction config of an existing graph, without materializing."""
    return self.construction_config if self.is_materialized else None

  @property
  def existing_graph_qat_config(self):
    """Return the QAT config of an existing graph, without materializing."""
    config = self.existing_graph_config
    return None if config is None else getattr(config._pydantic_config, "qwix_qat_config", None)

  @property
  def pending_qat_config(self):
    """Return the selected or existing graph's QAT config without materializing."""
    if self.existing_graph_config is not None:
      return self.existing_graph_qat_config
    checkpoint = self._checkpoint
    if checkpoint is not None and checkpoint["qat_config"] is not None:
      return checkpoint["qat_config"]
    if self._quantization_config is not None:
      return self._quantization_config
    if (checkpoint is not None and checkpoint["auto_config"]
        and "qwix_qat_config" not in self.runtime.public_config._explicit_keys):
      return None
    return self.runtime.public_config.qwix_qat_config

  @property
  def native_model(self):
    self.materialize(runtime=self.runtime)
    return self._native_model

  @native_model.setter
  def native_model(self, model):
    self._native_model = model

  def prepare_for_full_state_restore(self, qat_config, *, runtime):
    """Apply a selected full-state source's validated QAT config before materialization."""
    if runtime is not self.runtime:
      raise ValueError("A model must be prepared with its original runtime.")
    if not isinstance(qat_config, QwixQATConfig):
      raise TypeError("The selected full-state QAT config must be a QwixQATConfig.")
    if self._quantization_config is not None and self._quantization_config != qat_config:
      raise ValueError("The selected full-state QAT config differs from the requested QAT config.")
    public_config = runtime.public_config
    if "qwix_qat_config" in public_config._explicit_keys and public_config.qwix_qat_config != qat_config:
      raise ValueError("The selected full-state QAT config differs from the explicit runtime QAT config.")
    if self._checkpoint is not None:
      source_config = self._checkpoint["qat_config"]
      if source_config is not None and source_config != qat_config:
        raise ValueError("The selected full-state QAT config differs from the model checkpoint's QAT config.")
    if self.existing_graph_config is not None:
      if self.existing_graph_qat_config != qat_config:
        raise ValueError("The selected full-state QAT config differs from the existing model graph.")
      return self
    self._quantization_config = qat_config
    return self

  def materialize(self, *, runtime):
    """Construct or restore the model once using its runtime and mesh."""
    if runtime is not self.runtime:
      raise ValueError("A model must be materialized with its original runtime.")
    if self.is_materialized:
      return self
    if runtime.mesh is None:
      runtime.initialize()
    qat_config = self._quantization_config or runtime.public_config.qwix_qat_config
    if self._model_options is not None:
      if qat_config is not None and runtime.config.quantization and runtime.config.use_qwix_quantization:
        raise ValueError("Qwix QAT cannot be combined with an existing Qwix precision recipe.")
      options = self._model_options
      rngs = options["rngs"]
      if rngs is None:
        rngs = maxtext_utils_nnx.create_nnx_rngs(runtime.config, model_mode=options["model_mode"])
      model = model_factory.from_config(
          runtime.config, mesh=runtime.mesh, model_mode=options["model_mode"], rngs=rngs,
          quant_mode_str=options["quant_mode_str"],
      )
      if qat_config is not None:
        model, config = self._apply_qat(model, runtime.config, qat_config)
      else:
        config = runtime.config
      self._native_model, self.construction_config = model, config
      self._model_options = None
      return self
    checkpoint = self._checkpoint
    if (checkpoint["auto_config"] and checkpoint["qat_config"] is None and self._quantization_config is None
        and "qwix_qat_config" not in runtime.public_config._explicit_keys):
      qat_config = None
    checkpoint_path = resolve_checkpoint_path(checkpoint, runtime)
    source_qat_config = checkpoint["qat_config"]
    if source_qat_config is not None and qat_config is not None and source_qat_config != qat_config:
      raise ValueError("The requested Qwix QAT config differs from the checkpoint's QAT config.")
    if source_qat_config is not None:
      qat_config = source_qat_config
    if qat_config is not None and runtime.config.quantization and runtime.config.use_qwix_quantization:
      raise ValueError("Qwix QAT cannot be combined with an existing Qwix precision recipe.")
    native_config = HyperParameters(
        runtime.config._pydantic_config.model_copy(
            update={"load_parameters_path": str(checkpoint_path), "load_full_state_path": "",
                    "convert_checkpoint_if_possible": False}
        )
    )
    if source_qat_config is not None:
      native_config = scan_layers_sync.verify_and_sync_scan_layers(native_config)
      native_config = HyperParameters(native_config._pydantic_config.model_copy(update={"load_parameters_path": ""}))
    model, config = pretrained.from_pretrained(
        native_config, mesh=runtime.mesh, model_mode=checkpoint["model_mode"], rng_key=checkpoint["rng_key"],
        wrap_with_tunix_adapter=checkpoint["wrap_with_tunix_adapter"],
        tokenizer_pad_id=checkpoint["tokenizer_pad_id"], quant_mode_str=checkpoint["quant_mode_str"],
        return_effective_config=True,
    )
    if qat_config is not None:
      model, config = self._apply_qat(model, config, qat_config)
    if source_qat_config is not None:
      model_persistence.restore_qat_checkpoint(model, runtime, config, checkpoint_path)
    self._native_model, self.construction_config = model, config
    self._checkpoint = None
    return self

  def _apply_qat(self, model, config, qat_config):
    return model_qat_application.apply_qat(model, config, qat_config, self.runtime)

  def __call__(
      self, input_ids, *, position_ids=None, attention_mask=None, segment_ids=None, inputs_embeds=None,
      images=None, image_masks=None, videos=None, video_masks=None, video_grid_thw=None, audio_values=None,
      decoder_target_tokens=None, decoder_target_mask=None, enable_dropout=True,
  ):
    input_ids = jnp.asarray(input_ids)
    if position_ids is None:
      position_ids = jnp.broadcast_to(jnp.arange(input_ids.shape[-1]), input_ids.shape)
    if segment_ids is None:
      segment_ids = jnp.ones_like(input_ids) if attention_mask is None else jnp.asarray(attention_mask)
    elif attention_mask is not None:
      segment_ids = jnp.where(jnp.asarray(attention_mask) != 0, jnp.asarray(segment_ids), 0)
    media = fold_media(self.construction_config, {
        "images": images, "image_masks": image_masks, "videos": videos, "video_masks": video_masks,
        "video_grid_thw": video_grid_thw, "audio_values": audio_values,
    })
    return self.native_model(
        input_ids, position_ids, segment_ids, decoder_input_embeddings=inputs_embeds,
        encoder_images=media["images"], encoder_image_masks=media["image_masks"], encoder_videos=media["videos"],
        encoder_video_masks=media["video_masks"], encoder_video_grid_thw=media["video_grid_thw"],
        encoder_audios=media["audio_values"],
        decoder_target_tokens=decoder_target_tokens, decoder_target_mask=decoder_target_mask,
        enable_dropout=enable_dropout,
    )

  def generate(
      self, input_ids, generation_config=None, *, attention_mask=None, max_new_tokens=None, do_sample=None,
      top_k=None, top_p=None, temperature=None, rng=None, position_ids=None, mrope_deltas=None, images=None, image_masks=None,
      videos=None, video_masks=None, video_grid_thw=None, audio_values=None, audio_masks=None,
  ):
    """Return prompt and generated token IDs for each supplied sequence."""
    options = generation_config or self.generation_config
    options = options.model_copy(update={name: value for name, value in (
        ("max_new_tokens", max_new_tokens), ("do_sample", do_sample), ("top_k", top_k),
        ("top_p", top_p), ("temperature", temperature),
    ) if value is not None})
    return generate_sequences(
        self, input_ids, options, attention_mask=attention_mask, rng=rng, position_ids=position_ids,
        mrope_deltas=mrope_deltas, images=images, image_masks=image_masks, videos=videos, video_masks=video_masks,
        video_grid_thw=video_grid_thw, audio_values=audio_values, audio_masks=audio_masks,
    )

  @property
  def native_module(self):
    return self.native_model

  def generate_batch(self, inputs, generation_config=None, *, continuous_batching_config=None, rng=None):
    return generate_batch(self, inputs, generation_config or self.generation_config,
                          continuous_batching_config or ContinuousBatchingConfig(), rng)

  def save_pretrained(
      self, save_directory, *, format="maxtext", export_config=None,
  ):
    """Save the current model as a native checkpoint or Hugging Face model."""
    export_config = export_config or HuggingFaceExportConfig()
    return model_persistence.save_pretrained(self, save_directory, format=format, **export_config.model_dump())


class AutoModelForCausalLM:
  """Construct native MaxText models with an initialized runtime configuration."""

  @staticmethod
  def from_config(config, *, quantization_config=None, seed=0):
    """Create a wrapper that constructs an NNX model when materialized."""
    config = copy.deepcopy(config)
    if quantization_config is not None:
      config.quantization_config = quantization_config
    runtime, model_options = model_construction.prepare_from_config(
        config,
    )
    runtime.public_config = runtime.public_config.model_copy(update={"init_weights_seed": seed})
    runtime.config = runtime.public_config
    return MaxTextModel(
        None, runtime, model_options=model_options,
        quantization_config=config.quantization_config if isinstance(config.quantization_config, QwixQATConfig) else None,
        config=config,
    )

  @staticmethod
  def from_pretrained(
      pretrained_model_name_or_path, *, config=None, revision=None, token=None, cache_dir=None,
      local_files_only=False, quantization_config=None, fuse_qkv=None, fuse_mlp=None, precision_config=None,
  ):
    """Create a native model; token selects model snapshot access, not adapter credentials."""
    runtime, checkpoint = resolve_pretrained_source(
        pretrained_model_name_or_path, config=config, revision=revision, token=token,
        cache_dir=cache_dir, local_files_only=local_files_only, quantization_config=quantization_config,
        fuse_qkv=fuse_qkv, fuse_mlp=fuse_mlp, precision_config=precision_config,
    )
    qat = quantization_config if isinstance(quantization_config, QwixQATConfig) else None
    return MaxTextModel(None, runtime, checkpoint=checkpoint, quantization_config=qat, config=checkpoint["public_config"])
