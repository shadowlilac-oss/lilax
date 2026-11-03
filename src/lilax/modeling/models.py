"""Public native model construction APIs."""

try:
  import triton  # pylint: disable=unused-import
except ImportError:
  pass

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from etils import epath
from flax import nnx
import orbax.checkpoint as ocp

from lilax._backend.configs.pyconfig import HyperParameters
from lilax._backend.checkpoint_conversion.cpu_bootstrap import convert_on_cpu
from lilax._backend.checkpoint_conversion.to_huggingface import convert_to_huggingface
from lilax._backend.utils import maxtext_utils_nnx
from lilax._backend.utils.model_creation import model_factory, pretrained, scan_layers_sync
from lilax.configuration import AutoConfig, QwixQATConfig
from lilax.generation.engine import generate as generate_tokens
from lilax.modeling.hub import get_model_capabilities, resolve_model_name, resolve_model_snapshot
from lilax.runtime import Runtime
from lilax.modeling import persistence as model_persistence
from lilax.modeling import construction as model_construction
from lilax.modeling import qat_application as model_qat_application
from lilax.modeling.sources import resolve_checkpoint_path, resolve_pretrained_source


class MaxTextModel:
  """A native MaxText model bound to its initialized runtime."""

  def __init__(self, native_model, runtime, *, construction_config=None, checkpoint=None, model_options=None,
               quantization_config=None):
    self._native_model = native_model
    self.runtime = runtime
    self.construction_config = runtime.config if construction_config is None else construction_config
    self._checkpoint = checkpoint
    self._model_options = model_options
    self._quantization_config = quantization_config

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

  def __call__(self, *args, **kwargs):
    return self.native_model(*args, **kwargs)

  def generate(
      self, padded_tokens, true_length, *, max_new_tokens=None, rng=None, algorithm=None, topk=None,
      nucleus_topp=None, temperature=None, positions=None, mrope_deltas=None, images=None, image_masks=None,
      videos=None, video_masks=None, video_grid_thw=None, audio_values=None, audio_masks=None,
  ):
    """Generate token IDs for one padded prompt with the current model state."""
    return generate_tokens(
        self.native_model, self.runtime, padded_tokens, true_length, construction_config=self.construction_config,
        max_new_tokens=max_new_tokens, rng=rng,
        algorithm=algorithm, topk=topk, nucleus_topp=nucleus_topp, temperature=temperature, positions=positions,
        mrope_deltas=mrope_deltas, images=images, image_masks=image_masks, videos=videos, video_masks=video_masks,
        video_grid_thw=video_grid_thw, audio_values=audio_values, audio_masks=audio_masks,
    )

  @property
  def native_module(self):
    return self.native_model

  def save_pretrained(
      self, save_directory, *, format="maxtext", hf_model_path=None, revision=None, token=None, cache_dir=None,
      local_files_only=False, override_model_architecture=False,
  ):
    """Save the current model as a native checkpoint or Hugging Face model."""
    return model_persistence.save_pretrained(
        self, save_directory, format=format, hf_model_path=hf_model_path, revision=revision, token=token,
        cache_dir=cache_dir, local_files_only=local_files_only,
        override_model_architecture=override_model_architecture,
    )


class AutoModelForCausalLM:
  """Construct native MaxText models with an initialized runtime configuration."""

  @staticmethod
  def from_config_linen(config, *, runtime=None, model_mode="train", quant_mode_str="train"):
    """Construct the native deprecated Linen model without initializing its variables."""
    return model_construction.from_config_linen(
        config, runtime=runtime, model_mode=model_mode, quant_mode_str=quant_mode_str
    )

  @staticmethod
  def from_config(config, *, runtime=None, rngs=None, model_mode="train", quant_mode_str="train",
                  quantization_config=None):
    """Create a wrapper that constructs an NNX model when materialized."""
    runtime, model_options = model_construction.prepare_from_config(
        config, runtime=runtime, rngs=rngs, model_mode=model_mode, quant_mode_str=quant_mode_str
    )
    return MaxTextModel(
        None, runtime, model_options=model_options, quantization_config=quantization_config,
    )

  @staticmethod
  def from_pretrained(
      pretrained_model_name_or_path, *, config=None, runtime=None, revision=None, token=None, cache_dir=None,
      local_files_only=False, model_mode="train", rng_key=None, wrap_with_tunix_adapter=False,
      tokenizer_pad_id=None, quant_mode_str="train", lazy_load_tensors=True, eager_load_method="safetensors",
      save_dtype="bfloat16", simulated_cpu_devices_count=16, conversion_output_dir=None,
      quantization_config=None,
  ):
    """Create a native model; token selects model snapshot access, not adapter credentials."""
    runtime, checkpoint = resolve_pretrained_source(
        pretrained_model_name_or_path, config=config, runtime=runtime, revision=revision, token=token,
        cache_dir=cache_dir, local_files_only=local_files_only, model_mode=model_mode, rng_key=rng_key,
        wrap_with_tunix_adapter=wrap_with_tunix_adapter, tokenizer_pad_id=tokenizer_pad_id,
        quant_mode_str=quant_mode_str, lazy_load_tensors=lazy_load_tensors, eager_load_method=eager_load_method,
        save_dtype=save_dtype, simulated_cpu_devices_count=simulated_cpu_devices_count,
        conversion_output_dir=conversion_output_dir,
    )
    return MaxTextModel(None, runtime, checkpoint=checkpoint, quantization_config=quantization_config)
