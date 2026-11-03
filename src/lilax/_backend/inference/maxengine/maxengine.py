"""Implementation of Engine API for MaxText."""

from collections import defaultdict
from typing import Any, Callable
import functools
import os.path
import uuid
import warnings

from jax.experimental.layout import Format
from jax.sharding import PartitionSpec as P
import jax
import jax.numpy as jnp

if jax.__version_info__ >= (0, 6, 3):
  from jax.experimental.layout import Layout as DLL
else:
  from jax.experimental.layout import DeviceLocalLayout as DLL

from flax import nnx
from flax import struct
from flax.core.spmd import logical_axis_rules
from jetstream.engine import engine_api, token_utils
from jetstream.engine.tokenizer_pb2 import TokenizerParameters, TokenizerType

from lilax._backend.layers.quantization.configure import configure_quantization
from lilax._backend.inference import inference_utils
from lilax._backend.inference.maxengine import maxengine_adapters
from lilax._backend.inference.maxengine import maxengine_cache_sharding
from lilax._backend.inference.maxengine import maxengine_compilation
from lilax._backend.inference.maxengine import maxengine_decode_state
from lilax._backend.inference.maxengine import maxengine_generation
from lilax._backend.inference.maxengine import maxengine_insertion
from lilax._backend.inference.maxengine import maxengine_model
from lilax._backend.inference.maxengine import maxengine_model_transfer
from lilax._backend.inference.maxengine import maxengine_params
from lilax._backend.inference.maxengine import maxengine_prefill_cache
from lilax._backend.inference.maxengine import maxengine_tokenizer
from lilax._backend.utils import device_mesh
from lilax._backend.utils import lora_utils
from lilax._backend.utils import maxtext_utils
from lilax._backend.utils.diagnostics import pytree
from lilax._backend.utils.model_creation import model_factory
from lilax._backend.utils.tree_ops import partitioning
from lilax._backend.common.common_types import MODEL_MODE_PREFILL, DECODING_ACTIVE_SEQUENCE_INDICATOR, MODEL_MODE_AUTOREGRESSIVE

_DEEPSEEK_V4_CACHE_KEYS = maxengine_insertion.DEEPSEEK_V4_CACHE_KEYS

warnings.simplefilter("ignore", category=FutureWarning)
DecodeState = Any
Prefix = Any
PackedPrefix = Any
Params = Any
PRNGKeyType = Any


@struct.dataclass
class ExistingPrefix:
  """Represents a prefix that has already been processed."""

  cache: Any
  common_prefix_tokens: jax.Array


class MaxEngineConfig:
  """Engine specific config class to allow using multiple MaxEngine instances in an inference run.
  TODO: evaluate the need for this given the restructured pyconfig.py
  """

  def __init__(self, keys):
    self.__dict__["keys"] = keys

  def __getattr__(self, attr):
    if attr not in self.keys:
      raise ValueError(f"Requested key {attr}, not in config")
    return self.keys[attr]

  def __setattr__(self, attr, value):
    raise ValueError

  def get_keys(self):
    return self.keys


class MaxEngine(engine_api.Engine):
  """The computational core of the generative model server.

  Engine defines an API that models must adhere to as they plug into the
  JetStream efficient serving infrastructure.
  """

  def __init__(self, config: Any, devices: Any | None = None):
    self.config = config
    devices_array = device_mesh.create_device_mesh(config=config, devices=devices)
    self._mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

    quant = configure_quantization(config)
    nnx_quant_mode_str = "serve" if (quant is not None and config.checkpoint_is_quantized) else "train"
    _create_model = model_factory.get_nnx_create_model_fn(
        config, mesh=self._mesh, model_mode=MODEL_MODE_PREFILL, quant_mode_str=nnx_quant_mode_str
    )
    _create_model_ar = model_factory.get_nnx_create_model_fn(
        config, mesh=self._mesh, model_mode=MODEL_MODE_AUTOREGRESSIVE, quant_mode_str=nnx_quant_mode_str
    )
    self._nnx_quant_mode_str = nnx_quant_mode_str
    with logical_axis_rules(config.logical_axis_rules):
      abstract_model = nnx.eval_shape(_create_model)
      abstract_model_ar = nnx.eval_shape(_create_model_ar)
    self.model = abstract_model
    self.model_ar = abstract_model_ar
    graphdef, _, _, _ = nnx.split(abstract_model, nnx.Param, nnx.Cache, ...)
    self.graphdef = graphdef
    graphdef_ar, _, _, _ = nnx.split(abstract_model_ar, nnx.Param, nnx.Cache, ...)
    self.graphdef_ar = graphdef_ar
    self._create_model_fn = _create_model
    self._nnx_rest_state = None
    self.replicated_sharding = jax.sharding.NamedSharding(self._mesh, P(None))

    self.abstract_params = None
    self.prefill_kv_cache_annotations = None
    self.kv_cache_annotations = None
    self.kv_cache_annotations_named = None
    self.prefill_kv_cache_shardings = None
    self.kv_cache_shardings = None
    self.state_mesh_annotations = None
    self.decode_state_shapes = None
    self.decode_state_layouts = None
    self.param_layouts = None
    self.rng = None
    self._compiled_initialize_fn = None
    self._compiled_init_cache_fn = None

  def print_stats(self, label: str):
    pytree.print_mem_stats(label)
    pytree.print_cpu_ram_stats(label)

  def _abstract_model_for_mode(self, mode: str):
    """Pick the abstract NNX model whose cache vars match the requested mode."""
    if mode == MODEL_MODE_PREFILL:
      return self.model
    if mode == MODEL_MODE_AUTOREGRESSIVE:
      return self.model_ar
    raise ValueError(f"Expected mode to be MODEL_MODE_PREFILL or MODEL_MODE_AUTOREGRESSIVE, got {mode!r}.")

  def _nnx_cache_state_template(self, mode: str = MODEL_MODE_PREFILL) -> Any:
    """Empty nnx.State template for the model's nnx.Cache vars (PREFILL=batch 1, AR=batch N)."""
    src = self._abstract_model_for_mode(mode)
    _, cache_state, _ = nnx.split(src, nnx.Cache, ...)
    return cache_state

  def _nnx_init_cache_dict(self, mode: str = MODEL_MODE_PREFILL) -> dict:
    """Zero-filled pure-dict cache matching the abstract NNX model."""
    src = self._abstract_model_for_mode(mode)
    _, cache_state, _ = nnx.split(src, nnx.Cache, ...)
    cache_dict = nnx.to_pure_dict(cache_state)
    return jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), cache_dict)

  def _nnx_run_model(
      self,
      params,
      cache_dict,
      decoder_input_tokens,
      decoder_positions,
      *,
      decoder_segment_ids=None,
      enable_dropout=False,
      model_mode,
      previous_chunk=None,
      true_length=None,
      slot=None,
      encoder_images=None,
      encoder_image_masks=None,
      encoder_videos=None,
      encoder_video_masks=None,
      encoder_video_grid_thw=None,
      encoder_audios=None,
  ):
    """NNX equivalent of `model.apply(..., mutable=["cache"])`. Returns (logits, new_cache_dict)."""
    return maxengine_model.run_model(
        abstract_model=self._abstract_model_for_mode(model_mode),
        graphdef=self.graphdef_ar if model_mode == MODEL_MODE_AUTOREGRESSIVE else self.graphdef,
        rest_state=self._nnx_rest_state,
        params=params,
        cache_dict=cache_dict,
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        encoder_images=encoder_images,
        encoder_image_masks=encoder_image_masks,
        encoder_videos=encoder_videos,
        encoder_video_masks=encoder_video_masks,
        encoder_video_grid_thw=encoder_video_grid_thw,
        encoder_audios=encoder_audios,
        enable_dropout=enable_dropout,
        model_mode=model_mode,
        previous_chunk=previous_chunk,
        true_length=true_length,
        slot=slot,
    )

  def generate_aot(self, params: Params, decode_state: DecodeState, rng: PRNGKeyType | None = None):
    """Wrapper to generate for ahead of time compilation."""
    return self.generate(params=params, decode_state=decode_state, rng=rng)

  def _compile_generate_and_get_layouts(
      self, params: Any, decode_state: Any, rng_shape: Any, xla_flags: dict[str, Any] | None = None
  ) -> tuple[Any, Any, Any, Any]:
    """Optimal memory layout for params and decode_state."""
    return maxengine_compilation.compile_generate_and_get_layouts(
        self.generate_aot, params, decode_state, rng_shape, xla_flags
    )

  def _identity(self, x: Any) -> Any:
    """Avoids lambda that breaks JAX caching."""
    return x

  def _iterated_layout(self, arrays: Any, layouts: Any, xla_flags: dict[str, Any] | None = None) -> Any:
    """Lays out an array tensor by tensor to prevent OOMs."""

    return maxengine_compilation.iterated_layout(self._identity, arrays, layouts, xla_flags)

  def aot_compile(
      self, params: Params, pass_rng_shape: bool, xla_flags: dict[str, Any] | None = None
  ) -> tuple[Any, Params, Any]:
    """Ahead of time compilation of generate with auto layout, relayout parameters."""
    if pass_rng_shape:
      rng_shape = jax.ShapeDtypeStruct([4], jax.numpy.dtype("uint32"))
    else:
      rng_shape = None
    self.decode_state_shapes = jax.eval_shape(self.init_decode_state, rng_shape)

    generate_executable, self.param_layouts, _, self.decode_state_layouts = self._compile_generate_and_get_layouts(
        self.abstract_params, self.decode_state_shapes, rng_shape, xla_flags
    )
    return (
        generate_executable,
        self._iterated_layout(params, self.param_layouts),
        jax.jit(self.init_decode_state, in_shardings=(None), out_shardings=self.decode_state_layouts)
        .lower(rng_shape)
        .compile(),
    )

  def load_params(self, *args, params=None, model=None, rng: PRNGKeyType | None = None, **kwargs) -> Params:
    """Load Parameters from GCS or reshard given Parameters"""
    if model is not None and params is not None:
      raise ValueError("Pass either model or params, not both.")
    if rng is None:
      rng = jax.random.PRNGKey(0)
    return self._load_params_nnx(params=params, rng=rng, model=model)

  def _load_params_nnx(self, params, rng, model=None):
    """NNX equivalent of load_params: returns an nnx.Param state and populates KV cache shardings.

    Quantization handling:
      * `checkpoint_is_quantized=True`: model built in `serve` mode (no full
        kernel), `from_pretrained` reads `qrhs.frozen` from disk.
      * `checkpoint_is_quantized=False` + `quantization=...`: model built in
        `train` mode, full-precision kernel loaded; AQT layers quantize per
        forward. Same output as serve mode (absmax calibration), slower.
    """
    with logical_axis_rules(self.config.logical_axis_rules):
      graphdef, rest_state, concrete_model = maxengine_params.initialize_model_state(
          self.model, self._mesh, self._create_model_fn
      )
      self.graphdef = graphdef
      del concrete_model

    if model is not None:
      params_state = maxengine_model_transfer.transfer_model_state(
          model, self.model, self.model_ar, self._mesh, self.config.logical_axis_rules, rest_state
      )
      self._nnx_rest_state = rest_state
    elif params:
      params_state = maxengine_params.reshard_params(self.model, self.config, self._mesh, params)
      self._nnx_rest_state = rest_state
    else:
      params_state, nnx_model, loaded_rest_state, loaded_rest_dict, rest_dict = maxengine_params.load_checkpoint(
          self.config, self._mesh, self._nnx_quant_mode_str, rest_state
      )
      self._nnx_rest_state = rest_state
      del nnx_model, loaded_rest_state, loaded_rest_dict, rest_dict

    self.abstract_params = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=x.sharding)
        if isinstance(x, jax.Array)
        else None,
        params_state,
    )

    self.prefill_kv_cache_annotations = maxtext_utils.get_prefill_kv_cache_annotations_nnx(
        self.model, self.config, self._mesh
    )
    self.prefill_kv_cache_shardings = maxengine_cache_sharding.named_cache_shardings(
        self.prefill_kv_cache_annotations, self._mesh
    )
    if self.config.stack_prefill_result_cache and not self.config.scan_layers:
      self.prefill_kv_cache_shardings = maxengine_cache_sharding.add_prefill_stack_axis(
          self.prefill_kv_cache_shardings, self._mesh
      )
      self.prefill_kv_cache_shardings = maxengine_prefill_cache.stacked_prefill_cache_shardings(
          self.prefill_kv_cache_shardings
      )
    self.kv_cache_annotations = maxtext_utils.get_kv_cache_annotations_nnx(self.model_ar, self.config, self._mesh)
    self.kv_cache_shardings = maxengine_cache_sharding.named_cache_shardings(self.kv_cache_annotations, self._mesh)
    self.state_mesh_annotations = None

    self.print_stats("After load_params (NNX)")
    return params_state

  def load_single_adapter(self, adapter_path):
    """Load a single LoRA adapter from `adapter_path`.

    Expects `adapter_config.json` at the root and adapter weights under
    `<adapter_path>/0/items`. The returned `params` follows the same tree
    shape as `self.abstract_params` (NNX or Linen, depending on the engine).
    """
    params, config = maxengine_adapters.load_single_adapter(self.config, self.abstract_params, adapter_path)
    self.print_stats("After load_single_adapter.")

    return params, config

  def apply_adapter(self, base_params, adapter_config, adapter_params):
    """Apply the adapter params on the base params."""

    maxengine_adapters.apply_adapter(base_params, adapter_config, adapter_params)

  def unapply_adapter(self, base_params, adapter_config, adapter_params):
    """Unapply the adapter params from the merged params to get back the base params."""

    maxengine_adapters.unapply_adapter(base_params, adapter_config, adapter_params)

  def quantize_params(self, state, rng: PRNGKeyType | None = None):
    """Forward pass to quantize decode params."""
    if rng is None:
      rng = jax.random.PRNGKey(0)
    raise NotImplementedError(
        "Use load_params() on NNX — the convert step runs inside _load_params_nnx via "
        "_convert_and_quantize_nnx. quantize_params(state, rng) is the Linen API."
    )

  def _maybe_stack_prefill_result_cache(self, cache):
    """Stack the caches across the layers."""
    return maxengine_prefill_cache.stack_prefill_result_cache(self.config, cache)

  def _maybe_unstack_prefill_result_cache(self, cache):
    """Unstack the caches across the layers."""
    return maxengine_prefill_cache.unstack_prefill_result_cache(self.config, self.model, cache)

  def prefill_aot(
      self,
      params: Params,
      padded_tokens: jax.Array,
      true_length: int,
      rng: PRNGKeyType | None = None,
  ):
    """Wrapper for prefill for ahead-of-time compilation."""

    return self.prefill(
        params=params,
        padded_tokens=padded_tokens,
        true_length=true_length,
        rng=rng,
    )

  @functools.partial(
      jax.jit,
      static_argnums=(0,),
      static_argnames=("video_grid_thw", "return_prompt_logp", "algorithm", "topk", "nucleus_topp"),
  )
  def _prefill_jit(
      self,
      *,
      params: Params,
      existing_prefix: ExistingPrefix | None = None,
      padded_tokens: jax.Array,
      positions: jax.Array | None = None,
      mrope_deltas: jax.Array | None = None,
      images: jax.Array | None = None,
      image_masks: jax.Array | None = None,
      videos: jax.Array | None = None,
      video_masks: jax.Array | None = None,
      video_grid_thw: tuple[int, int, int] | None = None,
      audio_values: jax.Array | None = None,
      audio_masks: jax.Array | None = None,
      true_length: int,
      sampler: Callable[[Any], Any] | None = None,
      rng: PRNGKeyType | None = None,
      slot: int | None = None,
      return_prompt_logp: bool = False,
      algorithm: str | None = None,
      topk: int | None = None,
      nucleus_topp: float | None = None,
      temperature: float | None = None,
  ):
    """Performs a JIT-compiled prefill operation on a sequence of tokens.

    This function processes an input sequence (prompt) through the model to compute
    the key-value cache (KV cache). It supports chunked prefilling, where a long
    prompt can be processed in multiple calls by passing an `existing_prefix`.
    After processing the tokens, it samples the first token to begin the
    autoregressive decoding process.

    Args:
      params: The model parameters.
      existing_prefix: An optional `ExistingPrefix` containing the KV cache and
        tokens of a previously processed chunk. Used for chunked prefilling.
      padded_tokens: The input token sequence, padded to a fixed length.
      images: Optional input images for multimodal models.
      image_masks: Optional image masks for multimodal models with tiled images.
      audio_values: Optional input audio for multimodal models.
      audio_masks: Optional audio masks for multimodal models (currently unused).
      true_length: The actual length of `padded_tokens` before padding.
      sampler: A callable for custom sampling logic (currently unused).
      rng: JAX random number generator key for sampling.
      slot: The batch slot index for this request, used for paged attention.
      return_prompt_logp: If True, calculates and returns the log probabilities
        of the prompt tokens.
      algorithm: The sampling algorithm to use (e.g., 'greedy', 'composite').
        Overrides the engine's default.
      topk: The value for top-k sampling. Overrides the engine's default.
      nucleus_topp: The value for top-p (nucleus) sampling. Overrides the
        engine's default.
      temperature: The sampling temperature. Overrides the engine's default.

    Returns:
      A tuple containing:
        - A prefix dictionary with the computed KV cache, the logits for the
          next token, the first sampled token, and other metadata.
        - An `engine_api.ResultTokens` object containing the single sampled
          token to begin decoding.
    """

    start_position = 0
    previous_chunk = None
    input_params = params
    if existing_prefix is not None:
      if not self.use_chunked_prefill:
        raise ValueError("Using chunked prefill is needed for existing_prefix.")
      start_position = existing_prefix.common_prefix_tokens.shape[0]
      previous_chunk = jnp.expand_dims(existing_prefix.common_prefix_tokens, 0)

    full_true_length = start_position + true_length

    input_tokens = jnp.expand_dims(padded_tokens, 0)

    if positions is not None:
      if positions.ndim == 2:
        positions = jnp.expand_dims(positions, 1)
    else:
      positions = jnp.expand_dims(jnp.arange(start_position, start_position + input_tokens.shape[1]), 0)

    if self.config.use_multimodal and images is not None:
      if images.ndim == 3:
        images = images[jnp.newaxis, jnp.newaxis, ...]
        image_masks = image_masks[jnp.newaxis, jnp.newaxis, ...] if image_masks is not None else None
      elif images.ndim == 4:
        images = images[jnp.newaxis, ...]
        image_masks = image_masks[jnp.newaxis, ...] if image_masks is not None else None

    start_to_n = jnp.arange(start_position, start_position + input_tokens.shape[1])
    ones_to_keep = start_to_n < full_true_length
    one_d_output = ones_to_keep * DECODING_ACTIVE_SEQUENCE_INDICATOR
    sequence_indicator = jnp.expand_dims(one_d_output, 0)

    rng, new_rng = jax.random.split(rng)
    nnx_cache = (
        existing_prefix.cache if existing_prefix is not None else self._nnx_init_cache_dict(mode=MODEL_MODE_PREFILL)
    )
    with self._mesh, logical_axis_rules(self.config.logical_axis_rules):
      flat_logits, new_cache_dict = self._nnx_run_model(
          params=input_params,
          cache_dict=nnx_cache,
          decoder_input_tokens=input_tokens,
          decoder_positions=positions,
          decoder_segment_ids=sequence_indicator,
          encoder_images=images,
          encoder_image_masks=image_masks,
          encoder_videos=videos,
          encoder_video_masks=video_masks,
          encoder_video_grid_thw=video_grid_thw,
          encoder_audios=audio_values,
          enable_dropout=False,
          model_mode=MODEL_MODE_PREFILL,
          previous_chunk=previous_chunk,
          true_length=true_length,
          slot=slot,
      )
    new_vars = {"cache": new_cache_dict}
    if return_prompt_logp:
      prompt_logp = inference_utils.prompt_logprobs_from_prefill(flat_logits, input_tokens, true_length)
    else:
      prompt_logp = None

    generated_tokens = jnp.zeros((1, 1), dtype=jnp.int32)
    selected_logits = jax.lax.dynamic_slice(
        flat_logits,
        (0, true_length - 1, 0),
        (flat_logits.shape[0], 1, flat_logits.shape[2]),
    )
    selected_logits = jax.lax.with_sharding_constraint(selected_logits, self.replicated_sharding)

    rng, new_rng = jax.random.split(rng)
    first_generated_token = inference_utils.sampling(
        selected_logits,
        new_rng,
        algorithm if algorithm is not None else self.config.decode_sampling_strategy,
        topk=topk if topk is not None else self.config.decode_sampling_top_k,
        nucleus_topp=nucleus_topp if nucleus_topp is not None else self.config.decode_sampling_nucleus_p,
        temperature=temperature if temperature is not None else self.config.decode_sampling_temperature,
    )

    all_valid = jnp.ones(first_generated_token.shape, dtype=jnp.int8)
    if self.config.return_log_prob:
      token_logp = inference_utils.log_prob_of_chosen_token(selected_logits, first_generated_token)
    else:
      token_logp = jnp.zeros(first_generated_token.shape, dtype=jnp.float32)
    result = engine_api.ResultTokens(
        data=jnp.concatenate((first_generated_token, all_valid, generated_tokens), axis=1),
        tokens_idx=(0, 1),
        valid_idx=(1, 2),
        length_idx=(2, 3),
        log_prob=token_logp,
        samples_per_slot=1,
    )

    cache = new_vars["cache"]
    cache = self._maybe_stack_prefill_result_cache(cache)

    if mrope_deltas is not None:
      next_pos = jnp.full((1, 1), full_true_length, dtype=jnp.int32) + mrope_deltas
    else:
      next_pos = jnp.full((1, 1), full_true_length, dtype=jnp.int32)

    return {
        "logits": selected_logits,
        "cache": cache,
        "next_pos": next_pos,
        "generated_tokens": generated_tokens,
        "tokens": first_generated_token,
        "prompt_logp": prompt_logp,
        "token_logp": token_logp,
    }, result

  def prefill(
      self,
      *,
      params: Params,
      existing_prefix: ExistingPrefix | None = None,
      padded_tokens: jax.Array,
      positions: jax.Array | None = None,
      mrope_deltas: jax.Array | None = None,
      images: jax.Array | None = None,
      image_masks: jax.Array | None = None,
      videos: jax.Array | None = None,
      video_masks: jax.Array | None = None,
      video_grid_thw: jax.Array | None = None,
      audio_values: jax.Array | None = None,
      audio_masks: jax.Array | None = None,
      true_length: int,
      sampler: Callable[[Any], Any] | None = None,
      rng: PRNGKeyType | None = None,
      request_id: uuid.UUID | None = None,
      slot: int | None = None,
      return_prompt_logp: bool = False,
      algorithm: str | None = None,
      topk: int | None = None,
      nucleus_topp: float | None = None,
      temperature: float | None = None,
  ):
    """Public API for prefill that updates page state outside JIT."""
    if video_grid_thw is not None:
      flat_video_grid = jax.device_get(video_grid_thw).reshape(-1)
      if flat_video_grid.size != 3:
        raise ValueError(f"Decode currently supports one video grid with 3 values, got shape {video_grid_thw.shape}.")
      video_grid_thw = tuple(int(dim) for dim in flat_video_grid)

    if rng is None:
      if self.rng is None:
        self.rng = jax.random.PRNGKey(0)
      self.rng, rng = jax.random.split(self.rng)

    return self._prefill_jit(
        params=params,
        existing_prefix=existing_prefix,
        padded_tokens=padded_tokens,
        positions=positions,
        mrope_deltas=mrope_deltas,
        images=images,
        image_masks=image_masks,
        videos=videos,
        video_masks=video_masks,
        video_grid_thw=video_grid_thw,
        audio_values=audio_values,
        audio_masks=audio_masks,
        sampler=sampler,
        true_length=true_length,
        slot=slot,
        rng=rng,
        return_prompt_logp=return_prompt_logp,
        algorithm=algorithm,
        topk=topk,
        nucleus_topp=nucleus_topp,
        temperature=temperature,
    )

  def prefill_multisampling_aot(
      self,
      params: Params,
      padded_tokens: jax.Array,
      true_length: int,
      rng: PRNGKeyType | None = None,
      num_samples: int = 1,
      sampler: Callable[[Any], Any] | None = None,
      algorithm: str | None = None,
      topk: int | None = None,
      nucleus_topp: float | None = None,
      temperature: float | None = None,
  ):
    """Wrapper for multi-sampling prefill for ahead-of-time compilation."""
    return self.prefill_multisampling(
        params=params,
        padded_tokens=padded_tokens,
        true_length=true_length,
        sampler=sampler,
        rng=rng,
        num_samples=num_samples,
        algorithm=algorithm,
        topk=topk,
        nucleus_topp=nucleus_topp,
        temperature=temperature,
    )

  def prefill_multisampling(
      self,
      *,
      params: Params,
      padded_tokens: jax.Array,
      true_length: int,
      sampler: Callable[[Any], Any] | None = None,
      rng: PRNGKeyType | None = None,
      num_samples: int = 1,
      algorithm: str | None = None,
      topk: int | None = None,
      nucleus_topp: float | None = None,
      temperature: float | None = None,
  ):
    """Public API for prefill multisampling."""
    if rng is None:
      if self.rng is None:
        self.rng = jax.random.PRNGKey(0)
      self.rng, rng = jax.random.split(self.rng)

    return self._prefill_multisampling_jit(
        params=params,
        padded_tokens=padded_tokens,
        true_length=true_length,
        sampler=sampler,
        rng=rng,
        num_samples=num_samples,
        algorithm=algorithm,
        topk=topk,
        nucleus_topp=nucleus_topp,
        temperature=temperature,
    )

  @functools.partial(jax.jit, static_argnums=(0,), static_argnames=("num_samples", "algorithm", "topk", "nucleus_topp"))
  def _prefill_multisampling_jit(
      self,
      *,
      params: Params,
      padded_tokens: jax.Array,
      true_length: int,
      sampler: Callable[[Any], Any] | None = None,
      rng: PRNGKeyType | None = None,
      num_samples: int = 1,
      algorithm: str | None = None,
      topk: int | None = None,
      nucleus_topp: float | None = None,
      temperature: float | None = None,
  ) -> tuple[Prefix, Any]:
    """Computes a kv-cache and multiple first tokens for a request."""
    input_tokens = jnp.expand_dims(padded_tokens, 0)
    positions = jnp.expand_dims(jnp.arange(0, input_tokens.shape[1]), 0)

    zero_to_n = jnp.arange(0, padded_tokens.shape[0])
    ones_to_keep = zero_to_n < true_length
    one_d_output = ones_to_keep * DECODING_ACTIVE_SEQUENCE_INDICATOR
    sequence_indicator = jnp.expand_dims(one_d_output, 0)

    rng, new_rng = jax.random.split(rng)
    with self._mesh, logical_axis_rules(self.config.logical_axis_rules):
      flat_logits, new_cache_dict = self._nnx_run_model(
          params=params,
          cache_dict=self._nnx_init_cache_dict(mode=MODEL_MODE_PREFILL),
          decoder_input_tokens=input_tokens,
          decoder_positions=positions,
          decoder_segment_ids=sequence_indicator,
          enable_dropout=False,
          model_mode=MODEL_MODE_PREFILL,
      )
    new_vars = {"cache": new_cache_dict}

    next_pos = jnp.full((1, 1), true_length, dtype=jnp.int32)
    selected_logits = jax.lax.dynamic_slice(
        flat_logits,
        (0, true_length - 1, 0),
        (flat_logits.shape[0], 1, flat_logits.shape[2]),
    )
    selected_logits = jax.lax.with_sharding_constraint(selected_logits, self.replicated_sharding)

    first_generated_tokens = []
    token_logps = [] if self.config.return_log_prob else None
    for _ in range(num_samples):
      rng, new_rng = jax.random.split(rng)
      first_generated_token = inference_utils.sampling(
          selected_logits,
          new_rng,
          algorithm if algorithm is not None else self.config.decode_sampling_strategy,
          topk=topk if topk is not None else self.config.decode_sampling_top_k,
          nucleus_topp=nucleus_topp if nucleus_topp is not None else self.config.decode_sampling_nucleus_p,
          temperature=temperature if temperature is not None else self.config.decode_sampling_temperature,
      )
      first_generated_tokens.append(first_generated_token)
      if self.config.return_log_prob:
        p = inference_utils.log_prob_of_chosen_token(selected_logits, first_generated_token)
        token_logps.append(p)
    first_generated_tokens = jnp.concatenate(first_generated_tokens, axis=0)
    if self.config.return_log_prob:
      token_logps = jnp.concatenate(token_logps, axis=0)

    all_valid = jnp.ones((num_samples, 1), dtype=jnp.int8)
    generated_tokens = jnp.zeros((num_samples, 1), dtype=jnp.int32)
    result = engine_api.ResultTokens(
        data=jnp.concatenate((first_generated_tokens, all_valid, generated_tokens), axis=1),
        tokens_idx=(0, 1),
        valid_idx=(1, 2),
        length_idx=(2, 3),
        log_prob=token_logps,
        samples_per_slot=num_samples,
    )

    cache = new_vars["cache"]
    cache = self._maybe_stack_prefill_result_cache(cache)

    return {
        "logits": selected_logits,
        "cache": cache,
        "next_pos": next_pos,
        "generated_tokens": generated_tokens,
        "tokens": first_generated_tokens,
    }, result

  @functools.partial(
      jax.jit,
      static_argnums=(0,),
      static_argnames=("num_prompts", "return_prompt_logp", "algorithm", "topk", "nucleus_topp"),
  )
  def prefill_concat(
      self,
      *,
      params: Params,
      existing_prefix: ExistingPrefix | None = None,
      padded_tokens: jax.Array,
      decoder_positions: jax.Array,
      decoder_segment_ids: jax.Array,
      start_pos: jax.Array,
      true_lengths: jax.Array,
      num_prompts: int,
      sampler: Callable[[Any], Any] | None = None,
      rng: PRNGKeyType | None = None,
      return_prompt_logp: bool = False,
      algorithm: str | None = None,
      topk: int | None = None,
      nucleus_topp: float | None = None,
      temperature: float | None = None,
  ):
    """Computes a kv-cache for a packed request of concatenated prompts."""
    if existing_prefix:
      raise ValueError("We don't know what to do with existing_prefix")

    if rng is None:
      rng = jax.random.PRNGKey(0)
    input_tokens = jnp.expand_dims(padded_tokens, 0)
    decoder_positions = jnp.expand_dims(decoder_positions, 0)
    decoder_segment_ids = jnp.expand_dims(decoder_segment_ids, 0)
    rng, _ = jax.random.split(rng)
    with self._mesh, logical_axis_rules(self.config.logical_axis_rules):
      flat_logits, new_cache_dict = self._nnx_run_model(
          params=params,
          cache_dict=self._nnx_init_cache_dict(mode=MODEL_MODE_PREFILL),
          decoder_input_tokens=input_tokens,
          decoder_positions=decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          enable_dropout=False,
          model_mode=MODEL_MODE_PREFILL,
      )
    new_vars = {"cache": new_cache_dict}
    cache = new_vars["cache"]
    cache = self._maybe_stack_prefill_result_cache(cache)
    if return_prompt_logp:
      prompt_logp = inference_utils.prompt_logprobs_from_packed_prefill(
          flat_logits, input_tokens, decoder_positions, decoder_segment_ids, true_lengths
      )
    else:
      prompt_logp = None

    def process_packed_logits_and_caches(packed_flat_logits, idx):
      next_pos = jnp.full((1, 1), true_lengths[idx], dtype=jnp.int32)
      generated_tokens = jnp.zeros((1, 1), dtype=jnp.int32)
      selected_logits = jax.lax.dynamic_slice(
          packed_flat_logits,
          (0, start_pos[idx] + true_lengths[idx] - 1, 0),
          (packed_flat_logits.shape[0], 1, packed_flat_logits.shape[2]),
      )
      selected_logits = jax.lax.with_sharding_constraint(selected_logits, self.replicated_sharding)
      first_generated_token = inference_utils.sampling(
          selected_logits,
          rng,
          algorithm if algorithm is not None else self.config.decode_sampling_strategy,
          topk=topk if topk is not None else self.config.decode_sampling_top_k,
          nucleus_topp=nucleus_topp if nucleus_topp is not None else self.config.decode_sampling_nucleus_p,
          temperature=temperature if temperature is not None else self.config.decode_sampling_temperature,
      )
      all_valid = jnp.ones(first_generated_token.shape, dtype=jnp.int8)
      if self.config.return_log_prob:
        token_logp = inference_utils.log_prob_of_chosen_token(selected_logits, first_generated_token)
      else:
        token_logp = None
      result = engine_api.ResultTokens(
          data=jnp.concatenate((first_generated_token, all_valid, generated_tokens), axis=1),
          tokens_idx=(0, 1),
          valid_idx=(1, 2),
          length_idx=(2, 3),
          log_prob=token_logp,
          samples_per_slot=1,
      )

      return {
          "logits": selected_logits,
          "next_pos": next_pos,
          "generated_tokens": generated_tokens,
          "tokens": first_generated_token,
      }, result

    prefill_results = defaultdict(list)
    first_tokens = []
    for idx in range(num_prompts):
      prefill_result, first_token = process_packed_logits_and_caches(flat_logits, idx)
      for k, v in prefill_result.items():
        prefill_results[k].append(v)
      first_tokens.append(first_token)
    prefill_results = {k: jnp.stack(v) for k, v in prefill_results.items()}
    prefill_results["prompt_logp"] = prompt_logp
    return cache, prefill_results, first_tokens

  def generate(
      self,
      params: Params,
      decode_state: DecodeState,
      sampler: Callable[[Any], Any] | None = None,
      rng: PRNGKeyType | None = None,
      algorithm: str | None = None,
      topk: int | None = None,
      nucleus_topp: float | None = None,
      temperature: float | None = None,
  ):
    """Public API for generate that updates page state outside JIT."""
    if rng is None:
      if self.rng is None:
        self.rng = jax.random.PRNGKey(0)
      self.rng, rng = jax.random.split(self.rng)

    new_state, result = self._generate_jit(
        params=params,
        decode_state=decode_state,
        sampler=sampler,
        rng=rng,
        algorithm=algorithm,
        topk=topk,
        nucleus_topp=nucleus_topp,
        temperature=temperature,
    )
    return partitioning.unbox_logicallypartioned(new_state), result

  @functools.partial(
      jax.jit, static_argnums=(0,), donate_argnums=(2,), static_argnames=("algorithm", "topk", "nucleus_topp")
  )
  def _generate_jit(
      self,
      params: Params,
      decode_state: DecodeState,
      *,
      sampler: Callable[[Any], Any] | None = None,
      rng: PRNGKeyType | None = None,
      algorithm: str | None = None,
      topk: int | None = None,
      nucleus_topp: float | None = None,
      temperature: float | None = None,
  ):
    """Performs a single, JIT-compiled autoregressive decoding step."""
    return maxengine_generation.generate_step(
        self.config, self._mesh, self.replicated_sharding, self.kv_cache_shardings,
        self.model_ar, self.graphdef_ar, self._nnx_rest_state,
        params, decode_state, rng, algorithm, topk, nucleus_topp, temperature,
    )

  @functools.partial(
      jax.jit,
      static_argnums=(0,),
      donate_argnums=(
          1,
          2,
      ),
  )
  def bulk_insert(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slots: list[int],
  ) -> DecodeState:
    """Insert a single computed prefill cache into multiple slots in KV cache."""
    unboxed_prefix = partitioning.unbox_logicallypartioned(prefix)
    unboxed_prefix["cache"] = self._maybe_unstack_prefill_result_cache(unboxed_prefix["cache"])

    def copy(path, partial_cache, full_cache, annotations):
      return maxengine_insertion.copy_cache_leaf(path, partial_cache, full_cache, annotations, slots)

    inserted_cache = jax.tree_util.tree_map_with_path(
        copy,
        unboxed_prefix["cache"],
        decode_state["cache"],
        self.kv_cache_annotations_named,
    )
    for i, slot in enumerate(slots):
      decode_state["logits"] = jax.lax.dynamic_update_index_in_dim(
          decode_state["logits"], unboxed_prefix["logits"], slot, 0
      )
      decode_state["next_pos"] = jax.lax.dynamic_update_index_in_dim(
          decode_state["next_pos"], unboxed_prefix["next_pos"], slot, 0
      )
      decode_state["generated_tokens"] = jax.lax.dynamic_update_index_in_dim(
          decode_state["generated_tokens"],
          jnp.expand_dims(unboxed_prefix["generated_tokens"][i], axis=0),
          slot,
          0,
      )
      decode_state["tokens"] = jax.lax.dynamic_update_index_in_dim(
          decode_state["tokens"],
          jnp.expand_dims(unboxed_prefix["tokens"][i], axis=0),
          slot,
          0,
      )
      decode_state["token_logp"] = jax.lax.dynamic_update_index_in_dim(
          decode_state["token_logp"],
          jnp.expand_dims(unboxed_prefix["token_logp"][i], axis=0),
          slot,
          0,
      )

    inserted_logits = jax.lax.with_sharding_constraint(decode_state["logits"], self.replicated_sharding)
    inserted_generated_tokens = jax.lax.with_sharding_constraint(
        decode_state["generated_tokens"], self.replicated_sharding
    )
    inserted_next_pos = jax.lax.with_sharding_constraint(decode_state["next_pos"], self.replicated_sharding)
    inserted_tokens = jax.lax.with_sharding_constraint(decode_state["tokens"], self.replicated_sharding)
    inserted_cache = jax.lax.with_sharding_constraint(inserted_cache, self.kv_cache_shardings)
    inserted_token_logp = jax.lax.with_sharding_constraint(decode_state["token_logp"], self.replicated_sharding)
    return {
        "logits": inserted_logits,
        "cache": inserted_cache,
        "next_pos": inserted_next_pos,
        "generated_tokens": inserted_generated_tokens,
        "tokens": inserted_tokens,
        "token_logp": inserted_token_logp,
    }


  def release_pages(self, slot: int):
    """Releases pages associated with a specific slot (page group) via the PageManager."""
    print(f"Warning: release_pages called for slot {slot} but paged attention is not configured.")

  def get_prefix_destination_sharding(self) -> Any:
    return {
        "logits": self.replicated_sharding,
        "cache": self.prefill_kv_cache_shardings,
        "next_pos": self.replicated_sharding,
        "generated_tokens": self.replicated_sharding,
        "tokens": self.replicated_sharding,
        "token_logp": self.replicated_sharding,
    }

  def get_tokenizer(self) -> Any:
    """Return tokenizer parameters."""
    return maxengine_tokenizer.get_tokenizer(self.config, TokenizerType, TokenizerParameters)

  def build_tokenizer(self, metadata: Any):
    """Return a tokenizer"""
    return maxengine_tokenizer.build_tokenizer(metadata, TokenizerType, token_utils)

  def init_decode_state(
      self,
      *args,
      rng: PRNGKeyType | None = None,
      **kwargs,
  ) -> DecodeState:
    """Initialises any state which a generation step transforms."""
    return self._init_decode_state_nnx()

  def _init_decode_state_nnx(self) -> DecodeState:
    """NNX equivalent of init_decode_state. Returns a decode_state dict with a pure-dict cache."""
    batch = int(self.config.per_device_batch_size * self.mesh.size)
    vocab = self.config.vocab_size
    cache_dict_abs = self._nnx_init_cache_dict(mode=MODEL_MODE_AUTOREGRESSIVE)
    if self._compiled_init_cache_fn is None:
      self._compiled_init_cache_fn = maxengine_decode_state.compile_init_cache(cache_dict_abs, self.kv_cache_shardings)
    (cache,) = self._compiled_init_cache_fn()
    self.kv_cache_annotations_named = maxengine_decode_state.cache_annotations(self.model_ar)
    return maxengine_decode_state.init_decode_state(batch, vocab, cache)

  @property
  def max_concurrent_decodes(self) -> int:
    """Free slots."""
    return int(self.config.per_device_batch_size * self.mesh.size)

  @property
  def max_prefill_length(self) -> int:
    """Maximum prefill length."""
    return int(self.config.max_prefill_predict_length)

  @property
  def use_chunked_prefill(self) -> bool:
    """Whether to use chunked prefill."""
    return self.config.use_chunked_prefill

  @property
  def prefill_chunk_size(self) -> int:
    """Prefill chunk size."""
    return int(self.config.prefill_chunk_size)

  @property
  def samples_per_slot(self) -> int:
    """Number of samples per slot."""
    return 1

  @property
  def mesh(self) -> jax.sharding.Mesh:
    return self._mesh

  @property
  def colocated_cpus(self) -> None:
    """CPU devices colocated with the engine's accelerators."""
    raise NotImplementedError


  @functools.partial(jax.jit, static_argnums=(0,), donate_argnames=("prefix", "decode_state"))
  def _insert_jit(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slot: int,
      request_id: uuid.UUID | None = None,
  ) -> DecodeState:
    """Insert a single computed prefill cache into KV cache."""
    unboxed_prefix = partitioning.unbox_logicallypartioned(prefix)
    unboxed_prefix["cache"] = self._maybe_unstack_prefill_result_cache(unboxed_prefix["cache"])

    def copy(path, partial_cache, full_cache, annotations):
      return maxengine_insertion.copy_cache_leaf(path, partial_cache, full_cache, annotations, (slot,))

    inserted_cache = jax.tree_util.tree_map_with_path(
        copy,
        unboxed_prefix["cache"],
        decode_state["cache"],
        self.kv_cache_annotations_named,
    )

    inserted_logits = jax.lax.dynamic_update_index_in_dim(decode_state["logits"], unboxed_prefix["logits"], slot, 0)
    inserted_next_pos = jax.lax.dynamic_update_index_in_dim(
        decode_state["next_pos"], jnp.asarray(unboxed_prefix["next_pos"], dtype=decode_state["next_pos"].dtype), slot, 0
    )
    inserted_generated_tokens = jax.lax.dynamic_update_index_in_dim(
        decode_state["generated_tokens"],
        unboxed_prefix["generated_tokens"],
        slot,
        0,
    )
    inserted_tokens = jax.lax.dynamic_update_index_in_dim(decode_state["tokens"], unboxed_prefix["tokens"], slot, 0)
    inserted_token_logp = jax.lax.dynamic_update_index_in_dim(
        decode_state["token_logp"], unboxed_prefix["token_logp"], slot, 0
    )

    inserted_logits = jax.lax.with_sharding_constraint(inserted_logits, self.replicated_sharding)
    inserted_generated_tokens = jax.lax.with_sharding_constraint(inserted_generated_tokens, self.replicated_sharding)
    inserted_next_pos = jax.lax.with_sharding_constraint(inserted_next_pos, self.replicated_sharding)
    inserted_tokens = jax.lax.with_sharding_constraint(inserted_tokens, self.replicated_sharding)
    inserted_cache = jax.lax.with_sharding_constraint(inserted_cache, self.kv_cache_shardings)
    inserted_token_logp = jax.lax.with_sharding_constraint(inserted_token_logp, self.replicated_sharding)

    return {
        "logits": inserted_logits,
        "cache": inserted_cache,
        "next_pos": inserted_next_pos,
        "generated_tokens": inserted_generated_tokens,
        "tokens": inserted_tokens,
        "token_logp": inserted_token_logp,
    }

  def insert(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slot: int,
      request_id: uuid.UUID | None = None,
  ) -> DecodeState:
    """Non-JIT wrapper for inserting prefill cache."""

    updated_decode_state = self._insert_jit(
        prefix=prefix,
        decode_state=decode_state,
        slot=slot,
    )
    return updated_decode_state

  @functools.partial(
      jax.jit,
      static_argnums=(0,),
      static_argnames=(
          "num_prompts",
          "seq_len",
      ),
      donate_argnums=(
          1,
          2,
      ),
  )
  def insert_partial(
      self,
      prefix: PackedPrefix,
      decode_state: DecodeState,
      cache: Any,
      slots: jax.Array,
      *,
      start_indices: jax.Array,
      num_prompts: int,
      seq_len: int,
  ) -> DecodeState:
    """Insert into KV cache"""
    unboxed_prefix = partitioning.unbox_logicallypartioned(prefix)
    cache_unboxed = partitioning.unbox_logicallypartioned(cache)
    cache_unboxed = self._maybe_unstack_prefill_result_cache(cache_unboxed)
    start_idx = 0
    slot = slots[0]

    def copy(path, partial_cache, full_cache, annotations):
      return maxengine_insertion.copy_packed_cache_leaf(
          path, partial_cache, full_cache, annotations, self.config, slot, start_idx, i, num_prompts, seq_len
      )

    inserted_cache = decode_state["cache"]
    inserted_logits = decode_state["logits"]
    inserted_next_pos = decode_state["next_pos"]
    inserted_generated_tokens = decode_state["generated_tokens"]
    inserted_tokens = decode_state["tokens"]
    inserted_token_logp = decode_state["token_logp"]

    for i in range(num_prompts):
      start_idx = start_indices[i]
      slot = slots[i]
      inserted_cache = jax.tree_util.tree_map_with_path(
          copy, cache_unboxed, inserted_cache, self.kv_cache_annotations_named
      )
      inserted_logits = jax.lax.dynamic_update_index_in_dim(inserted_logits, unboxed_prefix["logits"][i, ...], slot, 0)
      inserted_next_pos = jax.lax.dynamic_update_index_in_dim(
          inserted_next_pos, unboxed_prefix["next_pos"][i, ...], slot, 0
      )
      inserted_generated_tokens = jax.lax.dynamic_update_index_in_dim(
          inserted_generated_tokens,
          unboxed_prefix["generated_tokens"][i, ...],
          slot,
          0,
      )
      inserted_tokens = jax.lax.dynamic_update_index_in_dim(inserted_tokens, unboxed_prefix["tokens"][i, ...], slot, 0)
      inserted_token_logp = jax.lax.dynamic_update_index_in_dim(
          inserted_token_logp, unboxed_prefix["token_logp"][i, ...], slot, 0
      )

    inserted_logits = jax.lax.with_sharding_constraint(inserted_logits, self.replicated_sharding)
    inserted_generated_tokens = jax.lax.with_sharding_constraint(inserted_generated_tokens, self.replicated_sharding)
    inserted_next_pos = jax.lax.with_sharding_constraint(inserted_next_pos, self.replicated_sharding)
    inserted_tokens = jax.lax.with_sharding_constraint(inserted_tokens, self.replicated_sharding)
    inserted_cache = jax.lax.with_sharding_constraint(inserted_cache, self.kv_cache_shardings)
    inserted_token_logp = jax.lax.with_sharding_constraint(inserted_token_logp, self.replicated_sharding)

    return {
        "logits": inserted_logits,
        "cache": inserted_cache,
        "next_pos": inserted_next_pos,
        "generated_tokens": inserted_generated_tokens,
        "tokens": inserted_tokens,
        "token_logp": inserted_token_logp,
    }


def set_engine_vars_from_base_engine(engine: MaxEngine, base_engine: MaxEngine, rng: PRNGKeyType):
  """Set sharding and cache attributes from an engine that loaded its checkpoint."""
  engine.state_mesh_annotations = base_engine.state_mesh_annotations
  engine.abstract_params = base_engine.abstract_params
  engine.kv_cache_annotations = maxtext_utils.get_kv_cache_annotations_nnx(engine.model_ar, engine.config, engine.mesh)
  engine.kv_cache_shardings = jax.tree_util.tree_map(
      lambda x: jax.sharding.NamedSharding(engine.mesh, x), engine.kv_cache_annotations
  )


def create_engine_from_config(config, devices=None):
  """Create an engine from its resolved native configuration and optional devices."""
  return MaxEngine(config, devices=devices)
