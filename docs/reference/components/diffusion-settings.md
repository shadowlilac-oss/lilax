# Diffusion component settings

Import `DiffusionConfig` from `lilax.generation.m3_config`. This is a Pydantic model. `DiffusionConfig.from_dict(raw: dict[str, Any])` accepts nested `autoencoder`, `scheduler`, `text_encoder`, `transformer`, and `pipeline` mappings, plus shared [M3 execution domains](m3-execution-settings.md) and the [training domain](m3-training-settings.md). Extra keys are allowed.

These are declared fields of nested Pydantic models, not separate public imports or constructor signatures. Fields described as retained store workflow metadata. They do not activate training or video generation through `from_dict`. The public workflow constructs components and passes supported settings explicitly.

## Autoencoder fields

The `autoencoder` domain declares VAE settings.

<details>
<summary>Parameters</summary>

- `vae_type` (`str`, optional, defaults to `"autoencoder_kl"`). The VAE component type.
- `vae_weights_dtype` (`Any`, optional, defaults to `"float32"`). The VAE weight dtype.
- `vae_dtype` (`Any`, optional, defaults to `"float32"`). VAE activation dtype.
- `replicate_vae` (`bool`, optional, defaults to `False`). Retained VAE replication across devices.
- `vae_decode_chunk` (`int`, optional, defaults to `1`). Retained latent frames per decode chunk.
- `vae_encode_chunk` (`int`, optional, defaults to `4`). Retained frames per encode chunk.
- `vae_spatial` (`int`, optional, defaults to `-1`). Retained VAE mesh spatial partition count.
- `vae_logical_axis_rules` (`tuple`, optional, defaults to `()`). Mapping of VAE logical axes to mesh axes.

</details>

## Scheduler fields

The `scheduler` domain declares diffusion schedule settings.

<details>
<summary>Parameters</summary>

- `scheduler_type` (`str`, optional, defaults to `"unipc_multistep"`). The scheduler.
- `scheduler_dtype` (`Any`, optional, defaults to `"float32"`). Computation dtype.
- `use_experimental_scheduler` (`bool`, optional, defaults to `True`). Retained experimental scheduler flag.
- `flow_shift` (`float`, optional, defaults to `5.0`). Flow schedule shift.
- `guidance_scale` (`float`, optional, defaults to `5.0`). Classifier-free prompt guidance strength.
- `guidance_rescale` (`float`, optional, defaults to `0.0`). Retained guided-noise rescaling.
- `num_inference_steps` (`int`, optional, defaults to `50`). Denoising steps.
- `diffusion_scheduler_config` (`dict`, optional, defaults to `{}`). Scheduler-specific configuration. Each instance receives a fresh value.
- `timestep_bias` (`dict`, optional, defaults to `{}`). Retained sampling-bias configuration. Each instance receives a fresh value.

</details>

## Text encoder fields

The `text_encoder` domain declares text conditioning settings.

<details>
<summary>Parameters</summary>

- `text_encoder_type` (`str`, optional, defaults to `"umt5"`). Text encoder component type.
- `text_encoder_dtype` (`Any`, optional, defaults to `"float32"`). Computation dtype.
- `max_sequence_length` (`int`, optional, defaults to `512`). Limit on token sequence length.
- `text_encoder_logical_axis_rules` (`tuple`, optional, defaults to `()`). Mapping of text-encoder logical axes to mesh axes.

</details>

## Diffusion transformer fields

The `transformer` domain extends [TransformerConfig](m3-execution-settings.md#transformerconfig) with diffusion fields.

<details>
<summary>Parameters</summary>

- `patch_size` (`tuple[int, int, int]`, optional, defaults to `(1, 2, 2)`). Temporal, height, and width latent patch dimensions.
- `in_channels` (`int`, optional, defaults to `16`). Input latent channels.
- `out_channels` (`int`, optional, defaults to `16`). Output latent channels.
- `text_dim` (`int`, optional, defaults to `4096`). Text conditioning width.
- `freq_dim` (`int`, optional, defaults to `256`). Timestep frequency embedding width.
- `qk_norm` (`str`, optional, defaults to `"rms_norm"`). Query and key normalization.
- `cross_attn_norm` (`bool`, optional, defaults to `True`). Flag for cross-attention input normalization.
- `flash_block_sizes` (`Optional[Any]`, optional, defaults to `None`). Tile sizes for compatible flash attention.
- `flash_min_seq_length` (`int`, optional, defaults to `4096`). The flash-attention sequence threshold.
- `ulysses_shards` (`int`, optional, defaults to `-1`). Ulysses partitions.
- `ulysses_attention_chunks` (`int`, optional, defaults to `1`). Ulysses chunks.
- `remat_policy` (`str`, optional, defaults to `"NONE"`). Compatible activation recomputation.
- `names_which_can_be_saved` (`list[str]`, optional, defaults to `[]`). Names of activations eligible for rematerialization retention. Each instance receives a fresh value.
- `names_which_can_be_offloaded` (`list[str]`, optional, defaults to `[]`). Names of activations eligible for host offload. Each instance receives a fresh value.
- `enable_jax_named_scopes` (`bool`, optional, defaults to `False`). Flag for JAX profiling scopes.

</details>

## Pipeline fields

The `pipeline` domain stores workflow settings. These fields do not provide a pretrained Lilax pipeline factory.
The [Wan component guide](../../guides/components/wan-components.md) shows available component construction.

<details>
<summary>Parameters</summary>

- `pretrained_model_name_or_path` (`str`, optional, defaults to `""`). A component repository or local path.
- `model_name` (`str`, optional, defaults to `""`). Model configuration.
- `model_type` (`str`, optional, defaults to `"T2V"`). The diffusion task, such as text-to-video.
- `prompt` (`str`, optional, defaults to `""`). Positive text conditioning.
- `prompt_2` (`str`, optional, defaults to `""`). Retained second prompt.
- `negative_prompt` (`str`, optional, defaults to `""`). Negative text conditioning.
- `prompt_file` (`str`, optional, defaults to `""`). Retained prompt file.
- `do_classifier_free_guidance` (`bool`, optional, defaults to `True`). Retained conditional and unconditional prediction combination.
- `height` (`int`, optional, defaults to `720`). Output height in pixels.
- `width` (`int`, optional, defaults to `1280`). Output width in pixels.
- `num_frames` (`int`, optional, defaults to `81`). Output frame count.
- `fps` (`int`, optional, defaults to `16`). Output frame rate.
- `seed` (`int`, optional, defaults to `0`). The random seed.
- `use_cfg_cache` (`bool`, optional, defaults to `False`). Retained classifier-free guidance caching.
- `use_batched_text_encoder` (`bool`, optional, defaults to `False`). Retained batched text encoding.
- `use_kv_cache` (`bool`, optional, defaults to `False`). Retained attention KV caching.
- `use_magcache` (`bool`, optional, defaults to `False`). Retained MagCache reuse.
- `magcache_thresh` (`float`, optional, defaults to `0.12`). Retained reuse threshold.
- `magcache_K` (`int`, optional, defaults to `2`). MagCache K. Public component construction does not consume it.
- `retention_ratio` (`float`, optional, defaults to `0.2`). Retained cache retention ratio.
- `mag_ratios_base` (`list`, optional, defaults to `[]`). Retained per-step MagCache ratios. Each instance receives a fresh value.
- `enable_lora` (`bool`, optional, defaults to `False`). Retained LoRA enablement.
- `lora_config` (`dict`, optional, defaults to `{}`). Retained adapter configuration. Each instance receives a fresh value.
- `controlnet_model_name_or_path` (`str`, optional, defaults to `""`). Retained ControlNet repository or path.
- `controlnet_from_pt` (`bool`, optional, defaults to `True`). Retained PyTorch ControlNet loading.
- `controlnet_conditioning_scale` (`float`, optional, defaults to `0.5`). Retained conditioning scale.
- `controlnet_image` (`str`, optional, defaults to `""`). Retained conditioning image location.
- `quantization` (`str`, optional, defaults to `""`). Quantization format.
- `use_qwix_quantization` (`bool`, optional, defaults to `False`). Qwix in compatible components.
- `qwix_module_path` (`str`, optional, defaults to `""`). Regular expression for module selection.
- `weight_quantization_calibration_method` (`str`, optional, defaults to `"absmax"`). Weight calibration.
- `act_quantization_calibration_method` (`str`, optional, defaults to `"absmax"`). Activation calibration.
- `bwd_quantization_calibration_method` (`str`, optional, defaults to `"absmax"`). Backward-gradient calibration.

</details>
