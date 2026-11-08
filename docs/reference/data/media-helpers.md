# Multimodal helper records and formatting

Import from `lilax.preprocessing.multimodal`. See [media helper guide](../../guides/data/media-helpers.md).

Configuration inputs below accept the matching model's runtime-source configuration, `model.runtime.public_config`, or resolved configuration, `model.runtime.resolve_config().config`. See [configuration objects](../execution/runtime.md#configuration-objects). The root `model.config` is not either execution configuration.

## Formatting and shapes

### reformat_prompt

```python
reformat_prompt(
	prompt, image_placeholder, model_name, num_images, video_placeholder="<|video|>",
	num_videos=0, audio_placeholder="<|audio|>", num_audios=0,
)
```

Returns text with family-specific media markers.

<details>
<summary>Parameters</summary>

- `prompt` (`str`). Prompt text.
- `image_placeholder` (`str`). Image marker.
- `model_name` (`str | object`). Family name or matching runtime-source or resolved configuration.
- `num_images` (`int`). Image count.
- `video_placeholder` (`str`, optional, defaults to `"<|video|>"`). Video marker.
- `num_videos` (`int`, optional, defaults to `0`). Video count.
- `audio_placeholder` (`str`, optional, defaults to `"<|audio|>"`). Audio marker.
- `num_audios` (`int`, optional, defaults to `0`). Audio count.

</details>

### reformat_response

```python
reformat_response(response, model_name)
```

Returns text with the family end-of-turn marker. Text-only or unrecognized paths return the supplied response unchanged.

<details>
<summary>Parameters</summary>

- `response` (`str`). Response text.
- `model_name` (`str | object`). Family name or matching runtime-source or resolved configuration.

</details>

### get_dummy_image_shape_for_init

```python
get_dummy_image_shape_for_init(model_name, batch_size=1, num_image_per_sequence=1)
```

Returns the family's initialization shape, or an empty tuple for unsupported vision families.

<details>
<summary>Parameters</summary>

- `model_name` (`str | object`). Family name or matching runtime-source or resolved configuration.
- `batch_size` (`int`, optional, defaults to `1`). Initialization batch size.
- `num_image_per_sequence` (`int`, optional, defaults to `1`). Image count per sequence.

</details>

## PreprocessorOutput

Import from `lilax.preprocessing.multimodal`. Dataclass containing processed image and audio arrays.

<details>
<summary>Parameters</summary>

- `pixel_values` (`np.ndarray | None`, optional, defaults to `None`). Processed images.
- `pixel_mask` (`np.ndarray | None`, optional, defaults to `None`). Valid image tiles.
- `aspect_ratios` (`np.ndarray | None`, optional, defaults to `None`). Image aspect-ratio metadata.
- `num_images` (`int`, optional, defaults to `0`). Image count.
- `audio_values` (`np.ndarray | None`, optional, defaults to `None`). Audio features.
- `audio_mask` (`np.ndarray | None`, optional, defaults to `None`). Valid audio segments.

</details>

### Qwen3OmniPreprocessorOutput

Import from `lilax.preprocessing.multimodal`. Dataclass extending `PreprocessorOutput` with Qwen media metadata. Inherited fields retain the defaults above.

<details>
<summary>Parameters</summary>

- `pixel_grid_thw` (`np.ndarray | None`, optional, defaults to `None`). Image patch grids.
- `num_videos` (`int`, optional, defaults to `0`). Video count.
- `video_values` (`np.ndarray | None`, optional, defaults to `None`). Processed videos.
- `video_grid_thw` (`np.ndarray | None`, optional, defaults to `None`). Video patch grids.
- `video_second_per_grid` (`np.ndarray | None`, optional, defaults to `None`). Seconds per temporal grid.
- `video_mask` (`np.ndarray | None`, optional, defaults to `None`). Video validity.
- `num_audios` (`int`, optional, defaults to `0`). Audio count.
- `audio_lengths` (`np.ndarray | None`, optional, defaults to `None`). Audio feature lengths.

</details>

## Resolved-configuration operations

The following operations require matching runtime-source or resolved execution configuration, rather than the root HF-style model configuration.
For processor-produced inputs, use [processing factories](data.md).

### preprocess_mm_data

```python
preprocess_mm_data(config)
```

Loads configured media paths. Unsupported families raise `ValueError`.
Returns the family preprocessor output record.

<details>
<summary>Parameters</summary>

- `config` (`object`). Matching runtime-source or resolved configuration with family settings and media paths.

</details>

### preprocess_image_for_training

```python
preprocess_image_for_training(image, config)
```

Returns a preprocessor record from supplied images. Unsupported image families raise `ValueError`.

<details>
<summary>Parameters</summary>

- `image` (`np.ndarray | list[np.ndarray] | str | list[str]`). Images to prepare through the selected family's preprocessor. The Qwen path accepts a single RGB `uint8` array shaped `(height, width, 3)` or a list of those arrays. Qwen video preprocessing also accepts a video path or list of video paths. Other families have their own input requirements.
- `config` (`object`). Matching runtime-source or resolved configuration with image family settings.

</details>

### get_image_offsets

```python
get_image_offsets(config, processor_output: PreprocessorOutput | None)
```

Returns added token-count information for recognized media families, or zero for an unsupported family.

<details>
<summary>Parameters</summary>

- `config` (`object`). Matching runtime-source or resolved configuration with family settings.
- `processor_output` (`PreprocessorOutput | None`). Family preprocessor record.

</details>

### prepare_text_for_image_fusion

```python
prepare_text_for_image_fusion(tokens, config, processor_output=None)
```

Returns tokens with expanded media placeholders. Unsupported multimodal models raise `ValueError`.

<details>
<summary>Parameters</summary>

- `tokens` (`list[int]`). Text token IDs.
- `config` (`object`). Matching runtime-source or resolved configuration with family settings.
- `processor_output` (`PreprocessorOutput | None`, optional, defaults to `None`). Media counts and metadata required by the selected family.

</details>

### get_dummy_audio_shape_for_init

```python
get_dummy_audio_shape_for_init(config)
```

Returns the family audio initialization shape, or an empty tuple without recognized audio configuration.

<details>
<summary>Parameters</summary>

- `config` (`object`). Matching runtime-source or resolved configuration with audio family settings.

</details>

### get_bidirectional_mask_vision

```python
get_bidirectional_mask_vision(config, decoder_input_tokens, is_video: bool = False)
```

Returns model-specific image or video token masks, or `None` when no matching mask applies.

<details>
<summary>Parameters</summary>

- `config` (`object`). Matching runtime-source or resolved configuration with family token conventions.
- `decoder_input_tokens` (`jax.Array | np.ndarray`). Decoder token IDs.
- `is_video` (`bool`, optional, defaults to `False`). Video-marker selection instead of image markers.

</details>

### get_bidirectional_mask_audio

```python
get_bidirectional_mask_audio(config, decoder_input_tokens)
```

Returns model-specific audio token masks, or `None` when no matching mask applies.

<details>
<summary>Parameters</summary>

- `config` (`object`). Matching runtime-source or resolved configuration with audio token conventions.
- `decoder_input_tokens` (`jax.Array | np.ndarray`). Decoder token IDs.

</details>

### downsample_video_mask_to_tokens

```python
downsample_video_mask_to_tokens(video_mask, config)
```

Returns a token mask using Qwen or Weaver reduction. `None` input returns `None`.
Another model family raises `ValueError`.

<details>
<summary>Parameters</summary>

- `video_mask` (`jax.Array | np.ndarray | None`). Frame-level video validity.
- `config` (`object`). Matching runtime-source or resolved configuration with video family settings.

</details>

## Multimodal position IDs

### get_rope_index

```python
get_rope_index(
	input_ids: np.ndarray, image_grid_thw: np.ndarray | None = None,
	video_grid_thw: np.ndarray | None = None, attention_mask: np.ndarray | None = None,
	use_audio_in_video: bool = False, audio_lengths: np.ndarray | None = None,
	second_per_grids: np.ndarray | None = None, spatial_merge_size: int = 2,
	position_id_per_seconds: int = 25, config=None,
) -> tuple[np.ndarray, np.ndarray]
```

Returns position IDs and position deltas. Absent configuration uses Qwen3-Omni special-token conventions.
Use the matching processor for another family's conventions.

<details>
<summary>Parameters</summary>

- `input_ids` (`np.ndarray`). Decoder token IDs.
- `image_grid_thw` (`np.ndarray | None`, optional, defaults to `None`). Image grids.
- `video_grid_thw` (`np.ndarray | None`, optional, defaults to `None`). Video grids.
- `attention_mask` (`np.ndarray | None`, optional, defaults to `None`). Valid text positions.
- `use_audio_in_video` (`bool`, optional, defaults to `False`). Audio inclusion within video positions.
- `audio_lengths` (`np.ndarray | None`, optional, defaults to `None`). Audio lengths.
- `second_per_grids` (`np.ndarray | None`, optional, defaults to `None`). Seconds per video grid.
- `spatial_merge_size` (`int`, optional, defaults to `2`). Spatial merge factor.
- `position_id_per_seconds` (`int`, optional, defaults to `25`). Temporal position scale.
- `config` (`object | None`, optional, defaults to `None`). Matching runtime-source or resolved configuration with family token conventions.

</details>
