# Vision and audio model settings

Return to the [configuration reference](configuration.md).

Import these configurations from `lilax.configuration`. Keep the settings loaded with the matching model family.
See [model configuration](../../guides/models/configuration.md) for loading and updating a focused domain.

These keyword fields show constructor defaults. A loaded checkpoint can supply different values.
Changing architecture dimensions or routing does not convert existing weights.

## VisionConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Configures vision encoder selection, dimensions, and freezing.

<details>
<summary>Parameters</summary>

- `use_multimodal` (`bool`, optional, defaults to `False`). Vision inputs for the matching family.
- `attention_for_vit` (`str`, optional, defaults to `"dot_product"`). Vision attention computation.
- `vision_encoder_block` (`VisionEncoderBlockType`, optional, defaults to `VisionEncoderBlockType.NONE`). Encoder architecture.
- `freeze_vision_encoder_params` (`bool`, optional, defaults to `True`). Encoder parameter exclusion from updates.
- `image_size_for_vit` (`int | list[int] | None`, optional, defaults to `896`). Input image dimensions.
- `posemb_type_for_vit` (`str`, optional, defaults to `"learn"`). Vision positional embedding type.
- `hidden_size_for_vit` (`int`, optional, defaults to `1408`). Encoder hidden width.
- `intermediate_size_for_vit` (`int`, optional, defaults to `5632`). Encoder MLP intermediate width.
- `num_attention_heads_for_vit` (`int`, optional, defaults to `16`). Attention head count.
- `num_channels_for_vit` (`int`, optional, defaults to `3`). Input channels, such as three for RGB.
- `tile_size_for_vit` (`int`, optional, defaults to `336`). Tile size.
- `patch_size_for_vit` (`int`, optional, defaults to `14`). Patch size.
- `conv_stride_for_vit` (`int`, optional, defaults to `14`). Patch embedding convolution stride.
- `num_hidden_layers_for_vit` (`int`, optional, defaults to `34`). Encoder layer count.
- `rope_theta_for_vit` (`int`, optional, defaults to `10000`). Vision RoPE theta.
- `vision_output_dim_for_vit` (`int`, optional, defaults to `4096`). Vision-to-language projection output width.
- `spatial_merge_size_for_vit` (`int`, optional, defaults to `2`). Spatial patch merge factor.
- `out_hidden_size_for_vit` (`int`, optional, defaults to `512`). ViT output width.
- `temporal_patch_size_for_vit` (`int`, optional, defaults to `2`). Video temporal patch size.
- `num_position_embeddings_for_vit` (`int`, optional, defaults to `1024`). Position embedding count.
- `deepstack_visual_indexes_for_vit` (`list[int]`, optional, defaults to `[]`). Layers providing deep visual features.
- `vision_output_length` (`int`, optional, defaults to `-1`). Encoder output soft-token count.

</details>

## VisionProjectorConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Configures the vision-to-text projection.

<details>
<summary>Parameters</summary>

- `projector_input_dim_for_vit` (`int`, optional, defaults to `4096`). Input width.
- `projector_output_dim_for_vit` (`int`, optional, defaults to `4096`). Output width.
- `pixel_shuffle_ratio_for_vit` (`float`, optional, defaults to `0.5`). Pixel shuffle ratio.
- `projector_dropout_for_vit` (`float`, optional, defaults to `0.0`). Projector dropout rate.
- `vision_projector_type` (`str`, optional, defaults to `"default"`). Projector type, either `default` or `customized_vision_projector`.
- `vision_connector_num_layers` (`int`, optional, defaults to `2`). Custom projector layer count.
- `vision_connector_hidden_size` (`int`, optional, defaults to `0`). Custom projector intermediate width. Zero uses the language model hidden width.
- `vision_connector_activation` (`str`, optional, defaults to `"gelu"`). Custom projector activation.
- `vision_connector_use_bias` (`bool`, optional, defaults to `True`). Custom projector bias.

</details>

## AudioConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Configures the audio encoder and freezing.

<details>
<summary>Parameters</summary>

- `use_audio` (`bool`, optional, defaults to `False`). Audio encoding.
- `freeze_audio_encoder_params` (`bool`, optional, defaults to `True`). Encoder parameter exclusion from updates.
- `d_model_for_audio` (`int`, optional, defaults to `256`). Encoder model width.
- `encoder_attention_heads_for_audio` (`int`, optional, defaults to `4`). Attention head count.
- `encoder_ffn_dim_for_audio` (`int`, optional, defaults to `512`). Feed-forward width.
- `encoder_layers_for_audio` (`int`, optional, defaults to `2`). Encoder layer count.
- `attention_dropout_for_audio` (`float`, optional, defaults to `0.0`). Attention dropout.
- `activation_dropout_for_audio` (`float`, optional, defaults to `0.0`). Activation dropout.
- `activation_function_for_audio` (`str`, optional, defaults to `"gelu"`). Activation function.
- `num_mel_bins_for_audio` (`int`, optional, defaults to `128`). Input mel-frequency bin count.
- `max_source_positions_for_audio` (`int`, optional, defaults to `1500`). Maximum encoder positions.
- `scale_embedding_for_audio` (`bool`, optional, defaults to `True`). Encoder embedding scaling.
- `n_window_for_audio` (`int`, optional, defaults to `50`). Audio processing window.
- `n_window_infer_for_audio` (`int`, optional, defaults to `800`). Inference window.
- `conv_chunksize_for_audio` (`int`, optional, defaults to `500`). Encoder convolution chunk size.
- `downsample_hidden_size_for_audio` (`int`, optional, defaults to `256`). Downsampling hidden width.
- `output_dim_for_audio` (`int`, optional, defaults to `512`). Encoder output width.
- `num_conv_layers_for_audio` (`int`, optional, defaults to `3`). Convolution layer count.
- `max_timescale_for_audio` (`float`, optional, defaults to `10000.0`). Position encoding maximum timescale.
- `max_sample_len_for_audio` (`int`, optional, defaults to `10000`). Maximum input sample length.

</details>

