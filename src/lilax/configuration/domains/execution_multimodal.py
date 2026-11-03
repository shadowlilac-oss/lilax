from pydantic import BaseModel, Field, model_validator

from lilax.configuration.domain_types import VisionEncoderBlockType


class MultimodalGeneral(BaseModel):
  """General configuration for Multimodal models."""

  use_multimodal: bool = Field(False, description="Enable multimodal capabilities.")
  attention_for_vit: str = Field("dot_product", description="The attention algorithm to use for vision encoder.")
  vision_encoder_block: VisionEncoderBlockType = Field(
      VisionEncoderBlockType.NONE, description="The style of VisionEncoderBlock to use (e.g., 'gemma3', 'llama4')."
  )
  freeze_vision_encoder_params: bool = Field(True, description="Freeze the parameters of the vision encoder.")
  freeze_audio_encoder_params: bool = Field(True, description="Freeze the parameters of the audio encoder.")
  use_audio: bool = Field(False, description="Enable audio encoder for multimodal models.")
  image_size_for_vit: int | list[int] | None = Field(896, description="Input image size for the Vision Transformer.")
  image_path: str = Field("", description="Path to an image for decoding.")
  image_placeholder: str = Field("<|image|>", description="Placeholder string for images in text prompts.")
  posemb_type_for_vit: str = Field("learn", description="Positional embedding type for the vision encoder.")
  max_num_images_per_example: int = Field(
      -1, description="Maximum number of images per example for training with image lists. -1 means no limit."
  )
  video_path: str = Field("", description="Path to a video for decoding.")
  audio_path: str = Field("", description="Path to an audio file for decoding.")
  video_placeholder: str = Field("<|video|>", description="Placeholder string for video in text prompts.")
  audio_placeholder: str = Field("<|audio|>", description="Placeholder string for audio in text prompts.")
  use_audio_in_video: bool = Field(False, description="Extract and use audio from video files.")
  use_mrope: bool = Field(False, description="Enable Multi-dimensional RoPE for Qwen3-Omni models.")
  mrope_section: list[int] = Field([24, 20, 20], description="Dimensions for temporal, height, width in MRoPE.")
  position_id_per_seconds: int = Field(25, description="Temporal granularity for MRoPE (tokens per second).")
  video_max_grid_t: int | None = Field(None, description="Maximum Qwen3-Omni/3.5 video grid size in temporal patches, before spatial merge.")
  video_max_grid_h: int | None = Field(None, description="Maximum Qwen3-Omni/3.5 video grid size in height patches, before spatial merge.")
  video_max_grid_w: int | None = Field(None, description="Maximum Qwen3-Omni/3.5 video grid size in width patches, before spatial merge.")

  @model_validator(mode="after")
  def validate_video_max_grid(self) -> "MultimodalGeneral":
    max_grid = (self.video_max_grid_t, self.video_max_grid_h, self.video_max_grid_w)
    if any(dim is None for dim in max_grid) and not all(dim is None for dim in max_grid):
      raise ValueError("video_max_grid_t, video_max_grid_h, and video_max_grid_w must be set together.")
    return self


class VisionTower(BaseModel):
  """Configuration for the Vision Tower (Encoder) in a multimodal model."""

  hidden_size_for_vit: int = Field(1408, description="Hidden size for the Vision Transformer.")
  intermediate_size_for_vit: int = Field(5632, description="Intermediate size for the Vision Transformer's MLP.")
  num_attention_heads_for_vit: int = Field(16, description="Number of attention heads in the Vision Transformer.")
  num_channels_for_vit: int = Field(3, description="Number of input channels for the Vision Transformer (e.g., 3 for RGB).")
  tile_size_for_vit: int = Field(336, description="Tile size for the Vision Transformer.")
  patch_size_for_vit: int = Field(14, description="Patch size for the Vision Transformer.")
  conv_stride_for_vit: int = Field(14, description="Convolutional stride for the Vision Transformer's patch embedding.")
  num_hidden_layers_for_vit: int = Field(34, description="Number of hidden layers in the Vision Transformer.")
  rope_theta_for_vit: int = Field(10000, description="RoPE theta value for the Vision Transformer.")
  vision_output_dim_for_vit: int = Field(4096, description="Final output dimension of the vision-to-language projection.")
  spatial_merge_size_for_vit: int = Field(2, description="Spatial merge factor for vision patches.")
  out_hidden_size_for_vit: int = Field(512, description="Output dimension of ViT.")
  temporal_patch_size_for_vit: int = Field(2, description="Temporal patch size for video inputs.")
  num_position_embeddings_for_vit: int = Field(1024, description="Number of position embeddings for ViT.")
  deepstack_visual_indexes_for_vit: list[int] = Field([], description="Layer indices to extract deep visual features.")
  vision_output_length: int = Field(-1, description="The output length (number of soft tokens) from the vision encoder.")


class VisionProjector(BaseModel):
  """Configuration for the Vision Projector in a multimodal model."""

  projector_input_dim_for_vit: int = Field(4096, description="Input dimension for the vision projector.")
  projector_output_dim_for_vit: int = Field(4096, description="Output dimension for the vision projector.")
  pixel_shuffle_ratio_for_vit: float = Field(0.5, description="Pixel shuffle ratio for the Vision Transformer.")
  projector_dropout_for_vit: float = Field(0.0, description="Dropout rate for the vision projector.")
  vision_projector_type: str = Field(
      "default", description="Type of the vision projector to use. Supported: 'default', 'customized_vision_projector'."
  )
  vision_connector_num_layers: int = Field(2, description="Number of layers in custom vision projector.")
  vision_connector_hidden_size: int = Field(0, description="Hidden size for custom vision projector intermediate layers. 0 defaults to LLM hidden size.")
  vision_connector_activation: str = Field("gelu", description="Activation function for custom vision projector.")
  vision_connector_use_bias: bool = Field(True, description="Whether to use bias in custom vision projector.")


class AudioEncoder(BaseModel):
  """Configuration for the Audio Encoder in a multimodal model."""

  d_model_for_audio: int = Field(256, description="Model dimension for the audio encoder.")
  encoder_attention_heads_for_audio: int = Field(4, description="Number of attention heads in the audio encoder.")
  encoder_ffn_dim_for_audio: int = Field(512, description="Feed-forward network dimension for the audio encoder.")
  encoder_layers_for_audio: int = Field(2, description="Number of encoder layers for audio.")
  attention_dropout_for_audio: float = Field(0.0, description="Attention dropout rate for audio encoder.")
  activation_dropout_for_audio: float = Field(0.0, description="Activation dropout rate for audio encoder.")
  activation_function_for_audio: str = Field("gelu", description="Activation function for audio encoder.")
  num_mel_bins_for_audio: int = Field(128, description="Number of mel-frequency bins for audio input.")
  max_source_positions_for_audio: int = Field(1500, description="Maximum source positions for audio encoder.")
  scale_embedding_for_audio: bool = Field(True, description="Whether to scale embeddings in audio encoder.")
  n_window_for_audio: int = Field(50, description="Window size for audio processing.")
  n_window_infer_for_audio: int = Field(800, description="Window size for audio inference.")
  conv_chunksize_for_audio: int = Field(500, description="Chunk size for convolutional layers in audio encoder.")
  downsample_hidden_size_for_audio: int = Field(256, description="Hidden size for downsampling in audio encoder.")
  output_dim_for_audio: int = Field(512, description="Output dimension for audio encoder.")
  num_conv_layers_for_audio: int = Field(3, description="Number of convolutional layers in audio encoder.")
  max_timescale_for_audio: float = Field(10000.0, description="Maximum timescale for audio positional encoding.")
  max_sample_len_for_audio: int = Field(10000, description="Maximum sample length for audio input.")


class Multimodal(MultimodalGeneral, VisionTower, VisionProjector, AudioEncoder):
  """Configurations for multimodal"""

  pass
