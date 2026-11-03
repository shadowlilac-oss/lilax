"""Qwen3 Omni vision encoder and projector."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from flax import nnx
import jax
import jax.numpy as jnp

from lilax._backend.common.common_types import Array, AttentionType, Config, DType
from lilax._backend.layers import initializers as max_initializers
from lilax._backend.layers.attentions import Attention
from lilax._backend.layers.embeddings import Qwen3OmniMoeVisionPosEmbedInterpolate
from lilax._backend.layers.linears import DenseGeneral


class Qwen3OmniMoeVisionPatchMerger(nnx.Module):
  """Vision patch merger that spatially merges patches using an MLP.

  Attributes:
      config: Config containing model parameters
      hidden_size: Hidden dimension after spatial merging
      use_postshuffle_norm: Whether to apply normalization after spatial shuffle
      dtype: Data type for computation
      weight_dtype: Data type for weights
      kernel_init: Initializer for kernel weights
      rngs: RNG state for initialization
      ln_q: LayerNorm before MLP
      mlp_0: First MLP layer
      mlp_2: Second MLP layer
  """

  def __init__(
      self,
      config: Config,
      use_postshuffle_norm: bool = False,
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      kernel_init: max_initializers.NdInitializer = max_initializers.nd_dense_init(1.0, "fan_in", "normal"),
      rngs: nnx.Rngs = None,  # pyrefly: ignore[bad-function-definition]
  ):
    """Initializes the Qwen3Omni vision patch merger.

    Args:
        config: Config containing model parameters
        use_postshuffle_norm: Whether to apply normalization after spatial shuffle
        dtype: Data type for computation
        weight_dtype: Data type for weights
        kernel_init: Initializer for kernel weights
        rngs: RNG state for initialization
    """
    self.config = config
    self.use_postshuffle_norm = use_postshuffle_norm
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.kernel_init = kernel_init
    self.rngs = rngs

    # Calculate hidden_size after spatial merge
    spatial_merge_size = config.spatial_merge_size_for_vit
    base_hidden_size = config.hidden_size_for_vit
    out_hidden_size = config.out_hidden_size_for_vit

    self.hidden_size = base_hidden_size * (spatial_merge_size**2)

    # LayerNorm before MLP
    ln_features = self.hidden_size if use_postshuffle_norm else base_hidden_size
    self.ln_q = nnx.LayerNorm(
        num_features=ln_features,
        epsilon=config.normalization_layer_epsilon,
        dtype=dtype,
        rngs=rngs,
    )

    # MLP layers: Linear -> GELU -> Linear
    self.mlp_0 = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=self.hidden_size,
        use_bias=True,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

    self.mlp_2 = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=out_hidden_size,
        use_bias=True,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

  def __call__(self, hidden: Array) -> Array:
    """
    Args:
        hidden: Input tensor of shape (batch, seq_len, base_hidden_size) after spatial reordering

    Returns:
        Output tensor of shape (batch, seq_len//merge_size**2, out_hidden_size) - spatially merged
    """
    # Get dimensions
    spatial_merge_size = self.config.spatial_merge_size_for_vit
    base_hidden_size = self.config.hidden_size_for_vit
    tokens_per_block = spatial_merge_size**2

    batch_size = hidden.shape[0]
    seq_len = hidden.shape[1]
    num_blocks = seq_len // tokens_per_block

    hidden = hidden.reshape(batch_size, num_blocks, tokens_per_block * base_hidden_size)

    # Apply layer norm
    if self.use_postshuffle_norm:
      hidden = self.ln_q(hidden)
    else:
      hidden_unmerged = hidden.reshape(batch_size, seq_len, base_hidden_size)
      hidden_unmerged = self.ln_q(hidden_unmerged)
      hidden = hidden_unmerged.reshape(batch_size, num_blocks, tokens_per_block * base_hidden_size)

    # MLP: Linear -> GELU -> Linear
    hidden = self.mlp_0(hidden)
    hidden = jax.nn.gelu(hidden)
    hidden = self.mlp_2(hidden)

    return hidden


class Qwen3OmniMoeVisionMLP(nnx.Module):
  """Vision MLP block with GELU activation.

  Attributes:
      config: Config containing model parameters
      hidden_size: Hidden dimension size
      intermediate_size: Intermediate dimension size
      dtype: Data type for computation
      weight_dtype: Data type for weights
      kernel_init: Initializer for kernel weights
      rngs: RNG state for initialization
      linear_fc1: First linear layer
      linear_fc2: Second linear layer
  """

  def __init__(
      self,
      config: Config,
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      kernel_init: max_initializers.NdInitializer = max_initializers.nd_dense_init(1.0, "fan_in", "normal"),
      rngs: nnx.Rngs = None,  # pyrefly: ignore[bad-function-definition]
  ):
    """Initializes the Qwen3Omni vision MLP.

    Args:
        config: Config containing model parameters
        dtype: Data type for computation
        weight_dtype: Data type for weights
        kernel_init: Initializer for kernel weights
        rngs: RNG state for initialization
    """
    self.config = config
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.kernel_init = kernel_init
    self.rngs = rngs

    self.hidden_size = config.hidden_size_for_vit
    self.intermediate_size = config.intermediate_size_for_vit

    self.linear_fc1 = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=self.intermediate_size,
        use_bias=True,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

    self.linear_fc2 = DenseGeneral(
        in_features_shape=self.intermediate_size,
        out_features_shape=self.hidden_size,
        use_bias=True,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

  def __call__(self, hidden_state: Array) -> Array:
    """
    Args:
        hidden_state: Input tensor of shape (..., hidden_size) - supports packed sequences

    Returns:
        Output tensor of shape (..., hidden_size)
    """
    hidden_state = self.linear_fc1(hidden_state)
    hidden_state = jax.nn.gelu(hidden_state)
    hidden_state = self.linear_fc2(hidden_state)
    return hidden_state


class Qwen3OmniMoeVisionPatchEmbed(nnx.Module):
  """3D convolution-based patch embedding for vision inputs.

  Attributes:
      config: Config containing model parameters
      patch_size: Spatial patch size
      temporal_patch_size: Temporal patch size
      in_channels: Number of input channels
      embed_dim: Embedding dimension
      dtype: Data type for computation
      weight_dtype: Data type for weights
      rngs: RNG state for initialization
      proj: Convolution projection layer
  """

  def __init__(
      self,
      config: Config,
      # Default to float32 for numerical stability in 3D convolutions on image/video inputs
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      rngs: nnx.Rngs = None,  # pyrefly: ignore[bad-function-definition]
  ):
    """Initializes the Qwen3Omni vision patch embedding.

    Args:
        config: Config containing model parameters
        dtype: Data type for computation (defaults to float32 for numerical stability)
        weight_dtype: Data type for weights (defaults to float32 for numerical stability)
        rngs: RNG state for initialization
    """
    self.config = config
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.rngs = rngs

    self.patch_size = config.patch_size_for_vit
    self.temporal_patch_size = config.temporal_patch_size_for_vit
    self.in_channels = config.num_channels_for_vit
    self.embed_dim = config.hidden_size_for_vit

    kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)

    self.proj = nnx.Conv(
        in_features=self.in_channels,
        out_features=self.embed_dim,
        kernel_size=kernel_size,
        strides=kernel_size,
        use_bias=True,
        dtype=dtype,
        param_dtype=weight_dtype,
        rngs=rngs,
    )

  def __call__(self, hidden_states: Array, video_mask: Array | None = None) -> tuple[Array, Array | None]:
    """
    Args:
        hidden_states: Input tensor of shape (batch, in_channels, temporal*patch_size, height*patch_size, width*patch_size)
        video_mask: Optional pixel-level mask with shape
          (batch, 1, temporal*patch_size, height*patch_size, width*patch_size).
    Returns:
        Tuple of:
        - Output tensor of shape (batch, T*H*W, embed_dim) where T, H, W are the number of patches
        - Attention mask of shape (batch, T*H*W), or None when video_mask is not provided
    """
    hidden_states = jnp.transpose(hidden_states, (0, 2, 3, 4, 1))
    hidden_states = self.proj(hidden_states)
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1] * hidden_states.shape[2] * hidden_states.shape[3]
    hidden_states = hidden_states.reshape(batch_size, seq_len, self.embed_dim)

    attention_mask = None
    if video_mask is not None:
      mask_patch_elements = self.temporal_patch_size * self.patch_size * self.patch_size
      attention_mask = video_mask.reshape(video_mask.shape[0], -1, mask_patch_elements).max(axis=-1).astype(jnp.int32)

    return hidden_states, attention_mask


class Qwen3OmniMoeVisionAttention(nnx.Module):
  """Vision attention layer wrapper.

  Attributes:
      config: Config containing model parameters
      attn: Underlying attention module
  """

  def __init__(self, config: Config, *, mesh=None, rngs: nnx.Rngs = None):  # pyrefly: ignore[bad-function-definition]
    """Initializes the Qwen3Omni vision attention layer.

    Args:
        config: Config containing model parameters
        mesh: JAX device mesh for sharding
        rngs: RNG state for initialization
    """
    self.config = config
    head_dim = self.config.hidden_size_for_vit // self.config.num_attention_heads_for_vit
    # Vision uses full SA, no kv cache
    self.attn = Attention(
        config=self.config,
        num_query_heads=self.config.num_attention_heads_for_vit,
        num_kv_heads=self.config.num_attention_heads_for_vit,
        head_dim=head_dim,
        max_target_length=self.config.num_position_embeddings_for_vit,
        attention_kernel=self.config.attention_for_vit,
        inputs_q_shape=(1, 1, self.config.hidden_size_for_vit),
        inputs_kv_shape=(1, 1, self.config.hidden_size_for_vit),
        float32_qk_product=self.config.float32_qk_product,
        float32_logits=self.config.float32_logits,
        dtype=self.config.dtype_mm,
        weight_dtype=self.config.weight_dtype,
        mesh=mesh,  # pyrefly: ignore[bad-argument-type]
        dropout_rate=0.0,
        attention_type=AttentionType.FULL,
        is_nope_layer=False,
        use_bias_in_projections=True,
        is_vision=True,
        use_qk_norm=False,
        query_pre_attn_scalar=head_dim ** (-0.5),
        model_mode="train",
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: Array,
      num_frames: int,
      height: int,
      width: int,
      attention_mask: Array | None = None,
      valid_grid: tuple[int, int, int] | None = None,
      deterministic: bool = True,
  ) -> Array:
    """
    Args:
        hidden_states: Input tensor of shape (batch, T*H*W, hidden_size)
        num_frames: Number of temporal frames (static)
        height: Height in patches (static)
        width: Width in patches (static)
        attention_mask: Optional mask identifying valid tokens in the padded sequence.
        valid_grid: Optional unpadded `(frames, height, width)` grid used for vision RoPE.
        deterministic: Whether to use deterministic mode (disable dropout)

    Returns:
        Output tensor of shape (batch, T*H*W, hidden_size)
    """
    # Pass through attention with static dimensions via rope_kwargs
    rope_kwargs = {
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "token_mask": attention_mask,
        "valid_grid": valid_grid,
    }
    output, _ = self.attn(
        inputs_q=hidden_states,
        inputs_kv=hidden_states,
        deterministic=deterministic,
        decoder_segment_ids=attention_mask,
        rope_kwargs=rope_kwargs,
    )

    return output


class Qwen3OmniMoeVisionBlock(nnx.Module):
  """Vision transformer block with attention and MLP.

  Attributes:
      config: Config containing model parameters
      ln1: LayerNorm before attention
      ln2: LayerNorm before MLP
      attn: Attention module
      mlp: First MLP layer
      mlp_out: Second MLP layer
  """

  def __init__(self, config: Config, *, mesh=None, rngs: nnx.Rngs = None):  # pyrefly: ignore[bad-function-definition]
    """Initializes the Qwen3Omni vision transformer block.

    Args:
        config: Config containing model parameters
        mesh: JAX device mesh for sharding
        rngs: RNG state for initialization
    """
    self.config = config
    hs = self.config.hidden_size_for_vit
    self.ln1 = nnx.LayerNorm(num_features=hs, epsilon=config.normalization_layer_epsilon, rngs=rngs)
    self.ln2 = nnx.LayerNorm(num_features=hs, epsilon=config.normalization_layer_epsilon, rngs=rngs)
    self.attn = Qwen3OmniMoeVisionAttention(config=config, mesh=mesh, rngs=rngs)
    self.mlp = DenseGeneral(
        in_features_shape=hs,
        out_features_shape=self.config.intermediate_size_for_vit,
        use_bias=True,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )
    self.mlp_out = DenseGeneral(
        in_features_shape=self.config.intermediate_size_for_vit,
        out_features_shape=hs,
        use_bias=True,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

  def __call__(
      self,
      x: Array,
      num_frames: int,
      height: int,
      width: int,
      attention_mask: Array | None = None,
      valid_grid: tuple[int, int, int] | None = None,
  ) -> Array:
    """
    Args:
        x: Input tensor of shape (batch, T*H*W, hidden_size)
        num_frames: Number of temporal frames (static)
        height: Height in patches (static)i
        width: Width in patches (static)

    Returns:
        Output tensor of shape (batch, T*H*W, hidden_size)
    """
    x = x + self.attn(
        self.ln1(x),
        num_frames=num_frames,
        height=height,
        width=width,
        attention_mask=attention_mask,
        valid_grid=valid_grid,
    )
    y = self.ln2(x)
    y = self.mlp(y)
    y = jax.nn.gelu(y)
    y = self.mlp_out(y)
    return x + y


class Qwen3OmniMoeVisionEncoder(nnx.Module):
  """Vision encoder with patch embedding, positional embedding, and transformer blocks.

  Attributes:
      config: Config containing model parameters
      patch_embed: Patch embedding module
      pos_embed_interpolate: Position embedding interpolation module
      blocks: List of transformer blocks
      merger_list: List of patch mergers for deep supervision
      spatial_merge_size: Size of spatial merging
      deep_idx: Indices of layers to extract deep features from
  """

  def __init__(self, config: Config, *, mesh=None, rngs: nnx.Rngs = None):  # pyrefly: ignore[bad-function-definition]
    """Initializes the Qwen3Omni vision encoder.

    Args:
        config: Config containing model parameters
        mesh: JAX device mesh for sharding
        rngs: RNG state for initialization
    """
    self.config = config
    self.patch_embed = Qwen3OmniMoeVisionPatchEmbed(config=config, rngs=rngs)

    num_pos = config.num_position_embeddings_for_vit
    hs = config.hidden_size_for_vit
    self.spatial_merge_size = config.spatial_merge_size_for_vit

    self.pos_embed_interpolate = Qwen3OmniMoeVisionPosEmbedInterpolate(
        num_position_embeddings=num_pos,
        hidden_size=hs,
        spatial_merge_size=self.spatial_merge_size,
        rngs=rngs,
    )

    self.depth = config.num_hidden_layers_for_vit

    # Use setattr with string names instead of nnx.List to avoid Orbax integer key bug
    for i in range(self.depth):
      block_name = f"blocks_{i}"
      block = Qwen3OmniMoeVisionBlock(config=config, mesh=mesh, rngs=rngs)
      setattr(self, block_name, block)

    self.deep_idx = tuple(config.deepstack_visual_indexes_for_vit)
    # Use setattr with string names instead of nnx.List to avoid Orbax integer key bug
    for i, _ in enumerate(self.deep_idx):
      merger_name = f"merger_{i}"
      merger = Qwen3OmniMoeVisionPatchMerger(config=config, use_postshuffle_norm=True, rngs=rngs)
      setattr(self, merger_name, merger)

  def __call__(
      self,
      hidden_states: Array,
      video_mask: Array | None = None,
      video_grid_thw: Array | tuple[int, int, int] | None = None,
      deterministic: bool = True,
  ):
    """
    Args:
        hidden_states: Input visual tokens of shape (batch, in_channels, T*patch_size, H*patch_size, W*patch_size)
        deterministic: Whether to use deterministic mode

    Returns:
        Tuple of:
        - encoder_output: shape (batch, T*H*W, hidden_size_for_vit)
        - deep_features: List of intermediate features, each of shape (batch, T*H*W, out_hidden_size)
    """
    batch_size, _, num_frames, height, width = hidden_states.shape
    num_frames = num_frames // self.config.temporal_patch_size_for_vit
    height = height // self.config.patch_size_for_vit
    width = width // self.config.patch_size_for_vit
    attention_mask = None
    if video_mask is not None:
      mask_patch_elements = (
          self.config.temporal_patch_size_for_vit * self.config.patch_size_for_vit * self.config.patch_size_for_vit
      )
      attention_mask = video_mask.reshape(batch_size, -1, mask_patch_elements).max(axis=-1).astype(jnp.int32)
    hidden_states = hidden_states.reshape(
        -1,
        self.config.num_channels_for_vit,
        self.config.temporal_patch_size_for_vit,
        self.config.patch_size_for_vit,
        self.config.patch_size_for_vit,
    )

    x, _ = self.patch_embed(hidden_states)
    x = x.reshape(batch_size, -1, self.config.hidden_size_for_vit)
    if attention_mask is not None and video_grid_thw is None:
      raise ValueError("video_grid_thw is required when video_mask is provided.")
    pos = self.pos_embed_interpolate(
        num_frames,
        height,
        width,
        video_grid_thw=video_grid_thw,  # pyrefly: ignore[bad-argument-type]
        attention_mask=attention_mask,
    )
    x = x + pos
    valid_grid = video_grid_thw

    h_traj = []
    for i in range(self.depth):
      block_name = f"blocks_{i}"
      blk = getattr(self, block_name)
      x = blk(
          x,
          num_frames=num_frames,
          height=height,
          width=width,
          attention_mask=attention_mask,
          valid_grid=valid_grid,
      )
      h_traj.append(x)

    deep_feats = []
    for i, idx in enumerate(self.deep_idx):
      h = h_traj[idx]
      merger_name = f"merger_{i}"
      merger = getattr(self, merger_name)
      deep_feat = merger(h)
      deep_feats.append(deep_feat)

    return x, deep_feats


class Qwen3OmniMoeVisionProjector(nnx.Module):
  """Projection layer that converts vision encoder output to model embedding space.

  Attributes:
      config: Config containing model parameters
      merger: Patch merger for spatial reduction
  """

  def __init__(self, config: Config, *, rngs: nnx.Rngs = None):  # pyrefly: ignore[bad-function-definition]
    """Initializes the Qwen3Omni vision projector.

    Args:
        config: Config containing model parameters
        rngs: RNG state for initialization
    """
    self.config = config
    self.merger = Qwen3OmniMoeVisionPatchMerger(config=config, use_postshuffle_norm=False, rngs=rngs)

  def __call__(self, hidden_states: Array) -> Array:
    """
    Args:
        hidden_states: Encoder output of shape (batch, T*H*W, hidden_size_for_vit)

    Returns:
        Projected output of shape (batch, T*H*W//merge_size**2, out_hidden_size_for_vit)
    """
    output = self.merger(hidden_states)
    return output


