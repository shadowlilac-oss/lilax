"""Embedding Layers."""

import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding

from flax import nnx

from lilax._backend.common.common_types import ShardMode, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN, Array, Config, DType, get_weight_dtype
from lilax._backend.layers.initializers import Initializer, default_embed_init
from lilax._backend.utils import max_logging
from lilax._backend.device_memory import memory_kinds
from lilax._backend.distributed.sharding import logical_to_mesh_axes, truncate_out_sharding


def _maybe_move_embedding_to_device(embedding_table: Array, config: Config) -> Array:
  """Moves embedding table to device if parameter offloading is enabled."""
  if config.parameter_memory_host_offload:
    max_logging.log("embeddings.py: Moving embedding parameter to device")
    return jax.device_put(embedding_table, memory_kinds.device_space())
  return embedding_table


class Embed(nnx.Module):
  """A parameterized function from integers [0, n) to d-dimensional vectors."""

  def __init__(
      self,
      num_embeddings: int,
      num_features: int,
      config: Config,
      mesh: Mesh,
      cast_input_dtype: None | DType = None,
      dtype: DType = jnp.float32,
      attend_dtype: None | DType = None,
      embedding_init: Initializer = default_embed_init,
      *,
      # Not used in Embed but passed in by nnx.bridge.to_linen.
      # TODO: Remove when bridge no longer needed
      rngs: nnx.Rngs,
  ):
    """Initializes the Embed module.

    Args:
      num_embeddings: The number of embeddings.
      num_features: The number of feature dimensions for each embedding.
      config: The model configuration.
      cast_input_dtype: The dtype to cast the input to, if any.
      dtype: The dtype of the embedding vectors.
      attend_dtype: The dtype for the `attend` method.
      embedding_init: The initializer for the embedding matrix.
      rngs: The random number generators for initialization.
    """
    self.num_embeddings = num_embeddings
    self.num_features = num_features
    self.config = config
    self.mesh = mesh
    self.cast_input_dtype = cast_input_dtype
    self.dtype = dtype
    self.attend_dtype = attend_dtype
    embed_weight_dtype = get_weight_dtype(self.config, "token_embedder")
    self.embedding = nnx.Param(
        embedding_init(
            rngs.params(),
            (self.num_embeddings, self.num_features),
            embed_weight_dtype,
        ),
        sharding=("vocab", "embed_vocab"),
    )

  def __call__(self, inputs: Array, model_mode: str = MODEL_MODE_TRAIN) -> Array:
    """Embeds the inputs along the last dimension.

    Args:
      inputs: input data, all dimensions are considered batch dimensions.

    Returns:
      Output which is embedded input data.  The output shape follows the input,
      with an additional `num_features` dimension appended.
    """
    cfg = self.config
    if self.cast_input_dtype:
      inputs = inputs.astype(self.cast_input_dtype)
    if not jnp.issubdtype(inputs.dtype, jnp.integer):
      raise ValueError("Input type must be an integer or unsigned integer.")

    embedding = jnp.asarray(
        _maybe_move_embedding_to_device(self.embedding.get_value(), self.config),
        self.dtype,
    )

    output_axis_names = (
        (
            "activation_embed_and_logits_batch",
            "prefill_activation_length",
            "activation_embed",
        )
        if model_mode == MODEL_MODE_PREFILL
        else (
            "activation_embed_and_logits_batch",
            "activation_length",
            "activation_embed",
        )
    )
    out_pspec = logical_to_mesh_axes(output_axis_names, self.mesh, rules=getattr(self.config, "logical_axis_rules", None))

    out_sharding = NamedSharding(self.mesh, out_pspec) if self.config.shard_mode == ShardMode.EXPLICIT else None
    if out_sharding is not None:
      out_sharding = truncate_out_sharding(out_sharding, inputs.ndim + 1)

    if cfg.use_iota_embed:
      iota = lax.iota(jnp.int32, self.num_embeddings)
      one_hot = jnp.array(inputs[..., jnp.newaxis] == iota, dtype=self.dtype)
      output = jnp.dot(one_hot, embedding, out_sharding=out_sharding)
    else:
      output = embedding.at[inputs].get(out_sharding=out_sharding)

    return output

  def embed_single_token(self, token_id: int) -> Array:
    """Returns the [num_features] embedding of one token id, bitwise equal to the row `__call__` returns."""
    embedding = jnp.asarray(
        _maybe_move_embedding_to_device(self.embedding.get_value(), self.config),
        self.dtype,
    )
    row = embedding[token_id]
    if self.config.use_iota_embed:
      row = jnp.where(row == 0, jnp.zeros_like(row), row)
    return row

  def attend(self, query: Array, out_sharding: NamedSharding | None = None) -> Array:
    """Attend over the embedding using a query array.

    Args:
      query: array with last dimension equal the feature depth `num_features` of the
        embedding.
      out_sharding: NamedSharding object indicating how the output gets sharded

    Returns:
      An array with final dim `num_embeddings` corresponding to the batched
      inner-product of the array of query vectors against each embedding.
      Commonly used for weight-sharing between embeddings and logit transform
      in NLP models.
    """
    embedding = self.embedding.get_value()
    attend_dtype = self.attend_dtype if self.attend_dtype is not None else self.dtype
    return attend_on_embedding(query, embedding, attend_dtype, self.config, out_sharding)


def attend_on_embedding(
    query: Array,
    embedding_table: Array,
    attend_dtype: DType,
    config: Config,
    out_sharding: NamedSharding | None = None,
) -> Array:
  """Attend over an embedding table using a query array.

  TODO: Remove this method when Embed bridge to Linen is no longer needed

  Args:
    query: An array with a last dimension equal to the feature depth of the embedding.
    embedding_table: The embedding table to attend over.
    attend_dtype: The data type for the attention computation.
    config: The model configuration, used to check for parameter offloading.
    out_sharding: NamedSharding object indicating the output sharding

  Returns:
    An array with a final dimension equal to `num_embeddings`, corresponding to the
    batched inner-product of the query vectors against each embedding.
  """
  # out_sharding must be None under auto shard_mode
  if config.shard_mode != ShardMode.EXPLICIT:
    out_sharding = None
  if out_sharding is not None:
    out_sharding = truncate_out_sharding(out_sharding, query.ndim)
  embedding_table = _maybe_move_embedding_to_device(embedding_table, config)
  # Contract over the table's feature axis instead of materializing `table.T`.
  # Under `shard_mode: explicit` the transposed table is a distinct typed value,
  # so XLA cannot fold it into the dot's dimension numbers: the (large) table
  # shard is cast to bf16 once for the input lookup in `Embed.__call__` and a
  # second time here, and the tied head's weight gradient comes out flipped.
  # Expressing the transpose as dimension numbers keeps both consumers on one
  # cast and leaves the gradient in the table's own axis order. Under `auto`
  # this is a no-op -- XLA already folded the `.T` away.
  return jnp.einsum(
      "...e,ve->...v",
      query,
      jnp.asarray(embedding_table, jnp.bfloat16),
      preferred_element_type=attend_dtype,
      out_sharding=out_sharding,
  )
