"""JAX implementation of the Multi Token Prediction https://arxiv.org/pdf/2412.19437"""

from typing import Type

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from lilax._backend.common.common_types import Config, DecoderBlockType, MODEL_MODE_TRAIN, ShardMode
from lilax._backend.layers import mhc
from lilax._backend.layers import moe
from lilax._backend.layers.linears import DenseGeneral
from lilax._backend.layers.nnx_decoders import NNXDecoderLayer
from lilax._backend.layers.normalizations import RMSNorm
from lilax._backend.models import deepseek4
from lilax._backend.models.deepseek_batchsplit.schedule import single_layer as deepseek_batchsplit_single_layer
from lilax._backend.utils import maxtext_utils
from lilax._backend.utils import sharding
from lilax._backend.utils.context_parallel import sequence_reorder
from lilax._backend.utils.globals import EPS


def _is_deepseek4(config: Config) -> bool:
  """Returns True when the model uses the DeepSeek-V4 decoder block.

  `decoder_block` may hold either the `DecoderBlockType` enum or its raw string value
  depending on how the config was constructed, so both are accepted.
  """
  return config.decoder_block in (DecoderBlockType.DEEPSEEK4, DecoderBlockType.DEEPSEEK4.value)


class mtp_losses(nnx.Intermediate):  # pylint: disable=invalid-name
  """Variable type for storing MTP loss components -> 'mtp_losses' collection."""


class mtp_acceptance(nnx.Intermediate):  # pylint: disable=invalid-name
  """Variable type for storing MTP acceptance predictions -> 'mtp_acceptance' collection."""


def _shift_left_one_cp_aware(x: jnp.ndarray, axis_name: str = "context") -> jnp.ndarray:
  """Left-shift x by 1 along axis=1, pulling the next token across CP ranks."""
  x_sharding = jax.typeof(x).sharding
  spec = x_sharding.spec.partitions
  if len(spec) > 1 and spec[1] is not None:
    gathered = jax.sharding.NamedSharding(x_sharding.mesh, jax.sharding.PartitionSpec(spec[0], None, *spec[2:]))
    return jax.reshard(_shift_left_one_cp_aware(jax.reshard(x, gathered), axis_name), x_sharding)

  local_rolled = jnp.roll(x, -1, axis=1)
  last_mask = jnp.arange(local_rolled.shape[1]) == local_rolled.shape[1] - 1
  last_mask = last_mask.reshape((1, -1) + (1,) * (local_rolled.ndim - 2))
  local_rolled = jnp.where(last_mask, 0, local_rolled)

  try:
    cp_size = jax.lax.axis_size(axis_name)
  except NameError:
    return local_rolled
  if cp_size == 1:
    return local_rolled

  cp_rank = jax.lax.axis_index(axis_name)
  first_token = jax.lax.dynamic_slice_in_dim(x, 0, 1, axis=1)
  perm = [(r, (r - 1) % cp_size) for r in range(cp_size)]
  next_first = jax.lax.ppermute(first_token, axis_name=axis_name, perm=perm)
  next_first = jnp.where(cp_rank == cp_size - 1, jnp.zeros_like(next_first), next_first)
  return jnp.where(last_mask, next_first, local_rolled)


def roll_and_mask(x: jnp.ndarray, shift: int = -1) -> jnp.ndarray:
  """Performs a leftward roll on sequence axis and masks invalid positions."""
  if shift == 0:
    return x
  if shift == -1:
    return _shift_left_one_cp_aware(x)
  return jnp.roll(x, shift, axis=1).at[:, shift:, ...].set(0)


def roll_and_mask_by_segment(x: jnp.ndarray, segment_ids: jnp.ndarray | None, shift: int = -1) -> jnp.ndarray:
  """Rolls sequence left within document boundaries defined by segment_ids."""
  assert shift == -1, f"roll_and_mask_by_segment only supports shift=-1, got {shift}"

  if segment_ids is None:
    return roll_and_mask(x, shift)

  rolled = _shift_left_one_cp_aware(x)
  seg_current = segment_ids
  seg_next = _shift_left_one_cp_aware(segment_ids)
  is_boundary = (seg_current != seg_next) | (seg_current == 0)
  mask = is_boundary[(...,) + (None,) * (x.ndim - 2)]
  return jnp.where(mask, 0, rolled)


class MultiTokenPredictionLayer(nnx.Module):
  """Multi-Token Prediction layer: normalize, concatenate, project, and transform."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      layer_number: int,
      transformer_layer_module: Type[NNXDecoderLayer],
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.layer_number = layer_number
    self.transformer_layer_module = transformer_layer_module
    self.rngs = rngs
    cfg = self.config

    self.embedding_norm = RMSNorm(
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.hidden_state_norm = RMSNorm(
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.projection_layer = DenseGeneral(
        in_features_shape=2 * cfg.emb_dim,
        out_features_shape=cfg.emb_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        use_bias=False,
        kernel_axes=("embed", None),
        rngs=rngs,
    )
    self.is_mhc = _is_deepseek4(cfg) and cfg.mhc_expansion_rate > 1
    if _is_deepseek4(cfg):
      self.transformer_layer = deepseek4.DeepSeek4DecoderLayer(
          config=cfg,
          mesh=mesh,
          model_mode=MODEL_MODE_TRAIN,
          rngs=rngs,
          layer_idx=cfg.base_num_decoder_layers + layer_number - 1,
          compress_ratio=0,
          is_hash_routing=False,
      )
    else:
      self.transformer_layer = transformer_layer_module(
          config=cfg,
          mesh=mesh,
          model_mode=MODEL_MODE_TRAIN,
          rngs=rngs,
      )

    if self.is_mhc:
      self.hc_head = mhc.DeepSeek4HyperHead(
          config=cfg,
          mesh=mesh,
          rngs=rngs,
      )

    self.final_norm = RMSNorm(
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )

  @property
  def embedding_norm(self):
    return getattr(self, f"mtp_{self.layer_number}_embedding_norm")

  @embedding_norm.setter
  def embedding_norm(self, module):
    setattr(self, f"mtp_{self.layer_number}_embedding_norm", module)

  @property
  def hidden_state_norm(self):
    return getattr(self, f"mtp_{self.layer_number}_hidden_state_norm")

  @hidden_state_norm.setter
  def hidden_state_norm(self, module):
    setattr(self, f"mtp_{self.layer_number}_hidden_state_norm", module)

  @property
  def projection_layer(self):
    return getattr(self, f"mtp_{self.layer_number}_projection")

  @projection_layer.setter
  def projection_layer(self, module):
    setattr(self, f"mtp_{self.layer_number}_projection", module)

  @property
  def transformer_layer(self):
    return getattr(self, f"mtp_{self.layer_number}_transformer_layer")

  @transformer_layer.setter
  def transformer_layer(self, module):
    setattr(self, f"mtp_{self.layer_number}_transformer_layer", module)

  @property
  def final_norm(self):
    return getattr(self, f"mtp_{self.layer_number}_final_norm")

  @final_norm.setter
  def final_norm(self, module):
    setattr(self, f"mtp_{self.layer_number}_final_norm", module)

  def __call__(
      self,
      prev_hidden_state: jnp.ndarray,
      target_token_embedding: jnp.ndarray,
      *,
      position_ids: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str = MODEL_MODE_TRAIN,
  ) -> jnp.ndarray:
    """Applies MTP combination, projection, and transformer processing."""
    target_token_embedding = sharding.maybe_shard_with_logical(
        target_token_embedding,
        ("activation_batch", "activation_length", "activation_embed"),
        self.mesh,
        self.config.shard_mode,
        sharding.get_logical_axis_rules(),
    )

    embedding_norm = self.embedding_norm(target_token_embedding)
    hidden_state_norm = self.hidden_state_norm(prev_hidden_state)

    if self.is_mhc:
      mhc_expand, _ = mhc.get_functions(self.config.mhc_expansion_rate)
      embedding_norm = mhc_expand(embedding_norm)

    if self.config.shard_mode == ShardMode.EXPLICIT:
      hidden_state_pspec = jax.sharding.NamedSharding(self.mesh, jax.typeof(hidden_state_norm).sharding.spec)
      embedding_norm = jax.reshard(embedding_norm, hidden_state_pspec)

    concatenated_features = jnp.concatenate([embedding_norm, hidden_state_norm], axis=-1)
    projected_features = self.projection_layer(concatenated_features)

    if self.config.decoder_block == DecoderBlockType.DEEPSEEK and self.config.use_batch_split_schedule:

      def extract_fn(x):
        if isinstance(x, nnx.variablelib.Variable):
          return sharding.maybe_shard_with_logical(
              x.value,
              x.sharding_names,
              self.mesh,
              shard_mode=self.config.shard_mode,
              rules=sharding.get_logical_axis_rules(),
          )
        return x

      output = deepseek_batchsplit_single_layer.batch_split_layer(
          inputs=projected_features,
          positions=position_ids,
          params=nnx.to_pure_dict(nnx.state(self, (nnx.Param, moe.MoEBiasVar)), extract_fn)[
              f"mtp_{self.layer_number}_transformer_layer"
          ],
          mesh=self.mesh,
          cfg=self.config,
      )
    else:
      output = self.transformer_layer(
          inputs=projected_features,
          decoder_segment_ids=decoder_segment_ids,
          decoder_positions=position_ids,
          deterministic=deterministic,
          model_mode=model_mode,
      )

    return output[0] if isinstance(output, tuple) else output


def _cross_entropy_with_integer_labels(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
  """Memory-efficient cross entropy loss from integer target labels."""
  log_sum_exp = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
  target_logits = jnp.take_along_axis(logits, labels[..., None], axis=-1)
  return jnp.squeeze(log_sum_exp - target_logits, axis=-1)


class MultiTokenPredictionBlock(nnx.Module):
  """Orchestrates DeepSeek Multi-Token Prediction (MTP) over K future positions."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      transformer_layer_module: Type[NNXDecoderLayer],
      decoder: nnx.Module,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.transformer_layer_module = transformer_layer_module
    self.decoder = decoder
    self.rngs = rngs if rngs is not None else nnx.Rngs(0)

    for k in range(1, config.mtp_num_layers + 1):
      layer = MultiTokenPredictionLayer(
          config=config,
          mesh=mesh,
          layer_number=k,
          transformer_layer_module=transformer_layer_module,
          rngs=rngs.fork(),
      )
      setattr(self, f"mtp_layer_{k}", layer)

  def __call__(
      self,
      shared_embedding,
      main_hidden_state,
      input_ids,
      target_ids,
      target_mask,
      *,
      position_ids,
      decoder_segment_ids,
      model_mode,
      deterministic,
      main_token_embeddings=None,
  ) -> dict:
    """Runs the MTP layers and sows their losses."""
    cfg = self.config
    cp_size = self.mesh.shape.get(cfg.context_sharding, 1) if self.mesh is not None else 1

    if target_mask is None:
      target_mask = jnp.ones_like(input_ids, dtype=jnp.int32)
    else:
      target_mask = jnp.asarray(target_mask)
      target_mask = (target_mask != 0).astype(jnp.int32)

    if target_ids is None:
      target_ids = jnp.ones_like(input_ids, dtype=jnp.int32)

    if sequence_reorder.reordered_cp_size(cfg, self.mesh) > 1:
      raise ValueError(
          "MTP does not support context_parallel_load_balance. "
          "DUAL_CHUNK_SWAP reorder breaks the ppermute-based neighbor "
          "fetch in _shift_left_one_cp_aware."
      )

    if cp_size > 1 and cfg.packing and cfg.mtp_num_layers > 1:
      raise ValueError(
          "MTP with CP and packing only supports mtp_num_layers=1. "
          "Multi-layer segment tracking alignment is not yet implemented."
      )

    mtp_hidden_state = main_hidden_state
    rolled_input_ids = input_ids
    rolled_target_ids = target_ids
    rolled_target_mask = target_mask != 0
    rolled_position_id = position_ids
    rolled_segment_ids = decoder_segment_ids

    mtp_losses_list = []
    mtp_weights_list = []
    mtp_preds_list = []
    mtp_masks_list = []

    rolled_token_embeddings = main_token_embeddings
    if main_token_embeddings is not None:
      token_zero_embedding = shared_embedding.embed_single_token(0)
      not_filled = jnp.ones(input_ids.shape, dtype=jnp.int32)

    for k in range(1, cfg.mtp_num_layers + 1):
      if main_token_embeddings is not None:
        is_filled = roll_and_mask_by_segment(not_filled, rolled_segment_ids) == 0
        rolled_token_embeddings = jnp.where(
            is_filled[..., None],
            token_zero_embedding,
            roll_and_mask_by_segment(rolled_token_embeddings, rolled_segment_ids),
        )
      rolled_input_ids = roll_and_mask_by_segment(rolled_input_ids, rolled_segment_ids)
      rolled_target_ids = roll_and_mask_by_segment(rolled_target_ids, rolled_segment_ids)
      rolled_target_mask = roll_and_mask_by_segment(rolled_target_mask, rolled_segment_ids)
      rolled_position_id = roll_and_mask_by_segment(rolled_position_id, rolled_segment_ids)
      if rolled_segment_ids is not None:
        rolled_segment_ids = roll_and_mask(rolled_segment_ids)

      embedding_kwargs = {} if main_token_embeddings is None else {"decoder_input_embeddings": rolled_token_embeddings}
      target_token_embedding = self.decoder._apply_embedding(
          shared_embedding,
          rolled_input_ids,
          rolled_position_id,
          deterministic,
          model_mode=self.decoder.model_mode,
          **embedding_kwargs,
      )

      mtp_layer = getattr(self, f"mtp_layer_{k}")
      mtp_hidden_state = mtp_layer(
          prev_hidden_state=mtp_hidden_state,
          target_token_embedding=target_token_embedding,
          position_ids=position_ids,
          decoder_segment_ids=decoder_segment_ids,
          deterministic=deterministic,
          model_mode=self.decoder.model_mode,
      )

      mtp_hidden_state_to_head = mtp_hidden_state
      if mtp_layer.is_mhc:
        mtp_hidden_state_to_head = mtp_layer.hc_head(mtp_hidden_state)

      normed_mtp_hidden_state = mtp_layer.final_norm(mtp_hidden_state_to_head)
      normed_mtp_hidden_state = sharding.maybe_shard_with_logical(
          normed_mtp_hidden_state,
          ("activation_batch", "activation_length", "activation_embed"),
          self.mesh,
          cfg.shard_mode,
          sharding.get_logical_axis_rules(),
      )

      mtp_logits = self.decoder.apply_output_head(
          shared_embedding,
          normed_mtp_hidden_state,
          deterministic,
          model_mode,
          normalize_y=False,
          reduce_mhc=False,
      )

      logits_logical_axes = (
          "activation_embed_and_logits_batch",
          "activation_length",
          "activation_vocab",
      )
      mtp_logits = sharding.maybe_shard_with_logical(
          mtp_logits,
          logits_logical_axes,
          self.mesh,
          cfg.shard_mode,
          debug_sharding=cfg.debug_sharding,
      )

      mtp_xent = _cross_entropy_with_integer_labels(mtp_logits, rolled_target_ids)
      mtp_xent = sharding.maybe_shard_with_logical(
          mtp_xent,
          ("activation_embed_and_logits_batch", "activation_length"),
          self.mesh,
          cfg.shard_mode,
          debug_sharding=cfg.debug_sharding,
      )
      mtp_xent_masked = mtp_xent * rolled_target_mask

      if model_mode == MODEL_MODE_TRAIN:
        mtp_losses_list.append(jnp.sum(mtp_xent_masked))
        mtp_weights_list.append(jnp.sum(rolled_target_mask).astype(jnp.float32))

      if cfg.mtp_eval_target_module == k:
        mtp_preds_list.append(jnp.argmax(mtp_logits, axis=-1).astype(jnp.float32))
        mtp_masks_list.append(rolled_target_mask)

    if mtp_losses_list:
      self.losses = mtp_losses(jnp.stack(mtp_losses_list))
      self.weights = mtp_losses(jnp.stack(mtp_weights_list))
    if mtp_preds_list:
      self.mtp_preds = mtp_acceptance(jnp.stack(mtp_preds_list))
      self.mtp_mask = mtp_acceptance(jnp.stack(mtp_masks_list))

    return {}


def calculate_mtp_loss(intermediate_outputs, config):
  """Calculates Multi-Token Prediction loss from intermediate outputs."""
  mtp_losses_data = maxtext_utils.get_nested_value(
      intermediate_outputs, ("mtp_losses", "mtp_block", "losses"), default=None
  )
  mtp_weights_data = maxtext_utils.get_nested_value(
      intermediate_outputs, ("mtp_losses", "mtp_block", "weights"), default=None
  )

  if mtp_losses_data is None:
    return 0.0

  if isinstance(mtp_losses_data, (tuple, list)):
    if not mtp_losses_data:
      return 0.0
    mtp_losses_array = jnp.array(mtp_losses_data)
    mtp_weights_array = jnp.array(mtp_weights_data)
  else:
    if mtp_losses_data.size == 0:
      return 0.0
    mtp_losses_array = mtp_losses_data
    mtp_weights_array = mtp_weights_data

  avg_mtp_loss = jnp.sum(mtp_losses_array) / (jnp.sum(mtp_weights_array) + EPS)
  return avg_mtp_loss * config.mtp_loss_scaling_factor


def calculate_mtp_acceptance_rate(intermediate_outputs, config):
  """Calculates MTP acceptance rate from intermediate outputs."""
  sown_data = maxtext_utils.get_nested_value(intermediate_outputs, ("mtp_acceptance", "mtp_block"), {})

  mtp_preds_raw = maxtext_utils.get_nested_value(sown_data, ("mtp_preds",), None)
  valid_mask_raw = maxtext_utils.get_nested_value(sown_data, ("mtp_mask",), None)

  mtp_preds = mtp_preds_raw[0] if isinstance(mtp_preds_raw, (tuple, list)) and mtp_preds_raw else mtp_preds_raw
  valid_mask = valid_mask_raw[0] if isinstance(valid_mask_raw, (tuple, list)) and valid_mask_raw else valid_mask_raw

  if mtp_preds is None or valid_mask is None:
    return 0.0

  mtp_preds = mtp_preds.astype(jnp.int32)
  main_model_preds = jnp.argmax(intermediate_outputs["logits"], axis=-1)

  rolled_main_preds = main_model_preds
  for _ in range(config.mtp_eval_target_module):
    rolled_main_preds = roll_and_mask(rolled_main_preds)

  correct_predictions = jnp.sum((mtp_preds == rolled_main_preds) * valid_mask)
  total_valid_tokens = jnp.sum(valid_mask)

  return (correct_predictions / (total_valid_tokens + EPS)) * 100
