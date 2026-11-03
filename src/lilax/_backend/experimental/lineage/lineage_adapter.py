"""Adapter between MaxText trainer state and the Lineage DSv3 model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import functools
from typing import Any, TYPE_CHECKING

from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from lilax._backend.utils import max_logging

if TYPE_CHECKING:
  from lilax._backend.experimental.lineage import dsv3_config
  from lilax._backend.experimental.lineage import dsv3_model
  from lilax._backend.experimental.lineage import quantization

_RAGGED_BUFFER_FACTOR = 1.2
PARAMS_KEY = "params"
ROUTER_BIAS_KEY = "router_bias"
MTP_ROUTER_BIAS_KEY = "mtp_router_bias"
EXPERT_PERMUTATIONS_KEY = "expert_permutations"
MTP_EXPERT_PERMUTATIONS_KEY = "mtp_expert_permutations"
MOE_BIAS_COLLECTION = "MoEBiasVar"
GA_STATS_KEY = "lineage_ga_stats"


def is_native(cfg: Any) -> bool:
  """Whether the configuration selects the Lineage-native model."""
  return bool(getattr(cfg, "use_lineage", False))


class LineageRouterBiasVar(nnx.Variable):
  """Non-trainable routed gate bias."""


RouterBiasVar = LineageRouterBiasVar


class LineageExpertPermutationVar(nnx.BatchStat):
  """Runtime per-layer expert permutation."""


ExpertPermutationVar = LineageExpertPermutationVar


def create_physical_device_mesh(ici_parallelism: Sequence[int], devices: Sequence[Any]) -> np.ndarray:
  from lilax._backend.experimental.lineage import dsv3_sharding

  return dsv3_sharding.create_physical_device_mesh(ici_parallelism, devices)


def create_device_mesh(
    ici_parallelism: Sequence[int], devices: Sequence[Any], dcn_parallelism: Sequence[int] | None = None,
    allow_split_physical_axes: bool | None = None,
) -> np.ndarray:
  from lilax._backend.experimental.lineage import dsv3_sharding

  return dsv3_sharding.create_device_mesh(
      ici_parallelism, devices, dcn_parallelism=dcn_parallelism,
      allow_split_physical_axes=allow_split_physical_axes, log_fn=max_logging.log,
  )


def get_capacity_factor(cfg: Any) -> float:
  if cfg.capacity_factor <= 0:
    raise ValueError(f"capacity_factor must be > 0, got: {cfg.capacity_factor}")
  return float(cfg.capacity_factor)


def get_quant_config(cfg: Any) -> quantization.QuantConfig | None:
  from lilax._backend.experimental.lineage import dsv3_config

  if cfg.lineage_quantization not in dsv3_config.QUANT_CONFIGS:
    raise ValueError(
        "lineage_quantization must be one of"
        f" {sorted(dsv3_config.QUANT_CONFIGS)}, got:"
        f" {cfg.lineage_quantization!r}"
    )
  return dsv3_config.QUANT_CONFIGS[cfg.lineage_quantization]


def uses_megatron_seq_aux_loss(cfg: Any) -> bool:
  return bool(
      getattr(cfg, "moe_use_megatron_seq_aux_loss", False)
      and getattr(cfg, "routed_score_func", "") == "sigmoid"
      and not getattr(cfg, "te_moe_block", False)
  )


def _router_dtype(cfg: Any) -> jax.typing.DTypeLike | None:
  return jnp.float32 if getattr(cfg, "float32_gate_logits", False) else None


def _router_bias_dtype(cfg: Any) -> jax.typing.DTypeLike | None:
  return _router_dtype(cfg) or getattr(cfg, "dtype", None)


def _stream_chunked_train_data(cfg: Any) -> bool:
  if getattr(cfg, "dataset_type", None) != "c4_mlperf":
    return False
  use_stream_chunking = getattr(cfg, "use_stream_chunking", None)
  if use_stream_chunking is None:
    use_stream_chunking = not getattr(cfg, "tokenize_train_data", True)
  return bool(use_stream_chunking)


def _mesh_scope(mesh: jax.sharding.Mesh):
  return jax.sharding.use_abstract_mesh(mesh.abstract_mesh)


def _unsupported_native_options(cfg: Any) -> list[str]:
  problems = []

  def require(ok: bool, message: str):
    if not ok:
      problems.append(message)

  get = functools.partial(getattr, cfg)
  require(get("mtp_num_layers", 0) >= 0, "mtp_num_layers must be >= 0")
  require(get("shared_experts", 1) == 1, "shared_experts must be 1")
  require(not get("quantization", None), "quantization (AQT/Qwix) is unsupported; use lineage_quantization")
  require(
      get("gradient_accumulation_steps", 1) == 1 or not get("use_tunix_gradient_accumulation", False),
      "use_tunix_gradient_accumulation is unsupported",
  )
  require(get("num_vocab_tiling", 1) <= 1, "num_vocab_tiling must be 1")
  require(not get("moe_dropless_fallback", None), "moe_dropless_fallback is unsupported")
  require(not get("log_required_ragged_buffer_factor", False), "log_required_ragged_buffer_factor is unsupported")
  require(not get("moe_log_max_load_ratio", False), "moe_log_max_load_ratio is unsupported")
  require(not get("use_indexer", False), "use_indexer is unsupported")
  require(get("training_objective", "causal_lm") == "causal_lm", "training_objective must be causal_lm")
  require(not get("use_multimodal", False), "use_multimodal is unsupported")
  require(not (get("enable_dropout", False) and get("dropout_rate", 0.0) > 0.0), "dropout is unsupported")
  require(not get("te_moe_block", False), "te_moe_block is unsupported")
  require(not get("use_qk_clip", False), "use_qk_clip is unsupported")
  require(not get("enable_diloco", False), "enable_diloco is unsupported")
  require(not getattr(get("lora", None), "enable_lora", False), "lora is unsupported")
  require(get("final_logits_soft_cap", None) is None, "final_logits_soft_cap is unsupported")
  require(
      not (get("logits_via_embedding", False) and not get("normalize_embedding_logits", True)),
      "logits_via_embedding requires normalize_embedding_logits",
  )
  require(str(get("rope_type", "yarn")).lower().endswith("yarn"), "rope_type must be yarn")
  require(get("rope_interleave", True), "rope_interleave must be true")
  require(get("routed_score_func", "sigmoid") == "sigmoid", "routed_score_func must be sigmoid")
  require(get("scan_layers", True), "scan_layers must be true")
  require(not get("parameter_memory_host_offload", False), "parameter_memory_host_offload is unsupported")
  return problems


def validate_supported(cfg: Any) -> None:
  problems = _unsupported_native_options(cfg)
  if problems:
    raise ValueError("use_lineage=True does not support this config: " + "; ".join(problems) + ".")


def dsv3_config_from_maxtext(cfg: Any, mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> dsv3_config.DSv3Config:
  """Builds and validates a Lineage configuration from a MaxText configuration."""
  from lilax._backend.experimental.lineage import dsv3_config
  from lilax._backend.experimental.lineage import dsv3_sharding

  num_decoder_layers = int(cfg.num_decoder_layers)
  num_dense_layers = int(cfg.first_num_dense_layers)
  router_dtype = _router_dtype(cfg)
  router_bias_dtype = _router_bias_dtype(cfg)
  model = dsv3_config.DSv3ModelConfig(
      vocab_size=int(cfg.vocab_size), emb_dim=int(cfg.emb_dim), num_dense_layers=num_dense_layers,
      num_sparse_layers=num_decoder_layers - num_dense_layers, num_mtp_layers=int(getattr(cfg, "mtp_num_layers", 0)),
      num_query_heads=int(cfg.num_query_heads), num_kv_heads=int(cfg.num_kv_heads),
      q_lora_rank=int(cfg.q_lora_rank), kv_lora_rank=int(cfg.kv_lora_rank),
      qk_head_dim=int(cfg.qk_nope_head_dim), rope_head_dim=int(cfg.qk_rope_head_dim), v_head_dim=int(cfg.v_head_dim),
      mscale=float(cfg.mscale), mlp_dim=int(cfg.mlp_dim), moe_mlp_dim=int(cfg.moe_mlp_dim),
      num_experts=int(cfg.num_experts), num_experts_per_tok=int(cfg.num_experts_per_tok),
      n_routing_groups=int(cfg.n_routing_groups), topk_routing_group=int(cfg.topk_routing_group),
      routed_scaling_factor=float(cfg.routed_scaling_factor), rope_theta=int(cfg.rope_max_timescale),
      max_position_embeddings=int(cfg.max_position_embeddings),
      original_max_position_embeddings=int(cfg.original_max_position_embeddings), rope_factor=int(cfg.rope_factor),
      beta_fast=int(cfg.beta_fast), beta_slow=int(cfg.beta_slow), norm_epsilon=float(cfg.normalization_layer_epsilon),
      tied_head=bool(getattr(cfg, "logits_via_embedding", False)), iota_embed=bool(getattr(cfg, "use_iota_embed", False)),
      dtype=jnp.dtype(cfg.dtype), weight_dtype=jnp.dtype(cfg.weight_dtype),
      router_dtype=None if router_dtype is None else jnp.dtype(router_dtype), router_bias_dtype=jnp.dtype(router_bias_dtype),
      head_dot_in_fp32=bool(getattr(cfg, "logits_dot_in_fp32", False)),
      cast_logits_to_fp32=bool(getattr(cfg, "cast_logits_to_fp32", True)),
  )
  sharding = dsv3_sharding.sharding_config(mesh)
  eval_sharding = None
  eval_pdb = float(getattr(cfg, "eval_per_device_batch_size", 0) or 0)
  if 0 < eval_pdb < 1:
    eval_sharding = dsv3_sharding.expert_axes_as_cp_config(sharding)
  expert_permutation = getattr(cfg, "lineage_expert_permutation", None)
  dq_reduction_steps = int(getattr(cfg, "lineage_dq_reduction_steps", 1))
  kernels = dsv3_config.DSv3KernelConfig(
      max_target_length=int(cfg.max_target_length), sa_block_q=int(cfg.sa_block_q), sa_block_kv=int(cfg.sa_block_kv),
      sa_block_kv_compute=int(cfg.sa_block_kv_compute), sa_block_q_dkv=int(cfg.sa_block_q_dkv),
      sa_block_kv_dkv=int(cfg.sa_block_kv_dkv), sa_block_kv_dkv_compute=int(cfg.sa_block_kv_dkv_compute),
      sa_q_layout=cfg.sa_q_layout, sa_k_layout=cfg.sa_k_layout, sa_v_layout=cfg.sa_v_layout,
      diag_band_skip=bool(getattr(cfg, "lineage_diag_band_skip", True)),
      dq_reduction_steps=None if dq_reduction_steps == 0 else dq_reduction_steps,
      capacity_factor=get_capacity_factor(cfg),
      ragged_buffer_factor=(float(cfg.ragged_buffer_factor)
                            if float(getattr(cfg, "ragged_buffer_factor", -1.0)) > 0.0 else _RAGGED_BUFFER_FACTOR),
      quant=get_quant_config(cfg), max_async_overlap_transform=True,
      expert_permutation=("none" if expert_permutation is None else str(expert_permutation)),
  )
  routed_bias_update_rate = float(cfg.routed_bias_update_rate) if getattr(cfg, "routed_bias", False) else 0.0
  training = dsv3_config.DSv3TrainingConfig(
      load_balance_loss_weight=float(getattr(cfg, "load_balance_loss_weight", 0.0) or 0.0),
      megatron_seq_aux_loss=uses_megatron_seq_aux_loss(cfg), routed_bias_update_rate=routed_bias_update_rate,
      z_loss_multiplier=float(getattr(cfg, "z_loss_multiplier", 0.0) or 0.0),
      mtp_loss_scaling_factor=float(getattr(cfg, "mtp_loss_scaling_factor", 0.1)),
      mtp_eval_target_module=int(getattr(cfg, "mtp_eval_target_module", 0)),
      mtp_reuse_input_embedding=bool(getattr(cfg, "mtp_reuse_input_embedding", False)),
  )
  lineage_cfg = dsv3_config.DSv3Config(
      model=model, sharding=sharding, kernels=kernels, training=training, eval_sharding=eval_sharding,
  )
  lineage_cfg.validate()
  return lineage_cfg


from_maxtext_config = dsv3_config_from_maxtext


def _path_key(entry: Any) -> str:
  if isinstance(entry, jax.tree_util.GetAttrKey):
    return entry.name
  if isinstance(entry, jax.tree_util.SequenceKey):
    return str(entry.idx)
  if isinstance(entry, jax.tree_util.DictKey):
    return str(entry.key)
  raise TypeError(f"Unsupported key path entry: {entry!r}")


def _param_tree(
    params: dsv3_model.DSv3Params, shardings: dsv3_model.DSv3Params,
) -> tuple[dict[str, Any], tuple[tuple[str, ...], ...]]:
  leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(params)
  sharding_leaves = jax.tree.leaves(shardings)
  if len(sharding_leaves) != len(leaves_with_path):
    raise ValueError(f"Got {len(sharding_leaves)} parameter shardings for" f" {len(leaves_with_path)} parameters.")
  flat = {}
  for (path, leaf), sharding in zip(leaves_with_path, sharding_leaves):
    key = tuple(_path_key(entry) for entry in path)
    flat[key] = nnx.Param(leaf, out_sharding=sharding, eager_sharding=False)
  return traverse_util.unflatten_dict(flat), tuple(flat)


def _state_var(var_type: type[nnx.Variable], value: jax.Array, sharding: jax.sharding.NamedSharding) -> nnx.Variable:
  return var_type(value, out_sharding=sharding, eager_sharding=False)


class LineageTransformer(nnx.Module):
  """NNX model with Lineage-layout parameters and mutable router state."""

  def __init__(
      self, config: Any, mesh: jax.sharding.Mesh, quant: Any = None,
      *, rngs: nnx.Rngs, model_mode: str = "train",
  ):
    if model_mode != "train":
      raise ValueError(f"use_lineage=True supports model_mode='train', got {model_mode!r}.")
    if quant is not None:
      raise ValueError("use_lineage=True does not support MaxText quantization.")
    validate_supported(config)
    from lilax._backend.experimental.lineage import dsv3_model
    from lilax._backend.experimental.lineage import ops

    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    cfg = dsv3_config_from_maxtext(config, mesh)
    if cfg.eval_sharding is not None:
      eval_sharding = cfg.eval_sharding
      token_spec = ops.physical_pspec(eval_sharding.token, eval_sharding.axis_mapping)
      override_keys = {"activation_embed_and_logits_batch", "activation_norm_length"}
      base_rules = list(getattr(config, "logical_axis_rules_for_eval", ()))
      new_rules = tuple(
          [("activation_embed_and_logits_batch", token_spec[0]), ("activation_norm_length", token_spec[1])]
          + [r for r in base_rules if r[0] not in override_keys]
      )
      if hasattr(config, "get_keys"):
        config.get_keys()["logical_axis_rules_for_eval"] = new_rules
      else:
        config.logical_axis_rules_for_eval = new_rules
    param_shardings, state_shardings = dsv3_model.dsv3_param_shardings(cfg, mesh)
    with _mesh_scope(mesh):
      params, state = dsv3_model.init_dsv3_params(rngs.params(), cfg, mesh)
    params_dict, self._param_paths = _param_tree(params, param_shardings)
    self.params = nnx.data(params_dict)
    assert state.router_bias is not None
    assert state_shardings.router_bias is not None
    self.router_bias = _state_var(RouterBiasVar, state.router_bias, state_shardings.router_bias)
    self.mtp_router_bias = nnx.data([
        _state_var(RouterBiasVar, bias, sharding)
        for bias, sharding in zip(state.mtp_router_bias, state_shardings.mtp_router_bias)
    ])
    if state.expert_permutations is not None:
      assert state_shardings.expert_permutations is not None
      self.expert_permutations = _state_var(
          ExpertPermutationVar, state.expert_permutations, state_shardings.expert_permutations,
      )
      self.mtp_expert_permutations = nnx.data([
          _state_var(ExpertPermutationVar, permutation, sharding)
          for permutation, sharding in zip(state.mtp_expert_permutations, state_shardings.mtp_expert_permutations)
      ])
    else:
      self.expert_permutations = None
      self.mtp_expert_permutations = nnx.data([])

  def lineage_config(self) -> dsv3_config.DSv3Config:
    return dsv3_config_from_maxtext(self.config, self.mesh)

  def lineage_params(self) -> dsv3_model.DSv3Params:
    from lilax._backend.experimental.lineage import dsv3_model

    param_shardings, _ = dsv3_model.dsv3_param_shardings(self.lineage_config(), self.mesh)
    treedef = jax.tree_util.tree_structure(param_shardings)
    flat = traverse_util.flatten_dict(self.params)
    if len(flat) != treedef.num_leaves or len(self._param_paths) != treedef.num_leaves:
      raise ValueError(
          f"Parameter leaf count mismatch: state has {len(flat)} leaves,"
          f" paths has {len(self._param_paths)}, treedef expects"
          f" {treedef.num_leaves}."
      )
    leaves = [flat[path].value for path in self._param_paths]
    return jax.tree_util.tree_unflatten(treedef, leaves)

  def lineage_state(self) -> dsv3_model.DSv3MutableState:
    from lilax._backend.experimental.lineage import dsv3_model

    return dsv3_model.DSv3MutableState(
        router_bias=self.router_bias.value,
        mtp_router_bias=tuple(bias.value for bias in self.mtp_router_bias),
        expert_permutations=(None if self.expert_permutations is None else self.expert_permutations.value),
        mtp_expert_permutations=tuple(permutation.value for permutation in self.mtp_expert_permutations),
    )

  def set_lineage_state(self, state: dsv3_model.DSv3MutableState) -> None:
    assert state.router_bias is not None
    self.router_bias.value = state.router_bias
    for var, bias in zip(self.mtp_router_bias, state.mtp_router_bias, strict=True):
      var.value = bias
    if self.expert_permutations is not None:
      assert state.expert_permutations is not None
      self.expert_permutations.value = state.expert_permutations
      for var, permutation in zip(self.mtp_expert_permutations, state.mtp_expert_permutations, strict=True):
        var.value = permutation

  def __call__(
      self, decoder_input_tokens: jax.Array, decoder_positions: jax.Array,
      decoder_segment_ids: jax.Array | None = None, **unused_kwargs: Any,
  ) -> jax.Array:
    from lilax._backend.experimental.lineage import dsv3_model

    cfg = self.lineage_config()
    batch = dsv3_model.DSv3Batch(
        inputs=decoder_input_tokens, targets=decoder_input_tokens, positions=decoder_positions,
        segment_ids=decoder_segment_ids, target_mask=jnp.ones_like(decoder_input_tokens),
    )
    with _mesh_scope(self.mesh):
      fwd = dsv3_model.dsv3_forward(self.lineage_params(), self.lineage_state(), batch, cfg, mesh=self.mesh)
      return dsv3_model.dsv3_logits(self.lineage_params(), fwd.hidden, cfg, mesh=self.mesh)


def accumulates_gradients(config: Any, is_train: bool = True) -> bool:
  return (
      is_train and int(getattr(config, "gradient_accumulation_steps", 1)) > 1
      and not getattr(config, "use_tunix_gradient_accumulation", False)
  )


def _ga_stats(aux: dsv3_model.DSv3LossAux, cfg: dsv3_config.DSv3Config) -> dict[str, Any]:
  from lilax._backend.experimental.lineage import dsv3_loss

  num_experts = cfg.model.num_experts
  stats: dict[str, Any] = {"z_loss_sum": aux.z_loss_sum}
  if cfg.training.routed_bias_update_rate > 0.0:
    stats["expert_counts"] = dsv3_loss.expert_counts(aux.router_aux, num_experts=num_experts)
    stats["mtp_expert_counts"] = tuple(dsv3_loss.expert_counts(a, num_experts=num_experts) for a in aux.mtp_router_aux)
  if cfg.kernels.expert_permutation != "none":
    stats["group_sizes"] = aux.router_aux.group_sizes
    stats["mtp_group_sizes"] = tuple(a.group_sizes for a in aux.mtp_router_aux)
  return stats


def loss_and_aux(
    model: LineageTransformer, config: Any, data: Mapping[str, jax.Array], *, is_train: bool,
) -> tuple[jax.Array, dict[str, Any]]:
  """Runs the model objective and exposes trainer auxiliary values."""
  from lilax._backend.experimental.lineage import dsv3_loss
  from lilax._backend.experimental.lineage import dsv3_model

  cfg = model.lineage_config()
  accumulate = accumulates_gradients(config, is_train)
  drop_segment_ids = is_train and _stream_chunked_train_data(config)
  batch = dsv3_model.DSv3Batch(
      inputs=data["inputs"], targets=data["targets"], positions=data["inputs_position"],
      segment_ids=None if drop_segment_ids else data["inputs_segmentation"], target_mask=data["targets_segmentation"],
  )
  with _mesh_scope(model.mesh):
    loss, aux = dsv3_model.dsv3_loss_and_aux(
        model.lineage_params(), model.lineage_state(), batch, cfg, mesh=model.mesh,
        train=is_train, update_state=not accumulate,
    )
  if is_train and not accumulate:
    model.set_lineage_state(aux.new_state)
  intermediate_outputs: dict[str, Any] = {}
  if not is_train and config.mtp_eval_target_module > 0:
    assert aux.mtp_preds is not None and aux.mtp_mask is not None
    intermediate_outputs["mtp_acceptance"] = {"mtp_block": {"mtp_preds": aux.mtp_preds, "mtp_mask": aux.mtp_mask}}
    intermediate_outputs["logits"] = aux.logits
  new_state = aux.new_state
  diag_bias_values = []
  diag_bias_updates = []
  if is_train and not accumulate and aux.router_bias_updates is not None:
    diag_bias_values.append(new_state.router_bias)
    diag_bias_updates.append(aux.router_bias_updates)
    diag_bias_values.extend(new_state.mtp_router_bias)
    diag_bias_updates.extend(aux.mtp_router_bias_updates)
  extra = {}
  if accumulate:
    weights = jnp.asarray(aux.total_weights, jnp.float32)
    loss = aux.xent_sum + (aux.mtp_loss + aux.moe_lb_loss) * weights
    extra[GA_STATS_KEY] = _ga_stats(aux, cfg)
  return loss, {
      "intermediate_outputs": intermediate_outputs, "xent_sum": aux.xent_sum,
      "z_loss": aux.z_loss_sum / (aux.total_weights + dsv3_loss.EPS), "total_weights": aux.total_weights,
      "moe_lb_loss": aux.moe_lb_loss, "indexer_loss": 0.0, "moe_bias_updates": None,
      "mtp_moe_bias_updates": None, "mtp_loss": aux.mtp_loss, "batch_stats": None,
      "has_moe_overflow": jnp.bool_(False), "diag_bias_values": diag_bias_values,
      "diag_bias_updates": diag_bias_updates, **extra,
  }


def finalize_accumulated_step(model: LineageTransformer, aux: dict[str, Any]) -> dict[str, Any]:
  """Applies the full-batch router state update after gradient accumulation."""
  from lilax._backend.experimental.lineage import dsv3_loss
  from lilax._backend.experimental.lineage import dsv3_model
  from lilax._backend.experimental.lineage import dsv3_types

  aux = dict(aux)
  stats = aux.pop(GA_STATS_KEY)
  cfg = model.lineage_config()
  aux["z_loss"] = stats["z_loss_sum"] / (aux["total_weights"] + dsv3_loss.EPS)
  aux["has_moe_overflow"] = jnp.bool_(False)
  bias_updates = None
  mtp_bias_updates = ()
  if "expert_counts" in stats:
    to_updates = functools.partial(
        dsv3_loss.load_balance_updates_from_counts,
        num_experts=cfg.model.num_experts, rate=cfg.training.routed_bias_update_rate,
    )
    bias_updates = to_updates(stats["expert_counts"])
    mtp_bias_updates = tuple(to_updates(c) for c in stats["mtp_expert_counts"])
  router_aux = mtp_router_aux = None
  if "group_sizes" in stats:
    as_aux = lambda g: dsv3_types.DSv3RouterAux(group_sizes=g, selected_experts=None, logits=None)
    router_aux = as_aux(stats["group_sizes"])
    mtp_router_aux = tuple(as_aux(g) for g in stats["mtp_group_sizes"])
  with _mesh_scope(model.mesh):
    new_state = dsv3_model.next_dsv3_state(
        model.lineage_state(), bias_updates=bias_updates, mtp_bias_updates=mtp_bias_updates,
        router_aux=router_aux, mtp_router_aux=mtp_router_aux, cfg=cfg, mesh=model.mesh,
    )
  model.set_lineage_state(new_state)
  if bias_updates is not None:
    aux["diag_bias_values"] = [new_state.router_bias, *new_state.mtp_router_bias]
    aux["diag_bias_updates"] = [bias_updates, *mtp_bias_updates]
  return aux


def maxtext_abstract_param_tree(config: Any, mesh: jax.sharding.Mesh) -> dict[str, Any]:
  from lilax._backend.experimental.lineage.interop import maxtext_layout

  return maxtext_layout.maxtext_abstract_param_tree(
      dsv3_config_from_maxtext(config, mesh), mesh, param_scan_axis=config.param_scan_axis,
  )


def maxtext_restore_target(config: Any, mesh: jax.sharding.Mesh) -> dict[str, Any]:
  tree = maxtext_abstract_param_tree(config, mesh)
  flat = traverse_util.flatten_dict(tree)
  params_flat = {}
  bias_flat = {}
  for path, val in flat.items():
    if path[-2:] == ("gate", "bias"):
      bias_flat[path] = val
    else:
      params_flat[path] = val
  target: dict[str, Any] = {"params": traverse_util.unflatten_dict(params_flat)}
  if bias_flat and getattr(config, "routed_bias", True):
    target[MOE_BIAS_COLLECTION] = traverse_util.unflatten_dict(bias_flat)
  return target


def _merge_restored_collections(restored: Mapping[str, Any]) -> dict[str, Any]:
  if hasattr(restored, "to_pure_dict"):
    restored = restored.to_pure_dict()
  if isinstance(restored, Mapping) and "model" in restored and len(restored) == 1:
    restored = restored["model"]
    if hasattr(restored, "to_pure_dict"):
      restored = restored.to_pure_dict()
  if isinstance(restored, Mapping) and "params" in restored:
    merged_flat = {}
    for collection in restored.values():
      if isinstance(collection, Mapping) and collection:
        merged_flat.update(traverse_util.flatten_dict(collection))
    return traverse_util.unflatten_dict(merged_flat)
  return dict(restored)


def lineage_state_pure_dict(params: dsv3_model.DSv3Params, state: dsv3_model.DSv3MutableState) -> dict[str, Any]:
  flat = {tuple(_path_key(entry) for entry in path): leaf for path, leaf in jax.tree_util.tree_leaves_with_path(params)}
  pure: dict[str, Any] = {PARAMS_KEY: traverse_util.unflatten_dict(flat)}
  if state.router_bias is not None:
    pure[ROUTER_BIAS_KEY] = state.router_bias
  if state.mtp_router_bias:
    pure[MTP_ROUTER_BIAS_KEY] = {i: bias for i, bias in enumerate(state.mtp_router_bias)}
  return pure


def restore_params(
    model_state: nnx.State, maxtext_params: Mapping[str, Any], config: Any, mesh: jax.sharding.Mesh,
) -> None:
  """Converts a restored MaxText-layout tree into Lineage model state."""
  from lilax._backend.experimental.lineage.interop import maxtext_layout

  cfg = dsv3_config_from_maxtext(config, mesh)
  tree = _merge_restored_collections(maxtext_params)
  if not getattr(config, "routed_bias", True):
    flat = traverse_util.flatten_dict(tree)
    bias_path = ("DeepSeekMoeBlock_0", "MoeBlock_0", "gate", "bias")
    flat[("decoder", "moe_layers", *bias_path)] = jnp.zeros(
        (cfg.model.num_sparse_layers, cfg.model.num_experts), dtype=cfg.model.router_bias_dtype
    )
    for depth in range(1, cfg.model.num_mtp_layers + 1):
      flat[("mtp_block", f"mtp_layer_{depth}", f"mtp_{depth}_transformer_layer", *bias_path)] = jnp.zeros(
          (cfg.model.num_experts,), dtype=cfg.model.router_bias_dtype
      )
    tree = traverse_util.unflatten_dict(flat)
  params, state = maxtext_layout.convert_maxtext_params(tree, cfg, mesh, param_scan_axis=config.param_scan_axis)
  nnx.replace_by_pure_dict(model_state, lineage_state_pure_dict(params, state))
