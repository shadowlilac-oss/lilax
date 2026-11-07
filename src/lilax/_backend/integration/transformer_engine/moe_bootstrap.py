"""Process-local Transformer Engine NCCL expert-parallel bootstrap for fused MoE."""

import jax

from lilax._backend.observability import max_logging

_te_moe_bootstrap_signature = None


def get_te_moe_recv_capacity_per_rank():
  """Return the receive capacity used by the process-local TE bootstrap."""
  if _te_moe_bootstrap_signature is None:
    raise RuntimeError("TE MoE EP has not been bootstrapped yet.")
  return _te_moe_bootstrap_signature[5]


def maybe_bootstrap_te_moe(config, mesh, shaped_batch):
  """Eagerly initialize Transformer Engine NCCL EP for fused MoE."""
  if not getattr(config, "te_moe_block", False):
    return
  if jax.local_device_count() != 1:
    raise ValueError(
        "te_moe_block=True requires one local device per process. Run MaxText with "
        "`test-maxtext.sh --multiprocess` or an equivalent one-GPU-per-process launcher."
    )
  try:
    from transformer_engine.jax.ep import ep_bootstrap
    from transformer_engine.jax.moe import get_moe_recv_capacity_per_rank, record_ep_bootstrap_signature_for_moe
  except ImportError as exc:
    raise ImportError("te_moe_block=True requires TransformerEngine with JAX EP MoE support.") from exc

  ep_axis = "expert"
  fsdp_axis = "fsdp"
  ep_size = mesh.shape.get(ep_axis, 1)
  fsdp_size = mesh.shape.get(fsdp_axis, 1)
  loaded_batch_size, sequence_length = shaped_batch["inputs"].shape[:2]
  ga_steps = config.gradient_accumulation_steps
  if loaded_batch_size % ga_steps != 0:
    raise ValueError(f"TE MoE EP loaded batch size ({loaded_batch_size}) must be divisible by GA steps ({ga_steps}).")
  batch_size = max(loaded_batch_size // ga_steps, config.micro_batch_size_to_train_on)
  if config.eval_interval > 0:
    batch_size = max(batch_size, config.micro_batch_size_to_eval_on)
  if batch_size <= 0 or batch_size % (fsdp_size * ep_size) != 0:
    raise ValueError(
        f"TE MoE EP per-call batch size ({batch_size}) must be positive and divisible by FSDP * EP ({fsdp_size * ep_size})."
    )
  if config.num_experts % ep_size != 0:
    raise ValueError(f"num_experts={config.num_experts} must be divisible by EP size={ep_size}.")

  max_tokens_per_rank = (batch_size // (fsdp_size * ep_size)) * sequence_length
  worst_case_recv_capacity_per_rank = get_moe_recv_capacity_per_rank(
      num_experts=config.num_experts,
      num_experts_per_tok=config.num_experts_per_tok,
      max_tokens_per_rank=max_tokens_per_rank,
      ep_size=ep_size,
  )
  recv_capacity_factor = None if config.ragged_buffer_factor <= 0 else config.ragged_buffer_factor
  recv_capacity_per_rank = get_moe_recv_capacity_per_rank(
      num_experts=config.num_experts,
      num_experts_per_tok=config.num_experts_per_tok,
      max_tokens_per_rank=max_tokens_per_rank,
      ep_size=ep_size,
      recv_capacity_factor=recv_capacity_factor,
  )
  drop_on_overflow = recv_capacity_per_rank < worst_case_recv_capacity_per_rank
  hidden_dim = config.moe_expert_input_dim if config.moe_expert_input_dim > 0 else config.emb_dim
  signature = (
      jax.process_count(), jax.process_index(), ep_size, config.num_experts,
      max_tokens_per_rank, recv_capacity_per_rank, hidden_dim, drop_on_overflow,
  )
  global _te_moe_bootstrap_signature
  if _te_moe_bootstrap_signature == signature:
    return
  if _te_moe_bootstrap_signature is not None:
    raise ValueError(
        f"TE MoE EP was already bootstrapped with {_te_moe_bootstrap_signature}, " f"but this run needs {signature}."
    )
  with jax.set_mesh(mesh), mesh:
    max_logging.log(
        "Bootstrapping TE MoE EP: "
        f"world={jax.process_count()} rank={jax.process_index()} ep={ep_size} "
        f"num_experts={config.num_experts} max_tokens_per_rank={max_tokens_per_rank} "
        f"recv_capacity_per_rank={recv_capacity_per_rank} hidden_dim={hidden_dim} "
        f"recv_capacity_factor={recv_capacity_factor} drop_on_overflow={drop_on_overflow}"
    )
    ep_bootstrap(
        world_size=jax.process_count(),
        rank=jax.process_index(),
        num_experts=config.num_experts,
        max_tokens_per_rank=max_tokens_per_rank,
        recv_capacity_per_rank=recv_capacity_per_rank,
        hidden_dim=hidden_dim,
        max_token_dtype=config.dtype,
        drop_on_overflow=drop_on_overflow,
    )
    record_ep_bootstrap_signature_for_moe(
        num_experts=config.num_experts,
        max_tokens_per_rank=max_tokens_per_rank,
        recv_capacity_per_rank=recv_capacity_per_rank,
        hidden_dim=hidden_dim,
        ep_size=ep_size,
    )
  _te_moe_bootstrap_signature = signature
