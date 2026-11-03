"""Decoder activation rematerialization policies."""

import warnings

import jax

from lilax._backend.utils import max_logging


def minimal_policy(with_context=False, with_quantization=False):
  """Helper for creating minimal checkpoint policies."""
  names = [
      "query_proj", "value_proj", "key_proj", "kv_proj", "qkv_proj", "out_proj",
      "mlpwi_0", "mlpwi_1", "mlpwi", "mlpwo",
  ]
  if with_context:
    names.append("context")
  if with_quantization:
    names.append("quantization")
  return jax.checkpoint_policies.save_only_these_names(*names)


def get_remat_policy(cfg, remat_policy, make_minimal_policy):
  """Get remat policy for jax.checkpoint."""
  policy = None
  if remat_policy and remat_policy != "none":
    if remat_policy in {"minimal_with_context", "minimal_flash"}:
      if remat_policy == "minimal_flash":
        max_logging.log("WARNING: 'minimal_flash' will be deprecated soon, please use 'minimal_with_context' instead.")
      policy = make_minimal_policy(with_context=True)
    elif remat_policy == "minimal":
      policy = make_minimal_policy()
    elif remat_policy == "minimal_with_quantization":
      if cfg.scan_layers:
        warnings.warn(
            "Scan layers can introduce overhead to checkpointed values that in some configurations is slower"
            "than not checkpointing at all. If you are using scan layers, benchmark with and without quantization "
            "checkpointing in your workflow to see which is faster. Without scan layers, checkpointing quantizations is "
            "beneficial for performance."
        )
      policy = make_minimal_policy(with_context=False, with_quantization=True)
    elif remat_policy == "minimal_with_context_and_quantization":
      if cfg.scan_layers:
        warnings.warn(
            "Scan layers can introduce overhead to checkpointed values that in some configurations is slower"
            "than not checkpointing at all. If you are using scan layers, benchmark with and without quantization "
            "checkpointing in your workflow to see which is faster. Without scan layers, checkpointing quantizations is "
            "beneficial for performance."
        )
      policy = make_minimal_policy(with_context=True, with_quantization=True)
    elif remat_policy == "save_dot_with_context_except_mlp":
      policy = jax.checkpoint_policies.save_only_these_names(
          "query_proj", "value_proj", "key_proj", "kv_proj", "qkv_proj", "context", "out_proj",
      )
    elif remat_policy == "save_dot_except_mlpwi":
      policy = jax.checkpoint_policies.save_only_these_names(
          "query_proj", "value_proj", "key_proj", "kv_proj", "qkv_proj", "out_proj", "mlpwo",
      )
    elif remat_policy == "save_dot_except_mlp":
      policy = jax.checkpoint_policies.save_only_these_names(
          "query_proj", "value_proj", "key_proj", "kv_proj", "qkv_proj", "out_proj",
      )
    elif remat_policy == "save_qkv_proj":
      policy = jax.checkpoint_policies.save_only_these_names("query_proj", "value_proj", "key_proj", "kv_proj", "qkv_proj")
    elif remat_policy == "qkv_proj_offloaded":
      policy = jax.checkpoint_policies.save_and_offload_only_these_names(
          names_which_can_be_saved=[],
          names_which_can_be_offloaded=["query_proj", "value_proj", "key_proj", "kv_proj"],
          offload_src="device",
          offload_dst="pinned_host",
      )
    elif remat_policy == "minimal_offloaded":
      policy = jax.checkpoint_policies.save_and_offload_only_these_names(
          names_which_can_be_saved=[],
          names_which_can_be_offloaded=[
              "query_proj", "value_proj", "key_proj", "kv_proj", "qkv_proj", "out_proj",
              "mlpwi_0", "mlpwi_1", "mlpwi", "mlpwo",
          ],
          offload_src="device",
          offload_dst="pinned_host",
      )
    elif remat_policy == "custom":
      policy = jax.checkpoint_policies.save_and_offload_only_these_names(
          names_which_can_be_saved=cfg.tensors_on_device,
          names_which_can_be_offloaded=cfg.tensors_to_offload,
          offload_src="device",
          offload_dst="pinned_host",
      )
    elif remat_policy == "save_out_proj":
      policy = jax.checkpoint_policies.save_only_these_names("out_proj")
    else:
      assert remat_policy == "full", "Remat policy needs to be on list of remat policies"
      policy = None
  return policy
