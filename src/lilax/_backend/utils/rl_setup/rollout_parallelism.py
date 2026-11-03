"""Resolve vLLM rollout parallelism sizes for the sampler devices."""


def get_rollout_kwargs_for_parallelism(sampler_config, num_sampler_devices):
  """Get rollout kwargs for vLLM rollout when using data parallelism."""
  dp = sampler_config.rollout_data_parallelism
  tp = sampler_config.rollout_tensor_parallelism
  ep = sampler_config.rollout_expert_parallelism
  num_auto = sum(1 for value in [tp, dp, ep] if value == -1)
  if num_auto > 1:
    raise ValueError(
        "At most one of rollout_tensor_parallelism, rollout_data_parallelism, "
        "rollout_expert_parallelism can be -1 (auto-derived).\n"
        f"Currently resolved values:\n"
        f"  - rollout_tensor_parallelism = {tp}\n"
        f"  - rollout_data_parallelism = {dp}\n"
        f"  - rollout_expert_parallelism = {ep}\n\n"
        "To fix this, you must explicitly define at least two of these parameters in your command line arguments.\n"
        "For example, try adding 'rollout_tensor_parallelism=4' to your command."
    )
  if dp == -1:
    if num_sampler_devices % (tp * ep) != 0:
      raise ValueError(
          f"num_sampler_devices({num_sampler_devices}) must be divisible by "
          f"rollout_tensor_parallelism({tp}) * rollout_expert_parallelism({ep}) "
          "when rollout_data_parallelism is -1."
      )
    dp = num_sampler_devices // tp // ep
  elif tp == -1:
    if num_sampler_devices % (dp * ep) != 0:
      raise ValueError(
          f"num_sampler_devices({num_sampler_devices}) must be divisible by "
          f"rollout_data_parallelism({dp}) * rollout_expert_parallelism({ep}) "
          "when rollout_tensor_parallelism is -1."
      )
    tp = num_sampler_devices // dp // ep
  elif ep == -1:
    if num_sampler_devices % (tp * dp) != 0:
      raise ValueError(
          f"num_sampler_devices({num_sampler_devices}) must be divisible by "
          f"rollout_tensor_parallelism({tp}) * rollout_data_parallelism({dp}) "
          "when rollout_expert_parallelism is -1."
      )
    ep = num_sampler_devices // tp // dp
  elif tp * dp * ep != num_sampler_devices:
    raise ValueError(
        f"rollout_tensor_parallelism({tp}) * "
        f"rollout_data_parallelism({dp}) * "
        f"rollout_expert_parallelism({ep}) "
        f"!= len(sampler_devices)({num_sampler_devices})"
    )
  rollout_kwargs = {}
  rollout_kwargs["tensor_parallel_size"] = tp
  rollout_kwargs["data_parallel_size"] = dp
  rollout_kwargs["expert_parallel_size"] = ep
  return rollout_kwargs
