"""Resolve explicitly selected Python execution controls over profile defaults."""


def execution_overrides(args):
  values = {}
  if args.parallelism_config is not None:
    for domain in (args.parallelism_config.mesh, args.parallelism_config.sharding):
      if domain is not None:
        values.update(domain.model_dump(exclude_unset=True))
  for domain in (args.diloco_config, args.remat_config):
    if domain is not None:
      values.update(domain.model_dump(exclude_unset=True))
  if "jax_distributed_initialization_timeout" in args.model_fields_set:
    values["jax_distributed_initialization_timeout"] = args.jax_distributed_initialization_timeout
  if "gradient_checkpointing" in args.model_fields_set:
    values["remat_policy"] = (
        args.remat_config.remat_policy if args.gradient_checkpointing and args.remat_config is not None
        else "full" if args.gradient_checkpointing else "none"
    )
  return values
