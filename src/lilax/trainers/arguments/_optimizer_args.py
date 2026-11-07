"""Map optimizer and scheduler configuration to their backend consumers."""


def optimizer_overrides(args):
  values = {
      "opt_type": args.optim, "adam_b1": args.adam_beta1, "adam_b2": args.adam_beta2,
      "adam_eps": args.adam_epsilon, "adam_weight_decay": args.weight_decay,
      "gradient_clipping_threshold": args.max_grad_norm, "lr_schedule_type": args.lr_scheduler_type,
      "warmup_steps_fraction": args.warmup_ratio,
  }
  if args.lr_scheduler_config is not None:
    scheduler = args.lr_scheduler_config
    values.update({
        "learning_rate_schedule_steps": scheduler.schedule_steps, "learning_rate_final_fraction": scheduler.final_fraction,
        "wsd_decay_steps_fraction": scheduler.wsd_decay_steps_fraction, "wsd_decay_style": scheduler.wsd_decay_style,
    })
  if args.optimizer_config is not None:
    optimizer = args.optimizer_config
    values.update({
        "adam_eps_root": optimizer.adam_eps_root, "adamw_mask": optimizer.adamw_mask or [],
        "mu_dtype": optimizer.mu_dtype, "trainable_parameters_mask": optimizer.trainable_parameters or [],
    })
    if optimizer.muon is not None:
      muon = optimizer.muon
      values.update({
          "muon_type": muon.implementation, "muon_beta": muon.beta, "muon_weight_decay": muon.weight_decay,
          "muon_consistent_rms": muon.consistent_rms, "muon_include_routers": muon.include_routers,
          "muon_use_all_to_all": muon.use_all_to_all,
      })
    if optimizer.spike_skip is not None:
      spike = optimizer.spike_skip
      values.update(skip_step_on_spikes=spike.enabled, skip_step_interval=spike.interval,
                    skip_step_scaling_factor=spike.scaling_factor)
  return values
