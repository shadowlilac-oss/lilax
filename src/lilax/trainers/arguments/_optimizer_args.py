"""Map optimizer and learning-rate schedule arguments to configuration overrides."""


def optimizer_overrides(args):
  return (
      ("lr_schedule_type", args.lr_schedule_type), ("warmup_steps_fraction", args.warmup_steps_fraction),
      ("learning_rate_schedule_steps", args.learning_rate_schedule_steps),
      ("learning_rate_final_fraction", args.learning_rate_final_fraction),
      ("wsd_decay_steps_fraction", args.wsd_decay_steps_fraction), ("wsd_decay_style", args.wsd_decay_style),
      ("gradient_clipping_threshold", args.gradient_clipping_threshold),
      ("opt_type", args.opt_type),
      ("adam_b1", args.adam_b1), ("adam_b2", args.adam_b2), ("adam_eps", args.adam_eps),
      ("adam_eps_root", args.adam_eps_root), ("adam_weight_decay", args.adam_weight_decay),
      ("adamw_mask", args.adamw_mask), ("mu_dtype", args.mu_dtype), ("muon_type", args.muon_type),
      ("muon_beta", args.muon_beta), ("muon_weight_decay", args.muon_weight_decay),
      ("muon_consistent_rms", args.muon_consistent_rms), ("muon_include_routers", args.muon_include_routers),
      ("muon_use_all_to_all", args.muon_use_all_to_all), ("skip_step_on_spikes", args.skip_step_on_spikes),
      ("skip_step_interval", args.skip_step_interval), ("skip_step_scaling_factor", args.skip_step_scaling_factor),
      ("trainable_parameters_mask", args.trainable_parameters_mask),
  )
