def validate_distillation_options(self):
  if self.distill_beta > 0.0 and not self.scan_layers:
    raise ValueError("a value of self.distill_beta > 0.0 requires self.scan_layers = True")
  if self.distill_alpha_end is not None and not 0.0 <= self.distill_alpha_end <= 1.0:
    raise ValueError(f"distill_alpha_end must be in [0, 1], got {self.distill_alpha_end}")
  if self.distill_temperature_end is not None and self.distill_temperature_end <= 0.0:
    raise ValueError(f"distill_temperature_end must be > 0, got {self.distill_temperature_end}")
  if self.distill_beta_end is not None and self.distill_beta_end < 0.0:
    raise ValueError(f"distill_beta_end must be >= 0, got {self.distill_beta_end}")
  if self.distill_beta == 0.0 and self.distill_beta_end is not None and self.distill_beta_end > 0.0:
    raise ValueError(
        f"distill_beta=0.0 but distill_beta_end={self.distill_beta_end}. "
        "Feature extraction is disabled when distill_beta starts at 0.0. "
        "Set distill_beta to a small positive value (e.g., 1e-6) to enable feature extraction."
    )
  for param_name, schedule, end_value in (
      ("distill_alpha", self.distill_alpha_schedule, self.distill_alpha_end),
      ("distill_temperature", self.distill_temperature_schedule, self.distill_temperature_end),
      ("distill_beta", self.distill_beta_schedule, self.distill_beta_end),
  ):
    if schedule != "constant" and end_value is None:
      raise ValueError(
          f"{param_name}_schedule is '{schedule}' but {param_name}_end is None. "
          f"Set {param_name}_end to a target value or use schedule='constant'."
      )
