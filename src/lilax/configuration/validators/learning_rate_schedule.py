from lilax.configuration.domain_types import LearningRateScheduleType


def validate_learning_rate_schedule_options(self):
  if self.lr_schedule_type == LearningRateScheduleType.WSD:
    total_fraction = self.warmup_steps_fraction + self.wsd_decay_steps_fraction
    if total_fraction > 1.0:
      raise ValueError(
          f"Invalid WSD schedule: warmup_steps_fraction ({self.warmup_steps_fraction}) + "
          f"wsd_decay_steps_fraction ({self.wsd_decay_steps_fraction}) must not exceed 1.0. "
          f"Current sum: {total_fraction}"
      )
