"""Detection of evaluation-specific kernel tile size selection."""

from flax.core.spmd import get_logical_axis_rules


def is_eval(config) -> bool:
  """Return whether evaluation-specific kernel tile sizes should be used."""
  eval_rules = getattr(config, "logical_axis_rules_for_eval", None)
  train_rules = getattr(config, "logical_axis_rules", None)
  return bool(eval_rules and eval_rules != train_rules and get_logical_axis_rules() == eval_rules)
