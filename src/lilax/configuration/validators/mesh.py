import logging
import os

import yaml

from lilax._backend.utils.globals import MAXTEXT_PKG_DIR
from lilax.configuration.domains.execution_mesh import CustomRule

logger = logging.getLogger("lilax._backend.configs.inert")


def resolve_custom_mesh_rules(self):
  def load_rule(rule: CustomRule) -> dict:
    path = os.path.join(MAXTEXT_PKG_DIR, "configs", "custom_mesh_and_rule", f"{rule.value}.yml")
    if not os.path.exists(path):
      raise ValueError(f"Custom mesh config file not found at {path}")
    with open(path, encoding="utf-8") as mesh_file:
      return yaml.safe_load(mesh_file) or {}

  def axes(rules: list, name: str) -> tuple[str, ...]:
    for rule in rules:
      if rule and len(rule) >= 2 and rule[0] == name:
        value = rule[1]
        if isinstance(value, str):
          return (value,)
        if isinstance(value, (list, tuple)):
          return tuple(axis for axis in value if isinstance(axis, str))
        break
    return ()

  if self.custom_mesh_and_rule is not CustomRule.DEFAULT:
    mesh_config = load_rule(self.custom_mesh_and_rule)
    for field in ("mesh_axes", "logical_axis_rules", "data_sharding"):
      if field in mesh_config:
        setattr(self, field, mesh_config[field])
    cp_axes = axes(self.logical_axis_rules, "activation_length")
    if cp_axes:
      self.context_sharding = cp_axes[0]

  if not self.override_logical_axis_rules:
    expert_axes = next(
        (rule[1] for rule in self.logical_axis_rules if rule and len(rule) >= 2 and rule[0] == "exp"), ()
    )
    if isinstance(expert_axes, str):
      expert_axes = (expert_axes,)
    elif isinstance(expert_axes, (list, tuple)):
      expert_axes = tuple(axis for axis in expert_axes if isinstance(axis, str))
    else:
      expert_axes = ()
    expert_rank = 1
    for axis in expert_axes:
      ici_size = getattr(self, f"ici_{axis}_parallelism", 1)
      dcn_size = getattr(self, f"dcn_{axis}_parallelism", 1)
      if ici_size == -1 or dcn_size == -1:
        expert_rank = -1
        break
      expert_rank *= ici_size * max(dcn_size, 1)
    if expert_rank == 1:
      disabled_flags = {
          "use_random_routing": False, "use_ragged_sort": False, "moe_dropless_fallback": None,
          "use_ring_of_experts": False, "num_moe_emb_chunks": 0,
      }
      if not self.te_moe_block:
        disabled_flags["ragged_buffer_factor"] = -1.0
        disabled_flags["eval_ragged_buffer_factor"] = -1.0
      for flag_name, disabled_value in disabled_flags.items():
        current = getattr(self, flag_name)
        if current != disabled_value:
          raise ValueError(f"When EP rank is 1, {flag_name} must be {disabled_value} (was {current}).")

  self._eval_rule_was_default = self.custom_mesh_and_rule_for_eval is CustomRule.DEFAULT
  if self._eval_rule_was_default:
    self.custom_mesh_and_rule_for_eval = self.custom_mesh_and_rule
    self.logical_axis_rules_for_eval = self.logical_axis_rules
  else:
    eval_config = load_rule(self.custom_mesh_and_rule_for_eval)
    self.logical_axis_rules_for_eval = eval_config.get("logical_axis_rules", self.logical_axis_rules)
    dropped_axes = [axis for axis in eval_config.get("mesh_axes", ()) if axis not in self.mesh_axes]
    if dropped_axes:
      logger.warning(
          "custom_mesh_and_rule_for_eval=%s declares mesh axes %s that are absent from the mesh built for "
          "custom_mesh_and_rule=%s; they are ignored and any eval rule referencing them is treated as replicated.",
          self.custom_mesh_and_rule_for_eval.value, dropped_axes, self.custom_mesh_and_rule.value,
      )
    train_axes = axes(self.logical_axis_rules, "activation_q_length")
    eval_axes = axes(self.logical_axis_rules_for_eval, "activation_q_length")
    if self.context_parallel_load_balance and train_axes != eval_axes:
      logger.warning(
          "context_parallel_load_balance=True but activation_q_length maps to %s under "
          "custom_mesh_and_rule=%s and to %s under custom_mesh_and_rule_for_eval=%s. The input pipeline "
          "reorders for the train extent only, so load balancing is disabled at eval; eval attention runs "
          "unbalanced (the last CP shard does ~2*cp/(cp+1) of the average work). Everything outside the "
          "sequence-quadratic term stays balanced.",
          list(train_axes), self.custom_mesh_and_rule.value, list(eval_axes), self.custom_mesh_and_rule_for_eval.value,
      )
  return self
