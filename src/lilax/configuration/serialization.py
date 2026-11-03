"""Serialize inert configuration and its explicit-key provenance."""

from lilax.configuration.schema import MaxTextConfig, RLConfig


def serialize_config(config: MaxTextConfig | RLConfig) -> dict:
  if isinstance(config, MaxTextConfig):
    serialized_config = config.model_copy(update=config.get_source_values())
  else:
    serialized_config = config
  payload = {
      "config_type": "rl" if isinstance(config, RLConfig) else "maxtext",
      "values": serialized_config.model_dump(mode="json"),
      "explicit_keys": sorted(config._explicit_keys),
  }
  if isinstance(config, MaxTextConfig):
    payload["eval_rule_was_default"] = config._eval_rule_was_default
  else:
    payload["profiler_synthesized"] = sorted(config._profiler_synthesized)
  return payload


def deserialize_config(payload: dict) -> MaxTextConfig | RLConfig:
  if payload["config_type"] == "maxtext":
    values = dict(payload["values"])
    if payload.get("eval_rule_was_default"):
      values["custom_mesh_and_rule_for_eval"] = ""
    config = MaxTextConfig._from_resolved(values, payload["explicit_keys"])
    if "eval_rule_was_default" in payload:
      config._eval_rule_was_default = payload["eval_rule_was_default"]
    return config
  config_class = {"maxtext": MaxTextConfig, "rl": RLConfig}[payload["config_type"]]
  config = config_class.model_validate(payload["values"], context={"resolved_values": True})
  config._explicit_keys = frozenset(payload["explicit_keys"])
  config._profiler_synthesized = frozenset(payload.get("profiler_synthesized", ()))
  object.__setattr__(config, "__pydantic_fields_set__", set(payload["explicit_keys"]))
  return config
