"""Python entry point for MaxText evaluation runners."""

from lilax._backend.eval.runner.eval_runner import (
    _build_results_path, _derive_from_maxtext_config, _load_config, _merge_config, run_eval,
)
from lilax._backend.eval.runner.harness_runner import run_harness
from lilax._backend.eval.runner.simple_evals_runner import run_simple_evals


def evaluate(cfg: dict, hf_token: str | None = None, *, runner: str = "evalchemy") -> dict:
  """Run an evaluation using a native runner configuration and optional Hugging Face token."""
  if runner == "eval":
    cfg = dict(cfg)
    config_path = cfg.pop("config", None)
    base_config_path = cfg.pop("base_config", None)
    base_cfg = _load_config(config_path) if config_path else {}
    if base_config_path:
      for key, value in _derive_from_maxtext_config(base_config_path).items():
        base_cfg.setdefault(key, value)
    cfg = _merge_config(base_cfg, cfg)
    if "max_tokens" not in cfg and "max_tokens_default" in cfg:
      cfg["max_tokens"] = cfg["max_tokens_default"]
    if "results_path" not in cfg:
      cfg["results_path"] = _build_results_path(cfg)
    return run_eval(cfg, hf_token=hf_token)
  if runner == "simple_evals":
    cfg = {"tasks": ["mmlu"], **cfg}
    if "results_path" not in cfg:
      cfg["results_path"] = _build_results_path(cfg)
    return run_simple_evals(cfg, hf_token=hf_token)
  if runner == "lm_eval":
    cfg = {"tasks": ["mmlu"], **cfg}
    if "results_path" not in cfg:
      cfg["results_path"] = _build_results_path(cfg)
    return run_harness(cfg, hf_token=hf_token)
  if runner == "evalchemy":
    cfg = {"backend": "evalchemy", "tasks": ["mmlu"], **cfg}
    if "results_path" not in cfg:
      cfg["results_path"] = _build_results_path(cfg)
    return run_harness(cfg, hf_token=hf_token)
  raise ValueError(f"Unknown evaluation runner: {runner}. Expected eval, lm_eval, evalchemy, or simple_evals.")
