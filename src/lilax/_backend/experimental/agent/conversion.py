"""Generate model-specific checkpoint conversion code."""

import json
from pathlib import Path

from lilax._backend.experimental.agent.base import BaseAgent


class CheckpointConversionAgent(BaseAgent):
  """Run the conversion agent's analysis, DSL review, and generation stages."""

  def __init__(
      self, directory, target_model="gemma3", max_retries=3, context_directory=None, resource_directory=None,
      model_id="gemini-2.5-pro",
  ):
    super().__init__(model_id=model_id)
    self.directory = Path(directory)
    self.resource_directory = Path(resource_directory) if resource_directory is not None else Path(__file__).parent
    self.target_model = target_model
    self.max_retries = max_retries
    self.context_directory = Path(context_directory) if context_directory is not None else (
        self.resource_directory / "context" / target_model
    )

  def _text(self, path):
    root = self.directory if path.startswith("outputs/") else self.resource_directory
    return (root / path).read_text(encoding="utf-8")

  def _json(self, path):
    return json.loads(self._text(path))

  def _prompt(self, name, **values):
    return self._text(f"prompts/{name}.txt").format(**values)

  def _generate(self, prompt):
    return self.generate_text(prompt)

  def _write(self, name, text):
    output = self.directory / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    (output / name).write_text(text, encoding="utf-8")

  def _parameters(self):
    with (self.context_directory / "maxtext_params.json").open("rt", encoding="utf-8") as file:
      maxtext_params = json.load(file)
    with (self.context_directory / "hf_params.json").open("rt", encoding="utf-8") as file:
      hf_params = json.load(file)
    return maxtext_params, hf_params

  def analyze_model_structures(self):
    maxtext_params, hf_params = self._parameters()
    prompt = self._prompt(
        "01_analysis", target_model=self.target_model, maxtext_params_json=json.dumps(maxtext_params, indent=2),
        hf_params_json=json.dumps(hf_params, indent=2), dsl=self._text("context/dsl.txt"),
        pitfalls=self._text("prompts/04_pitfalls.txt"),
    )
    analysis = self._generate(prompt)
    self._write("analysis.txt", analysis)
    return analysis

  def verify_dsl(self):
    prompt = self._prompt(
        "04_dsl", analysis=self._text("outputs/analysis.txt"), target_model=self.target_model,
        dsl=self._text("context/dsl.txt"),
    )
    proposed_dsl = self._generate(prompt)
    self._write("proposed_dsl.txt", proposed_dsl)
    return proposed_dsl

  def plan_conversion(self):
    analysis = self._text("outputs/analysis.txt")
    dsl = self._text("context/dsl.txt")
    feedback = ""
    for attempt in range(1, self.max_retries + 1):
      plan = self._generate(self._prompt("01_plan", analysis=analysis, dsl=dsl, feedback=feedback))
      feedback = self._generate(self._prompt("01_plan_check", analysis=analysis, plan=plan))
      print(f"  Validator Call {attempt}...")
      print(feedback)
      if "yes" in feedback.lower():
        self._write("plan.json", json.dumps(plan, ensure_ascii=False, indent=4))
        return plan
    raise RuntimeError(f"Max attempts tried for {self.max_retries}")

  def _generate_mapping(self, name, generate_args, check_args, output):
    feedback = ""
    for attempt in range(1, self.max_retries + 1):
      candidate = self._generate(self._prompt(name, **generate_args, feedback=feedback))
      feedback = self._generate(self._prompt(f"{name}_check", **check_args, code=candidate))
      print(f"  Validator Call {attempt}...")
      print(feedback)
      if "passed" in feedback.lower():
        self._write(output, candidate)
        return candidate
    raise RuntimeError(f"Max attempts tried for {name}")

  def generate_param_mapping(self):
    maxtext_params, hf_params = self._parameters()
    analysis = self._text("outputs/analysis.txt")
    maxtext_json = json.dumps(maxtext_params, indent=2)
    hf_json = json.dumps(hf_params, indent=2)
    return self._generate_mapping(
        "03_param_mapping",
        {"target_model": self.target_model, "analysis": analysis, "maxtext_params_json": maxtext_json,
         "hf_params_json": hf_json, "pitfalls": self._text("prompts/04_pitfalls.txt"),
         "request_options": {"timeout": 300}},
        {"maxtext_params_json": maxtext_json, "hf_params_json": hf_json, "analysis": analysis},
        "param_mapping.py",
    )

  def generate_shape_mapping(self):
    _, hf_params = self._parameters()
    hf_json = json.dumps(hf_params, indent=2)
    return self._generate_mapping(
        "05_shape_mapping",
        {"target_model": self.target_model, "hf_params_json": hf_json,
         "analysis": self._text("outputs/analysis.txt"), "pitfalls": self._text("prompts/04_pitfalls.txt")},
        {"hf_params_json": hf_json}, "hf_shape.py",
    )

  def generate_hook_functions(self):
    prompt = self._prompt(
        "04_hook_fn_dsl", plan=self._json("outputs/plan.json"), target_model=self.target_model,
        dsl=self._text("context/dsl.txt"), pitfalls=self._text("prompts/04_pitfalls.txt"),
        param_mapping_code=self._text("outputs/param_mapping.py"),
    )
    code = self._generate(prompt)
    self._write("hook_fn.py", code)
    return code

  def analyze(self):
    self.analyze_model_structures()
    return self.verify_dsl()

  def generate(self):
    self.plan_conversion()
    self.generate_param_mapping()
    self.generate_shape_mapping()
    return self.generate_hook_functions()
