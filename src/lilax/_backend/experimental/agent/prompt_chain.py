"""Baseline prompt-chain checkpoint conversion agent."""

import json

from lilax._backend.experimental.agent.conversion import CheckpointConversionAgent


class BaselinePromptChain(CheckpointConversionAgent):
  """Generate conversion mappings and hooks using the baseline prompt chain."""

  def run_chain(self, max_retries=3):
    maxtext_params, hf_params = self._parameters()
    maxtext_json = json.dumps(maxtext_params, indent=2)
    hf_json = json.dumps(hf_params, indent=2)
    prompts = {
        name: self._text(f"prompts/{name}.txt")
        for name in (
            "01_analysis", "03_param_mapping", "03_param_mapping_check", "04_hook_fn_prompt_chain",
            "04_pitfalls", "05_shape_mapping", "05_shape_mapping_check",
        )
    }

    print("Step 1: Analyzing model structures...")
    analysis = self._generate(prompts["01_analysis"].format(
        target_model=self.target_model, maxtext_params_json=maxtext_json, hf_params_json=hf_json,
        dsl=None, pitfalls=prompts["04_pitfalls"],
    ))

    print("Step 2: Generating and verifying parameter mapping function...")
    param_mapping = None
    feedback = ""
    for attempt in range(1, max_retries + 1):
      print(f"  Attempt {attempt}...")
      candidate = self._generate(prompts["03_param_mapping"].format(
          target_model=self.target_model, analysis=analysis, pitfalls=None, maxtext_params_json=maxtext_json,
          hf_params_json=hf_json, feedback=feedback, request_options={"timeout": 300},
      ))
      feedback = self._generate(prompts["03_param_mapping_check"].format(
          maxtext_params_json=maxtext_json, hf_params_json=hf_json, code=candidate, analysis=analysis,
      ))
      print(f"  Validator Call {attempt}...")
      print(feedback)
      if "passed" in feedback:
        param_mapping = candidate
        print("  Passed Validator...")
        break
      if attempt == max_retries:
        raise RuntimeError("Max attempts tried")
    try:
      self._write("param_mapping.py", param_mapping)
    except IOError as error:
      print(f"Error saving hook functions file: {error}")

    shape_mapping = None
    feedback = ""
    for attempt in range(1, max_retries + 1):
      print("Step 3: Generating HF weights shape mapping function...")
      candidate = self._generate(prompts["05_shape_mapping"].format(
          target_model=self.target_model, hf_params_json=hf_json, analysis=analysis, feedback=feedback, pitfalls=None,
      ))
      feedback = self._generate(prompts["05_shape_mapping_check"].format(hf_params_json=hf_json, code=candidate))
      print(f"  Validator Call {attempt}...")
      print(feedback)
      if "yes" in feedback.lower():
        shape_mapping = candidate
        print("  Passed Validator...")
        break
      if attempt == max_retries:
        raise RuntimeError("Max attempts tried")
    try:
      self._write("hf_shape.py", shape_mapping)
    except IOError as error:
      print(f"Error saving hook functions file: {error}")

    print("Step 4: Generating layerwise transformation hook functions...")
    hook = self._generate(prompts["04_hook_fn_prompt_chain"].format(
        target_model=self.target_model, analysis=analysis, param_mapping=param_mapping,
    ))
    try:
      self._write("hook_fn.py", hook)
    except IOError as error:
      print(f"Error saving hook functions file: {error}")
