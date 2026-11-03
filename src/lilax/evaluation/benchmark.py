"""Python APIs for the original MMLU and multimodal benchmarks."""

import jax

from lilax._backend.benchmarks.mmlu import mmlu_eval
from lilax._backend.benchmarks.multimodal import multimodal_eval
from lilax.generation.engine import MaxEngine


def evaluate_mmlu(config, *, prompt_template=mmlu_eval.DEFAULT_PROMPT_TEMPLATE, devices=None):
  """Load a configured checkpoint and evaluate it on the MMLU test split."""
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  engine = MaxEngine(config, devices=devices)
  mmlu_eval.validate_config(engine.config)
  return mmlu_eval.evaluate(engine, engine.load_params(), engine.config, prompt_template)


def evaluate_multimodal(config, *, num_examples=-1, tmp_results_file="mm_eval_results.csv",
                        remove_tmp_results=True, ckpt_type="base", image_resize=-1, devices=None):
  """Load a configured checkpoint and evaluate it on the configured ChartQA split."""
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  engine = MaxEngine(config, devices=devices)
  multimodal_eval.validate_config(engine.config)
  return multimodal_eval.evaluate(engine, engine.load_params(), engine.config, num_examples=num_examples,
                                  tmp_results_file=tmp_results_file, remove_tmp_results=remove_tmp_results,
                                  ckpt_type=ckpt_type, image_resize=image_resize)
