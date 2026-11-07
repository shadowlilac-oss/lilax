"""Native benchmark algorithms applied to a loaded model and supplied data."""

import jax
from jetstream.engine import tokenizer_pb2
from jetstream.engine.token_utils import HuggingFaceTokenizer

from lilax._backend.benchmarks.mmlu import mmlu_eval
from lilax._backend.benchmarks.multimodal import multimodal_eval
from lilax._backend.configs.pyconfig import HyperParameters
from lilax._backend.experimental.omni_pipeline import eval_sft_omni
from lilax.generation._engine import MaxEngine
from lilax._processing.preferences import processing_overrides
from lilax.data_pipeline.trainer import validate_eval_dataset


class _BenchmarkTokenizer(HuggingFaceTokenizer):
  def __init__(self, processing_class):
    self.tokenizer = getattr(processing_class, "tokenizer", processing_class)
    self.metadata = tokenizer_pb2.TokenizerParameters(
        use_chat_template=self.tokenizer.init_kwargs.get("lilax_processing", {}).get("use_chat_template", False),
    )


def _benchmark_model(model, processing_class, eval_dataset, *, multimodal=False):
  validate_eval_dataset(eval_dataset)
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  model.materialize(runtime=model.runtime)
  options = processing_overrides(processing_class)
  if multimodal:
    options["hf_path"] = "HuggingFaceM4/ChartQA"
  config = HyperParameters(model.construction_config._pydantic_config.model_copy(update=options))
  engine = MaxEngine(config, devices=model.runtime.devices)
  return engine, engine.load_params(model=model.native_model), config, _BenchmarkTokenizer(processing_class)


def evaluate_mmlu(model, *, processing_class, eval_dataset, prompt_template=mmlu_eval.DEFAULT_PROMPT_TEMPLATE):
  engine, params, config, tokenizer = _benchmark_model(model, processing_class, eval_dataset)
  return mmlu_eval.evaluate(engine, params, config, prompt_template, eval_dataset=eval_dataset, tokenizer=tokenizer)


def evaluate_multimodal(model, *, processing_class, eval_dataset, num_examples=-1,
                        tmp_results_file="mm_eval_results.csv", remove_tmp_results=True, image_resize=-1):
  engine, params, config, tokenizer = _benchmark_model(model, processing_class, eval_dataset, multimodal=True)
  return multimodal_eval.evaluate(
      engine, params, config, eval_dataset=eval_dataset, tokenizer=tokenizer, num_examples=num_examples,
      tmp_results_file=tmp_results_file, remove_tmp_results=remove_tmp_results, image_resize=image_resize,
      ckpt_type="sft" if getattr(model.config, "finetuning_task", None) == "sft" else "base",
  )


def evaluate_omni(model, *, processing_class, eval_dataset, num_examples=-1,
                  tmp_results_file="omni_eval_results.csv", remove_tmp_results=True, image_resize=-1):
  engine, params, config, tokenizer = _benchmark_model(model, processing_class, eval_dataset, multimodal=True)
  return eval_sft_omni.evaluate(
      engine, params, config, eval_dataset=eval_dataset, tokenizer=tokenizer, num_examples=num_examples,
      tmp_results_file=tmp_results_file, remove_tmp_results=remove_tmp_results, image_resize=image_resize,
  )
