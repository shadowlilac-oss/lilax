"""Typed benchmark evaluation against a model or a running serving endpoint."""

from contextlib import nullcontext

from lilax._backend.eval.runner.eval_runner import run_eval
from lilax._backend.eval.runner.harness_runner import run_harness
from lilax._backend.eval.runner.simple_evals_runner import run_simple_evals
from lilax.data_pipeline.trainer import validate_eval_dataset
from lilax.serving.configuration import EvaluationServerConfig
from lilax.serving.endpoint import ServingEndpoint
from .configuration import DatasetEvalConfig, EvalOutputConfig, HarnessEvalConfig, SimpleEvalsConfig


def evaluate(model, *, config: DatasetEvalConfig | HarnessEvalConfig | SimpleEvalsConfig,
             output: EvalOutputConfig, processing_class=None, eval_dataset=None, compute_metrics=None, token=None):
  validate_eval_dataset(eval_dataset)
  if not isinstance(config, (DatasetEvalConfig, HarnessEvalConfig, SimpleEvalsConfig)):
    raise TypeError("config must be DatasetEvalConfig, HarnessEvalConfig or SimpleEvalsConfig.")
  chat = isinstance(config, SimpleEvalsConfig) or isinstance(config, HarnessEvalConfig) and config.backend == "evalchemy"
  if isinstance(model, ServingEndpoint):
    endpoint = model
    if chat and not endpoint.config.enable_chat_api:
      raise ValueError("This evaluation requires EvaluationServerConfig(enable_chat_api=True).")
    endpoint.base_url
    context = nullcontext(endpoint)
  else:
    endpoint = ServingEndpoint(model, config=EvaluationServerConfig(
        max_model_len=model.config.max_position_embeddings, enable_chat_api=chat,
    ), processing_class=processing_class)
    context = endpoint
  processor = processing_class if processing_class is not None else endpoint.processing_class
  endpoint._token = token
  source = endpoint.model.runtime.public_config
  values = config.model_dump(exclude={"generation_config"}) | output.model_dump() | {
      "model_name": endpoint.model_name, "hf_path": source.tokenizer_path,
      "max_model_len": endpoint.config.max_model_len, "max_num_seqs": endpoint.config.max_num_seqs,
  }
  if isinstance(config, HarnessEvalConfig) and config.generation_config is not None:
    generation = config.generation_config
    values["gen_kwargs"] = {"do_sample": generation.do_sample,
                            "temperature": generation.temperature if generation.do_sample else 0.0,
                            "top_k": generation.top_k}
    if generation.max_new_tokens is not None:
      values["gen_kwargs"]["max_gen_toks"] = generation.max_new_tokens
    if generation.top_p >= 0:
      values["gen_kwargs"]["top_p"] = generation.top_p
    if generation.eos_token_id is not None:
      eos = generation.eos_token_id
      values["gen_kwargs"]["stop_token_ids"] = [eos] if isinstance(eos, int) else eos
  with context:
    if isinstance(config, DatasetEvalConfig):
      return run_eval(values, token, endpoint=endpoint, processing_class=processor,
                      eval_dataset=eval_dataset, compute_metrics=compute_metrics)
    if isinstance(config, HarnessEvalConfig):
      return run_harness(values, token, endpoint=endpoint, eval_dataset=eval_dataset, compute_metrics=compute_metrics)
    return run_simple_evals(values, token, endpoint=endpoint, eval_dataset=eval_dataset, compute_metrics=compute_metrics)
