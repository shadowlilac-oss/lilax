"""Next-token and candidate scoring with a native vLLM engine."""

import json
import os
from typing import Any, Sequence

import jax
import numpy as np
import transformers
from vllm import LLM
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams

from lilax._backend.inference import vllm_decode
import lilax._backend.integration.vllm.maxtext_vllm_adapter as adapter
from lilax._backend.observability import max_logging

DEFAULT_TOP_K = 20
TokenIds = Sequence[int]
PromptLike = str | TokenIds


def setup_scoring_environment() -> None:
  """Configure the in-process TPU engine as in the native scoring entry point."""
  os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
  os.environ["SKIP_JAX_PRECOMPILE"] = "1"
  os.environ["NEW_MODEL_DESIGN"] = "1"
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )


class VllmScorer:
  """Score next tokens using an existing vLLM LLM."""

  def __init__(self, llm: LLM, default_top_k: int = DEFAULT_TOP_K):
    self.llm = llm
    self.tokenizer = llm.get_tokenizer()
    if default_top_k < 1:
      raise ValueError("default_top_k must be at least 1")
    self.default_top_k = default_top_k
    max_logprobs = llm.llm_engine.model_config.max_logprobs
    if max_logprobs != -1 and default_top_k > max_logprobs:
      max_logging.log(
          f"VllmScorer: default_top_k={default_top_k} exceeds engine max_logprobs={max_logprobs}; "
          "clamping. Pass max_logprobs=<k> to create_vllm_llm to raise the limit."
      )
      self.default_top_k = max_logprobs

  def to_token_ids(self, prompt: PromptLike, add_special_tokens: bool = False) -> list[int]:
    if isinstance(prompt, str):
      return list(self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens))
    return [int(token) for token in prompt]

  def _tokens_prompts(self, prompts: Sequence[PromptLike], add_special_tokens: bool) -> list[TokensPrompt]:
    return [TokensPrompt(prompt_token_ids=self.to_token_ids(prompt, add_special_tokens)) for prompt in prompts]

  def next_token_logprobs(
      self, prompts: Sequence[PromptLike], top_k: int | None = None, add_special_tokens: bool = False
  ) -> list[dict[int, float]]:
    k = self.default_top_k if top_k is None else top_k
    max_logprobs = self.llm.llm_engine.model_config.max_logprobs
    if k < 1 or (max_logprobs != -1 and k > max_logprobs):
      raise ValueError(f"top_k must be positive and within the engine max_logprobs limit ({max_logprobs})")
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=k, detokenize=False)
    outputs = self.llm.generate(self._tokens_prompts(prompts, add_special_tokens), sampling_params, use_tqdm=False)
    results = []
    for output in outputs:
      step_logprobs = output.outputs[0].logprobs
      if not step_logprobs:
        raise RuntimeError("vLLM did not return logprobs; make sure the engine was created with max_logprobs>=k.")
      results.append({int(token_id): float(logprob.logprob) for token_id, logprob in step_logprobs[0].items()})
    return results

  def greedy_next_token(self, prompts: Sequence[PromptLike], add_special_tokens: bool = False) -> list[int]:
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0, detokenize=False)
    outputs = self.llm.generate(self._tokens_prompts(prompts, add_special_tokens), sampling_params, use_tqdm=False)
    return [int(output.outputs[0].token_ids[0]) for output in outputs]

  def exact_candidate_logprobs(
      self, prompts: Sequence[PromptLike], candidate_token_ids: Sequence[TokenIds], add_special_tokens: bool = False
  ) -> list[np.ndarray]:
    if len(prompts) != len(candidate_token_ids):
      raise ValueError("prompts and candidate_token_ids must have the same length")
    for candidates in candidate_token_ids:
      for candidate in candidates:
        _validate_token_id(self.tokenizer, candidate)
    requests: list[TokensPrompt] = []
    owners: list[tuple[int, int]] = []
    for index, (prompt, candidates) in enumerate(zip(prompts, candidate_token_ids)):
      ids = self.to_token_ids(prompt, add_special_tokens)
      for candidate_index, candidate in enumerate(candidates):
        requests.append(TokensPrompt(prompt_token_ids=ids + [int(candidate)]))
        owners.append((index, candidate_index))
    if not requests:
      return [np.zeros((0,), dtype=np.float64) for _ in prompts]
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1, detokenize=False)
    outputs = self.llm.generate(requests, sampling_params, use_tqdm=False)
    results = [np.full((len(candidates),), np.nan, dtype=np.float64) for candidates in candidate_token_ids]
    for (index, candidate_index), output, request in zip(owners, outputs, requests):
      candidate = request["prompt_token_ids"][-1]
      prompt_logprobs = output.prompt_logprobs
      if not prompt_logprobs or prompt_logprobs[-1] is None or candidate not in prompt_logprobs[-1]:
        raise RuntimeError(f"vLLM did not return a prompt logprob for candidate token {candidate} (prompt {index}).")
      results[index][candidate_index] = float(prompt_logprobs[-1][candidate].logprob)
    return results

  def candidate_logprobs(
      self,
      prompts: Sequence[PromptLike],
      candidate_token_ids: Sequence[TokenIds],
      method: str = "auto",
      top_k: int | None = None,
      add_special_tokens: bool = False,
      *,
      top_k_logprobs: Sequence[dict[int, float]] | None = None,
  ) -> list[np.ndarray]:
    if method not in ("auto", "top_k", "exact"):
      raise ValueError(f"Unknown method {method!r}; expected 'auto', 'top_k' or 'exact'.")
    if len(prompts) != len(candidate_token_ids):
      raise ValueError("prompts and candidate_token_ids must have the same length")
    for candidates in candidate_token_ids:
      for candidate in candidates:
        _validate_token_id(self.tokenizer, candidate)
    if method == "exact":
      return self.exact_candidate_logprobs(prompts, candidate_token_ids, add_special_tokens)
    if top_k_logprobs is not None and len(top_k_logprobs) != len(prompts):
      raise ValueError("prompts and top_k_logprobs must have the same length")
    top = (
        self.next_token_logprobs(prompts, top_k=top_k, add_special_tokens=add_special_tokens)
        if top_k_logprobs is None else top_k_logprobs
    )
    results = []
    missing = []
    for index, (distribution, candidates) in enumerate(zip(top, candidate_token_ids)):
      scores = np.array([distribution.get(int(candidate), np.nan) for candidate in candidates], dtype=np.float64)
      if np.isnan(scores).any():
        missing.append(index)
      results.append(scores)
    if missing:
      if method == "top_k":
        raise RuntimeError(
            f"Candidates outside top-{top_k or self.default_top_k} for prompts {missing}; "
            "increase top_k / max_logprobs or use method='exact' or 'auto'."
        )
      max_logging.log(f"VllmScorer: falling back to exact prompt_logprobs scoring for prompts {missing}.")
      exact = self.exact_candidate_logprobs(
          [prompts[index] for index in missing], [candidate_token_ids[index] for index in missing], add_special_tokens
      )
      for index, scores in zip(missing, exact):
        results[index] = scores
    return results

  def candidate_probabilities(
      self, prompts: Sequence[PromptLike], candidate_token_ids: Sequence[TokenIds], **kwargs: Any
  ) -> list[np.ndarray]:
    return [_candidate_probabilities(scores) for scores in self.candidate_logprobs(prompts, candidate_token_ids, **kwargs)]


def _parse_candidates(raw: Any) -> list[Any]:
  if raw is None or raw == "":
    return []
  if isinstance(raw, str):
    raw = [candidate for candidate in raw.split(",") if candidate != ""]
  return list(raw)


def _validate_token_id(tokenizer: Any, token_id: int) -> None:
  if isinstance(token_id, (bool, np.bool_)) or not isinstance(token_id, (int, np.integer)):
    raise ValueError("Candidate token IDs must be integers")
  if not 0 <= token_id < len(tokenizer):
    raise ValueError(f"Candidate token ID {token_id} is outside the tokenizer vocabulary")


def candidate_token_id(
    tokenizer: Any, prompt_text: str, prompt_ids: Sequence[int], candidate: Any, add_special_tokens: bool = False
) -> int:
  if add_special_tokens and prompt_ids and prompt_ids[-1] == tokenizer.eos_token_id:
    raise ValueError("The tokenizer appended EOS to the prompt; use add_special_tokens=false for next-token scoring.")
  if isinstance(candidate, bool) or not isinstance(candidate, (int, str)):
    raise ValueError("Candidates must be text strings or integer token IDs")
  if isinstance(candidate, int):
    _validate_token_id(tokenizer, candidate)
    return candidate
  combined = tokenizer.encode(prompt_text + candidate, add_special_tokens=add_special_tokens)
  suffix = combined[len(prompt_ids) :]
  if list(combined[: len(prompt_ids)]) != list(prompt_ids) or len(suffix) != 1:
    raise ValueError(
        f"Candidate {candidate!r} is not a single token at the prompt boundary (got {len(suffix)} tokens: {suffix})."
    )
  return int(suffix[0])


def load_requests(
    config, tokenizer: Any, candidates: Any = "", prompts_file: str = "", add_special_tokens: bool = False,
    enable_thinking: bool = False,
) -> list[dict[str, Any]]:
  requests = []
  if prompts_file:
    with open(prompts_file, "r", encoding="utf-8") as file:
      for line in file:
        line = line.strip()
        if line:
          item = json.loads(line)
          requests.append({"prompt": item["prompt"], "candidates": _parse_candidates(item.get("candidates"))})
  else:
    prompt = config.prompt
    if config.use_chat_template:
      prompt = tokenizer.apply_chat_template(
          vllm_decode.build_chat_messages(config), tokenize=False, add_generation_prompt=True,
          add_special_tokens=False, enable_thinking=enable_thinking,
      )
    requests.append({"prompt": prompt, "candidates": _parse_candidates(candidates)})
  for request in requests:
    request["prompt_ids"] = tokenizer.encode(request["prompt"], add_special_tokens=add_special_tokens)
    request["candidate_ids"] = [
        candidate_token_id(tokenizer, request["prompt"], request["prompt_ids"], candidate, add_special_tokens)
        for candidate in request["candidates"]
    ]
  return requests


def _candidate_probabilities(logprobs: np.ndarray) -> np.ndarray:
  if len(logprobs) == 0:
    return np.zeros((0,), dtype=np.float64)
  weights = np.exp(logprobs - np.max(logprobs))
  return weights / np.sum(weights)


def score_requests(
    scorer: VllmScorer, requests: list[dict[str, Any]], top_k: int = DEFAULT_TOP_K, method: str = "auto"
) -> list[dict[str, Any]]:
  tokenizer = scorer.tokenizer
  prompt_ids = [request["prompt_ids"] for request in requests]
  top = scorer.next_token_logprobs(prompt_ids, top_k=top_k)
  with_candidates = [index for index, request in enumerate(requests) if request["candidate_ids"]]
  candidate_scores = {}
  if with_candidates:
    scores = scorer.candidate_logprobs(
        [prompt_ids[index] for index in with_candidates],
        [requests[index]["candidate_ids"] for index in with_candidates],
        method=method, top_k=top_k, top_k_logprobs=[top[index] for index in with_candidates],
    )
    candidate_scores = dict(zip(with_candidates, scores))
  results = []
  for index, request in enumerate(requests):
    distribution = top[index]
    greedy_id = max(distribution, key=distribution.get)
    result = {
        "prompt": request["prompt"],
        "num_prompt_tokens": len(request["prompt_ids"]),
        "greedy_token": {
            "id": greedy_id, "text": tokenizer.decode([greedy_id]), "logprob": distribution[greedy_id]
        },
        "top_k": [
            {"id": token_id, "text": tokenizer.decode([token_id]), "logprob": logprob}
            for token_id, logprob in sorted(distribution.items(), key=lambda item: item[1], reverse=True)[:top_k]
        ],
    }
    if index in candidate_scores:
      logprobs = candidate_scores[index]
      probabilities = _candidate_probabilities(logprobs)
      result["candidates"] = [
          {"candidate": str(candidate), "id": token_id, "logprob": float(logprob), "prob_among_candidates": float(prob)}
          for candidate, token_id, logprob, prob in zip(
              request["candidates"], request["candidate_ids"], logprobs, probabilities
          )
      ]
      result["prediction"] = str(request["candidates"][int(np.argmax(logprobs))])
    results.append(result)
  return results


def log_results(results: Sequence[dict[str, Any]]) -> None:
  for index, result in enumerate(results, start=1):
    heading = f"Result {index}"
    if "prediction" in result:
      heading += f" — prediction: {result['prediction']}"
    lines = [heading, f"Prompt ({result['num_prompt_tokens']} tokens): {result['prompt']!r}"]
    candidates = result.get("candidates", [])
    if candidates:
      ranked = sorted(candidates, key=lambda option: option["prob_among_candidates"], reverse=True)
      option_width = max(len("Option"), *(len(repr(option["candidate"])) for option in ranked))
      lines.append(f"{'Option':<{option_width}}  {'Option-prob':>12}  {'Log-prob':>10}")
      for option in ranked:
        lines.append(
            f"{option['candidate']!r:<{option_width}}  {option['prob_among_candidates']:>12.2%}  "
            f"{option['logprob']:>10.4f}"
        )
    greedy = result["greedy_token"]
    lines.append(f"Greedy next token: {greedy['text']!r} (id={greedy['id']}, log-prob={greedy['logprob']:.4f})")
    if result["top_k"]:
      lines.append("Top-k next tokens:")
      for token in result["top_k"]:
        lines.append(f"  {token['text']!r} (id={token['id']}, log-prob={token['logprob']:.4f})")
    max_logging.log("\n".join(lines))


def score(
    config, *, candidates: Any = "", prompts_file: str = "", output_file: str = "", top_k: int = DEFAULT_TOP_K,
    method: str = "auto", add_special_tokens: bool = False, enable_prefix_caching: bool = False,
    repeat: int = 1, enable_thinking: bool = False,
) -> list[dict[str, Any]]:
  if top_k < 1:
    raise ValueError("top_k must be at least 1.")
  if repeat < 1:
    raise ValueError("repeat must be at least 1.")
  if method not in ("auto", "top_k", "exact"):
    raise ValueError(f"Unknown method {method!r}; expected 'auto', 'top_k' or 'exact'.")
  adapter.register(config)
  tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer_path, token=config.hf_access_token)
  requests = load_requests(config, tokenizer, candidates, prompts_file, add_special_tokens, enable_thinking)
  llm = vllm_decode.create_vllm_llm(config, max_logprobs=top_k, enable_prefix_caching=enable_prefix_caching)
  scorer = VllmScorer(llm, default_top_k=top_k)
  for index in range(repeat):
    results = score_requests(scorer, requests, top_k=top_k, method=method)
    if repeat > 1:
      summary = [
          [round(candidate["logprob"], 6) for candidate in result["candidates"]]
          if "candidates" in result else round(result["greedy_token"]["logprob"], 6)
          for result in results
      ]
      max_logging.log(f"[repeat {index + 1}/{repeat}] candidate/greedy logprobs per prompt: {summary}")
  log_results(results)
  if output_file:
    with open(output_file, "w", encoding="utf-8") as file:
      for result in results:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")
    max_logging.log(f"Wrote {len(results)} results to {output_file}")
  return results
