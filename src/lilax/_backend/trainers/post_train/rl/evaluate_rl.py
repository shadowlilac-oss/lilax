import collections
import json
import re
from typing import Any, Callable, Optional

from tqdm.auto import tqdm
from tunix.rl.rollout.base_rollout import RolloutConfig

from lilax._backend.trainers.post_train.rl import utils_rl
from lilax._backend.utils import max_logging


def generate_responses(tmvp_config, prompts, rl_cluster, num_passes=1):
  multiple_call_responses = [[] for _ in range(len(prompts))]
  eval_strategy = tmvp_config.generation_configs[tmvp_config.eval_sampling_strategy]
  for pass_index in range(num_passes):
    responses = rl_cluster.rollout.generate(
        prompts,
        rollout_config=RolloutConfig(
            max_tokens_to_generate=tmvp_config.max_target_length - tmvp_config.max_prefill_predict_length,
            temperature=eval_strategy["eval_temperature"], top_k=eval_strategy["eval_top_k"],
            top_p=eval_strategy["eval_top_p"],
        ),
    ).text
    if tmvp_config.debug:
      max_logging.log(f"Pass {pass_index+1}/{num_passes}, responses: {responses}")
    for idx, response in enumerate(responses):
      multiple_call_responses[idx].append(response)
  return multiple_call_responses


def _score_single(
    extracted_response: str, raw_response: str, answers: list[str], tmvp_config: Any, match_format: re.Pattern[str],
) -> tuple[bool, bool, bool]:
  has_correct_format = match_format.search(raw_response) is not None
  try:
    is_correct, is_partially_correct = utils_rl.check_correctness(extracted_response, answers, tmvp_config)
    if tmvp_config.debug:
      max_logging.log(f"Result has_correct_format: {has_correct_format}")
      max_logging.log(f"Result is_correct: {is_correct}")
      max_logging.log(f"Result is_partially_correct: {is_partially_correct}")
  except Exception as error:
    is_correct, is_partially_correct = False, False
    if tmvp_config.debug:
      max_logging.log(f"Evaluation Exception: {error} — SKIPPED")
  return is_correct, is_partially_correct, has_correct_format


def score_responses(tmvp_config, question, responses, answers):
  if tmvp_config.debug:
    max_logging.log("========================================")
    max_logging.log(f"Evaluation Question: {question}")
    max_logging.log(f"Evaluation Answer: {answers}")
    max_logging.log(f"Evaluation Responses: {responses}")
    max_logging.log("========================================")
  eval_mode = getattr(tmvp_config, "eval_mode", "pass")
  match_format = utils_rl.get_match_format_regex(tmvp_config)
  extracted_responses = [utils_rl.extract_answer(response, tmvp_config) for response in responses]
  if not extracted_responses:
    return False, False, False
  if eval_mode == "maj":
    counter = collections.Counter(extracted_responses)
    majority = counter.most_common(1)[0][0]
    if tmvp_config.debug:
      max_logging.log(f"Majority Response: {majority} (Count: {counter[majority]})")
    has_correct_format = any(
        match_format.search(responses[idx]) is not None
        for idx, response in enumerate(extracted_responses) if response == majority
    )
    is_correct, is_partially_correct, _ = _score_single(majority, responses[0], answers, tmvp_config, match_format)
    return is_correct, is_partially_correct, has_correct_format
  if eval_mode == "pass":
    result = False, False, False
    for extracted, response in zip(extracted_responses, responses):
      result = _score_single(extracted, response, answers, tmvp_config, match_format)
      if all(result):
        return result
    return result
  if eval_mode == "pass_at_1":
    scores = [
        _score_single(extracted, response, answers, tmvp_config, match_format)
        for extracted, response in zip(extracted_responses, responses)
    ]
    n_samples = len(scores)
    frac_correct = sum(score[0] for score in scores) / n_samples
    frac_partial = sum(score[1] for score in scores) / n_samples
    frac_format = sum(score[2] for score in scores) / n_samples
    if tmvp_config.debug:
      max_logging.log(f"{frac_correct*n_samples:.0f}/{n_samples} correct")
      max_logging.log(f"{frac_partial*n_samples:.0f}/{n_samples} partial")
      max_logging.log(f"{frac_format*n_samples:.0f}/{n_samples} format")
    return frac_correct, frac_partial, frac_format
  raise ValueError(f"Unknown eval_mode: {eval_mode!r}")


def _compute_row_reward(reward_fns, prompt, question, responses, answer, row_idx):
  if not responses:
    return 0.0, 0
  try:
    score_sum = 0.0
    for response in responses:
      for fn in reward_fns:
        scores = fn(prompts=[prompt], completions=[response], answer=[answer], question=[question])
        if scores:
          score_sum += float(scores[0])
    return score_sum, len(responses)
  except Exception as error:
    max_logging.log(f"[eval-reward] reward_fn failed on row {row_idx}: {error!r}")
    return 0.0, 0


def evaluate(
    tmvp_config, dataset, rl_cluster, num_passes=1, corr_lst=False, make_lst=False,
    reward_fns: Optional[list[Callable[..., Any]]] = None,
):
  response_lst = []
  corr = 0
  partially_corr = 0
  corr_format = 0
  total = 0
  reward_sum = 0.0
  reward_count = 0
  use_reward = bool(reward_fns)
  for batch in tqdm(dataset):
    answers = batch["answer"]
    questions = batch["question"]
    prompts = batch["prompts"]
    multiple_call_responses = generate_responses(tmvp_config, prompts, rl_cluster, num_passes=num_passes)
    for question, responses, answer, prompt in zip(questions, multiple_call_responses, answers, prompts):
      answer_list = list(dict.fromkeys(json.loads(answer)))
      is_correct, is_partially_correct, has_correct_format = score_responses(tmvp_config, question, responses, answer_list)
      if use_reward:
        row_sum, row_count = _compute_row_reward(reward_fns, prompt, question, responses, answer, total)
        reward_sum += row_sum
        reward_count += row_count
      corr += is_correct
      partially_corr += is_partially_correct
      corr_format += has_correct_format
      if make_lst:
        if corr_lst and is_correct:
          response_lst.append((question, answer_list, responses))
        elif not corr_lst and not is_correct:
          response_lst.append((question, answer_list, responses))
      total += 1
      if total % 10 == 0:
        max_logging.log(
            f"===> {corr=}, {total=}, {corr / total * 100=}, "
            f"{partially_corr / total * 100=}, {corr_format / total * 100=}"
        )
  to_return = (
      corr, total, corr / total * 100 if total > 0 else 0,
      partially_corr / total * 100 if total > 0 else 0, corr_format / total * 100 if total > 0 else 0,
      reward_sum / reward_count if (use_reward and reward_count > 0) else 0.0,
  )
  return to_return, response_lst
