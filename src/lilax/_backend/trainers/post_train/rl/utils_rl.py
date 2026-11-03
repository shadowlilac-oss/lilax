import importlib.util
import itertools
import json
import os
import re
import sys
import uuid
from typing import Any, Callable, Optional

from etils import epath
from math_verify import parse
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
import numpy as np
import optax
from tunix.rl.agentic.parser.chat_template_parser import parser as agentic_chat_template_parser

from lilax._backend.trainers.post_train.rl.hooks import RLTrainingHooks
from lilax._backend.trainers.post_train.rl.math_verify_pool import math_verify_pool, verify_math_worker
from lilax._backend.utils import max_logging


EPSILON = 1e-6
FALLBACK_ANSWER = "-1000000"
SUBSTITUTIONS = [
    ("\\\\", "\\"), ("\\tfrac", "\\frac"), ("\\dfrac", "\\frac"), ("an ", ""), ("a ", ""),
    (".$", "$"), ("\\$", ""), (r"\ ", ""), (" or ", ","), (" and ", ","), ("million", "*10^6"),
    ("billion", "*10^9"), ("trillion", "*10^12"), (" ", ""), ("mbox", "text"),
    (",\\text{and}", ","), ("\\text{and}", ","), ("\\text{m}", "\\text{}"),
]
UNITS = [
    "yard", "foot", "feet", "mile", "day", "week", "month", "year", "hour", "minute", "second",
    "centimeter", "meter", "cm", "mm", "km", "inch", "degree", "pound", "cent", "mph",
]
REMOVED_EXPRESSIONS = [
    "\\left", "\\right", "\\!", "square", "ways", "integers", "dollars", "units", "\\ldots", "sue",
    "points", "digits", "gm", "meals", "edges", "students", "childrentickets", "multiples", "\\text{s}",
    "\\text{.}", "\\text{\ns}", "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}", r"\mathrm{th}",
    r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]


def math_verify_func(items, scores, timeout=300, trainer_config=None):
  if not items:
    return scores
  num_procs = None
  if trainer_config is not None:
    timeout = getattr(trainer_config, "math_verify_timeout", timeout)
    num_procs = getattr(trainer_config, "math_verify_num_procs", None)
  return math_verify_pool(trainer_config, items, scores, timeout=timeout, num_procs=num_procs, log_fn=max_logging.log)


def boxed(value: str) -> str:
  return "\\boxed{" + value + "}" if not value.startswith("\\boxed{") else value


def get_match_format_regex(tmvp_config: Any) -> re.Pattern[str]:
  reasoning_start = "" if getattr(tmvp_config, "reasoning_start_token_in_prompt", False) else re.escape(tmvp_config.reasoning_start_token)
  match_format = re.compile(
      rf"{reasoning_start}.*?{re.escape(tmvp_config.reasoning_end_token)}.*?"
      rf"{re.escape(tmvp_config.solution_start_token)}(.+?){re.escape(tmvp_config.solution_end_token)}",
      flags=re.MULTILINE | re.DOTALL,
  )
  if tmvp_config.debug:
    match_format.search(
        f"{tmvp_config.reasoning_start_token}Let me"
        f" think!{tmvp_config.reasoning_end_token}{tmvp_config.solution_start_token}2{tmvp_config.solution_end_token}"
    )
  return match_format


def get_answer_fallback_regex(tmvp_config: Any) -> re.Pattern[str]:
  return re.compile(
      rf"{re.escape(tmvp_config.solution_start_token)}(.+?){re.escape(tmvp_config.solution_end_token)}",
      flags=re.MULTILINE | re.DOTALL,
  )


def match_format_exactly(prompts: list[str], completions: list[str], tmvp_config: Any, **kargs: Any) -> list[float]:
  scores = []
  match_format = get_match_format_regex(tmvp_config)
  for completion in completions:
    score = 0
    if match_format.search(completion) is not None:
      score += tmvp_config.reward_exact_format_match
    scores.append(score)
  return scores


def match_format_approximately(prompts: list[str], completions: list[str], tmvp_config: Any, **kargs: Any) -> list[float]:
  scores = []
  for completion in completions:
    score = 0
    score += (
        tmvp_config.reward_partial_format_match
        if getattr(tmvp_config, "reasoning_start_token_in_prompt", False) or completion.count(tmvp_config.reasoning_start_token) == 1
        else tmvp_config.penalty_incorrect_format
    )
    score += tmvp_config.reward_partial_format_match if completion.count(tmvp_config.reasoning_end_token) == 1 else tmvp_config.penalty_incorrect_format
    score += tmvp_config.reward_partial_format_match if completion.count(tmvp_config.solution_start_token) == 1 else tmvp_config.penalty_incorrect_format
    score += tmvp_config.reward_partial_format_match if completion.count(tmvp_config.solution_end_token) == 1 else tmvp_config.penalty_incorrect_format
    scores.append(score)
  return scores


def normalize_final_answer(final_answer: str) -> str:
  final_answer = final_answer.split("=")[-1]
  final_answer = re.sub(r"([0-9]) +([0-9])", r"\1+\2", final_answer)
  for before, after in SUBSTITUTIONS:
    final_answer = final_answer.replace(before, after)
  for unit in UNITS:
    final_answer = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", final_answer)
  for expr in REMOVED_EXPRESSIONS:
    final_answer = final_answer.replace(expr, "")
  final_answer = re.sub(
      r".*?(\d+)?\s*\$\s*(\d+)?\s*(\\frac\{.*?\}\{.*?\}|\d+/\d+)\s*\$.*",
      lambda match: f"${whole}{match.group(3)}$" if (whole := (match.group(1) or match.group(2))) else f"${match.group(3)}$",
      final_answer,
  )
  final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
  final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
  final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
  final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
  final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
  final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
  final_answer = final_answer.replace("$", "")
  if final_answer.startswith("."):
    final_answer = "0" + final_answer
  final_answer = final_answer.replace("{.", "{0.")
  if len(final_answer) >= 2 and final_answer[0] == "{" and final_answer[-1] == "}":
    final_answer = final_answer[1:-1]
  try:
    value = float(final_answer)
    if abs(value - round(value)) < 1e-7:
      final_answer = str(int(round(value)))
  except (ValueError, OverflowError):
    pass
  if final_answer.replace(",", "").isdigit():
    final_answer = final_answer.replace(",", "")
  return final_answer


def preprocess_math_string(text: str) -> str:
  text = normalize_final_answer(text).strip()
  return fix_latex_escaping(text)


def fix_latex_escaping(text: str) -> str:
  escape_fixes = [
      ("\f", "rac", r"\frac"), ("\n", "ewline", r"\newline"), ("\n", "e", r"\ne"),
      ("\t", "heta", r"\theta"), ("\t", "an", r"\tan"), ("\t", "o", r"\to"),
      ("\t", "imes", r"\times"), ("\t", "ext", r"\text"), ("\t", "extbf", r"\textbf"),
      ("\t", "extit", r"\textit"), ("\r", "ightarrow", r"\rightarrow"),
      ("\r", "ightarrow", r"\Rightarrow"), ("\b", "eta", r"\beta"), ("\b", "ar", r"\bar"),
      ("\b", "inom", r"\binom"), ("\b", "oxed", r"\boxed"), ("\a", "lpha", r"\alpha"),
      ("\a", "pprox", r"\approx"), ("\v", "ec", r"\vec"),
  ]
  for escape_char, suffix, latex_cmd in escape_fixes:
    if escape_char in text:
      text = text.replace(escape_char + suffix, latex_cmd)
  latex_commands = [
      "frac", "sqrt", "pi", "theta", "alpha", "beta", "gamma", "delta", "sum", "int", "infty", "cdot",
      "times", "div", "pm", "mp", "leq", "geq", "neq", "approx", "equiv", "sin", "cos", "tan", "log",
      "ln", "exp", "lim", "to", "rightarrow", "leftarrow", "Rightarrow", "Leftarrow", "overline", "underline",
      "hat", "bar", "vec", "dot", "ddot", "mathbb", "mathbf", "mathrm", "text", "textbf", "textit", "boxed",
      "left", "right", "choose", "binom",
  ]
  for cmd in latex_commands:
    text = re.sub(rf"(?<!\\)\b{cmd}\b", rf"\\{cmd}", text)
  return text


def check_numbers(prompts: list[str], completions: list[str], answer: list[str], tmvp_config: Any, **kargs: Any) -> list[float]:
  question = kargs["question"]
  extracted_responses = [extract_answer(completion, tmvp_config) for completion in completions]
  true_answers = [list(dict.fromkeys(json.loads(acceptable_answers))) for acceptable_answers in answer]
  scores = [tmvp_config.penalty_incorrect_format] * len(completions)
  math_verify_queue = []
  for gen_idx, (guess, unique_answers) in enumerate(zip(extracted_responses, true_answers)):
    if guess is None:
      continue
    if guess == FALLBACK_ANSWER:
      scores[gen_idx] = tmvp_config.penalty_incorrect_answer
      continue
    has_exact_match = False
    for true_answer in unique_answers:
      if guess == true_answer:
        scores[gen_idx] = max(scores[gen_idx], tmvp_config.reward_exact_answer)
        has_exact_match = True
      elif guess.strip() == true_answer.strip():
        scores[gen_idx] = max(scores[gen_idx], tmvp_config.reward_white_space_format_match)
        has_exact_match = True
    if not has_exact_match:
      norm_guess = preprocess_math_string(guess)
      norm_answers = [boxed(preprocess_math_string(true_answer)) for true_answer in unique_answers]
      math_verify_queue.append((gen_idx, norm_answers, [boxed(norm_guess)]))
  if math_verify_queue:
    scores = math_verify_func(math_verify_queue, scores, trainer_config=tmvp_config)
  if tmvp_config.debug:
    debug_log_path = epath.Path(tmvp_config.base_output_directory) / tmvp_config.run_name / "debug_rl_logs"
    debug_log_path.mkdir(parents=True, exist_ok=True)
    log_file = debug_log_path / f"check_numbers_{uuid.uuid4().hex}.txt"
    log_content = (
        "START ============================\n" f"Question: {question[0]}\n" f"Answer: {answer[0]}\n"
        f"Response: {completions[0]}\n" f"Extracted: {extracted_responses[0]}\n" f"Reward Score: {scores[0]}\n"
        "END ==============================\n"
    )
    log_file.write_text(log_content)
  return scores


_BOXED_PREFIX = "\\boxed"
_BRACE_REGEX = re.compile(r"[{}]")


def _last_balanced_boxed(content: str) -> str | None:
  best = None
  stack = []
  for match in _BRACE_REGEX.finditer(content):
    index = match.start()
    if content[index] == "{":
      stack.append(index)
    elif stack:
      opening = stack.pop()
      if opening >= len(_BOXED_PREFIX) and content.startswith(_BOXED_PREFIX, opening - len(_BOXED_PREFIX)):
        best = content[opening + 1:index].strip()
  return best


def extract_answer(response: str, tmvp_config: Any) -> str:
  answer_tag_regex = get_answer_fallback_regex(tmvp_config)
  answer_matches = answer_tag_regex.findall(response)
  content = answer_matches[-1] if answer_matches else response
  boxed_answer = _last_balanced_boxed(content)
  if boxed_answer is not None:
    return boxed_answer
  match = re.search(r"\\boxed\s*\{?\s*([a-zA-Z0-9\.,\-]+)\s*\}?", content)
  if match:
    return match.group(1).strip()
  fallback_matches = answer_tag_regex.findall(response)
  if fallback_matches:
    return fallback_matches[-1].strip()
  return FALLBACK_ANSWER


def extract_hash_answer(text: str) -> str | None:
  if "####" not in text:
    return None
  return text.split("####")[1].strip()


def check_correctness(extracted_response: str, acceptable_answers: list[str], tmvp_config: Any) -> tuple[bool, bool]:
  norm_response = preprocess_math_string(extracted_response)
  norm_answers = [preprocess_math_string(answer) for answer in acceptable_answers]
  score = verify_math_worker([boxed(answer) for answer in norm_answers], [boxed(norm_response)])
  if score > 0.0:
    return True, True
  is_partially_correct = False
  try:
    predictions = parse(boxed(norm_response), (ExprExtractionConfig(), LatexExtractionConfig()))
    golds = list(itertools.chain.from_iterable(
        parse(boxed(answer), (ExprExtractionConfig(), LatexExtractionConfig())) for answer in norm_answers
    ))
    is_partially_correct = any(
        0.9 <= (float(pred) + EPSILON) / (float(gold) + EPSILON) <= 1.1 for pred in predictions for gold in golds
    )
  except:
    if tmvp_config.debug:
      max_logging.log(
          f"check_correctness failed for extracted response: {extracted_response} and answers: {acceptable_answers}"
      )
  return False, is_partially_correct


def get_optimizer(tmvp_config: Any) -> optax.GradientTransformation:
  lr_schedule_steps = getattr(tmvp_config, "learning_rate_schedule_steps", -1)
  config_steps = getattr(tmvp_config, "steps", -1)
  if lr_schedule_steps is not None and lr_schedule_steps > 0 and lr_schedule_steps != config_steps:
    schedule_steps = lr_schedule_steps
  else:
    schedule_steps = tmvp_config.train_steps
  schedule = optax.schedules.warmup_cosine_decay_schedule(
      init_value=0.0, peak_value=tmvp_config.learning_rate,
      warmup_steps=int(tmvp_config.warmup_steps_fraction * schedule_steps), decay_steps=schedule_steps, end_value=0.0,
  )

  def make_optimizer(learning_rate):
    transforms = []
    if tmvp_config.gradient_clipping_threshold > 0:
      transforms.append(optax.clip_by_global_norm(max_norm=tmvp_config.gradient_clipping_threshold))
    transforms.append(optax.adamw(
        learning_rate=learning_rate, b1=tmvp_config.adam_b1, b2=tmvp_config.adam_b2,
        weight_decay=tmvp_config.adam_weight_decay,
    ))
    return optax.chain(*transforms)
  return optax.inject_hyperparams(make_optimizer)(learning_rate=schedule)


def format_maxtext_messages(messages: list[str], template_config: dict[str, Any], tmvp_config: Any) -> list[dict[str, str]]:
  if template_config is None:
    raise ValueError("template_config cannot be None for format_maxtext_messages.")
  formatted_messages = []
  for msg in messages:
    formatted_content = template_config["TEMPLATE"].format(
        system_prompt=template_config["SYSTEM_PROMPT"].format(
            reasoning_start_token=tmvp_config.reasoning_start_token, reasoning_end_token=tmvp_config.reasoning_end_token,
            solution_start_token=tmvp_config.solution_start_token, solution_end_token=tmvp_config.solution_end_token,
        ), question=msg,
    )
    formatted_messages.append({"role": "user", "content": formatted_content})
  return formatted_messages


def process_answer(question: str, answer: str, question_type: str) -> list[str]:
  if question_type == "MCQ":
    return process_mcq(question, answer)
  return [answer, answer]


def process_mcq(question: str, answer: str) -> list[str]:
  pattern = r"""
    (?:\(?([A-E])\)?|\\(?:text|mathrm|textbf)\{([A-E])\})
    [\s\.\:\}\$]*\s*(.*?)
    (?=\s*(?:\\quad|\\qquad|\\hspace|\n|\r)\s*(?:\(?|\\(?:text|mathrm|textbf))|$)
  """
  matches = re.findall(pattern, question, re.DOTALL | re.VERBOSE)
  options = {}
  for match in matches:
    letter = match[0] or match[1]
    value = match[2].strip()
    options[letter] = re.sub(r"\\q?quad|\\textbf{|[~$]", "", value).strip()
  acceptable_answers = [answer]
  if answer in options:
    acceptable_answers.append(options[answer])
  else:
    for letter, value in options.items():
      if normalize_final_answer(answer) == normalize_final_answer(value):
        acceptable_answers.append(letter)
        break
  return acceptable_answers


def process_data(dataset_name: str, model_tokenizer: Any, template_config: dict[str, Any], tmvp_config: Any, x: dict[str, Any]) -> dict[str, str]:
  def to_str(val):
    if isinstance(val, bytes):
      return val.decode("utf-8")
    return str(val)
  for key in ["problem", "prompt", "question"]:
    if key in x:
      question = to_str(x[key])
      break
  for key in ["answer", "solution", "expected_answer"]:
    if key in x:
      answer = to_str(x[key])
      break
  if "extra_info" in x and isinstance(x["extra_info"], dict) and "raw_problem" in x["extra_info"]:
    question = to_str(x["extra_info"]["raw_problem"])
  if "reward_model" in x and isinstance(x["reward_model"], dict) and "ground_truth" in x["reward_model"]:
    answer = to_str(x["reward_model"]["ground_truth"])
  if dataset_name == "openai/gsm8k":
    answer = extract_hash_answer(answer)
  question_type = "default"
  if "question_type" in x:
    question_type = to_str(x["question_type"])
  processed_answer = process_answer(question, answer, question_type)
  formatted_messages = format_maxtext_messages([question], template_config, tmvp_config)
  prompts = model_tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
  return {"prompts": prompts, "question": question, "answer": json.dumps(processed_answer)}


def get_correctness_metrics(prompts: Any, completions: Any, rewards: np.ndarray, advantages: Any, **kwargs: Any) -> dict[str, tuple[float | int, Callable[..., Any]]]:
  del prompts, completions, advantages, kwargs
  solve_all = (rewards > 0.1).all()
  solve_none = (rewards == 0).all()
  solve_partial = (~solve_all) and (~solve_none)
  solve_ratio = (rewards > 0.1).mean()
  return {
      "rewards/solve_all": (1 if solve_all else 0, np.mean),
      "rewards/solve_none": (1 if solve_none else 0, np.mean),
      "rewards/solve_partial": (1 if solve_partial else 0, np.mean),
      "rewards/solve_ratio": (solve_ratio, np.mean),
  }


class MaxTextChatParser(agentic_chat_template_parser.DefaultChatTemplateParser):
  def __init__(self, model_tokenizer: Any, template_config: dict[str, Any], tmvp_config: Any) -> None:
    super().__init__(model_tokenizer)
    self.template_config = template_config
    self.tmvp_config = tmvp_config

  def parse(self, messages: list[dict[str, str]], add_generation_prompt: bool = False, is_first_msg: bool = False) -> str:
    formatted_messages = format_maxtext_messages(messages, self.template_config, self.tmvp_config)
    return super().parse(
        messages=formatted_messages, add_generation_prompt=add_generation_prompt, is_first_msg=is_first_msg
    )


def install_training_hooks(
    rl_cluster: Any, trainer_config: Any, test_dataset: Any, reward_fns: Optional[list[Callable[..., Any]]] = None,
    goodput_recorder: Any = None, *, evaluator: Callable,
) -> None:
  eval_interval = int(getattr(trainer_config, "eval_interval", 0))
  num_test_batches = int(getattr(trainer_config, "num_test_batches", 0))
  eval_enabled = num_test_batches > 0 and eval_interval > 0
  goodput_enabled = bool(getattr(trainer_config, "enable_goodput_recording", False))
  if not eval_enabled and not goodput_enabled:
    return
  try:
    actor = rl_cluster.actor_trainer
    if getattr(actor, "training_hooks", None) is None:
      actor.training_hooks = RLTrainingHooks(
          rl_cluster, trainer_config, test_dataset, eval_interval, reward_fns, goodput_recorder=goodput_recorder,
          evaluator=evaluator,
      )
      max_logging.warning(f"[intermediate-eval] hook installed: evaluate(...) will fire every {eval_interval} outer steps.")
    else:
      max_logging.warning(
          "[intermediate-eval] actor.training_hooks already set; skipping install (chain manually if you need both)."
      )
  except Exception as error:
    max_logging.warning(f"[intermediate-eval] install failed: {error!r}")


def _exec_user_module(module_name: str, module_path: str, register: bool = False):
  spec = importlib.util.spec_from_file_location(module_name, module_path)
  if spec is None or spec.loader is None:
    raise ValueError(f"Cannot import {module_path!r}: not a valid python file")
  module = importlib.util.module_from_spec(spec)
  if register:
    sys.modules[module_name] = module
  try:
    spec.loader.exec_module(module)
  except BaseException:
    if register:
      sys.modules.pop(module_name, None)
    raise
  return module


def _module_name_is_free(module_name: str, module_path: str) -> bool:
  if not module_name.isidentifier() or module_name in sys.modules:
    return False
  try:
    spec = importlib.util.find_spec(module_name)
  except (ImportError, ValueError):
    return False
  return spec is None or os.path.abspath(spec.origin or "") == module_path


def _load_worker_importable_module(module_path: str, function_name: str):
  module_path = os.path.abspath(module_path)
  module_name = os.path.splitext(os.path.basename(module_path))[0]
  loaded = sys.modules.get(module_name)
  if loaded is not None and os.path.abspath(getattr(loaded, "__file__", None) or "") == module_path:
    return loaded
  if not _module_name_is_free(module_name, module_path):
    max_logging.warning(
        f"{module_path!r} is loaded as a private module because the name {module_name!r} is taken or is not a valid"
        " module name; its functions cannot be pickled into reward worker processes and will be evaluated in the"
        " parent process. Rename the file to parallelize them."
    )
    return _exec_user_module(f"_user_module_{function_name}", module_path)
  module = _exec_user_module(module_name, module_path, register=True)
  directory = os.path.dirname(module_path)
  if directory not in sys.path:
    sys.path.append(directory)
  return module


def load_custom_callable(module_path: str, function_name: str, importable_by_workers: bool = False) -> Callable:
  if not os.path.isfile(module_path):
    raise ValueError(f"Cannot import {module_path!r}: file does not exist")
  if importable_by_workers:
    module = _load_worker_importable_module(module_path, function_name)
  else:
    module = _exec_user_module(f"_user_module_{function_name}", module_path)
  fn = getattr(module, function_name, None)
  if fn is None:
    raise ValueError(f"{module_path!r} does not define a function named {function_name!r}")
  return fn
