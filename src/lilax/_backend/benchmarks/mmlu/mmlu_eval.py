"""Native MMLU evaluation using the original MaxText benchmark behavior."""

import collections
import re

import datasets
from tqdm import tqdm

from lilax._backend.benchmarks.mmlu.mmlu_categories import categories, subcategories
from lilax._backend.utils import max_logging

DEFAULT_PROMPT_TEMPLATE = """The following are multiple choice questions (with answers) about {subject}.

{question}
{choices}
Answer:"""


def construct_prompt(subject, question, choices, prompt_template=DEFAULT_PROMPT_TEMPLATE):
  choices_text = "\n".join(f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(choices))
  return prompt_template.format(subject=subject.replace("_", " "), question=question, choices=choices_text)


def parse_answer(output):
  match = re.search(r"Answer:\s*([A-D])|(?:The answer is)\s*([A-D])", output, re.IGNORECASE)
  return match.group(1) or match.group(2) if match else None


def validate_config(config):
  assert not config.load_full_state_path, "Decode doesn't operate on full states! Convert to parameter checkpoint first."


def evaluate(engine, params, config, prompt_template=DEFAULT_PROMPT_TEMPLATE):
  validate_config(config)
  tokenizer = engine.build_tokenizer(engine.get_tokenizer())
  prefill_length = getattr(config, "max_prefill_predict_length", 1024)
  target_length = getattr(config, "max_target_length", 2048)
  correct_count = total_count = 0
  subject_correct, subject_total = collections.defaultdict(int), collections.defaultdict(int)
  subcat_correct, subcat_total = collections.defaultdict(int), collections.defaultdict(int)
  dataset = datasets.load_dataset("lighteval/mmlu", "all", split="test")

  for idx, example in enumerate(tqdm(dataset, desc="Evaluating MMLU dataset")):
    subject, choices = example["subject"], example["choices"]
    prompt = construct_prompt(subject, example["question"], choices, prompt_template)
    tokens, true_length = tokenizer.encode(prompt, is_bos=True, prefill_lengths=[prefill_length])
    if true_length > prefill_length:
      max_logging.log(f"Warning: Prompt length {true_length} exceeds max prefill length {prefill_length}. Truncating.")
      tokens, true_length = tokens[:prefill_length], prefill_length
    assert config.quantization != "fp8", "fp8 on NVIDIA GPUs is not supported in decode.py yet"
    assert config.quantization != "nanoo_fp8", "NANOO fp8 on AMD MI300/MI325 GPUs is not supported in decode.py yet"
    prefill_result, first_token = engine.prefill(params=params, padded_tokens=tokens, true_length=true_length)
    decode_state = engine.insert(prefill_result, engine.init_decode_state(), slot=0)
    sampled_tokens = [first_token.get_result_at_slot(0).tokens.item()]
    predicted_answer = ""
    for _ in range(prefill_length, target_length):
      output = tokenizer.decode(sampled_tokens)
      predicted_answer = parse_answer(prompt + output)
      if predicted_answer:
        break
      decode_state, sampled_token = engine.generate(params, decode_state)
      sampled_tokens.append(sampled_token.get_result_at_slot(0).tokens.item())
      if sampled_tokens[-1] == tokenizer.eos_id:
        break
    if not predicted_answer:
      max_logging.log(f"Could not extract an answer from the model's output for example {total_count + 1}")
    elif predicted_answer not in {chr(ord('A') + i) for i in range(len(choices))}:
      max_logging.log(f"Invalid or missing predicted answer for subject '{subject}' in example {total_count + 1}")
    correct_answer = chr(65 + example["answer"])
    max_logging.log(f"{total_count + 1} | {prompt}\n[Model output] {output}\n"
                    f"[Correct answer] {correct_answer}, Matching: {predicted_answer == correct_answer}")
    if predicted_answer == correct_answer:
      correct_count += 1
      subject_correct[subject] += 1
    total_count += 1
    subject_total[subject] += 1
    if idx % 50 == 0:
      max_logging.log(f" Accuracy: {correct_count / total_count:.4f} | Processed: {total_count}/{len(dataset)}")

  if total_count:
    max_logging.log(f"\nFinal accuracy on MMLU dataset: {correct_count / total_count:.4f}")
  else:
    max_logging.log("No valid predictions were made.")
  subject_acc = {subject: subject_correct[subject] / subject_total[subject] for subject in subject_total}
  for subject in subject_acc:
    if subject in subcategories:
      for label in subcategories[subject]:
        subcat_correct[label] += subject_correct[subject]
        subcat_total[label] += subject_total[subject]
    else:
      max_logging.log(f"Warning: Subject '{subject}' not found in subcategories.")
  subcat_acc = {label: subcat_correct[label] / total for label, total in subcat_total.items()}
  max_logging.log("\nSubcategory Accuracies:")
  for label, accuracy in subcat_acc.items():
    max_logging.log(f"Accuracy for subcategory '{label}': {accuracy:.4f}")
  cat_correct, cat_total = collections.defaultdict(int), collections.defaultdict(int)
  for category, labels in categories.items():
    for label in labels:
      cat_correct[category] += subcat_correct[label]
      cat_total[category] += subcat_total[label]
  cat_acc = {category: cat_correct[category] / total if total else None for category, total in cat_total.items()}
  max_logging.log("\nCategory Accuracies:")
  for category, accuracy in cat_acc.items():
    value = f"{accuracy:.4f}" if accuracy is not None else "No data available."
    max_logging.log(f"Accuracy for category '{category}': {value}")
  return {"accuracy": correct_count / total_count if total_count else None, "subject_accuracy": subject_acc,
          "subcategory_accuracy": subcat_acc, "category_accuracy": cat_acc}
