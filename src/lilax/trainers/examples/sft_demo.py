"""GSM8K evaluation and LoRA SFT workflow from the MaxText demo notebooks."""

import json
from importlib.resources import files
import re

import datasets
from tqdm.auto import tqdm

from lilax import SFTTrainer


DATASET_NAME = "openai/gsm8k"
DATASET_DATA_DIR = "main"
DATASET_TRAIN_SPLIT = "train"
DATASET_TEST_SPLIT = "test"
DATASET_DATA_COLUMN = ["question", "answer"]
TRAIN_STEPS = 10000
SEED = 42
BATCH_SIZE = 4
NUM_TEST_SAMPLES = 1320
MAX_TOKENS_TO_GENERATE = 768
MAX_PROMPT_LENGTH = 256
EVALUATION_CONFIG = {"temperature": 1e-4, "top_k": 1, "top_p": 1.0}
REASONING_START = "<reasoning>"
REASONING_END = "</reasoning>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"
MATCH_FORMAT = re.compile(
    rf"^.*?" rf"{REASONING_START}.+?{REASONING_END}.*?" rf"{ANSWER_START}(.+?){ANSWER_END}" rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)
MATCH_ANSWER = re.compile(rf"{ANSWER_START}.*?([\d\.\,\$]{{1,}})", flags=re.MULTILINE | re.DOTALL)


def get_test_dataset(processing_class, *, token=None):
  """Load and format the shuffled GSM8K test split as an HF dataset."""
  template_config = json.loads(files("lilax").joinpath("_backend/examples/chat_templates/math_qa.json").read_text())
  dataset = datasets.load_dataset(
      DATASET_NAME, data_dir=DATASET_DATA_DIR, split=DATASET_TEST_SPLIT, token=token,
  )
  separator = template_config["REASONING_ANSWER_SEPARATOR"]

  def format_example(x):
    _, answer = x["answer"].split(separator) if separator in x["answer"] else (None, None)
    return {
        "question": x["question"],
        "prompt": processing_class.apply_chat_template(
            [{"role": "user", "content": template_config["PROMPT_TEMPLATE"].format(question=x["question"].strip())}],
            tokenize=False, add_generation_prompt=True,
        ),
        "target_answer": answer,
    }

  return dataset.shuffle(seed=SEED).map(format_example)


def evaluate_model(dataset, model, processing_class, debug=True):
  """Return the demo's exact, within-10-percent, and format percentages."""
  total, total_correct, total_partially_correct, total_correct_format = 0, 0, 0, 0
  for batch in tqdm(dataset):
    inputs = processing_class(batch["prompt"], padding=True, return_tensors="np")
    if any(sum(mask) > MAX_PROMPT_LENGTH for mask in inputs["attention_mask"]):
      raise ValueError(f"A demo prompt exceeds MAX_PROMPT_LENGTH={MAX_PROMPT_LENGTH}.")
    sequences = model.generate(
        **inputs, max_new_tokens=MAX_TOKENS_TO_GENERATE, do_sample=True, **EVALUATION_CONFIG,
    )
    responses = processing_class.batch_decode(sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    for i, question in enumerate(batch["question"]):
      if debug:
        print("========================================")
        print(f"Question: {question}")
        print("----------------------------------------")
        print(f"Model Generated Response: {responses[i]}")
        print("----------------------------------------")
        print(f"Target Response: {batch['target_answer'][i]}")
        print("========================================")
      is_correct, is_partially_correct, has_correct_format = score_response(
          target=batch["target_answer"][i], prediction=responses[i], debug=debug
      )
      total_correct += is_correct
      total_partially_correct += is_partially_correct
      total_correct_format += has_correct_format
      total += 1
  return {
      "correct": (total_correct / total) * 100,
      "partially_correct": (total_partially_correct / total) * 100,
      "correct_format": (total_correct_format / total) * 100,
  }


def safe_string_to_float(text):
  """Remove commas, spaces, and dollar signs before numeric conversion."""
  return text.replace(",", "").replace(" ", "").replace("$", "")


def score_response(target, prediction, debug=True):
  """Apply the source demo's numeric and response-format scoring."""
  is_correct, is_partially_correct, has_correct_format = False, False, False
  extracted_response = guess.group(1) if (guess := MATCH_ANSWER.search(prediction)) is not None else ""
  extracted_response = safe_string_to_float(extracted_response)
  target = safe_string_to_float(target)
  try:
    if float(extracted_response.strip()) == float(target.strip()):
      is_correct = True
    ratio = float(extracted_response.strip()) / float(target.strip())
    if 0.9 <= ratio <= 1.1:
      is_partially_correct = True
    if MATCH_FORMAT.search(prediction) is not None:
      has_correct_format = True
  except (ValueError, TypeError, ZeroDivisionError) as e:
    if debug:
      print("Evaluation exception: ", e)
  return is_correct, is_partially_correct, has_correct_format


def finetune_and_evaluate(model, args, train_dataset, test_dataset, processing_class, debug=True):
  """Evaluate, train, and reevaluate the same public model, including an attached PEFT adapter."""
  trainer = SFTTrainer(model=model, args=args, train_dataset=train_dataset, processing_class=processing_class)
  before = evaluate_model(test_dataset.batch(BATCH_SIZE), model, processing_class, debug=debug)
  output = trainer.train()
  after = evaluate_model(test_dataset.batch(BATCH_SIZE), trainer.model, processing_class, debug=debug)
  return trainer, output, before, after
