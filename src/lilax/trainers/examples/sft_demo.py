"""GSM8K evaluation and LoRA SFT workflow from the MaxText demo notebooks."""

import re

import datasets
import grain
import jax
from flax import nnx
from tqdm.auto import tqdm
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout.vllm_rollout import VllmRollout

from lilax._backend.checkpoint_conversion.to_huggingface import _get_lora_delta
from lilax._backend.input_pipeline import instruction_data_processing
from lilax._backend.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from lilax._backend.trainers.post_train.sft import train_sft


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


def get_test_dataset(config, tokenizer, data_template_path):
  """Load and format the shuffled GSM8K test split as a Grain map dataset."""
  template_config = instruction_data_processing.load_data_template_from_file(data_template_path)
  dataset = datasets.load_dataset(
      DATASET_NAME, data_dir=DATASET_DATA_DIR, split=DATASET_TEST_SPLIT, token=config.hf_access_token,
  )
  return grain.MapDataset.source(dataset).shuffle(seed=SEED).map(
      lambda x: {
          "question": x["question"],
          "prompt": tokenizer.apply_chat_template(
              [{"role": "user", "content": template_config["PROMPT_TEMPLATE"].format(question=x["question"].strip())}],
              tokenize=False, add_generation_prompt=True,
          ),
          "target_answer": instruction_data_processing.extract_reasoning_and_answer(
              x["answer"], template_config["REASONING_ANSWER_SEPARATOR"]
          )[1],
      }
  )


def evaluate_model(dataset, vllm_rollout, debug=True):
  """Return the demo's exact, within-10-percent, and format percentages."""
  rollout_config = base_rollout.RolloutConfig(
      max_tokens_to_generate=MAX_TOKENS_TO_GENERATE, max_prompt_length=MAX_PROMPT_LENGTH,
      temperature=EVALUATION_CONFIG["temperature"], top_p=EVALUATION_CONFIG["top_p"],
      top_k=EVALUATION_CONFIG["top_k"], data_type="bfloat16",
  )
  total, total_correct, total_partially_correct, total_correct_format = 0, 0, 0, 0
  for batch in tqdm(dataset):
    batch_response = vllm_rollout.generate(batch["prompt"], rollout_config)
    for i, question in enumerate(batch["question"]):
      if debug:
        print("========================================")
        print(f"Question: {question}")
        print("----------------------------------------")
        print(f"Model Generated Response: {batch_response.text[i]}")
        print("----------------------------------------")
        print(f"Target Response: {batch['target_answer'][i]}")
        print("========================================")
      is_correct, is_partially_correct, has_correct_format = score_response(
          target=batch["target_answer"][i], prediction=batch_response.text[i], debug=debug
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


def setup_trainer(trainer):
  """Prepare a public Tunix SFTTrainer for the notebook's pre-training evaluation."""
  train_iterator, eval_iterator = trainer._data_iterators()
  trainer.native_trainer, mesh = train_sft.setup_trainer_state(
      trainer.config, model=trainer.model.native_model, mesh=trainer.model.runtime.mesh,
      train_iterator=train_iterator, eval_iterator=eval_iterator,
  )
  trainer.model.native_model = trainer.native_trainer.model
  return trainer.native_trainer, mesh


def create_rollout(trainer, tokenizer, model_version, mesh):
  """Create the notebook's Tunix vLLM rollout for a prepared native trainer."""
  return VllmRollout(
      model=TunixMaxTextAdapter(trainer.model), tokenizer=tokenizer, cache_config_or_size=1280, mesh=mesh,
      rollout_config=base_rollout.RolloutConfig(
          rollout_vllm_model_version=model_version, rollout_vllm_hbm_utilization=0.3,
          rollout_vllm_init_with_random_weights=True, rollout_vllm_tpu_backend_type="jax",
          data_type=jax.numpy.bfloat16, max_tokens_to_generate=512,
      ),
  )


def train_trainer(trainer, mesh):
  """Train the prepared SFTTrainer and enforce the notebook's step count."""
  trainer.native_trainer = train_sft.train_model(trainer.config, trainer.native_trainer, mesh)
  trainer.model.native_model = trainer.native_trainer.model
  if trainer.native_trainer.train_steps < trainer.config.steps:
    raise RuntimeError(
        f"Training ended prematurely! Expected {trainer.config.steps} steps, "
        f"but only completed {trainer.native_trainer.train_steps} steps."
    )
  return trainer.native_trainer


def sync_lora_for_evaluation(trainer, config, vllm_rollout):
  """Apply the notebook's in-place state merge and hand the filtered state to vLLM.

  This retains the existing LoRA graph and its interception; it does not unload the adapter.
  Calling it again applies the delta again.
  """
  _, state = nnx.split(trainer.model)
  flat_params = jax.device_put(state.to_pure_dict(), jax.devices("cpu")[0])
  scaling = config.lora.lora_alpha / config.lora.lora_rank
  official_dict = {}
  for path, val in jax.tree_util.tree_flatten_with_path(flat_params)[0]:
    p_names = [str(p.key if hasattr(p, "key") else str(p).strip("['']")) for p in path]
    official_dict["params-" + "-".join(p_names)] = val
  merge_count = 0
  for off_key, val in official_dict.items():
    if off_key.endswith("-kernel"):
      delta = _get_lora_delta(off_key, official_dict, scaling)
      if delta is not None:
        merged_val = val + delta.reshape(val.shape) if delta.size == val.size else val + delta
        original_path = off_key.replace("params-", "").split("-")
        curr = flat_params
        for part in original_path[:-1]:
          curr = curr[part]
        curr[original_path[-1]] = merged_val
        merge_count += 1
        print(f"[Merged on CPU] {off_key}")
  nnx.update(trainer.model, nnx.State(jax.device_put(flat_params)))
  full_updated_state = nnx.state(trainer.model)
  clean_state = nnx.State({k: v for k, v in full_updated_state.items() if "lora" not in str(k)})
  vllm_rollout.update_params(clean_state)
  return merge_count
