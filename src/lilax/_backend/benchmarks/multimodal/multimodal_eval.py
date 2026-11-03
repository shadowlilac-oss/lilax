"""Native ChartQA evaluation using the original multimodal benchmark behavior."""

import os
from dataclasses import dataclass, field
from datetime import datetime

import absl.logging
import datasets
import jax
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from lilax._backend.multimodal import processor as mm_processor
from lilax._backend.trainers.post_train.rl import utils_rl
from lilax._backend.utils import gcs_utils, max_logging

SUPPORTED_DATASETS = ["HuggingFaceM4/ChartQA"]
DEFAULT_PROMPT_TEMPLATE = """You are an expert at answering questions based
on provided charts. Your task is to extract the exact answer from the
given context or determine that it's not present.
For numerical answers, provide only the number.
For text answers, provide only the exact text.
For judgement questions, respond with "Yes" or "No".
If not found, output "N/A".
Your output must be only the exact answer within <answer></answer>, with no extra contents.

Example:
Question: What is the capital of France?
Your answer: <answer>Paris</answer>

Chart: {image_placeholder} Question: {question}
"""


@dataclass
class DebugConfig:
  rl: bool = False


@dataclass
class TmvpConfig:
  solution_start_token: str = "<answer>"
  solution_end_token: str = "</answer>"
  debug: DebugConfig = field(default_factory=DebugConfig)


@dataclass
class ParsedDatasetExample:
  question: str | None = None
  image_np: np.ndarray | None = None
  choices: list[str] | None = None
  answer: str | None = None


def parse_dataset_example(example, hf_dataset_name, image_resize=-1):
  if hf_dataset_name != "HuggingFaceM4/ChartQA":
    raise ValueError(f"Unsupported dataset: {hf_dataset_name}")
  parsed = ParsedDatasetExample(question=example["query"], image_np=np.asarray(example["image"].convert("RGB")),
                                answer=example["label"][0])
  if image_resize != -1:
    image = Image.fromarray(parsed.image_np).resize((image_resize, image_resize))
    parsed.image_np = np.asarray(image.convert("RGB"))
  return parsed


def construct_prompt(example, config, ckpt_type="base", system_message=None):
  image_placeholder = config.image_placeholder
  choices = "\n".join(f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(example.choices or []))
  if ckpt_type == "base":
    prompt = DEFAULT_PROMPT_TEMPLATE.format(image_placeholder=image_placeholder, question=example.question,
                                            choices=choices if choices else "N/A")
    if config.use_multimodal and "qwen" in str(config.decoder_block).lower():
      prompt = mm_processor.reformat_prompt(prompt, image_placeholder, config.model_name, num_images=1)
  elif ckpt_type == "sft":
    prompt = mm_processor.reformat_prompt(example.question, image_placeholder, config.model_name,
                                         num_images=1 if config.use_multimodal else 0)
  else:
    raise ValueError(f"Unsupported ckpt_type: {ckpt_type}")
  return system_message + "\n\n" + prompt if system_message else prompt


def validate_config(config):
  assert not config.load_full_state_path, "Decode doesn't operate on full states! Convert to parameter checkpoint first."
  assert config.hf_path, "For benchmark evaluation, please specify the HuggingFace dataset name using hf_path."
  assert config.hf_path in SUPPORTED_DATASETS, f"Unsupported dataset {config.hf_path}. Supported: {SUPPORTED_DATASETS}."


def evaluate(engine, params, config, *, num_examples=-1, tmp_results_file="mm_eval_results.csv",
             remove_tmp_results=True, ckpt_type="base", image_resize=-1):
  absl.logging.set_verbosity(absl.logging.INFO)
  validate_config(config)
  tokenizer = engine.build_tokenizer(engine.get_tokenizer())
  max_prefill_length = getattr(config, "max_prefill_predict_length", 1024)
  max_target_length = getattr(config, "max_target_length", 2048)
  eos_token_ids = {tokenizer.eos_id} if getattr(tokenizer, "eos_id", None) is not None else set()
  if hasattr(tokenizer, "stop_tokens"):
    eos_token_ids.update(tokenizer.stop_tokens)
  if hasattr(tokenizer, "tokenizer"):
    vocab = tokenizer.tokenizer.get_vocab()
    for token in ("<end_of_turn>", "<eos>", "<|im_end|>", "<|endoftext|>"):
      if token in vocab:
        eos_token_ids.add(vocab[token])

  timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  result_gcs_path = f"{config.base_output_directory}/{timestamp}.csv" if config.base_output_directory else None
  max_logging.log(f"Results will be saved to {tmp_results_file} and uploaded to GCS: {result_gcs_path}")
  results_data = []
  tmvp_config = TmvpConfig()
  correct_count = total_count = 0
  dataset = datasets.load_dataset(config.hf_path, "default", split=config.hf_eval_split)
  for idx, example in enumerate(tqdm(dataset, desc=f"Evaluating {config.hf_path} dataset")):
    parsed = parse_dataset_example(example, config.hf_path, image_resize)
    prompt = construct_prompt(parsed, config, ckpt_type)
    processor_output = mm_processor.preprocess_image_for_training(parsed.image_np, config)
    prefill_length = config.max_prefill_predict_length - mm_processor.get_image_offsets(config, processor_output)
    print("\n" + "*" * 50)
    is_bos = config.add_bos and getattr(tokenizer, "bos_id", None) is not None
    tokens, true_length = tokenizer.encode(prompt, is_bos=is_bos, prefill_lengths=[prefill_length])
    position_ids = mrope_position_deltas = None
    if config.use_multimodal:
      tokens = mm_processor.prepare_text_for_image_fusion(tokens=tokens, config=config, processor_output=processor_output)
      true_length += mm_processor.get_image_offsets(config=config, processor_output=processor_output)
      if config.use_mrope:
        from lilax._backend.multimodal import processor_qwen3_omni

        position_ids, mrope_position_deltas = processor_qwen3_omni.get_rope_index(
            input_ids=tokens[np.newaxis, :], image_grid_thw=processor_output.pixel_grid_thw,
            video_grid_thw=processor_output.video_grid_thw, attention_mask=np.ones_like(tokens)[np.newaxis, :],
            use_audio_in_video=config.use_audio and getattr(processor_output, "num_videos", 0) > 0,
            audio_lengths=processor_output.audio_lengths, second_per_grids=processor_output.video_second_per_grid,
            spatial_merge_size=config.spatial_merge_size_for_vit, position_id_per_seconds=config.position_id_per_seconds,
            config=config,
        )
    if true_length > max_prefill_length:
      max_logging.log(f"Warning: Prompt length {true_length} exceeds max prefill length {max_prefill_length}. Truncating.")
      tokens, true_length = tokens[:max_prefill_length], max_prefill_length
    assert config.quantization != "fp8", "fp8 on NVIDIA GPUs is not supported in decode.py yet"
    assert config.quantization != "nanoo_fp8", "NANOO fp8 on AMD MI300/MI325 GPUs is not supported in decode.py yet"
    prefill_result, first_token = engine.prefill(
        params=params, padded_tokens=tokens, positions=position_ids, mrope_deltas=mrope_position_deltas,
        images=processor_output.pixel_values if config.use_multimodal else None,
        image_masks=getattr(processor_output, "pixel_mask", None)
        if config.use_multimodal and "llama4" in config.model_name else None,
        audio_values=getattr(processor_output, "audio_values", None) if config.use_audio else None,
        audio_masks=getattr(processor_output, "audio_mask", None) if config.use_audio else None,
        true_length=true_length, slot=0,
    )
    decode_state = engine.insert(prefill_result, engine.init_decode_state(), slot=0)
    sampled_tokens = [first_token.get_result_at_slot(0).tokens.item()]
    predicted_answer = ""
    for _ in range(max_prefill_length, max_target_length):
      output = tokenizer.decode(sampled_tokens)
      predicted_answer = utils_rl.extract_answer(output, tmvp_config)
      if predicted_answer != utils_rl.FALLBACK_ANSWER:
        break
      decode_state, sampled_token = engine.generate(params, decode_state)
      sampled_tokens.append(sampled_token.get_result_at_slot(0).tokens.item())
      if sampled_tokens[-1] in eos_token_ids:
        break
    if predicted_answer == utils_rl.FALLBACK_ANSWER:
      predicted_answer = output
    is_correct, _ = utils_rl.check_correctness(predicted_answer, [parsed.answer], tmvp_config)
    max_logging.log(f"{total_count + 1} | {parsed.question}\n[Model output] {output}\n"
                    f"[Label answer] {parsed.answer}\nMatching: {is_correct}")
    results_data.append({"question ID": total_count + 1, "question": parsed.question, "label": parsed.answer,
                         "output": output, "is_correct": is_correct})
    if is_correct:
      correct_count += 1
    total_count += 1
    max_logging.log(f"Running accuracy: {correct_count / total_count:.4f} | Processed: {total_count}/{len(dataset)}")
    if num_examples != -1 and total_count >= num_examples:
      break
    if idx % 100 == 0 and result_gcs_path is not None and jax.process_index() == 0:
      pd.DataFrame(results_data).to_csv(tmp_results_file, index=False)
      gcs_utils.upload_blob(result_gcs_path, tmp_results_file)
      max_logging.log(f"Uploaded the results file to GCS bucket: {result_gcs_path}")

  if total_count:
    max_logging.log(f"\nFinal accuracy on {config.hf_path} dataset: {correct_count / total_count:.4f}")
  else:
    max_logging.log("No valid predictions were made.")
  if result_gcs_path is not None and jax.process_index() == 0:
    pd.DataFrame(results_data).to_csv(tmp_results_file, index=False)
    max_logging.log(f"Saved predictions to {tmp_results_file}")
    gcs_utils.upload_blob(result_gcs_path, tmp_results_file)
    max_logging.log(f"Uploaded the results file to GCS bucket: {result_gcs_path}")
    if remove_tmp_results and os.path.exists(tmp_results_file):
      os.remove(tmp_results_file)
      max_logging.log(f"Removed temporary results file: {tmp_results_file}")
  return {"accuracy": correct_count / total_count if total_count else None, "results": results_data,
          "results_path": result_gcs_path}
