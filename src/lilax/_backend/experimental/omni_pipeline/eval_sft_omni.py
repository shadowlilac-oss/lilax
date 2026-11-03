"""Omni multimodal checkpoint evaluation."""

from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path

import datasets
import jax
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from lilax._backend.configs.runtime import resolve_runtime_config
from lilax._backend.inference.maxengine import maxengine
from lilax._backend.multimodal import processor as mm_processor
from lilax._backend.trainers.post_train.rl import utils_rl
from lilax._backend.utils import gcs_utils, max_logging
from lilax._backend.utils.globals import MAXTEXT_PKG_DIR
from lilax.configuration import MaxTextConfig


@dataclass
class DebugConfig:
  rl: bool = False


@dataclass
class TmvpConfig:
  solution_start_token: str = "<answer>"
  solution_end_token: str = "</answer>"
  debug: DebugConfig = field(default_factory=DebugConfig)


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


def evaluate_sft_omni(
    config_path=None, *, num_examples=-1, tmp_results_file="omni_eval_results.csv", remove_tmp_results=True,
    ckpt_type="sft", image_resize=-1, split="test", **overrides,
):
  """Evaluate an Omni checkpoint against the native multimodal benchmark."""
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  if config_path is None:
    config_path = Path(MAXTEXT_PKG_DIR) / "experimental" / "omni_pipeline" / "configs" / "sft-maxtext-omni-gemma3-qwen3.yml"
  options = {
      "override_model_config": True,
      "per_device_batch_size": 1,
      "async_checkpointing": False,
      "hf_eval_files": "",
      "hf_train_files": "",
      "hf_path": "HuggingFaceM4/ChartQA",
      "hf_eval_split": split or "test",
      **overrides,
  }
  config = resolve_runtime_config(MaxTextConfig(base_config=config_path, **options))
  if not config.load_parameters_path:
    raise ValueError("Please specify a checkpoint path using load_parameters_path=<path>.")
  if getattr(config, "hf_access_token", ""):
    os.environ["HF_TOKEN"] = config.hf_access_token
  assert not config.load_full_state_path, (
      "Decode doesn't operate on full states! Convert to parameter checkpoint"
      " first. Using generate_param_only_checkpoint."
  )
  assert config.hf_path, "For benchmark evaluation, please specify the HuggingFace dataset name using the hf_path config field."
  assert config.hf_path in ["HuggingFaceM4/ChartQA"], (
      f"Unsupported dataset {config.hf_path}. Supported datasets are: ['HuggingFaceM4/ChartQA']."
      " Please add support for your desired dataset in the code of multimodal_eval.py."
  )

  engine = maxengine.MaxEngine(config)
  params = engine.load_params()
  tokenizer = engine.build_tokenizer(engine.get_tokenizer())
  max_prefill_predict_length = getattr(config, "max_prefill_predict_length", 1024)
  max_target_length = getattr(config, "max_target_length", 2048)
  eos_token_ids = {tokenizer.eos_id} if getattr(tokenizer, "eos_id", None) is not None else set()
  if hasattr(tokenizer, "stop_tokens"):
    eos_token_ids.update(tokenizer.stop_tokens)
  if hasattr(tokenizer, "tokenizer"):
    vocab = tokenizer.tokenizer.get_vocab()
    for token in ("<end_of_turn>", "<eos>", "<|im_end|>", "<|endoftext|>"):
      if token in vocab:
        eos_token_ids.add(vocab[token])

  correct_count = 0
  total_count = 0
  timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  result_gcs_path = f"{config.base_output_directory}/{timestamp}.csv" if config.base_output_directory else None
  max_logging.log(f"Results will be saved to {tmp_results_file} and uploaded to GCS: {result_gcs_path}")
  results_data = []
  tmvp_config = TmvpConfig()
  test_ds = datasets.load_dataset(config.hf_path, "default", split=config.hf_eval_split)
  for idx, example in enumerate(tqdm(test_ds, desc=f"Evaluating {config.hf_path} dataset")):
    question = example["query"]
    image_np = np.asarray(example["image"].convert("RGB"))
    answer = example["label"][0]
    if image_resize != -1:
      image = Image.fromarray(image_np).resize((image_resize, image_resize))
      image_np = np.asarray(image.convert("RGB"))
    if ckpt_type == "base":
      prompt = DEFAULT_PROMPT_TEMPLATE.format(image_placeholder=config.image_placeholder, question=question, choices="N/A")
      if config.use_multimodal and "qwen" in str(config.decoder_block).lower():
        prompt = mm_processor.reformat_prompt(prompt, config.image_placeholder, config.model_name, num_images=1)
    elif ckpt_type == "sft":
      prompt = mm_processor.reformat_prompt(
          question, config.image_placeholder, config.model_name, num_images=1 if config.use_multimodal else 0,
      )
    else:
      raise ValueError(f"Unsupported ckpt_type: {ckpt_type}")

    processor_output = mm_processor.preprocess_image_for_training(image_np, config)
    prefill_length = config.max_prefill_predict_length - mm_processor.get_image_offsets(
        config=config, processor_output=processor_output,
    )
    print("\n" + "*" * 50)
    is_bos = config.add_bos and getattr(tokenizer, "bos_id", None) is not None
    tokens, true_length = tokenizer.encode(prompt, is_bos=is_bos, prefill_lengths=[prefill_length])
    position_ids = None
    mrope_position_deltas = None
    if config.use_multimodal:
      tokens = mm_processor.prepare_text_for_image_fusion(tokens=tokens, config=config, processor_output=processor_output)
      true_length += mm_processor.get_image_offsets(config=config, processor_output=processor_output)
      if config.use_mrope:
        # Loads the decord/GCS video dependencies here, as the former processor import did.
        from lilax._backend.multimodal.qwen3_omni.video import loading  # pylint: disable=unused-import
        from lilax._backend.multimodal.qwen3_omni import positions

        position_ids, mrope_position_deltas = positions.get_rope_index(
            input_ids=tokens[np.newaxis, :], image_grid_thw=processor_output.pixel_grid_thw,
            video_grid_thw=processor_output.video_grid_thw, attention_mask=np.ones_like(tokens)[np.newaxis, :],
            use_audio_in_video=config.use_audio and getattr(processor_output, "num_videos", 0) > 0,
            audio_lengths=processor_output.audio_lengths, second_per_grids=processor_output.video_second_per_grid,
            spatial_merge_size=config.spatial_merge_size_for_vit,
            position_id_per_seconds=config.position_id_per_seconds, config=config,
        )
    if true_length > max_prefill_predict_length:
      max_logging.log(
          f"Warning: Prompt length {true_length} exceeds max prefill length {max_prefill_predict_length}. Truncating."
      )
      tokens = tokens[:max_prefill_predict_length]
      true_length = max_prefill_predict_length
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
    slot = 0
    decode_state = engine.insert(prefill_result, engine.init_decode_state(), slot=slot)
    sampled_tokens = [first_token.get_result_at_slot(slot).tokens.item()]
    predicted_answer = ""
    for _ in range(max_prefill_predict_length, max_target_length):
      output = tokenizer.decode(sampled_tokens)
      predicted_answer = utils_rl.extract_answer(output, tmvp_config)
      if predicted_answer != utils_rl.FALLBACK_ANSWER:
        break
      decode_state, sampled_token = engine.generate(params, decode_state)
      sampled_tokens.append(sampled_token.get_result_at_slot(slot).tokens.item())
      if sampled_tokens[-1] in eos_token_ids:
        break
    if predicted_answer == utils_rl.FALLBACK_ANSWER:
      predicted_answer = output
    is_correct, _ = utils_rl.check_correctness(predicted_answer, [answer], tmvp_config)
    max_logging.log(
        f"{total_count + 1} | {question}\n[Model output] {output}\n[Label answer] {answer}\nMatching: {is_correct}"
    )
    results_data.append({"question ID": total_count + 1, "question": question, "label": answer,
                         "output": output, "is_correct": is_correct})
    if is_correct:
      correct_count += 1
    total_count += 1
    max_logging.log(f"Running accuracy: {correct_count / total_count:.4f} | Processed: {total_count}/{len(test_ds)}")
    if num_examples != -1 and total_count >= num_examples:
      break
    if idx % 100 == 0 and result_gcs_path is not None and jax.process_index() == 0:
      pd.DataFrame(results_data).to_csv(tmp_results_file, index=False)
      gcs_utils.upload_blob(result_gcs_path, tmp_results_file)
      max_logging.log(f"Uploaded the results file to GCS bucket: {result_gcs_path}")
  if total_count > 0:
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
