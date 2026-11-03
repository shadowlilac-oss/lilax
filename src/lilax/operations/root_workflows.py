"""ShareGPT workload estimates from inference microbenchmark timings."""

import json


PREFILL_BUCKET_SIZE_TO_MS = {64: 9.174, 128: 11.087, 256: 18.468, 512: 29.128, 1024: 58.386}
SYSTEM_TIME_PER_DECODE_TOKEN_MS = 0.32591875
MAX_INPUT_TOKENS = 1024
MAX_OUTPUT_TOKENS = 1024


def next_power_of_2(x):
  return 1 if x == 0 else 2 ** (x - 1).bit_length()


def tokens_in_input_str(s):
  return int(1.3 * len(s.split()))


def get_prefill_and_generate_times(filename=""):
  if filename == "":
    return PREFILL_BUCKET_SIZE_TO_MS, SYSTEM_TIME_PER_DECODE_TOKEN_MS

  with open(filename, "rt", encoding="utf8") as f:
    microbenchmark_results = json.load(f)
  prefill_bucket_size_to_ms = {
      int(k): round(v["prefill_time_in_ms"], 3) for k, v in microbenchmark_results["Prefill"].items()
  }
  return prefill_bucket_size_to_ms, microbenchmark_results["AutoRegressive"]["ar_step_in_ms_per_seq"]


def get_conversations_from_file(filename, max_input_tokens, max_output_tokens):
  with open(filename, "rt", encoding="utf8") as f:
    loaded_share_gpt = json.load(f)
  convo_token_numbers = []
  for example in loaded_share_gpt:
    if len(example["conversations"]) < 2:
      continue
    num_input_tokens = tokens_in_input_str(example["conversations"][0]["value"])
    num_output_tokens = tokens_in_input_str(example["conversations"][1]["value"])
    convo_token_numbers.append((num_input_tokens, num_output_tokens))

  num_convos = len(convo_token_numbers)
  kept_convos = [c for c in convo_token_numbers if c[0] <= max_input_tokens and c[1] <= max_output_tokens]
  mean_input = sum(c[0] for c in kept_convos) / len(kept_convos)
  mean_output = sum(c[1] for c in kept_convos) / len(kept_convos)
  print(f"Kept {len(kept_convos)} of {num_convos} total convos. {len(100 * kept_convos) / num_convos:.3f}%")
  print(f"Out of kept convos, mean input tokens: {mean_input:.3f}, mean output tokens: {mean_output:.3f}")
  return kept_convos


def compute_times(conversations, prefill_bucket_size_to_ms, system_time_per_decode_token_ms, verbose=False):
  total_prefill_system_ms = 0
  total_generate_system_ms = 0
  for input_tok, output_tok in conversations:
    bucket = max(128, next_power_of_2(input_tok))
    generate_system_ms = output_tok * system_time_per_decode_token_ms
    prefill_system_ms = prefill_bucket_size_to_ms[bucket]
    total_prefill_system_ms += prefill_system_ms
    total_generate_system_ms += generate_system_ms
    if verbose:
      print(f"{(input_tok, output_tok)} {bucket}, {prefill_system_ms:.2f}, {generate_system_ms:.2f}")

  total_prefill_time_seconds = total_prefill_system_ms / 1000
  total_generate_time_seconds = total_generate_system_ms / 1000
  total_time_s = total_prefill_time_seconds + total_generate_time_seconds
  print(
      f"\nTotal time {total_time_s:.3f} seconds: "
      f"\n\tPrefill time: {total_prefill_time_seconds:.3f} seconds"
      f"\n\tGenerate time: {total_generate_time_seconds:.3f} seconds"
  )
  return total_time_s, total_prefill_time_seconds, total_generate_time_seconds


def get_num_tokens_in_convos(conversations):
  return sum(c[0] for c in conversations), sum(c[1] for c in conversations)


def analyze_sharegpt(convo_file, mb_timing_file="", verbose=False):
  """Estimate a ShareGPT workload using the original microbenchmark timing format."""
  conversations = get_conversations_from_file(convo_file, MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS)
  total_input_tokens, total_output_tokens = get_num_tokens_in_convos(conversations)
  prefill_times, generate_time = get_prefill_and_generate_times(mb_timing_file)
  total_time, prefill_time, generation_time = compute_times(conversations, prefill_times, generate_time, verbose)
  print(f"Output {total_output_tokens} tokens in {total_time:.3f} seconds = {total_output_tokens / total_time:.3f} out tok/s")
  return total_input_tokens, total_output_tokens, total_time, prefill_time, generation_time
