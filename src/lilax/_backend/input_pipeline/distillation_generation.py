"""Generate conversational response-based distillation datasets."""

import asyncio
import json
import os

import grpc
from datasets import Dataset
from huggingface_hub import create_repo, get_full_repo_name, repo_exists, upload_file
from jetstream.core.proto import jetstream_pb2, jetstream_pb2_grpc
from transformers import AutoTokenizer
import tqdm.asyncio

from lilax._backend.input_pipeline import distillation_data_processing
from lilax._backend.utils import gcs_utils, max_logging


async def _send_request(request, stub, tokenizer, num_generations, progress_bar):
  decode_request = jetstream_pb2.DecodeRequest(
      token_content=jetstream_pb2.DecodeRequest.TokenContent(token_ids=request.prompt_token_ids),
      max_tokens=request.max_output_tokens,
      num_samples=num_generations,
      has_bos=True,
  )
  completion_tokens = [[] for _ in range(num_generations)]
  async for response in stub.Decode(decode_request):
    for index, sample in enumerate(response.stream_content.samples):
      completion_tokens[index].extend(sample.token_ids)

  outputs = []
  for tokens in completion_tokens:
    completion = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    outputs.append({
        "prompt": [{"role": "user", "content": request.prompt}],
        "completion": [{"role": "assistant", "content": completion}],
        "actual_completion": [{"role": "assistant", "content": request.actual_completion}],
    })
  progress_bar.update(1)
  return outputs


async def _run_inference(requests, tokenizer, num_generations, server_address):
  progress_bar = tqdm.asyncio.tqdm(total=len(requests))
  progress_bar.set_description(f"Running inference on {len(requests)} prompts")
  service_config = json.dumps({
      "methodConfig": [{
          "name": [{}],
          "retryPolicy": {
              "maxAttempts": 5,
              "initialBackoff": "0.2s",
              "maxBackoff": "1s",
              "backoffMultiplier": 2,
              "retryableStatusCodes": ["UNAVAILABLE"],
          },
      }],
  })
  options = [
      ("grpc.keepalive_timeout_ms", 10000),
      ("grpc.enable_retries", 1),
      ("grpc.service_config", service_config),
  ]
  async with grpc.aio.insecure_channel(server_address, options=options) as channel:
    stub = jetstream_pb2_grpc.OrchestratorStub(channel)
    outputs = await asyncio.gather(*(
        _send_request(request, stub, tokenizer, num_generations, progress_bar) for request in requests
    ))
  progress_bar.close()
  return [output for outputs_for_prompt in outputs for output in outputs_for_prompt]


def generate_distillation_data(
    dataset_path, data_split, data_columns, tokenizer_path, batch_size, num_batches, *,
    hf_access_token=None, dataset_type="huggingface", use_chat_template=False,
    max_prefill_length=256, max_target_length=2048, num_generations=1,
    server_address="localhost:9000", upload, hf_repo_id=None, gcs_bucket=None,
    gcs_data_path=None, remove_local_dataset_files=False,
):
  """Generate and upload one Parquet dataset per batch using a JetStream server."""
  assert max_prefill_length < max_target_length, "Maximum length of prompt should be less than maximum target length."
  if upload == "upload-to-hf":
    if hf_repo_id is None:
      raise ValueError("hf_repo_id is required for upload-to-hf")
  elif upload == "upload-to-gcs":
    if gcs_bucket is None or gcs_data_path is None:
      raise ValueError("gcs_bucket and gcs_data_path are required for upload-to-gcs")
  else:
    raise ValueError("upload must be upload-to-hf or upload-to-gcs")
  dataset = distillation_data_processing.load_dataset(dataset_path, data_split, hf_access_token, dataset_type)
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token=hf_access_token)

  for batch_num in range(num_batches):
    start = batch_num * batch_size
    batch = Dataset.from_dict(dataset[start : start + batch_size])
    batch = distillation_data_processing.process_dataset(batch, data_columns)
    requests = distillation_data_processing.filter_dataset(
        batch, tokenizer, max_target_length, max_prefill_length, use_chat_template
    )
    data = asyncio.run(_run_inference(requests, tokenizer, num_generations, server_address))
    filename = f"distillation-data-{batch_num}.parquet"
    Dataset.from_list(data).to_parquet(filename)
    if upload == "upload-to-hf":
      full_repo_name = get_full_repo_name(model_id=hf_repo_id, token=hf_access_token)
      if not repo_exists(repo_id=full_repo_name, repo_type="dataset", token=hf_access_token):
        create_repo(repo_id=hf_repo_id, repo_type="dataset", private=True, token=hf_access_token)
      upload_file(
          repo_id=full_repo_name, repo_type="dataset", path_or_fileobj=filename,
          path_in_repo=f"data/{filename}", commit_message=f"Uploading dataset batch number {batch_num}",
          token=hf_access_token,
      )
    elif upload == "upload-to-gcs":
      data_path = gcs_utils.add_trailing_slash(gcs_data_path)
      gcs_utils.upload_blob(f"gs://{gcs_bucket}/{data_path}{filename}", filename)
    if remove_local_dataset_files and os.path.exists(filename):
      try:
        os.remove(filename)
      except OSError as error:
        max_logging.log(f"Unable to remove local dataset file {filename}: {error}")
