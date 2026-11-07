"""MLPerf OpenOrca summarization dataset."""

from __future__ import annotations

import datasets as hf_datasets

from lilax._backend.eval.datasets.base import BenchmarkDataset, SampleRequest

_SYSTEM_PROMPT = "You are a helpful assistant. Summarize the following conversation."


class MlperfOpenOrcaDataset(BenchmarkDataset):
  name = "mlperf_openorca"

  def sample_requests(self, num_samples, tokenizer, *, dataset=None) -> list[SampleRequest]:
    ds = hf_datasets.load_dataset("Open-Orca/OpenOrca", split="train", streaming=True) if dataset is None else dataset
    requests = []
    for row in ds:
      if not row.get("response", "").strip():
        continue
      system_prompt = row.get("system_prompt", _SYSTEM_PROMPT) or _SYSTEM_PROMPT
      question = row["question"]
      reference = row["response"]
      if tokenizer is not None:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
      else:
        prompt = f"{system_prompt}\n\nUser: {question}\nAssistant:"
      requests.append(SampleRequest(prompt=prompt, reference=reference))
      if num_samples is not None and len(requests) >= num_samples:
        break
    return requests
