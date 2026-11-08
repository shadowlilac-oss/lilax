# Prepare teacher generation requests

Use the data pipeline helpers to turn conversational examples into requests for a teacher service.

```python
from lilax import AutoTokenizer
from lilax.data import Dataset
from lilax.data_pipeline.api import filter_distillation_dataset, process_distillation_dataset

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
dataset = Dataset.from_list([{"messages": [
	{"role": "user", "content": "What does a checkpoint store?"},
	{"role": "assistant", "content": "It stores model state for later use."},
]}])
prepared = process_distillation_dataset(dataset, ["messages"])
requests = filter_distillation_dataset(
	prepared, tokenizer, max_target_length=512, max_prefill_length=256,
	use_chat_template=True,
)
```

The lengths illustrate request construction. Requests retain the original completion and a token limit for the generated answer. The helper keeps the first user text and first assistant text after conversion; it does not preserve a multi-turn conversation as a single prompt.

## Generate and upload teacher responses

The service workflow writes Parquet files and requires an upload destination. Supply `dataset_id` and `split` for your conversational Hub dataset, `server_address` for your teacher, `output_repo` for the destination Hub dataset, `token` for Hub access, and your selected `batch_size` and `batches` count.

Run a compatible JetStream teacher at the supplied address. The following sends generation requests, writes local files, and uploads dataset commits. Local dataset files remain by default; set `remove_local_dataset_files=True` to remove them after upload. A missing Hub dataset repository is created as private.

```python
from lilax.data_pipeline.api import generate_distillation_data

generate_distillation_data(
	dataset_id, split, ["messages"], "Qwen/Qwen3-8B", batch_size, batches,
	hf_access_token=token, server_address=server_address,
	upload="upload-to-hf", hf_repo_id=output_repo,
)
```

See [request and upload reference](../../reference/data/distillation-requests.md).
