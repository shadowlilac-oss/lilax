# Prepare specialized training corpora

These operations prepare files and may download large datasets or invoke external tools.

## Prepare LLaVA video metadata

This downloads metadata and writes Parquet. A GCS destination uses local staging and removes the staged folder after upload. Set `download_videos=True` only when video archive extraction is wanted.

```python
from lilax.data_operations.llava_video import prepare_llava_video_178k

prepare_llava_video_178k("./llava-metadata", download_videos=False)
```

## Generate response-based distillation records

```python
from lilax.data_operations.distillation_vllm import generate_distillation_data

generate_distillation_data(
	hf_access_token=None, teacher_model="Qwen/Qwen3-8B", output_file="./responses.parquet",
	use_chat_template=True,
)
```

This loads a vLLM teacher, generates assistant responses, and writes Parquet. It produces response records, not the offline top-k logits used by `TeacherLogitsDataset`.

## Copy C4 in Cloud Storage

This invokes `gcloud storage cp` and requires that tool and cloud access. Replace the destination with your bucket.

```python
from lilax.data_operations.c4 import copy_c4

copy_c4("gs://your-c4-destination")
```

See [corpus preparation reference](../../reference/data/data-preparation.md).
