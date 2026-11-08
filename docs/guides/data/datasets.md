# Load and prepare datasets

Use the sections below for ordinary dataset loading and preparation. The [standalone loader](../../reference/data/standalone-data.md) also exposes a configured batch-loading workload that allocates model and optimizer state.

Pass a Hugging Face `Dataset` or `IterableDataset` to a trainer. Choose the columns required by the training task.

## Load local or Hub records

For a JSON Lines file containing a `text` column:

```python
from lilax.data import load_dataset

dataset = load_dataset("json", data_files="./train.jsonl", split="train")
```

For a Hub dataset:

```python
dataset = load_dataset("trl-lib/Capybara", split="train")
```

## Stream records

```python
stream = load_dataset("json", data_files="./train.jsonl", split="train", streaming=True)
print(next(iter(stream)))
```

Streaming returns an iterable dataset. Keep evaluation finite so an evaluation pass can finish.

## Create records in memory

```python
from lilax.data import Dataset

dataset = Dataset.from_list([
	{"text": "A learning rate controls the size of an optimizer update."},
	{"text": "Gradient accumulation combines gradients across microbatches."},
])
```

See [SFT](../training/sft.md) for text and prompt-completion examples, [DPO](../training/dpo.md) for preferences, and [GRPO](../training/grpo.md) for reward-based records.

## Supply synthetic training input

```python
from lilax.data import SyntheticDataset

training_data = SyntheticDataset(reuse_example_batch=0)
```

Pass this descriptor as `train_dataset` to `Trainer`. Synthetic data is unbounded and raises `ValueError` for evaluation.

## Train a tokenizer from files

```python
from lilax.data import train_tokenizer_from_files

train_tokenizer_from_files(
	"./corpus/*.parquet", data_column="text", assets_path="./tokenizer-assets", vocab_model_name="corpus",
)
```

This trains SentencePiece assets from Parquet, ArrayRecord, or TFRecord. The tokenizer must match the model's vocabulary and token IDs. This operation does not resize a pretrained model.

See the [data reference](../../reference/data/data.md).
