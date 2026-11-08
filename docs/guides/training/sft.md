# Supervised fine-tuning

Train a model on text or prompt-completion records with `SFTTrainer`.

## Train on text

Constructing `SFTTrainer` initializes runtime placement and the training mesh before `train()`, while weight restoration remains deferred. Set device and mesh options first. See [runtime controls](../../reference/execution/runtime.md).

```python
from lilax import AutoModelForCausalLM, AutoTokenizer, SFTConfig, SFTTrainer
from lilax.data import Dataset

model_id = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = Dataset.from_list([
	{"text": "Gradient accumulation combines gradients across microbatches."},
	{"text": "A checkpoint lets you resume a training run."},
])
args = SFTConfig(output_dir="./sft", max_steps=10, per_device_train_batch_size=1, max_length=256, packing=False)
trainer = SFTTrainer(model, args=args, train_dataset=dataset, processing_class=tokenizer)
result = trainer.train()
trainer.save_model("./sft-model")
```

The return value has this illustrative shape. Angle brackets stand for values from your run, not measured results:

```text
TrainOutput(global_step=<completed optimizer steps>, training_loss=<mean loss>, metrics={"train_loss": <mean loss>})
```

Read the loss with `result.metrics["train_loss"]`. These short records and numerical choices illustrate API use.

## Choose a record layout

Text SFT recognizes these column layouts. Keep exactly one matching layout in the dataset, or supply `formatting_func`.

| Columns | Example record | Preparation and loss boundary |
| --- | --- | --- |
| `text`, or `dataset_text_field` | `{"text": "A checkpoint stores training state."}` | One text sequence. All text is completion text, even with `completion_only_loss=True`. |
| `prompt`, `completion` as strings | `{"prompt": "What is a checkpoint?", "completion": "Saved state."}` | Concatenates the strings and marks the prompt boundary. |
| `prompt`, `completion` as message lists | `{"prompt": [{"role": "user", "content": "Hello"}], "completion": [{"role": "assistant", "content": "Hi"}]}` | Combines the messages and applies the tokenizer's chat template. |
| `question`, `answer` as strings | `{"question": "What is a checkpoint?", "answer": "Saved state."}` | Converts the pair to user and assistant messages, then applies the chat template. |
| `messages` | `{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]}` | Applies the chat template and identifies prompt and assistant completion segments. |

For media, use [paired SFT records](../data/processing.md#supply-paired-records-for-sft).

## Use prompt-completion records

Replace the dataset above with explicit prompt and completion columns:

```python
dataset = Dataset.from_list([
	{"prompt": "What is gradient accumulation?", "completion": "Combining gradients across microbatches."},
	{"prompt": "What is a checkpoint?", "completion": "Saved state used to resume training."},
])
trainer = SFTTrainer(model, args=args, train_dataset=dataset, processing_class=tokenizer)
```

`completion_only_loss=True` masks prompt targets. Set it to `False` to train on prompt tokens too.

## Use conversations

```python
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = Dataset.from_list([{"messages": [
	{"role": "user", "content": "What is a checkpoint?"},
	{"role": "assistant", "content": "Saved training state."},
]}])
trainer = SFTTrainer(model, args=args, train_dataset=dataset, processing_class=tokenizer)
```

The `messages` column selects conversational preparation. Keep any system message at index zero. `tokenizer.chat_template` supplies the formatting contract and identifies prompt and assistant completion segments. If the tokenizer has no template, assign the model's actual template to `tokenizer.chat_template` before constructing the trainer. `use_chat_template=True` does not select this SFT path.

## Load conversations from the Hub

Using the model, tokenizer, and arguments above, load the `messages` column from [Capybara](https://huggingface.co/datasets/trl-lib/Capybara):

```python
from lilax.data import load_dataset

dataset = load_dataset("trl-lib/Capybara", split="train").select_columns(["messages"])
trainer = SFTTrainer(model, args=args, train_dataset=dataset, processing_class=tokenizer)
result = trainer.train()
print(result.global_step, result.metrics["train_loss"])
```

The printed values come from your run. Inspect formatted text before training with `tokenizer.apply_chat_template(dataset[0]["messages"], tokenize=False)`.

## Adapt your text columns

For a single text column with another name, set `dataset_text_field`:

```python
dataset = Dataset.from_list([{"content": "A checkpoint stores training state."}])
args = SFTConfig(output_dir="./sft", dataset_text_field="content", max_steps=10, packing=False)
trainer = SFTTrainer(model, args=args, train_dataset=dataset, processing_class=tokenizer)
```

For computed text, supply a formatting function:

```python
dataset = Dataset.from_list([{"question": "What is a checkpoint?", "answer": "Saved training state."}])

def format_record(record):
	return record["question"] + "\n" + record["answer"]

trainer = SFTTrainer(model, args=args, train_dataset=dataset, processing_class=tokenizer, formatting_func=format_record)
```

A formatting function returns a string or list of strings. It replaces the original columns with one text field, so all returned text is trained as completion text. Use the native `question` and `answer` layout above to retain the prompt boundary. For paired media, transform the dataset before training instead. See [paired image and video SFT records](../data/processing.md#supply-paired-records-for-sft) for the supported columns and processor setup.

## Pack and evaluate

`packing=True` combines examples into training sequences. `max_length` controls SFT sequence length through packing or pad-or-trim preparation. `None` keeps the configured model length. `use_truncation=True` does not enable tokenizer truncation for SFT. Supply a finite `eval_dataset` and positive `eval_steps` to evaluate during training. Call `trainer.evaluate()` for an explicit evaluation pass.

See [logged metrics](../../reference/training/trainers.md#logged-metrics) for training history and tracker output.

## Evaluate before and after SFT with the public demo

`lilax.trainers.examples.sft_demo` downloads and caches GSM8K, evaluates a model, trains it in place, and evaluates it again. Start a separate setup before materializing the model. Reuse your training `dataset` in an SFT record format above, but create fresh model, tokenizer, and argument objects:

```python
from lilax import AutoConfig, AutoModelForCausalLM, AutoTokenizer, SFTConfig
from lilax.trainers.examples.sft_demo import finetune_and_evaluate, get_test_dataset

model_id = "Qwen/Qwen3-8B"
config = AutoConfig.from_pretrained(model_id)
config.max_position_embeddings = 1024
model = AutoModelForCausalLM.from_pretrained(model_id, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.runtime.initialize(overrides={"max_prefill_predict_length": 256})
args = SFTConfig(
	output_dir="./sft-demo", max_steps=10, per_device_train_batch_size=1, max_length=1024, packing=False,
)
test_dataset = get_test_dataset(tokenizer).filter(
	lambda record: len(tokenizer(record["prompt"])["input_ids"]) <= 256,
)
trainer, output, before, after = finetune_and_evaluate(
	model, args, dataset, test_dataset, tokenizer, debug=False,
)
print(before, after)
```

The explicit 256-token prefill capacity and 1024-token target accommodate the helper's fixed request for 768 new tokens. A target of 1024 alone is insufficient if the configured prefill capacity is larger. Changing `args` after a model graph exists does not resize that graph. Training allocates model and optimizer state on the configured devices and uses `args` for output and checkpoints.

The evaluator raises `ValueError` for any formatted prompt over 256 tokens. `get_test_dataset` does not filter those records, so the example filters the tokenized prompts before evaluation. Use a nonempty filtered dataset. The full GSM8K split is not guaranteed to fit. To train adapters, attach the [LoRA adapter](lora.md#evaluate-before-and-after-sft-with-the-public-demo) to this fresh model before the helper call.

Each evaluation dictionary contains `correct`, `partially_correct`, and `correct_format` percentages from your run. For evaluation alone, pass a nonempty batched dataset such as `test_dataset.batch(4)` to `evaluate_model`. See the [demo reference](../../reference/training/sft-demo.md) for fixed generation settings, prompt limits, result fields, and scoring rules.

Continue with [LoRA](lora.md), [callbacks](callbacks.md), or [saving and resuming](checkpoints.md). See [trainer reference](../../reference/training/trainers.md) for arguments and limits.
