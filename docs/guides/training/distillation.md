# Distill a teacher model

Train a student using a teacher model or saved teacher logits.

## Train with an online teacher

Training writes checkpoints under `output_dir`. The final `save_model` call replaces Lilax-owned native weights and updates named configuration and processing files while preserving unrelated destination files.

Constructing `DistillationTrainer` initializes the student's training mesh and the teacher runtime before `train()`, while weight restoration remains deferred. Set device and mesh options first. See [runtime controls](../../reference/execution/runtime.md).

```python
from lilax import AutoModelForCausalLM, AutoTokenizer, DistillationConfig, DistillationTrainer
from lilax.data import Dataset

model_id = "Qwen/Qwen3-8B"
student = AutoModelForCausalLM.from_pretrained(model_id)
teacher = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = Dataset.from_list([{"text": "A checkpoint preserves training state."}])
args = DistillationConfig(output_dir="./distillation", max_steps=10, per_device_train_batch_size=1)
trainer = DistillationTrainer(student, teacher, args=args, train_dataset=dataset, processing_class=tokenizer)
result = trainer.train()
trainer.save_model("./student")
```

The return value has this illustrative shape. Angle brackets represent values from your run:

```text
TrainOutput(global_step=<completed optimizer steps>, training_loss=<mean loss>, metrics={"train_loss": <mean loss>})
```

Read the loss with `result.metrics["train_loss"]`. See [logged metrics](../../reference/training/trainers.md#logged-metrics) for history and tracker output.

The same supported model is used twice to demonstrate wiring. Account for both models' memory. Student and teacher must use the same mesh and compatible batch settings.

## Save teacher logits

Before training, use a trainer with an online teacher to produce offline records. This writes ArrayRecord files under `local_tmp_dir` and uploads them if `gcs_upload_path` is supplied:

```python
trainer.generate_teacher_logits(dataset=dataset, top_k=128, local_tmp_dir="./teacher-logits")
```

`steps_per_file`, upload workers, and serialization workers control record output. `gcs_upload_path` optionally selects a cloud destination.

## Train from saved records

Use an ArrayRecord file pattern for the generated files:

```python
from lilax.data import TeacherLogitsDataset

student = AutoModelForCausalLM.from_pretrained(model_id)
offline = DistillationTrainer(
	student, None, args=args, train_dataset=TeacherLogitsDataset("./teacher-logits/*.array_record"),
	processing_class=tokenizer,
)
offline.train()
```

## Vary the losses

`alpha` interpolates between softened teacher KL loss and hard-label cross-entropy. `temperature` controls probability softening and the soft loss includes its square as a scale factor. `beta` weights the added feature loss. Select `feature_loss_type="cosine"` or `"l2"` and optional `layer_indices`.

Each coefficient accepts a constant, linear, or cosine schedule. A linear or cosine schedule requires its matching end value. `alpha_end` must be in `[0, 1]`, `temperature_end` must be positive, and `beta_end` must be nonnegative. Positive initial `beta` requires scanning. Starting at `beta=0` and scheduling toward a positive `beta_end` is rejected because feature extraction starts disabled. To schedule positive feature loss, the initial `beta` must already be positive.

Using the online teacher, student, tokenizer, and dataset above, this bounded composition moves `alpha` from its default to the soft-loss-only endpoint. Temperature remains at its default and feature loss remains disabled. These values demonstrate composition, not a tuning recommendation:

```python
args = DistillationConfig(
	output_dir="./scheduled-distillation", max_steps=10, per_device_train_batch_size=1,
	alpha=0.5, alpha_end=1.0, alpha_schedule="linear",
	temperature=1.0, temperature_end=1.0, temperature_schedule="cosine",
	beta=0.0, beta_schedule="constant",
)
trainer = DistillationTrainer(student, teacher, args=args, train_dataset=dataset, processing_class=tokenizer)
result = trainer.train()
```

## Copy teacher weights and select student parameters

Use `LearnToInitConfig` through `DistillationConfig.learn_to_init`. The same supported `Qwen/Qwen3-8B` architecture on both sides gives the embedding tensors identical shapes. The model graph has a `token_embedder` whose parameter is `embedding`, so the path is `token_embedder/embedding`.

The following training call copies the teacher embedding values into the student and freezes student parameters outside that path. Copying changes the student's weights during setup. Training writes checkpoints under the selected output directory. `enabled=False` retains copy and parameter-filter controls without requesting structural attention finalization:

```python
from lilax.training import LearnToInitConfig

student = AutoModelForCausalLM.from_pretrained(model_id)
teacher = AutoModelForCausalLM.from_pretrained(model_id)
initialization = LearnToInitConfig(
	enabled=False,
	weights_copy_map={r"token_embedder/(embedding)": r"token_embedder/\1"},
	student_params_to_update=[r"^token_embedder/embedding$"],
)
args = DistillationConfig(
	output_dir="./embedding-distillation", max_steps=10, per_device_train_batch_size=1,
	learn_to_init=initialization,
)
trainer = DistillationTrainer(student, teacher, args=args, train_dataset=dataset, processing_class=tokenizer)
result = trainer.train()
```

Copy-map keys match teacher paths with `re.fullmatch`. Values expand captured groups into student destinations. Student sharing maps instead alias one student parameter object at another student path. Trainable-parameter filters use regex search. See [map contracts and structural limits](../../reference/training/trainers.md#learn-to-init-maps-and-parameter-selection) before changing paths or shapes. With public `AutoModelForCausalLM` models, `enabled=True` raises `AssertionError` during finalization after the distillation training loop. Keep `enabled=False` for this copy and filter workflow.
