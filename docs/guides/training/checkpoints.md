# Save and resume training

Save model weights for loading and deployment. Save training state to resume the optimizer and training step.

## Save model weights

The default native save writes weights under `_lilax_model` in the destination. Repeated saves replace Lilax-owned weights and update configuration files while preserving unrelated files, checkpoints, and logs. An existing `_lilax_model` child without a Lilax ownership declaration is not forcibly replaced.

```python
from lilax import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model.save_pretrained("./weights")
tokenizer.save_pretrained("./weights")
restored = AutoModelForCausalLM.from_pretrained("./weights", local_files_only=True)
```

## Export Hugging Face weights

Using the model above:

Export temporarily writes a full native weight copy under `~/tmp/lilax/`, even when the destination is on another filesystem. Reserve space on the home filesystem. The temporary directory is removed when export exits. Export can also download the registered model repository snapshot for its HF configuration, unless a local source or cached snapshot is used.

```python
model.save_pretrained("./hf-weights", format="huggingface")
tokenizer.save_pretrained("./hf-weights")
```

Export requires a registered export mapping and compatible projection layout. Qwix QAT export raises `NotImplementedError`.

When saving native and Hugging Face weights to the same local directory, public loading and conversion select the latest successfully saved format. External Orbax clients read native weights at `<save_directory>/_lilax_model`. See [model saving](../../reference/training/checkpoints.md#model-saving).

## Save full training state

Continue with `model`, `tokenizer`, `dataset`, and `args` from the first [SFT example](sft.md#train-on-text). The update preserves its ten-step limit, batch size of one, length of 256, and disabled packing. Set checkpoint policy before constructing the trainer:

```python
from lilax import SFTTrainer
from lilax.checkpoints import TrainingCheckpointConfig

args = args.model_copy(update={
	"run_name": "sft-run", "save_steps": 100,
	"checkpoint_config": TrainingCheckpointConfig(max_num_checkpoints_to_keep=2),
})
trainer = SFTTrainer(model, args=args, train_dataset=dataset, processing_class=tokenizer)
result = trainer.train()
trainer.save_state()
```

Full-state checkpoints go under `./sft/sft-run/checkpoints/<step>/`. `trainer.save_state()` requests a full-state checkpoint after the ten-step run.

`trainer.save_model()` saves model weights and the processing class, not a resumable optimizer state. With no destination, it saves under `args.output_dir` while preserving run checkpoints, metrics, TensorBoard logs, and converted weights under `lilax-models/`. Repeated saves replace Lilax-owned model weights and update named configuration and processing files. To select another destination:

```python
trainer.save_model("./sft-weights")
```

## Resume a run

Keep the same `output_dir="./sft"` and `run_name="sft-run"` when constructing a trainer to resume this run:

```python
trainer.train(resume_from_checkpoint=True)
```

`True` selects the latest checkpoint under `./sft/sft-run/checkpoints/`, not directly under `output_dir`. An empty `run_name` resolves to `JOBSET_NAME` when set, otherwise `<model_name>_<YYYY-MM-DD-HH-MM>`. A later process can therefore select a different run. Resume raises `ValueError` if no checkpoint is available.

Use `trainer.train(resume_from_checkpoint="./sft/sft-run/checkpoints")` to select that checkpoint root. A path ending in a numeric step selects that step within its parent directory. Alternatively use `trainer.train(checkpoint_step=result.global_step)` to select the saved step from the training result above. Do not combine `checkpoint_step` with `resume_from_checkpoint`.

## Convert weights on CPU

```python
from lilax.checkpoints import CheckpointConversionConfig, convert_checkpoint

destination = convert_checkpoint(
	"Qwen/Qwen3-8B", "./converted-weights", config=CheckpointConversionConfig(save_dtype="bfloat16"),
)
```

For a supported saved Lilax weight checkpoint, `fused_qkv` and `fused_mlp` request projection-layout conversion. The destination must not exist. Full training checkpoints and QAT layout conversion are unsupported.

The shell form is `lilax convert source destination`, with optional `--fused-qkv`, `--no-fused-qkv`, `--fused-mlp`, and `--no-fused-mlp`. It calls `convert_checkpoint` without a `config` argument and accepts the same sources, including supported Hub repositories, local snapshots, and Lilax weight-only checkpoints.

See the [checkpoint reference](../../reference/training/checkpoints.md).
