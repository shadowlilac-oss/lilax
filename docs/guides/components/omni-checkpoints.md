# Evaluate a stitched Omni checkpoint

The experimental Omni workflow loads a stitched checkpoint and runs its single-sample ChartQA evaluation. It is separate from Hub model loading.

Supply `checkpoint_path` as your stitched checkpoint compatible with the Omni configuration. The calls resolve the runtime, initialize JAX, restore model weights, download ChartQA and tokenizer files, and log predictions. Evaluation also resets Python's global random seed to `42`.

```python
from lilax.generation.omni import load_omni_config, run_evaluation

runtime = load_omni_config(checkpoint_path)
run_evaluation(checkpoint_path, runtime)
```

`config_path` on `load_omni_config` selects another compatible configuration file. Runtime configuration is resolved for single-sample execution. `run_evaluation` returns `None` and logs up to five randomly selected validation samples by default. It logs a dataset-loading error and returns if ChartQA cannot be loaded.

For standard supported multimodal models, use [AutoProcessor](../data/processing.md) and [model generation](../inference/generation.md). See [Omni reference](../../reference/components/omni-checkpoints.md) for the component entry points.
