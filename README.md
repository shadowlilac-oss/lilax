# Lilax

Lilax brings a Hugging Face-style Python API to MaxText, a JAX library for training
and running large models on TPUs and GPUs.

> **Work in progress:** The APIs are source-authored; verification is limited to selected paths.
> Installation, runtime behavior, and feature coverage may vary by model and platform.

## Install

Python 3.12 or newer is required. For TPU, install from the repository root:

```sh
uv pip install './lilax[tpu]' --resolution=lowest
```

The NVIDIA GPU image `sha256:a77a1ac9ad9d132ccd7683572e711dbc4232109573c9d27b48b3817b87a0b4bc`
completed a 100-step SFT run with batch size 4, finite losses, and final loss 1.3456947803497314.
The run logged cuDNN runtime 9.26 versus compiled 9.27 incompatibility; it does not establish
general compatibility. The exact pinned Tunix package was installed without dependencies as an
explicit exception to its incompatible JAX version bound; its other requirements were resolved
normally. See
[container installation](TRAINING_OPERATIONS.md#container-installation).

Model loading also requires the `generation` extra. Training input pipelines require the
`tfds` extra. Optional training dependencies have known version conflicts, and a complete
installation across all paths has not been verified. See [implementation status](IMPLEMENTATION.md) and
[coverage](COVERAGE.md) before choosing an environment.

## Load a model

```python
from lilax import AutoModelForCausalLM

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
    )
```

Hugging Face conversion uses the Trainer's output directory or the local Hugging Face cache.
For multiple JAX processes, set a shared Trainer `output_dir` before first using the model.
The model graph is constructed when first used. Lilax supports Hugging Face loading
for selected models, not every model on the Hub.

## Fine-tune a model

Use `SFTConfig` for training settings and pass the dataset and tokenizer to `SFTTrainer`:

```python
from datasets import load_dataset
from lilax import AutoModelForCausalLM, AutoTokenizer, SFTConfig, SFTTrainer

if __name__ == "__main__":
    model_id = "Qwen/Qwen3-8B"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir="./checkpoints",
            max_steps=100,
            per_device_train_batch_size=1,
            max_length=1024,
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model("./model")
```

The Trainer recognizes the dataset's `messages` column and owns packing and completion loss masks.
The example's Qwen3-8B training has not been executed. The earlier 100-step run used a different
model and batch size and does not verify this API integration.

Use `jaxcoord config` to write a reusable execution profile, then run
`jaxcoord launch --config-file jaxcoord.yaml train.py`. See [execution profiles](EXECUTION_PROFILE.md)
for CPU/GPU/TPU placement, DiLoCo and Python/checkpoint precedence.
The [public API reference](PUBLIC_API.md) lists inspected constructors and arguments.
The [migration ledger](API_MIGRATION.md) records removed and relocated arguments.

Pass finite data to `eval_dataset` and call `trainer.evaluate()` for manual evaluation.
`eval_on_start` and positive `eval_steps` select startup and periodic evaluation.
Evaluation consumes the supplied dataset without a batch cap. `SyntheticDataset` supports training
only; selecting it for evaluation raises an error. Finite HF iterable datasets need no known length.

For Tunix SFT, DPO, distillation and GRPO, `save_only_lora_params` and
`restore_only_lora_params` in `TrainingCheckpointConfig` default to `None` (selection follows LoRA
enablement). Set either to `False` for full-model checkpointing. QAT with LoRA requires both
to be `False`; adapter-only checkpoints omit QAT statistics. Supply a matching base model
when restoring adapter-only checkpoints. The native SFT lifecycle does not support these controls.

### OpenEnv with agentic GRPO

`GRPOTrainer` can create an environment for each rollout and expose its public methods as tools.
For example, with the OpenEnv Echo client installed and a configured model and training dataset:

```python
from echo_env import EchoEnv
from lilax import GRPOConfig, GRPOTrainer
from lilax.rollout import AgenticRolloutConfig, RolloutConfig


class EchoTools:
    def __init__(self):
        self.env = EchoEnv(base_url=ENV_URL).sync()
        self.reward = 0.0

    def reset(self) -> None:
        self.env.reset()
        self.reward = 0.0

    def echo(self, message: str) -> str:
        """Echo a message.

        Args:
            message: Message to echo.

        Returns:
            The echoed message.
        """
        result = self.env.call_tool("echo_message", message=message)
        self.reward = float(len(result))
        return result

    def get_reward(self) -> float:
        return self.reward

    def close(self) -> None:
        self.env.close()


trainer = GRPOTrainer(
    model=model,
    args=GRPOConfig(output_dir="./grpo-checkpoints"),
    train_dataset=train_dataset,
    processing_class=tokenizer,
    environment_factory=EchoTools,
    rollout_config=RolloutConfig(engine="vllm"),
    agentic_config=AgenticRolloutConfig(enabled=True),
)
trainer.train()
```

Set `ENV_URL` to an Echo environment server. This example defines its own reward as the length of
the last echoed message; it is not an environment-provided reward.

Use the bundled Qwen3 chat template and its `<|im_end|>` EOS on `processing_class`.
Supply HF records prepared with `lilax.data.prepare_rl_dataset`; there is no public raw iterator argument.
`reset()` returns a string appended to the prompt, or `None`. Public synchronous
methods other than `reset`, `get_reward`, and `close` become tools; their typed arguments and
docstrings must support Transformers' tool schema conversion. The model can call multiple tools per
turn and continues until it answers without a tool call or reaches the rollout limit. Async tools,
multiple environment factories, and native RL evaluation are not supported.
Tunix labels a normally completed rollout `SUCCEEDED`; that label does not report whether the
environment's task was solved. Rollouts filtered at a context limit do not receive a final reward.

`get_reward()` is called before closing each completed environment. Alternatively provide a callable
or list of callables in `reward_funcs`; these receive batched `prompts` and `completions` strings and
available dataset fields, and return one numeric reward per prompt. They can also accept `environments`,
the corresponding factory instances in batch order. Environments have completed `close()` before the
reward call: their retained attributes are available, but their clients are not necessarily usable.
`close()` must be synchronous. The dataset fields `environments` and `_lilax_environment_token`
are reserved when batch reward functions are configured.
If close times out or has not completed, batch reward computation fails instead of reading that instance.
Set `reward_num_workers` to 0 or 1 for this path; environment instances are not sent to worker processes.
Tunix currently supplies the first assistant response as the completion, not the full conversation.
If both reward sources are provided, their scores are added; do not score the same signal twice.
No configured default reward functions are added for `environment_factory`.
For non-environment training, `reward_funcs` replaces the configured reward functions as before.

This path has not been runtime-verified. Tunix attempts to close the environment after each episode
with a bounded wait. A timed-out synchronous call can continue while cleanup waits; closing the client
does not guarantee remote work has stopped. The existing callback-based bridge remains available.

## More

- See the [training operations guide](TRAINING_OPERATIONS.md) for multi-node GPU launch, quantization,
  reporting, and profiling.
- Use `DPOTrainer`, `GRPOTrainer`, or `DistillationTrainer` for other post-training workflows.
- See [coverage](COVERAGE.md) for available model, training, and infrastructure APIs and their limits.
