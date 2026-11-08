# Train with GRPO

Train a policy using generated completions and rewards. Choose rollout generation separately from training arguments.

For the vLLM rollout used below, complete the [TPU-vLLM setup](../../getting-started/environment.md#tpu-vllm-workflows) before constructing the trainer.

## Train with math rewards

Training writes checkpoints under `output_dir`. The final `save_model` call replaces Lilax-owned native weights and updates named configuration and processing files while preserving unrelated destination files.

Constructing `GRPOTrainer` allocates trainer and sampler devices and initializes the training mesh before `train()`, while weight restoration remains deferred. Set device and mesh options first. See [runtime controls](../../reference/execution/runtime.md).

```python
from lilax import AutoModelForCausalLM, AutoTokenizer, GRPOConfig, GRPOTrainer, GenerationConfig
from lilax.data import load_dataset, prepare_rl_dataset
from lilax.rollout import MathRewardsConfig, RolloutConfig

model_id = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
dataset = load_dataset("openai/gsm8k", "main", split="train")
dataset = prepare_rl_dataset(dataset, processing_class=tokenizer)
args = GRPOConfig(output_dir="./grpo", num_batches=4, batch_size=1, rewards=MathRewardsConfig())
rollout = RolloutConfig(
	engine="vllm", max_prompt_length=256,
	generation_config=GenerationConfig(max_new_tokens=128, do_sample=True, temperature=1.0),
)
trainer = GRPOTrainer(model, args=args, train_dataset=dataset, processing_class=tokenizer, rollout_config=rollout)
result = trainer.train()
trainer.save_model("./policy")
```

The return value has this illustrative shape. Angle brackets represent values from your run:

```text
TrainOutput(global_step=<completed optimizer steps>, training_loss=<mean loss>, metrics={"train_loss": <mean loss>})
```

Read the loss with `result.metrics["train_loss"]`. See [logged metrics](../../reference/training/trainers.md#logged-metrics) for history and tracker output.

Use an environment with the required trainer and rollout devices. The values above illustrate the API, not resource sizing or reward tuning.

## Prepare your own records

Prepared records contain `prompts`, `question`, and `answer`. Otherwise, the default converter selects question and answer columns, applies name-specific GSM8K answer extraction, and formats prompts with a packaged math template. See [column precedence and formatting](../../reference/data/data.md#rl-preparation). Supply `RLDataOptions(processor=...)` to replace that conversion. The processor receives one record and returns a dictionary with those fields.

```python
from lilax.data import Dataset, RLDataOptions

records = Dataset.from_list([{"prompts": "What is 2 + 2?", "question": "What is 2 + 2?", "answer": "4"}])
dataset = prepare_rl_dataset(records, processing_class=tokenizer, options=RLDataOptions(train_fraction=1.0))
```

Training duration derives from `num_batches`, `batch_size`, dataset fraction, and epochs. GRPO does not accept a numeric `max_steps` or `gradient_accumulation_steps`.

## Supply a custom reward

This toy callback demonstration rewards the number of characters in each completion. It demonstrates the callable contract, not a recommended training objective. Accept `prompts`, `completions`, and additional execution keywords through `**kwargs`, and return one number per completion. Using the model, arguments, dataset, tokenizer, and vLLM rollout above:

```python
def character_count_reward(prompts, completions, **kwargs):
	return [float(len(completion)) for completion in completions]

print(character_count_reward(prompts=["Say hello."], completions=["Hello!"]))
# [6.0]

trainer = GRPOTrainer(
	model, reward_funcs=character_count_reward, args=args, train_dataset=dataset,
	processing_class=tokenizer, rollout_config=rollout,
)
result = trainer.train()
```

Custom `reward_funcs` replace the built-in reward stack selected by `args.rewards`. Multiple reward callables are summed. Additional dataset values can arrive through `**kwargs`. See the [reward callable contract](../../reference/training/rollouts.md#rewards).

## Pack rollout sequences

For vLLM, a positive `max_seq_token_per_tpu` enables packed training rows. Choose the row budget for your workload. It must fit a maximal prompt plus completion. With the existing prompt capacity of 256 and generation length of 128, the minimum valid row budget is `256 + 128 = 384` tokens. This example derives that lower bound from the rollout configuration rather than choosing a hardware rate:

```python
row_budget = rollout.max_prompt_length + rollout.generation_config.max_new_tokens
packed_args = GRPOConfig(
	output_dir="./packed-grpo", num_batches=4, batch_size=1,
	rewards=MathRewardsConfig(), max_seq_token_per_tpu=row_budget,
)
trainer = GRPOTrainer(
	model, args=packed_args, train_dataset=dataset, processing_class=tokenizer, rollout_config=rollout,
)
result = trainer.train()
```

Leave microbatch fields at `-1` for automatic sizing. See [loss, aggregation, and microbatch semantics](../../reference/training/trainers.md#grpoconfig-and-grpotrainer) for accepted settings and dependency-specific training errors.

## Select rollout controls

Compose `RolloutConfig` with `RolloutClusterConfig` for device allocation, `VLLMRolloutConfig` for rollout batching, and `WeightSyncConfig` for weight transfer. These settings apply to the vLLM rollout workflow.

The `"maxtext"` engine has a fixed length reward. It rejects custom reward functions, evaluation input, agentic environments, LoRA, and Qwix QAT. It requires stochastic generation and one iteration per rollout batch. See [rollout reference](../../reference/training/rollouts.md) before selecting an engine.

## Supply a reference policy

`ref_model` supplies a separate policy for the KL penalty controlled by `GRPOConfig.beta`. It is not the rollout generator or a reward model. If omitted, the vLLM workflow creates a reference from the policy's construction configuration or clones the actor when loading only once.

Using the model, dataset, arguments, tokenizer, and rollout configuration above:

```python
reference = AutoModelForCausalLM.from_pretrained(model_id)
trainer = GRPOTrainer(
	model, ref_model=reference, args=args, train_dataset=dataset,
	processing_class=tokenizer, rollout_config=rollout,
)
```

Pass a model object rather than a repository string. An uninitialized reference runtime uses the policy's mesh. The `"maxtext"` engine requires the actor and reference to use the same mesh.

## Evaluate and generate

For the vLLM workflow, supply finite `eval_dataset` and `GRPOEvaluationConfig` through `GRPOConfig.evaluation_config`. After training, `trainer.generate(prompts, num_passes=1)` generates responses. Calling it before a completed training setup raises `RuntimeError`.

Use the same name for `sampling_strategy` and the corresponding `generation_configs` key. With the existing model, tokenizer, arguments, and rollout, prepare a held-out split and compose evaluation before trainer construction:

```python
from lilax.training import GRPOEvaluationConfig

validation = prepare_rl_dataset(
	load_dataset("openai/gsm8k", "main", split="test"), processing_class=tokenizer,
)
evaluation = GRPOEvaluationConfig(
	sampling_strategy="validation", generation_configs={"validation": GenerationConfig(do_sample=False)},
)
eval_args = args.model_copy(update={"evaluation_config": evaluation})
trainer = GRPOTrainer(
	model, args=eval_args, train_dataset=dataset, eval_dataset=validation,
	processing_class=tokenizer, rollout_config=rollout,
)
```

Only temperature, top-k, and top-p translate from these named configurations. Their `max_new_tokens` and `eos_token_id` fields are ignored. Evaluation uses the runtime's target length minus prompt capacity, which the rollout above sets to `128` tokens. See [evaluation configuration](../../reference/training/rollouts.md#grpo-evaluation).

See [environment bindings](environments.md), [trainer reference](../../reference/training/trainers.md), and [rollout reference](../../reference/training/rollouts.md).
