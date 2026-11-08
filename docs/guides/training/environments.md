# Bind a synchronous OpenEnv environment

Use `environment_factory` for a synchronous tool environment, or `agentic_config.agent` to bind an existing OpenEnv client.

Before either example, complete the [TPU vLLM workflow setup](../../getting-started/environment.md#tpu-vllm-workflows). These examples use that integration recipe.

## Train with a tool environment

This adapts the counter task in the official [TRL GRPO environments example](https://huggingface.co/docs/trl/grpo_trainer#environments). The environment samples its target in `reset()` and rewards reaching that target. It is an API example, not a training recipe or a reward recommendation.

```python
import random

from lilax import AutoModelForCausalLM, AutoTokenizer, GRPOConfig, GRPOTrainer
from lilax.data import Dataset, prepare_rl_dataset
from lilax.rollout import AgenticRolloutConfig, RolloutConfig

class IncrementEnv:
	def reset(self) -> str:
		self.counter = 0
		self.target = random.randint(1, 6)
		return f"Increment the counter by {self.target}."

	def increment(self, step: int) -> int:
		"""Add a value to the counter.

		Args:
			step: Value to add.

		Returns:
			The updated counter.
		"""
		self.counter += step
		return self.counter

	def get_reward(self) -> float:
		return float(self.counter == self.target)

model_id = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
records = Dataset.from_list([{"prompts": "", "question": "", "answer": ""}])
dataset = prepare_rl_dataset(records, processing_class=tokenizer)
trainer = GRPOTrainer(
	model, args=GRPOConfig(output_dir="./counter-grpo"), train_dataset=dataset,
	processing_class=tokenizer, environment_factory=IncrementEnv,
	rollout_config=RolloutConfig(engine="vllm"), agentic_config=AgenticRolloutConfig(enabled=True),
)
trainer.train()
```

The empty record supplies the required dataset fields. During agentic training, the dataset's `question` supplies the base prompt and the string from `reset()` is appended. Unlike the current TRL example, Lilax requires `train_dataset`, derives duration from data settings, and calls both the factory and `reset()` without arguments. Dataset columns do not become `reset()` arguments.

`reset()` returns a string or `None`. `increment(step=2)` returns an integer counter value. Tool results become strings in the conversation. `get_reward()` returns a numeric episode reward. Training starts with `trainer.train()` and writes training artifacts under `./counter-grpo`.

Use a tokenizer with the bundled Qwen3 chat template and `<|im_end|>` as its EOS token. Lilax validates both requirements. The tools use Qwen `<tool_call>` messages. A response with no tool calls ends the episode.

Public methods become tools, except `reset`, `get_reward`, and `close`. Methods whose names start with `_` are excluded. Give each tool typed arguments and a Google-style docstring so its JSON schema can be constructed. Tools, `get_reward`, and optional `close()` must be synchronous. Supply `get_reward()` when you do not supply `reward_funcs`.

For a trainer-owned reward that reads the environment, the keyword is `environments`. In an environment class without `get_reward()`, supply this callable as `reward_funcs` to score the same counter state:

```python
def counter_reward(prompts, completions, environments, **kwargs):
	return [float(env.counter == env.target) for env in environments]
```

The returned reward list has one value per completion. Environment objects reach the reward function after `close()` completes, so closing must preserve state needed for rewards.

Do not use dataset fields `environments` or `_lilax_environment_token`. They are reserved for environment reward delivery. Combining `environment_factory` with `reward_funcs` requires `reward_num_workers` to be `0` or `1`. The `"maxtext"` engine rejects environment factories, and `evaluate()` with an environment factory raises `NotImplementedError`.

## Bind an existing OpenEnv client

Bind an environment-specific client and message conversion functions for agentic GRPO. The client must expose synchronous `reset`, `step`, and `close` methods.

### Illustrate the client protocol in process

This toy client adapts `IncrementEnv` above without an OpenEnv package or server. Its result record has `observation`, `reward`, `done`, and `metadata` fields. For this illustration, one integer response performs one increment and ends the episode. The reward remains `float(counter == target)`.

Reuse `IncrementEnv`, `AutoModelForCausalLM`, `GRPOConfig`, `GRPOTrainer`, `RolloutConfig`, `model_id`, `tokenizer`, and `dataset` from the first example. Use a fresh model so the first trainer's initialized mesh is not reused. The client binding uses a different conversation parser from the tool environment and requires a data template file.

The example writes `counter-openenv-template.json` in the current directory, replacing an existing file with
that name. Training writes artifacts under `./counter-openenv-grpo`.

```python
import json
from dataclasses import dataclass
from pathlib import Path

from lilax.openenv import OpenEnvAgent, OpenEnvEnvironment, TerminalStatus
from lilax.rollout import AgentBinding, AgenticRolloutConfig

@dataclass
class CounterResult:
	observation: str
	reward: float
	done: bool
	metadata: dict | None = None

class CounterClient:
	def __init__(self):
		self.env = IncrementEnv()

	def reset(self):
		return CounterResult(self.env.reset(), reward=0.0, done=False)

	def step(self, action: int):
		counter = self.env.increment(action)
		return CounterResult(str(counter), reward=self.env.get_reward(), done=True)

	def close(self):
		pass  # This in-process client owns no external resources.

def client_factory(task, *, group_id, pair_index):
	return CounterClient()

def reset_kwargs(task, *, group_id, pair_index):
	return {}  # IncrementEnv samples its own target during reset.

def reward_from_result(result):
	return result.reward

def action_from_response(text):
	return int(text.strip())

def observation_to_text(observation):
	return observation

def status_mapper(observation, reward, info):
	return TerminalStatus.SUCCEEDED

template_path = Path("./counter-openenv-template.json").resolve()
template_path.write_text(
	json.dumps({"SYSTEM_PROMPT": "", "TEMPLATE": "{question}"}), encoding="utf-8",
)
client_model = AutoModelForCausalLM.from_pretrained(model_id)
client_model.runtime.public_config.data_template_path = str(template_path)

system_prompt = "Reply with only the integer to add to the counter. Do not include reasoning or tool calls."
agentic_config = AgenticRolloutConfig(
	enabled=True, system_prompt=system_prompt,
	agent=AgentBinding(
		agent_class=OpenEnvAgent, environment_class=OpenEnvEnvironment,
		agent_kwargs={
			"system_prompt": system_prompt, "action_from_response": action_from_response,
			"observation_to_text": observation_to_text, "status_mapper": status_mapper,
		},
		environment_kwargs={
			"client_factory": client_factory, "reset_kwargs": reset_kwargs,
			"reward_from_result": reward_from_result, "max_steps": 1,
		},
	)
)
trainer = GRPOTrainer(
	client_model, args=GRPOConfig(output_dir="./counter-openenv-grpo"), reward_funcs=[],
	train_dataset=dataset, processing_class=tokenizer,
	rollout_config=RolloutConfig(engine="vllm"), agentic_config=agentic_config,
)
result = trainer.train()
print(result.global_step)
print(result.metrics)
```

The neutral template avoids injecting math-task instructions. Set `data_template_path` on the model's runtime
configuration before constructing `GRPOTrainer`. The printed values are the completed training step and
training metrics, not the counter reward.

`client_factory` creates a fresh counter per episode. `reset_kwargs` returns the keyword arguments for
`client.reset()`. Reset must return `done=False`. `observation_to_text` receives the client's observation.

`action_from_response` converts the model's response to the integer passed to `client.step()`. A response
with reasoning, tool markup, or other non-integer text raises `ValueError`.

`SUCCEEDED` means that the client completed normally, even when the counter reward is `0.0`. The mapper runs
only for terminal observations and must return a member other than `RUNNING`. `reward_funcs=[]` keeps the
environment reward without adding Lilax's default math rewards.

### Connect an external client

Replace `CounterClient` and the conversion functions with those for your environment. Install that environment's client separately and supply its real connection details. Keep `reset`, `step`, and `close` synchronous. Use the vLLM rollout workflow. The `"maxtext"` engine rejects agentic configurations.

The collector closes each episode's environment. The adapter serializes `close()` with client operations, so close waits for an operation holding the lock. A timed-out reset or step can still run and have side effects. Closing the client does not acknowledge cancellation on the environment server.

See [rollout reference](../../reference/training/rollouts.md).
