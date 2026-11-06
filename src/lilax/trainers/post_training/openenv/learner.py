"""Batched environment rewards for factory-owned agentic rollouts."""

import asyncio
import threading

from tunix.rl.agentic import agentic_grpo_learner
from tunix.rl.agentic.trajectory import trajectory_collect_engine

from lilax.trainers.post_training.openenv.registry import ENVIRONMENT_TOKEN


class FactoryCollectEngine(trajectory_collect_engine.TrajectoryCollectEngine):
  async def _run_with_timing(self, func, *args, timeout=None):
    try:
      return await super()._run_with_timing(func, *args, timeout=timeout)
    except asyncio.TimeoutError:
      if func == self.env.close:
        self.env.mark_close_timeout()
      raise


class FactoryGRPOLearner(agentic_grpo_learner.GRPOLearner):
  def __init__(self, *, environment_registry, **kwargs):
    self.environment_registry = environment_registry
    self.reward_context = threading.local()
    super().__init__(**kwargs)

  def _build_orchestrator(self):
    orchestrator = super()._build_orchestrator()
    orchestrator.engine_cls = FactoryCollectEngine
    return orchestrator

  def _process_results(self, trajectories, mode, expected_step=None):
    tokens = []
    try:
      for item in trajectories:
        token = item.traj["original_input"].pop(ENVIRONMENT_TOKEN)
        if not isinstance(token, list) or len(token) != 1 or not isinstance(token[0], str):
          raise RuntimeError("Invalid environment reward token in trajectory.")
        tokens.append(token[0])
      self.reward_context.environments = self.environment_registry.resolve(tokens)
      return super()._process_results(trajectories, mode, expected_step)
    finally:
      self.environment_registry.release(tokens)
      if hasattr(self.reward_context, "environments"):
        del self.reward_context.environments

  def _compute_rewards(self, prompts, completions, mode, expected_step=None, **kwargs):
    return super()._compute_rewards(
        prompts, completions, mode, expected_step=expected_step,
        environments=self.reward_context.environments, **kwargs,
    )

  def train(self, train_dataset, eval_dataset=None, skip_jit=False):
    try:
      return super().train(train_dataset, eval_dataset=eval_dataset, skip_jit=skip_jit)
    finally:
      self.environment_registry.shutdown()
