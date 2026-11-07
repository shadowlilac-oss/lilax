"""Settings for the existing math-reward functions."""

from pydantic import BaseModel, ConfigDict

from lilax.configuration.domains.data_rl import RLSpecialTokens


class RewardSpecialTokens(RLSpecialTokens):
  model_config = ConfigDict(extra="forbid")


class MathRewardsConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  exact_answer: float = 5.0
  exact_format_match: float = 3.0
  white_space_format_match: float = 1.5
  partial_format_match: float = 0.5
  ratio_guess_to_answer_high: float = 0.5
  ratio_guess_to_answer_low: float = 0.25
  incorrect_format: float = -0.5
  incorrect_answer: float = -1.0
  math_verify_timeout: int = 300
  math_verify_num_procs: int | None = None
  special_tokens: RewardSpecialTokens | None = None

  def _source_values(self):
    values = {("penalty_" if name.startswith("incorrect_") else "reward_") + name: getattr(self, name)
              for name in ("exact_answer", "exact_format_match", "white_space_format_match", "partial_format_match",
                           "ratio_guess_to_answer_high", "ratio_guess_to_answer_low", "incorrect_format", "incorrect_answer")}
    values.update(math_verify_timeout=self.math_verify_timeout, math_verify_num_procs=self.math_verify_num_procs)
    if self.special_tokens is not None:
      values.update(self.special_tokens.model_dump())
    return values
