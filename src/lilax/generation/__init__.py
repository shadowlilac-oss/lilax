"""Generation preferences and completion records."""

from .configuration import ContinuousBatchingConfig, GenerationConfig
from lilax._backend.inference.offline_engine import CompletionOutput
from lilax._backend.inference.inference_utils import (
    log_prob_of_chosen_token, prompt_logprobs_from_packed_prefill, prompt_logprobs_from_prefill, sampling,
)

__all__ = ["ContinuousBatchingConfig", "GenerationConfig", "CompletionOutput", "log_prob_of_chosen_token",
           "prompt_logprobs_from_packed_prefill", "prompt_logprobs_from_prefill", "sampling"]
