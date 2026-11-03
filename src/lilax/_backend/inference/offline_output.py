import numpy as np


def completion_token_ids(token_outputs):
  return np.array([token_output.token for token_output in token_outputs]).flatten()


def completion_logprobs(token_outputs):
  return np.array([token_output.log_prob.flatten() for token_output in token_outputs]).flatten()
