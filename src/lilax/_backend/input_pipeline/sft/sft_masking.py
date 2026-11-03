"""Prompt/completion concatenation and target masking for SFT."""

import dataclasses

import grain.python as grain
import numpy as np


@dataclasses.dataclass
class SFTPromptMasking(grain.MapTransform):
  """Construct inputs and targets for SFT training. Concat prompt and completion to generate inputs.
  Causal SFT can mask prompt targets with ``unk_id``. Block diffusion keeps
  clean same-position targets and emits role-derived completion metadata.
  """

  def __init__(
      self,
      text_column_name,
      completion_only,
      max_target_length,
      unk_id=0,
      training_objective="causal_lm",
  ):
    if training_objective not in ("causal_lm", "block_diffusion"):
      raise ValueError(f"Unsupported training objective: {training_objective}")
    self.text_column_name = text_column_name
    self.completion_only = completion_only
    self.max_target_length = max_target_length
    self.unk_id = unk_id
    self.training_objective = training_objective

  def map(self, element):
    """
    Maps a single dataset element to an SFT training instance.
    It concatenates the prompt and completion to form the `inputs` sequence.
    For the `targets` sequence:
    - If `self.completion_only` is `True`, causal SFT masks the prompt portion
      with `self.unk_id`; block-diffusion SFT keeps clean targets and records
      completion eligibility separately.
    - If `self.completion_only` is `False`, the target sequence is
      identical to the input sequence.
    """
    is_block_diffusion = self.training_objective == "block_diffusion"
    inputs, targets, completion_mask = [], [], []
    for i, text in enumerate(element[self.text_column_name]):
      inputs += text
      is_prompt = element["is_prompt"][i]
      targets += [self.unk_id] * len(text) if self.completion_only and is_prompt and not is_block_diffusion else text
      completion_mask += [not is_prompt] * len(text)
    output = {
        "inputs": np.asarray(inputs[: self.max_target_length], dtype=np.int32),
        "targets": np.asarray(targets[: self.max_target_length], dtype=np.int32),
    }
    if is_block_diffusion:
      output["completion_mask"] = np.asarray(completion_mask[: self.max_target_length], dtype=np.int32)
    return output


@dataclasses.dataclass
class SFTPromptMaskingVision(grain.MapTransform):
  """SFT prompt masking for multimodal"""

  def __init__(self, query_column, response_column, max_target_length, pad_id):
    self.query_column = query_column
    self.response_column = response_column
    self.max_target_length = max_target_length
    self.pad_id = pad_id

  def map(self, element):
    inputs = np.concatenate((element[self.query_column], element[self.response_column]))
    targets = np.concatenate(
        (
            np.asarray([self.pad_id] * len(element[self.query_column])),
            element[self.response_column],
        )
    )
    return {
        "inputs": np.asarray(inputs[: self.max_target_length], dtype=np.int32),
        "targets": np.asarray(targets[: self.max_target_length], dtype=np.int32),
        "images": element["images"],
    }
