"""Adapt a caller's Hugging Face batch to the backend's causal token layout."""

import numpy as np
import grain.python as grain

from lilax._backend.input_pipeline.transforms.target_shift import shift_left
from lilax._backend.input_pipeline.dpo_utils import DPODataFormatting


class CausalCollator:
  def __init__(self, data_collator, ignored_ids=(), *, shift=True):
    self.data_collator = data_collator
    self.ignored_ids = ignored_ids
    self.shift = shift
    self.pad_id = ignored_ids[0] if ignored_ids and ignored_ids[0] is not None else 0

  def __call__(self, examples):
    batch = {key: np.asarray(value) for key, value in self.data_collator(examples).items()}
    if "inputs" in batch:
      return batch
    inputs = batch.pop("input_ids")
    attention_mask = batch.pop("attention_mask", np.ones_like(inputs, dtype=np.int32))
    segment_ids = batch.pop("segment_ids", attention_mask)
    labels = batch.pop("labels", inputs)
    loss_mask = (labels != -100) & (attention_mask != 0) & ~np.isin(labels, self.ignored_ids)
    targets = np.where(loss_mask, labels, self.pad_id)
    positions = batch.pop("position_ids", np.broadcast_to(np.arange(inputs.shape[1]), inputs.shape))
    return batch | {
        "inputs": inputs, "inputs_position": positions.astype(np.int32),
        "inputs_segmentation": np.where(attention_mask, segment_ids, 0).astype(np.int32),
        "targets": shift_left(targets, self.pad_id) if self.shift else targets,
        "targets_position": positions.astype(np.int32),
        "targets_segmentation": (shift_left(np.where(loss_mask, segment_ids, 0).astype(np.int32), 0)
                                 if self.shift else np.where(loss_mask, segment_ids, 0).astype(np.int32)),
    }


class BatchCollator(grain.MapTransform):
  def __init__(self, data_collator, ignored_ids, *, shift=True):
    self.collator = CausalCollator(data_collator, ignored_ids, shift=shift)
    self.pad_id = ignored_ids[0]

  def map(self, batch):
    examples = []
    for index in range(batch["inputs"].shape[0]):
      example = {key: value[index] for key, value in batch.items() if key not in (
          "inputs", "targets", "inputs_position", "targets_position", "inputs_segmentation", "targets_segmentation",
      )}
      example.update({
          "input_ids": batch["inputs"][index], "position_ids": batch["inputs_position"][index],
          "attention_mask": (batch["inputs_segmentation"][index] != 0).astype(np.int32),
          "segment_ids": batch["inputs_segmentation"][index],
          "labels": np.where((batch["targets_segmentation"][index] != 0)
                             & (batch["targets"][index] != self.pad_id), batch["targets"][index], -100),
      })
      examples.append(example)
    return self.collator(examples)


class PreferenceBatchCollator(grain.MapTransform):
  def __init__(self, data_collator, pad_id, max_length, max_prompt_length):
    self.data_collator = data_collator
    self.formatter = DPODataFormatting(pad_id, max_length, ("prompt", "chosen", "rejected"), max_prompt_length)

  def map(self, batch):
    examples = [{role + "_ids": batch[role + "_ids"][index][batch[role + "_mask"][index] != 0].tolist()
                 for role in ("prompt", "chosen", "rejected")} for index in range(len(batch["prompt_ids"]))]
    output = {key: np.asarray(value) for key, value in self.data_collator(examples).items()}
    if "prompt_ids" in output:
      return output
    tokens, attention, completion = output["input_ids"], output["attention_mask"], output["completion_mask"]
    size = len(examples)
    records = []
    for index in range(size):
      prompt = tokens[index][(attention[index] != 0) & (completion[index] == 0)]
      chosen = tokens[index][(attention[index] != 0) & (completion[index] != 0)]
      rejected = tokens[index + size][(attention[index + size] != 0) & (completion[index + size] != 0)]
      records.append(self.formatter.map({"prompt": prompt, "chosen": chosen, "rejected": rejected}))
    return {key: np.stack([record[key] for record in records]) for key in records[0]}
