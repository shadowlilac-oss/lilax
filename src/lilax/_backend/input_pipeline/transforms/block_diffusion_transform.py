"""Grain adapter for block-diffusion input corruption."""

import dataclasses

import grain.python as grain
import numpy as np

from lilax._backend.diffusion.block_diffusion import corruption as block_diffusion_corruption


@dataclasses.dataclass
class BlockDiffusionCorruption(grain.RandomMapTransform):
  """Adapts block-diffusion corruption to the Grain batch contract."""

  block_size: int
  mask_id: int
  min_noise: float = 1.0e-3
  logit_alignment: str = "same_position"
  canvas_policy: str = "all_masked"
  axis: int = 1
  completion_only: bool = False

  def random_map(self, element, rng: np.random.Generator):
    """Corrupts inputs while preserving clean targets and input metadata."""
    inputs = np.asarray(element["inputs"])
    targets = np.asarray(element["targets"])
    targets_segmentation = np.asarray(element["targets_segmentation"])
    if inputs.shape != targets.shape or inputs.shape != targets_segmentation.shape:
      raise ValueError(
          "inputs, targets, and targets_segmentation must have identical shapes, got "
          f"{inputs.shape}, {targets.shape}, and {targets_segmentation.shape}"
      )
    validity_mask = targets_segmentation != 0
    raw_completion_mask = element.get("completion_mask")
    if raw_completion_mask is None:
      if self.completion_only:
        if "completion_mask" not in element:
          raise ValueError("completion-only block diffusion requires an explicit completion_mask")
        raise ValueError("completion-only block diffusion requires a non-None completion_mask")
      completion_mask = validity_mask
    else:
      completion_mask = np.asarray(raw_completion_mask) != 0
    if completion_mask.shape != inputs.shape:
      raise ValueError(f"completion_mask must match inputs shape; received {completion_mask.shape} and {inputs.shape}")
    if np.any(completion_mask & ~validity_mask):
      raise ValueError("completion_mask must be a subset of valid target positions")
    can_validate_block_order = (
        self.completion_only
        and self.block_size > 0
        and inputs.ndim > 0
        and -inputs.ndim <= self.axis < inputs.ndim
        and inputs.shape[self.axis % inputs.ndim] > 0
    )
    if can_validate_block_order:
      axis = self.axis % inputs.ndim
      sequence_length = inputs.shape[axis]
      block_count = (sequence_length + self.block_size - 1) // self.block_size
      padded_length = block_count * self.block_size

      def _as_blocks(mask):
        rows = np.moveaxis(mask, axis, -1).reshape(-1, sequence_length)
        padded = np.pad(rows, ((0, 0), (0, padded_length - sequence_length)))
        return padded.reshape(rows.shape[0], block_count, self.block_size)

      validity_blocks = _as_blocks(validity_mask)
      completion_blocks = _as_blocks(completion_mask)
      seen_completion = np.maximum.accumulate(completion_blocks, axis=-1)
      if np.any(seen_completion & validity_blocks & ~completion_blocks):
        raise ValueError(
            "completion-only block diffusion cannot place a prompt token after "
            "a completion token in the same diffusion block; split the example "
            "or align the next prompt to a block boundary"
        )
    supervision_mask = completion_mask if self.completion_only else validity_mask
    result = block_diffusion_corruption.corrupt_tokens(
        inputs,
        supervision_mask,
        rng,
        block_size=self.block_size,
        mask_id=self.mask_id,
        min_noise=self.min_noise,
        logit_alignment=self.logit_alignment,
        canvas_policy=self.canvas_policy,
        axis=self.axis,
    )
    output = dict(element)
    output["inputs"] = result.inputs
    output["targets"] = targets
    output["corruption_mask"] = result.corruption_mask.astype(targets_segmentation.dtype)
    output["targets_loss_mask"] = result.targets_loss_mask.astype(targets_segmentation.dtype)
    return output
