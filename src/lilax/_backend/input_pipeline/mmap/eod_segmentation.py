"""Megatron-compatible EOD-derived segmentation, positions, and input/target splits for mmap samples."""

import dataclasses

import grain.python as grain
import numpy as np


def megatron_min_segment_length(config) -> int:
  """Return the Megatron-compatible short-segment merge threshold.

  The threshold is ``max_target_length // packing_max_segments_per_sample``.
  It is meaningful only when attention resets at document boundaries; a
  non-positive divisor disables merging.
  """
  if not config.reset_attention_mask:
    return 0
  divisor = config.packing_max_segments_per_sample
  if divisor <= 0:
    return 0
  return config.max_target_length // divisor


def _merge_short_segments_np(segmentation: np.ndarray, position: np.ndarray, min_seg_len: int) -> None:
  """Greedily merge short EOD-derived segments in place.

  This follows Megatron's packed-sequence boundary rule. A candidate boundary
  is retained only when it is strictly more than ``min_seg_len`` tokens after
  the previous retained boundary. Dropped boundaries merge their tokens into
  the preceding retained segment, with segment IDs and position IDs rebuilt
  as a single contiguous sequence.

  Args:
    segmentation: One-dimensional, one-indexed segment IDs.
    position: One-dimensional position IDs corresponding to ``segmentation``.
    min_seg_len: Boundary-merging threshold. Values of zero or one are no-ops.
  """
  seq_len = len(segmentation)
  if min_seg_len <= 1 or seq_len == 0:
    return
  boundaries = np.flatnonzero(np.diff(segmentation)) + 1
  seg_starts = [0, *boundaries.tolist()]
  if len(seg_starts) <= 1:
    return
  kept = [0]
  for start in seg_starts[1:]:
    if start - kept[-1] > min_seg_len:
      kept.append(start)
  if len(kept) == len(seg_starts):
    return
  kept.append(seq_len)
  for seg_idx in range(len(kept) - 1):
    s, e = kept[seg_idx], kept[seg_idx + 1]
    segmentation[s:e] = seg_idx + 1
    position[s:e] = np.arange(e - s, dtype=position.dtype)


@dataclasses.dataclass
class GenerateDocSegmentIds(grain.MapTransform):
  """Generate EOD-aware segmentation and position arrays for ``mmap`` samples.

  The input data must already contain document-ending EOD tokens, normally
  from Megatron preprocessing with ``--append-eod``. ``MMapSampleIndexDataSource``
  concatenates and windows those tokens; it does not insert EOD tokens itself.
  For every input token field, this transform adds ``<field>_segmentation``
  and ``<field>_position``.

  Args:
    eod_id: Token ID that marks the end of a document.
    reset_attention_mask: Controls document-boundary attention and positions.

      * ``True`` (default) -- attention resets at every document boundary.
        EOD belongs to the preceding document (same segment ID), and a new
        segment starts after EOD.  Positions continue through EOD and reset
        after EOD.

        ::

            tokens:        [tok tok tok EOD tok tok EOD tok tok tok tok tok]
            segmentation:  [ 1   1   1   1   2   2   2   3   3   3   3   3]
            positions:     [ 0   1   2   3   0   1   2   0   1   2   3   4]

      * ``False`` -- cross-document attention is allowed. Positions are a
        continuous ``arange`` over the full sample, and all tokens use
        segment ID ``1`` except masked EOD positions.

    eod_mask_loss: When ``reset_attention_mask`` is False, controls whether
      EOD tokens receive segmentation value zero and are excluded from loss.
      In the ``mmap`` pipeline, ``ShiftData`` subsequently masks shifted EOD
      labels because EOD is also the batch padding sentinel.
    min_segment_length: Optional threshold used to merge adjacent short
      EOD-derived segments when attention resets are enabled.
  """

  def __init__(
      self, eod_id: int, reset_attention_mask: bool = True, eod_mask_loss: bool = False, min_segment_length: int = 0
  ):
    self.eod_id = eod_id
    self.reset_attention_mask = reset_attention_mask
    self.eod_mask_loss = eod_mask_loss
    self.min_segment_length = min_segment_length

  def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Apply EOD-based segmentation and loss masking to each column."""
    for col, tokens in list(element.items()):
      seq_len = tokens.shape[0]
      is_eod = tokens == self.eod_id
      if self.reset_attention_mask:
        segmentation = np.zeros(seq_len, dtype=np.int32)
        position = np.zeros(seq_len, dtype=np.int32)
        seg_id = 1
        pos_in_doc = 0
        for i in range(seq_len):
          if is_eod[i]:
            segmentation[i] = seg_id
            position[i] = pos_in_doc
            pos_in_doc += 1
            seg_id += 1
            pos_in_doc = 0
          else:
            segmentation[i] = seg_id
            position[i] = pos_in_doc
            pos_in_doc += 1
        if self.min_segment_length > 0:
          _merge_short_segments_np(segmentation, position, self.min_segment_length)
      else:
        if self.eod_mask_loss:
          segmentation = np.where(is_eod, np.int32(0), np.int32(1))
        else:
          segmentation = np.ones(seq_len, dtype=np.int32)
        position = np.arange(seq_len, dtype=np.int32)
      element[f"{col}_segmentation"] = segmentation
      element[f"{col}_position"] = position
    return element


@dataclasses.dataclass
class MegatronSplitInputsTargets(grain.MapTransform):
  """Build Megatron-compatible pretraining fields from an ``L + 1`` sample.

  Input is ``{"text": tokens}``, where ``len(tokens) == L + 1``. The output
  contains length-``L`` fields with ``inputs = tokens[:-1]`` and
  ``targets = tokens[1:]``. Unlike a padded left-shift, this preserves the
  final real target and its loss contribution.

  This gives valid prediction targets at every position without padding,
  unlike the ShiftData approach which wastes the last position.

  Attention, loss, and position annotations are constructed independently to
  match Megatron's separate attention-mask, loss-mask, and position-ID rules:

  - ``inputs_segmentation`` (attention): controlled by ``reset_attention_mask``.
    When True, EOD belongs to its preceding document (same segment ID) and
    a new segment starts after EOD; when False, all tokens share segment 1.
  - ``targets_segmentation`` (loss): controlled by ``eod_mask_loss`` only.
    When True, positions where ``inputs == eod_id`` get segmentation=0
    (excluded from loss); when False, all positions get segmentation=1.
  - ``inputs_position`` (RoPE): controlled by ``reset_attention_mask``.
    When True, EOD continues the preceding document's position counter and
    resets to 0 after EOD; when False, positions are sequential.

  Args:
    eod_id: Token ID marking the end of a document.
    reset_attention_mask: When True, EOD ends the current attention segment
      and positions restart at zero for the following token.
    eod_mask_loss: When True, positions whose input token is EOD have loss
      mask zero.
    no_attnmask_dataset_ids: Optional dataset IDs for samples that disable
      attention-mask reset regardless of the global setting.
    min_segment_length: Optional threshold for merging short EOD-derived
      attention segments.
    emit_dataset_id: Whether to emit a per-token ``dataset_id`` field when
      the source sample provides one.
  """

  def __init__(
      self,
      eod_id: int,
      reset_attention_mask: bool = True,
      eod_mask_loss: bool = False,
      no_attnmask_dataset_ids: set[int] | None = None,
      min_segment_length: int = 0,
      emit_dataset_id: bool = False,
  ):
    self.eod_id = eod_id
    self.reset_attention_mask = reset_attention_mask
    self.eod_mask_loss = eod_mask_loss
    self.no_attnmask_dataset_ids = no_attnmask_dataset_ids or set()
    self._has_no_attnmask = bool(self.no_attnmask_dataset_ids)
    self.min_segment_length = min_segment_length
    self.emit_dataset_id = emit_dataset_id

  def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Split tokens into input/target pairs with EOD-based segmentation."""
    tokens = element["text"]
    inputs = tokens[:-1]
    targets = tokens[1:]
    seq_len = inputs.shape[0]
    is_eod = inputs == self.eod_id
    effective_reset = self.reset_attention_mask
    if self._has_no_attnmask:
      dataset_id = element.get("dataset_id", None)
      if dataset_id is not None and int(dataset_id) in self.no_attnmask_dataset_ids:
        effective_reset = False
    if effective_reset:
      input_segmentation = np.zeros(seq_len, dtype=np.int32)
      position = np.zeros(seq_len, dtype=np.int32)
      seg_id = 1
      pos_in_doc = 0
      for i in range(seq_len):
        if is_eod[i]:
          input_segmentation[i] = seg_id
          position[i] = pos_in_doc
          pos_in_doc = 0
          seg_id += 1
        else:
          input_segmentation[i] = seg_id
          position[i] = pos_in_doc
          pos_in_doc += 1
      if self.min_segment_length > 0:
        _merge_short_segments_np(input_segmentation, position, self.min_segment_length)
    else:
      input_segmentation = np.ones(seq_len, dtype=np.int32)
      position = np.arange(seq_len, dtype=np.int32)
    if self.eod_mask_loss:
      target_segmentation = np.where(is_eod, np.int32(0), np.int32(1))
    else:
      target_segmentation = np.ones(seq_len, dtype=np.int32)
    result = {
        "inputs": inputs,
        "targets": targets,
        "inputs_segmentation": input_segmentation,
        "targets_segmentation": target_segmentation,
        "inputs_position": position,
        "targets_position": position,
    }
    if self.emit_dataset_id and "dataset_id" in element:
      result["dataset_id"] = np.full(seq_len, int(element["dataset_id"]), dtype=np.int32)
    return result
