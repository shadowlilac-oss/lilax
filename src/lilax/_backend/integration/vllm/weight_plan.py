"""Planning for direct MaxText rollout weight conversion."""

import logging
import re
from typing import Any, List, Mapping, Optional, Tuple

from lilax._backend.integration.vllm.weight_tree import _is_non_weight_path


class WeightPlanBuilder:
  """Builds conversion plans from trainer and rollout weight paths."""

  def _scanned_candidates(
      self, key_tuple: Tuple[Any, ...], prefix_len: int, suffix_start: int, slot: int
  ) -> List[Tuple[Tuple[Any, ...], Optional[int]]]:
    prefix, suffix = key_tuple[:prefix_len], key_tuple[suffix_start:]
    if self.cycle > 1:
      if slot == self.cycle - 1:
        candidates = [(prefix + ("layers", "global_layer") + suffix, None)]
      else:
        candidates = [(prefix + ("layers", "local_layers") + suffix, slot)]
      return candidates + [
          (prefix + ("layers", f"layer_{slot}") + suffix, None),
          (prefix + ("layers", str(slot)) + suffix, None),
          (prefix + ("layers", slot) + suffix, None),
      ]
    return [(prefix + ("layers",) + suffix, None)]

  def _build_target_free_plan(self, src_flat: Mapping[Tuple[Any, ...], Any]) -> list:
    plan = []
    consumed_wi_1 = set()
    for src_key in src_flat:
      if _is_non_weight_path(src_key):
        continue
      if src_key in consumed_wi_1:
        continue
      if "layers" not in src_key:
        plan.append(self._plan_entry_type(src_key, (src_key,), None, "identity"))
        continue
      idx = src_key.index("layers")
      prefix = src_key[:idx]
      rest = src_key[idx + 1 :]
      if self.cycle == 1:
        slot_refs = ((0, None),)
        suffix = rest
      else:
        slot_token = rest[0]
        match = re.fullmatch(r"layer_(\d+)", slot_token) if isinstance(slot_token, str) else None
        if slot_token == "local_layers":
          slot_refs = tuple((slot, slot) for slot in range(self.cycle - 1))
        elif slot_token == "global_layer":
          slot_refs = ((self.cycle - 1, None),)
        elif match:
          slot_refs = ((int(match.group(1)), None),)
        elif isinstance(slot_token, str) and slot_token.isdigit():
          slot_refs = ((int(slot_token), None),)
        elif isinstance(slot_token, int):
          slot_refs = ((slot_token, None),)
        else:
          raise self._plan_error_type(f"Unexpected slot token {slot_token!r} in key {src_key}")
        suffix = rest[1:]
      is_wi_0 = bool(suffix and suffix[-1] == "wi_0")
      wi_1_key = src_key[:-1] + ("wi_1",) if is_wi_0 else None
      fuse_moe = self.prefuse_moe_weights and is_wi_0 and (wi_1_key in src_flat)
      if fuse_moe:
        consumed_wi_1.add(wi_1_key)
        for slot, local_index in slot_refs:
          for block in range(self.num_blocks):
            global_idx = block * self.cycle + slot
            tgt_key = prefix + (f"layers_{global_idx}",) + suffix[:-1] + ("wi",)
            plan.append(self._plan_entry_type(tgt_key, (src_key, wi_1_key), block, "fuse_moe", local_index))
      elif self.prefuse_moe_weights and suffix and suffix[-1] == "wi_1" and (src_key[:-1] + ("wi_0",) in src_flat):
        continue
      else:
        for slot, local_index in slot_refs:
          for block in range(self.num_blocks):
            global_idx = block * self.cycle + slot
            tgt_key = prefix + (f"layers_{global_idx}",) + suffix
            plan.append(self._plan_entry_type(tgt_key, (src_key,), block, "slice", local_index))
    return plan

  def _build_plan(
      self,
      src_flat: Mapping[Tuple[Any, ...], Any],
      tgt_flat: Mapping[Tuple[Any, ...], Any],
      skip_paths: frozenset = frozenset(),
  ) -> list:
    plan = []
    unmatched: List[Tuple[Any, ...]] = []
    consumed: set = set()
    for tgt_key in tgt_flat:
      if tgt_key in skip_paths or _is_non_weight_path(tgt_key):
        continue
      if tgt_key in src_flat:
        plan.append(self._plan_entry_type(tgt_key, (tgt_key,), None, "identity"))
        consumed.add(tgt_key)
        continue
      layer_idx, prefix_len, suffix_start = self._find_unrolled_layer(tgt_key)
      if layer_idx < 0:
        unmatched.append(tgt_key)
        continue
      if layer_idx >= self.num_decoder_layers:
        raise self._plan_error_type(
            f"Target key {'.'.join(map(str, tgt_key))} refers to layer "
            f"{layer_idx}, beyond num_decoder_layers={self.num_decoder_layers}."
        )
      slot, block = layer_idx % self.cycle, layer_idx // self.cycle
      candidates = self._scanned_candidates(tgt_key, prefix_len, suffix_start, slot)
      matched = next(((candidate, local_index) for candidate, local_index in candidates if candidate in src_flat), None)
      if matched is not None:
        matched_key, local_index = matched
        plan.append(self._plan_entry_type(tgt_key, (matched_key,), block, "slice", local_index))
        consumed.add(matched_key)
        continue
      if tgt_key and tgt_key[-1] == "wi":
        for candidate, local_index in candidates:
          wi_0, wi_1 = candidate[:-1] + ("wi_0",), candidate[:-1] + ("wi_1",)
          if wi_0 in src_flat and wi_1 in src_flat:
            plan.append(self._plan_entry_type(tgt_key, (wi_0, wi_1), block, "fuse_moe", local_index))
            consumed.update((wi_0, wi_1))
            break
        else:
          unmatched.append(tgt_key)
        continue
      unmatched.append(tgt_key)
    self._validate_plan(src_flat, tgt_flat, unmatched, consumed)
    return plan

  def _validate_plan(self, src_flat, tgt_flat, unmatched, consumed) -> None:
    if unmatched:
      shown = "\n  ".join(".".join(map(str, key)) for key in sorted(unmatched)[:40])
      raise self._plan_error_type(
          f"{len(unmatched)} rollout parameter(s) have no source in the trainer "
          f"state and would keep vLLM's random dummy weights:\n  {shown}" + ("\n  ..." if len(unmatched) > 40 else "")
      )
    unused = [
        key for key in src_flat
        if key not in consumed
        and not _is_non_weight_path(key)
        and not any(marker in ".".join(map(str, key)) for marker in self.allow_unused_source_keys)
    ]
    if unused:
      shown = "\n  ".join(".".join(map(str, key)) for key in sorted(unused)[:40])
      raise self._plan_error_type(
          f"{len(unused)} trainer parameter(s) were not transferred to the "
          f"rollout. This usually means a layer was renamed on one side:\n  {shown}"
          + ("\n  ..." if len(unused) > 40 else "")
          + "\nPass `allow_unused_source_keys` if the omission is intentional."
      )
    logging.info(
        "MaxTextToMaxTextConverter: plan covers %d rollout parameters from %d trainer parameters.",
        len(tgt_flat), len(consumed),
    )
