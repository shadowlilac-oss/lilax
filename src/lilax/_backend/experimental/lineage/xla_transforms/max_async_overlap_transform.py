# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom XLA post-scheduler transformation for DSv3."""

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
import dataclasses
import json
import re
from typing import Any

from jax._src.lib import hlo as _hlo
import jax.extend.xla as jex_xla
from tensorflow.compiler.xla.service import hlo_pb2  # pylint: disable=g-direct-tensorflow-import

_DEFAULT_TRANSFORM_NAME = "max_async_overlap_transform"
_ASYNC_PAIRS: Mapping[str, str] = {
    "kAsyncStart": "kAsyncDone",
    "kAllGatherStart": "kAllGatherDone",
    "kAllReduceStart": "kAllReduceDone",
    "kCollectivePermuteStart": "kCollectivePermuteDone",
    "kSend": "kSendDone",
    "kRecv": "kRecvDone",
}
_ASYNC_START_OPCODES = frozenset(_ASYNC_PAIRS)
_ASYNC_DONE_OPCODES = frozenset(_ASYNC_PAIRS.values())
_DONE_TO_START_OPCODE: Mapping[str, str] = {done: start for start, done in _ASYNC_PAIRS.items()}

_DATA_FORMATTING_OPCODES = frozenset(
    {
        "kAdd", "kAfterAll", "kBitcast", "kBitcastConvert", "kBroadcast", "kClamp", "kConcatenate",
        "kConstant", "kConvert", "kCopy", "kCopyDone", "kCopyStart", "kDynamicSlice", "kDynamicUpdateSlice",
        "kGetTupleElement", "kIota", "kOptimizationBarrier", "kPad", "kReshape", "kSlice", "kSubtract",
        "kTranspose", "kTuple", "kAnd", "kCompare", "kMultiply", "kPartitionId", "kReduceWindow", "kSelect",
        "kShiftRightLogical",
    }
)
_DATA_FORMATTING_CUSTOM_CALLS = frozenset({"AllocateBuffer", "ZeroCrop"})
_FUSION_INTERNAL_EXTRA_OPCODES = frozenset({"kParameter", "kReduce"})

_CONTROL_PREDECESSORS_RE = re.compile(r"control-predecessors=\{([^}]*)\}")
_CALLS_RE = re.compile(r"\bcalls=(%?[\w.\-]+)")
_CUSTOM_CALL_TARGET_RE = re.compile(r'custom_call_target="([^"]*)"')
_BODY_RE = re.compile(r"\bbody=(%?[\w.\-]+)")
_INSTRUCTION_NAME_RE = re.compile(r"%([\w.\-]+)")
_ASYNC_THREAD_RE = re.compile(r'async_execution_thread="([^"]+)"')
_CORE_IDS_RE = re.compile(r'"core_ids":\s*\[([^\]]*)\]')
_ENTRY_RE = re.compile(r"^ENTRY %([\w.\-]+)", re.MULTILINE)

_MEGASCALE_HANDLER = "xla_megascale_runtime"
_DCN_SEND_QUEUE = "dcn:send"
_DCN_RECV_QUEUE = "dcn:recv"
_HOST_H2D_QUEUE = "host:h2d"
_HOST_D2H_QUEUE = "host:d2h"
_HOST_MEMORY_SPACE = "S(5)"
_SERIALIZED_QUEUES = frozenset({_HOST_H2D_QUEUE, _HOST_D2H_QUEUE})
_AGGREGATE_DEFER = "AGGREGATE_STATUS_DEFER"
_AGGREGATE_ISSUE = "AGGREGATE_STATUS_ISSUE"
_AGGREGATED_SEND_RECV_CONFIG = "aggregated_send_recv_config"
_MAX_SEND_RECV_AGGREGATION = 40


class ScheduleTransformError(Exception):
  """Raised when the transform cannot produce a valid schedule."""


@dataclasses.dataclass(frozen=True)
class ScheduleGraph:
  """One computation's schedule as plain data."""

  order: tuple[str, ...]
  opcodes: Mapping[str, str]
  operands: Mapping[str, tuple[str, ...]]
  users: Mapping[str, tuple[str, ...]]
  control_preds: Mapping[str, tuple[str, ...]]
  control_succs: Mapping[str, tuple[str, ...]]
  data_formatting: frozenset[str]
  pinned: frozenset[str]
  param_prefix_len: int
  root: str
  queues: Mapping[str, str]
  glue: Mapping[str, str]

  def positions(self) -> dict[str, int]:
    return {name: i for i, name in enumerate(self.order)}


def make_schedule_graph(
    order: Sequence[str],
    opcodes: Mapping[str, str],
    operands: Mapping[str, Sequence[str]],
    control_preds: Mapping[str, Sequence[str]] | None = None,
    data_formatting: Iterable[str] = (),
    queues: Mapping[str, str] | None = None,
    glue: Mapping[str, str] | None = None,
    serialized_queues: Iterable[str] = (),
    fences: Iterable[str] = (),
) -> ScheduleGraph:
  """Build a graph, deriving users, control successors, and pinned instructions."""
  order = tuple(order)
  if not order:
    raise ScheduleTransformError("Cannot build a graph from an empty schedule.")
  present = set(order)
  if len(present) != len(order):
    raise ScheduleTransformError("Schedule contains duplicate instructions.")

  control_preds = control_preds or {}
  queues = {k: v for k, v in (queues or {}).items() if k in present}
  serialized_queues = frozenset(serialized_queues)
  fences = frozenset(fences)
  operand_map: dict[str, tuple[str, ...]] = {}
  control_pred_lists: dict[str, list[str]] = {}
  user_lists: dict[str, list[str]] = {name: [] for name in order}
  control_succ_lists: dict[str, list[str]] = {name: [] for name in order}

  retired: dict[str, list[str]] = {}
  fence: str | None = None
  for name in order:
    operand_map[name] = tuple(o for o in operands.get(name, ()) if o in present)
    preds = [p for p in control_preds.get(name, ()) if p in present]
    queue = queues.get(name)
    if queue is not None and queue in serialized_queues:
      preds += [d for d in retired.get(queue, ()) if d not in preds]
    if opcodes.get(name) in _ASYNC_START_OPCODES and fence is not None:
      if fence not in preds:
        preds.append(fence)
    control_pred_lists[name] = preds
    if name in fences:
      fence = name
    if opcodes.get(name) in _ASYNC_DONE_OPCODES and operand_map[name]:
      done_queue = queues.get(operand_map[name][0])
      if done_queue is not None and done_queue in serialized_queues:
        retired.setdefault(done_queue, []).append(name)
  fence = None
  for name in reversed(order):
    if name in fences:
      fence = name
    elif fence is not None and opcodes.get(name) in _ASYNC_DONE_OPCODES:
      if name not in control_pred_lists[fence]:
        control_pred_lists[fence].append(name)

  control_pred_map = {name: tuple(control_pred_lists[name]) for name in order}
  for name in order:
    for operand in operand_map[name]:
      user_lists[operand].append(name)
    for pred in control_pred_map[name]:
      control_succ_lists[pred].append(name)

  root = order[-1]
  param_prefix_len = 0
  for name in order:
    if opcodes.get(name) != "kParameter":
      break
    param_prefix_len += 1
  pinned = frozenset([name for name in order if opcodes.get(name) == "kParameter"] + [root])

  return ScheduleGraph(
      order=order, opcodes=dict(opcodes), operands=operand_map,
      users={name: tuple(user_lists[name]) for name in order}, control_preds=control_pred_map,
      control_succs={name: tuple(control_succ_lists[name]) for name in order},
      data_formatting=frozenset(n for n in data_formatting if n in present), pinned=pinned,
      param_prefix_len=param_prefix_len, root=root, queues=queues,
      glue={k: v for k, v in (glue or {}).items() if k in present and v in present},
  )


@dataclasses.dataclass(frozen=True)
class AsyncPair:
  """An async start paired with its async done."""

  start: str
  done: str


@dataclasses.dataclass(frozen=True)
class Probe:
  """Where a bundle could move to, and what stopped it."""

  slot: int
  blocker: str | None


@dataclasses.dataclass(frozen=True)
class RescheduleResult:
  """Everything the core computed, for both application and debugging."""

  order: tuple[str, ...]
  pairs: tuple[AsyncPair, ...]
  start_bundles: Mapping[str, frozenset[str]]
  done_bundles: Mapping[str, frozenset[str]]
  start_probes: Mapping[str, Probe]
  done_probes: Mapping[str, Probe]
  start_bubble_order: tuple[str, ...]
  done_bubble_order: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class FifoViolation:
  """A pair of async ops whose starts and dones complete out of order."""

  group: str
  earlier_start: str
  later_start: str
  earlier_done: str
  later_done: str


def find_async_pairs(graph: ScheduleGraph) -> list[AsyncPair]:
  """Find async start/done pairs by matching a done to its first operand."""
  for name in graph.order:
    if graph.opcodes[name] == "kAsyncUpdate":
      raise ScheduleTransformError(f"kAsyncUpdate is not supported (instruction {name}).")

  pairs: list[AsyncPair] = []
  for name in graph.order:
    opcode = graph.opcodes[name]
    if opcode not in _ASYNC_DONE_OPCODES:
      continue
    operands = graph.operands[name]
    expected = _DONE_TO_START_OPCODE[opcode]
    if not operands or graph.opcodes.get(operands[0]) != expected:
      raise ScheduleTransformError(
          f"Async done {name} ({opcode}) has no {expected} as operand 0;" f" operands={operands}."
      )
    pairs.append(AsyncPair(start=operands[0], done=name))
  return pairs


def bundle_for_start(graph: ScheduleGraph, start: str) -> frozenset[str]:
  """Return start plus its transitive data-formatting producers."""
  return _closure(graph, start, graph.operands)


def bundle_for_done(graph: ScheduleGraph, done: str) -> frozenset[str]:
  """Return done plus its transitive data-formatting consumers."""
  return _closure(graph, done, graph.users)


def _closure(graph: ScheduleGraph, seed: str, edges: Mapping[str, tuple[str, ...]]) -> frozenset[str]:
  """Grow a bundle from seed along edges through data-formatting operations."""
  bundle = {seed}
  stack = [seed]
  while stack:
    current = stack.pop()
    for neighbor in edges[current]:
      if neighbor in bundle:
        continue
      if neighbor in graph.pinned:
        continue
      if neighbor not in graph.data_formatting:
        continue
      bundle.add(neighbor)
      stack.append(neighbor)
  return frozenset(bundle)


def earliest_slot(graph: ScheduleGraph, bundle: frozenset[str], positions: Mapping[str, int]) -> Probe:
  """Return the earliest legal start slot and its blocker."""
  slot = graph.param_prefix_len
  blocker: str | None = None
  for member in bundle:
    for dep in (*graph.operands[member], *graph.control_preds[member]):
      if dep in bundle:
        continue
      if positions[dep] + 1 > slot:
        slot = positions[dep] + 1
        blocker = dep
  return Probe(slot=slot, blocker=blocker)


def latest_slot(graph: ScheduleGraph, bundle: frozenset[str], positions: Mapping[str, int]) -> Probe:
  """Return the latest legal end slot and its blocker."""
  slot = positions[graph.root] - 1
  blocker: str | None = graph.root
  for member in bundle:
    for dep in (*graph.users[member], *graph.control_succs[member]):
      if dep in bundle:
        continue
      if positions[dep] - 1 < slot:
        slot = positions[dep] - 1
        blocker = dep
  return Probe(slot=slot, blocker=blocker)


def prune_bundle_for_start(
    graph: ScheduleGraph, seed: str, bundle: frozenset[str], positions: Mapping[str, int], min_slot: int = 0,
) -> tuple[frozenset[str], Probe]:
  """Shrink a start bundle until no member would have to move later."""
  current = set(bundle)
  while True:
    probe = earliest_slot(graph, frozenset(current), positions)
    target = max(probe.slot, min_slot)
    dropped = {m for m in current if positions[m] < target}
    if seed in dropped:
      if positions[seed] < probe.slot:
        raise ScheduleTransformError(
            f"Async start {seed} at {positions[seed]} precedes its own bundle" f" lower bound {probe.slot}."
        )
      dropped.discard(seed)
    if not dropped:
      return frozenset(current), probe
    current -= dropped


def prune_bundle_for_done(
    graph: ScheduleGraph, seed: str, bundle: frozenset[str], positions: Mapping[str, int],
    max_slot: int | None = None,
) -> tuple[frozenset[str], Probe]:
  """Shrink a done bundle until no member would have to move earlier."""
  current = set(bundle)
  while True:
    probe = latest_slot(graph, frozenset(current), positions)
    target = probe.slot if max_slot is None else min(probe.slot, max_slot)
    dropped = {m for m in current if positions[m] > target}
    if not dropped:
      return frozenset(current), probe
    if seed in dropped:
      alone = frozenset({seed})
      if earliest_slot(graph, alone, positions).slot > target:
        raise ScheduleTransformError(
            f"Async done {seed} at {positions[seed]} cannot move back to its" f" queue cap {target}."
        )
      return alone, latest_slot(graph, alone, positions)
    current -= dropped


def _cluster_start_slots(
    dones: Sequence[str], start_slots: Mapping[str, int], start_of_done: Mapping[str, str],
    glue: Mapping[str, str] | None = None,
) -> dict[str, int]:
  """Return the start slot each done is ranked by."""
  followers: dict[str, list[str]] = {}
  for done in dones:
    anchor = (glue or {}).get(done)
    if anchor is not None:
      followers.setdefault(anchor, []).append(done)
  slots: dict[str, int] = {}
  for done in dones:
    candidates = [-1]
    for member in (done, *followers.get(done, ())):
      start = start_of_done.get(member)
      if start in start_slots:
        candidates.append(start_slots[start])
    slots[done] = max(candidates)
  return slots


def order_completions(
    dones: Sequence[str], done_probes: Mapping[str, Probe], start_slots: Mapping[str, int],
    start_of_done: Mapping[str, str], glue: Mapping[str, str] | None = None,
) -> list[str]:
  """Order async dones as the queues should complete them."""
  ranked_starts = _cluster_start_slots(dones, start_slots, start_of_done, glue)

  def key(done: str) -> tuple[int, int, int, str]:
    done_slot = done_probes[done].slot
    start_slot = ranked_starts[done]
    return (start_slot + done_slot, done_slot, start_slot, done)

  return sorted(dones, key=key)


def order_dones(
    dones: Sequence[str], done_probes: Mapping[str, Probe], start_slots: Mapping[str, int],
    start_of_done: Mapping[str, str], total: int, glue: Mapping[str, str] | None = None,
) -> list[str]:
  """Order async dones as they should appear in the final schedule."""
  ranked_starts = _cluster_start_slots(dones, start_slots, start_of_done, glue)

  def key(done: str) -> tuple[int, int, str]:
    distance_from_end = total - 1 - done_probes[done].slot
    return (-distance_from_end, ranked_starts[done], done)

  return sorted(dones, key=key)


def fifo_constrained_slots(
    start_probes: Mapping[str, Probe], done_order: Sequence[str], start_of_done: Mapping[str, str],
    queues: Mapping[str, str],
) -> dict[str, int]:
  """Raise start slots so each hardware queue issues in completion order."""
  slots = {start: probe.slot for start, probe in start_probes.items()}
  last_on_queue: dict[str, int] = {}
  for done in done_order:
    start = start_of_done.get(done)
    if start is None or start not in slots:
      continue
    queue = queues.get(start)
    if queue is None:
      continue
    floor = last_on_queue.get(queue)
    if floor is not None and slots[start] < floor:
      slots[start] = floor
    last_on_queue[queue] = slots[start]
  return slots


def order_starts(
    starts: Sequence[str], start_slots: Mapping[str, int], done_order: Sequence[str], done_of_start: Mapping[str, str],
) -> list[str]:
  """Order async starts for the bubbling phase."""
  done_rank = {done: i for i, done in enumerate(done_order)}

  def key(start: str) -> tuple[int, int]:
    done = done_of_start.get(start)
    rank = done_rank.get(done, -1) if done is not None else -1
    return (-start_slots[start], -rank)

  return sorted(sorted(starts, reverse=True), key=key)


def defer_glued(names: Sequence[str], glue: Mapping[str, str]) -> list[str]:
  """Move glued followers to the back of a bubbling order."""
  return [n for n in names if n not in glue] + [n for n in names if n in glue]


def apply_glue_order(names: Sequence[str], glue: Mapping[str, str]) -> list[str]:
  """Reorder names so each glued follower sits directly before its anchor."""
  followers: dict[str, list[str]] = {}
  for name in names:
    anchor = glue.get(name)
    if anchor is not None:
      followers.setdefault(anchor, []).append(name)
  ordered: list[str] = []
  for anchor in names:
    if anchor in glue:
      continue
    ordered.extend(followers.get(anchor, ()))
    ordered.append(anchor)
  return ordered


def move_bundle_to_start_slot(order: Sequence[str], bundle: frozenset[str], slot: int) -> list[str]:
  """Move a bundle directly after the instruction at slot minus one."""
  members = [name for name in order if name in bundle]
  rest = [name for name in order if name not in bundle]
  index = slot - sum(1 for name in order[:slot] if name in bundle)
  return rest[:index] + members + rest[index:]


def move_bundle_to_end_slot(order: Sequence[str], bundle: frozenset[str], slot: int) -> list[str]:
  """Move a bundle so its last member sits at slot."""
  members = [name for name in order if name in bundle]
  rest = [name for name in order if name not in bundle]
  tail = len(order) - 1 - slot
  split = len(rest) - tail
  return rest[:split] + members + rest[split:]


def move_bundle_before(order: Sequence[str], bundle: frozenset[str], anchor: str) -> list[str]:
  """Move a bundle immediately before its anchor."""
  if anchor in bundle:
    raise ScheduleTransformError(f"Cannot park a bundle against {anchor}, which is one of its members.")
  members = [name for name in order if name in bundle]
  rest = [name for name in order if name not in bundle]
  index = rest.index(anchor)
  return rest[:index] + members + rest[index:]


def bubble_starts_left(
    graph: ScheduleGraph, order: Sequence[str], bubble_order: Sequence[str],
    bundles: Mapping[str, frozenset[str]], floors: Mapping[str, int] | None = None,
) -> tuple[list[str], dict[str, frozenset[str]]]:
  """Bubble each start bundle as far left as legal."""
  floors = floors or {}
  current = list(order)
  taken: dict[str, frozenset[str]] = {}
  for start in bubble_order:
    positions = {name: i for i, name in enumerate(current)}
    anchor = graph.glue.get(start)
    if anchor is not None:
      floor = positions[anchor]
    elif floors.get(start, 0) > 0:
      floor = positions[graph.order[floors[start] - 1]] + 1
    else:
      floor = 0
    bundle, probe = prune_bundle_for_start(graph, start, bundles[start], positions, floor)
    if anchor is not None and probe.slot <= positions[anchor]:
      current = move_bundle_before(current, bundle, anchor)
    else:
      current = move_bundle_to_start_slot(current, bundle, max(probe.slot, floor))
    taken[start] = bundle
  return current, taken


def bubble_dones_right(
    graph: ScheduleGraph, order: Sequence[str], bubble_order: Sequence[str],
    bundles: Mapping[str, frozenset[str]],
) -> tuple[list[str], dict[str, frozenset[str]]]:
  """Bubble each done bundle as far right as legal."""
  issue = {name: i for i, name in enumerate(order)}
  start_of = {done: graph.operands[done][0] for done in bubble_order}
  current = list(order)
  taken: dict[str, frozenset[str]] = {}
  placed: list[str] = []
  for done in bubble_order:
    positions = {name: i for i, name in enumerate(current)}
    queue = graph.queues.get(start_of[done])
    cappers = [
        other for other in placed
        if queue is not None and graph.queues.get(start_of[other]) == queue
        and issue[start_of[other]] > issue[start_of[done]]
    ]
    capper = min(cappers, key=positions.__getitem__, default=None)
    cap = None if capper is None else positions[capper] - 1
    bundle, probe = prune_bundle_for_done(graph, done, bundles[done], positions, cap)
    slot = probe.slot if cap is None else min(probe.slot, cap)
    anchor = graph.glue.get(done)
    if anchor is not None and positions[anchor] - 1 <= slot:
      current = move_bundle_before(current, bundle, anchor)
    elif capper is not None and slot < probe.slot:
      current = move_bundle_before(current, bundle, capper)
    else:
      current = move_bundle_to_end_slot(current, bundle, slot)
    taken[done] = bundle
    placed.append(done)
  return current, taken


def reschedule(graph: ScheduleGraph) -> RescheduleResult:
  """Run the full bubbling algorithm over a graph."""
  pairs = find_async_pairs(graph)
  done_of_start = {p.start: p.done for p in pairs}
  start_of_done = {p.done: p.start for p in pairs}
  starts = [p.start for p in pairs]
  dones = [p.done for p in pairs]
  positions = graph.positions()
  start_bundles: dict[str, frozenset[str]] = {}
  start_probes: dict[str, Probe] = {}
  for start in starts:
    bundle, probe = prune_bundle_for_start(graph, start, bundle_for_start(graph, start), positions)
    start_bundles[start] = bundle
    start_probes[start] = probe
  done_bundles: dict[str, frozenset[str]] = {}
  done_probes: dict[str, Probe] = {}
  for done in dones:
    bundle, probe = prune_bundle_for_done(graph, done, bundle_for_done(graph, done), positions)
    done_bundles[done] = bundle
    done_probes[done] = probe
  completion_order = order_completions(
      dones, done_probes, {start: probe.slot for start, probe in start_probes.items()}, start_of_done, graph.glue,
  )
  final_done_order = apply_glue_order(completion_order, graph.glue)
  start_slots = fifo_constrained_slots(start_probes, final_done_order, start_of_done, graph.queues)
  start_bubble_order = defer_glued(order_starts(starts, start_slots, final_done_order, done_of_start), graph.glue)
  order, moved_starts = bubble_starts_left(graph, graph.order, start_bubble_order, start_bundles, start_slots)
  positions = {name: i for i, name in enumerate(order)}
  for done in dones:
    bundle, probe = prune_bundle_for_done(graph, done, bundle_for_done(graph, done), positions)
    done_bundles[done] = bundle
    done_probes[done] = probe
  done_order = order_dones(
      dones, done_probes, {start: positions[start] for start in starts}, start_of_done, len(graph.order), graph.glue,
  )
  done_bubble_order = defer_glued(apply_glue_order(done_order, graph.glue), graph.glue)
  order, moved_dones = bubble_dones_right(graph, order, done_bubble_order, done_bundles)
  return RescheduleResult(
      order=tuple(order), pairs=tuple(pairs), start_bundles=moved_starts, done_bundles=moved_dones,
      start_probes=start_probes, done_probes=done_probes, start_bubble_order=tuple(start_bubble_order),
      done_bubble_order=tuple(done_bubble_order),
  )


def validate_order(graph: ScheduleGraph, order: Sequence[str]) -> None:
  """Check that order is a legal schedule for graph."""
  if len(order) != len(graph.order) or set(order) != set(graph.order):
    missing = set(graph.order) - set(order)
    extra = set(order) - set(graph.order)
    raise ScheduleTransformError(
        f"Schedule changed instruction set (missing={sorted(missing)[:5]},"
        f" extra={sorted(extra)[:5]}, len {len(order)} vs {len(graph.order)})."
    )
  positions = {name: i for i, name in enumerate(order)}
  for name in order:
    for operand in graph.operands[name]:
      if positions[operand] >= positions[name]:
        raise ScheduleTransformError(f"{name} is scheduled before its operand {operand}.")
    for pred in graph.control_preds[name]:
      if positions[pred] >= positions[name]:
        raise ScheduleTransformError(f"{name} is scheduled before its control predecessor {pred}.")
  if order[-1] != graph.root:
    raise ScheduleTransformError(f"Root {graph.root} is not last; {order[-1]} is.")
  prefix = tuple(order[: graph.param_prefix_len])
  if prefix != graph.order[: graph.param_prefix_len]:
    raise ScheduleTransformError(
        "Leading parameter run was disturbed:" f" {prefix[:5]} vs {graph.order[:graph.param_prefix_len][:5]}."
    )


def fifo_violations(
    order: Sequence[str], pairs: Iterable[AsyncPair], group_of: Callable[[AsyncPair], str | None],
) -> list[FifoViolation]:
  """Find async pairs on one queue that complete out of issue order."""
  positions = {name: i for i, name in enumerate(order)}
  grouped: dict[str, list[AsyncPair]] = {}
  for pair in pairs:
    group = group_of(pair)
    if group is None:
      continue
    grouped.setdefault(group, []).append(pair)
  violations: list[FifoViolation] = []
  for group in sorted(grouped):
    members = sorted(grouped[group], key=lambda p: positions[p.start])
    for earlier, later in zip(members, members[1:]):
      if positions[earlier.done] > positions[later.done]:
        violations.append(
            FifoViolation(
                group=group, earlier_start=earlier.start, later_start=later.start,
                earlier_done=earlier.done, later_done=later.done,
            )
        )
  return violations


def aggregation_statuses(
    order: Sequence[str], operands: Mapping[str, Sequence[str]], control_preds: Mapping[str, Sequence[str]],
    transfers: Iterable[str], dones: Iterable[str], free: Iterable[str], channel_ids: Mapping[str, int],
    max_group_size: int = _MAX_SEND_RECV_AGGREGATION,
) -> dict[str, str]:
  """Recompute AggregateSendRecv's DEFER/ISSUE stamps for a final schedule."""
  transfers = frozenset(transfers)
  dones = frozenset(dones)
  free = frozenset(free)
  statuses: dict[str, str] = {}
  group: set[str] = set()
  members: list[str] = []
  channels: set[int] = set()

  def issue() -> None:
    for i, name in enumerate(members):
      last = i == len(members) - 1
      statuses[name] = _AGGREGATE_ISSUE if last else _AGGREGATE_DEFER
    group.clear()
    members.clear()
    channels.clear()

  for name in order:
    if name in free:
      continue
    if name not in transfers and name not in dones:
      issue()
      continue
    if any(p in group and p in dones for p in control_preds.get(name, ())):
      issue()
    channel = channel_ids.get(name)
    if name in transfers:
      if any(o in group for o in operands.get(name, ())) or len(members) >= max_group_size or channel in channels:
        issue()
      members.append(name)
    group.add(name)
    if channel is not None:
      channels.add(channel)
  issue()
  return statuses


def stranded_defers(order: Sequence[str], statuses: Mapping[str, str]) -> list[str]:
  """Return DEFER'd operations with no subsequent ISSUE."""
  stranded: list[str] = []
  for name in order:
    status = statuses.get(name)
    if status == _AGGREGATE_ISSUE:
      stranded.clear()
    elif status == _AGGREGATE_DEFER:
      stranded.append(name)
  return stranded


def _parse_control_predecessors(instruction_text: str) -> tuple[str, ...]:
  """Extract control predecessor names from an instruction's to_string()."""
  match = _CONTROL_PREDECESSORS_RE.search(instruction_text)
  if match is None:
    return ()
  return tuple(_INSTRUCTION_NAME_RE.findall(match.group(1)))


def _fusion_is_data_formatting(
    instruction_text: str, computations_by_name: Mapping[str, "_hlo.HloComputation"], cache: dict[str, bool],
) -> bool:
  """Return whether a fusion contains only data-formatting operations."""
  match = _CALLS_RE.search(instruction_text)
  if match is None:
    return False
  callee = match.group(1).lstrip("%")
  if callee in cache:
    return cache[callee]
  computation = computations_by_name.get(callee)
  if computation is None:
    cache[callee] = False
    return False
  cache[callee] = False
  allowed = _DATA_FORMATTING_OPCODES | _FUSION_INTERNAL_EXTRA_OPCODES
  result = True
  for inner in computation.instructions():
    opcode = inner.opcode.name
    if opcode in allowed:
      continue
    if opcode == "kFusion" and _fusion_is_data_formatting(inner.to_string(), computations_by_name, cache):
      continue
    if opcode == "kCustomCall" and _custom_call_is_data_formatting(inner.to_string()):
      continue
    result = False
    break
  cache[callee] = result
  return result


def _custom_call_is_data_formatting(instruction_text: str) -> bool:
  """Return whether a custom call only reserves or reshapes storage."""
  match = _CUSTOM_CALL_TARGET_RE.search(instruction_text)
  return match is not None and match.group(1) in _DATA_FORMATTING_CUSTOM_CALLS


def async_queue(opcode: str, instruction_text: str) -> str | None:
  """Return the hardware FIFO on which an async start is issued, if any."""
  if _MEGASCALE_HANDLER in instruction_text:
    if opcode == "kSend":
      return _DCN_SEND_QUEUE
    if opcode == "kRecv":
      return _DCN_RECV_QUEUE
  if opcode == "kAsyncStart" and _HOST_MEMORY_SPACE in instruction_text:
    if " dynamic-slice-start(" in instruction_text:
      return _HOST_H2D_QUEUE
    if " dynamic-update-slice-start(" in instruction_text:
      return _HOST_D2H_QUEUE
  thread = _ASYNC_THREAD_RE.search(instruction_text)
  cores = _CORE_IDS_RE.search(instruction_text)
  if thread is None or cores is None:
    return None
  return f"{thread.group(1)}:{cores.group(1).strip()}"


def megascale_glue(sequence: Sequence["_hlo.HloInstruction"]) -> dict[str, str]:
  """Pair each megascale DCN recv with its send, and their dones."""
  recv_by_token: dict[str, str] = {}
  send_by_token: dict[str, str] = {}
  done_of: dict[str, str] = {}
  for inst in sequence:
    opcode = inst.opcode.name
    if opcode in ("kSendDone", "kRecvDone"):
      operands = inst.operands()
      if operands:
        done_of[operands[0].name] = inst.name
      continue
    if opcode not in ("kSend", "kRecv"):
      continue
    if _MEGASCALE_HANDLER not in inst.to_string():
      continue
    tokens = [o.name for o in inst.operands() if o.opcode.name == "kAfterAll"]
    if len(tokens) != 1:
      continue
    by_token = recv_by_token if opcode == "kRecv" else send_by_token
    by_token[tokens[0]] = inst.name

  glue: dict[str, str] = {}
  for token, recv in recv_by_token.items():
    send = send_by_token.get(token)
    if send is None:
      continue
    glue[recv] = send
    send_done = done_of.get(send)
    recv_done = done_of.get(recv)
    if send_done is not None and recv_done is not None:
      glue[send_done] = recv_done
  return glue


def _build_graph(
    sequence: Sequence["_hlo.HloInstruction"], computations_by_name: Mapping[str, "_hlo.HloComputation"],
    fusion_cache: dict[str, bool], async_bodies: Collection[str] = (),
) -> ScheduleGraph:
  """Build a graph from a scheduled instruction sequence."""
  order = [inst.name for inst in sequence]
  opcodes = {inst.name: inst.opcode.name for inst in sequence}
  operands = {inst.name: [o.name for o in inst.operands()] for inst in sequence}
  control_preds: dict[str, tuple[str, ...]] = {}
  data_formatting: list[str] = []
  queues: dict[str, str] = {}
  fences: list[str] = []
  for inst in sequence:
    text = inst.to_string()
    control_preds[inst.name] = _parse_control_predecessors(text)
    opcode = inst.opcode.name
    if opcode in _DATA_FORMATTING_OPCODES:
      data_formatting.append(inst.name)
    elif opcode == "kFusion" and _fusion_is_data_formatting(text, computations_by_name, fusion_cache):
      data_formatting.append(inst.name)
    elif opcode == "kCustomCall" and _custom_call_is_data_formatting(text):
      data_formatting.append(inst.name)
    if opcode in _ASYNC_START_OPCODES:
      queue = async_queue(opcode, text)
      if queue is not None:
        queues[inst.name] = queue
    elif opcode == "kWhile" and _while_body_name(text) in async_bodies:
      fences.append(inst.name)
  return make_schedule_graph(
      order=order, opcodes=opcodes, operands=operands, control_preds=control_preds,
      data_formatting=data_formatting, queues=queues, glue=megascale_glue(sequence),
      serialized_queues=_SERIALIZED_QUEUES, fences=fences,
  )


def _while_body_name(instruction_text: str) -> str | None:
  """Return the body computation named by a kWhile instruction, if any."""
  match = _BODY_RE.search(instruction_text)
  return None if match is None else match.group(1).lstrip("%")


def while_body_names(module: "_hlo.HloModule") -> set[str]:
  """Return the names of every computation used as a while body."""
  names: set[str] = set()
  for computation in module.make_nonfusion_computations():
    for inst in computation.instructions():
      if inst.opcode.name != "kWhile":
        continue
      body = _while_body_name(inst.to_string())
      if body is not None:
        names.add(body)
  return names


def async_issuing_body_names(module: "_hlo.HloModule") -> set[str]:
  """Return while bodies that issue async work directly or through nested loops."""
  issues_directly: dict[str, bool] = {}
  nested_bodies: dict[str, list[str]] = {}
  for computation in module.make_nonfusion_computations():
    issues_directly[computation.name] = False
    nested_bodies[computation.name] = []
    for inst in computation.instructions():
      opcode = inst.opcode.name
      if opcode in _ASYNC_START_OPCODES:
        issues_directly[computation.name] = True
      elif opcode == "kWhile":
        body = _while_body_name(inst.to_string())
        if body is not None:
          nested_bodies[computation.name].append(body)
  memo: dict[str, bool] = {}

  def issues(name: str) -> bool:
    if name not in memo:
      memo[name] = False
      memo[name] = issues_directly.get(name, False) or any(issues(nested) for nested in nested_bodies.get(name, ()))
    return memo[name]

  bodies = set().union(*nested_bodies.values())
  return {body for body in bodies if issues(body)}


def entry_computation_name(module: "_hlo.HloModule") -> str:
  """Return the name of the module's entry computation."""
  options = _hlo.HloPrintOptions()
  options.print_backend_config = False
  options.print_extra_attributes = False
  options.print_metadata = False
  options.print_operand_shape = False
  options.print_program_shape = False
  options.print_result_shape = False
  match = _ENTRY_RE.search(module.to_string(options))
  if match is None:
    raise ScheduleTransformError(f"Module {module.name} has no entry.")
  return match.group(1)


def default_computation_filter(module: "_hlo.HloModule") -> Callable[["_hlo.HloComputation"], bool]:
  """Return a predicate selecting the entry and every while body."""
  selected = while_body_names(module) | {entry_computation_name(module)}
  return lambda computation: computation.name in selected


def transform_module(
    module: "_hlo.HloModule",
    computation_filter: Callable[["_hlo.HloModule"], Callable[["_hlo.HloComputation"], bool]] | None = None,
) -> list[tuple[str, ScheduleGraph, RescheduleResult]]:
  """Rewrite the schedule of every selected computation in module."""
  schedule = module.schedule()
  if schedule is None:
    return []
  filter_factory = computation_filter or default_computation_filter
  selects = filter_factory(module)
  computations_by_name = {c.name: c for c in module.computations()}
  async_bodies = async_issuing_body_names(module)
  fusion_cache: dict[str, bool] = {}
  results: list[tuple[str, ScheduleGraph, RescheduleResult]] = []
  for computation in module.make_nonfusion_computations():
    if not selects(computation):
      continue
    try:
      sequence = schedule.sequence(computation)
    except Exception:  # pylint: disable=broad-except
      continue
    if not sequence:
      continue
    graph = _build_graph(sequence, computations_by_name, fusion_cache, async_bodies)
    result = reschedule(graph)
    validate_order(graph, result.order)
    by_name = {inst.name: inst for inst in sequence}
    schedule.set_sequence(computation, [by_name[name] for name in result.order])
    results.append((computation.name, graph, result))
  schedule.verify()
  module.set_schedule(schedule)
  return results


def restamp_send_recv_aggregation(serialized_hlo: bytes, computation_names: Iterable[str]) -> bytes:
  """Recompute AggregateSendRecv's DEFER/ISSUE stamps on rewritten schedules."""
  names = frozenset(computation_names)
  proto = hlo_pb2.HloModuleProto.FromString(serialized_hlo)
  for computation in proto.computations:
    if computation.name not in names:
      continue
    if computation.id not in proto.schedule.sequences:
      continue
    by_id = {inst.id: inst for inst in computation.instructions}
    sequence = [by_id[i] for i in proto.schedule.sequences[computation.id].instruction_ids]
    name_of = {inst.id: inst.name for inst in computation.instructions}
    configs: dict[str, dict[str, Any]] = {}
    transfers: set[str] = set()
    for inst in sequence:
      if inst.opcode not in ("send", "recv") or not inst.backend_config:
        continue
      config = json.loads(inst.backend_config)
      if _AGGREGATED_SEND_RECV_CONFIG in config:
        configs[inst.name] = config
        transfers.add(inst.name)
    dones = {
        inst.name for inst in sequence
        if inst.opcode in ("send-done", "recv-done") and inst.operand_ids
        and name_of.get(inst.operand_ids[0]) in transfers
    }
    free = {
        inst.name for inst in sequence
        if inst.opcode in ("parameter", "after-all") and not inst.operand_ids and not inst.control_predecessor_ids
    }
    order = [inst.name for inst in sequence]
    statuses = aggregation_statuses(
        order=order,
        operands={inst.name: [name_of[i] for i in inst.operand_ids] for inst in sequence},
        control_preds={inst.name: [name_of[i] for i in inst.control_predecessor_ids] for inst in sequence},
        transfers=transfers,
        dones=dones,
        free=free,
        channel_ids={
            inst.name: inst.channel_id for inst in sequence
            if (inst.name in transfers or inst.name in dones) and inst.channel_id
        },
    )
    stranded = stranded_defers(order, statuses)
    if stranded:
      raise ScheduleTransformError(f"DEFER'd transfers {stranded} in {computation.name} have no ISSUE" " after them.")
    for inst in sequence:
      status = statuses.get(inst.name)
      config = configs.get(inst.name)
      if status is None or config is None:
        continue
      if config[_AGGREGATED_SEND_RECV_CONFIG].get("status") == status:
        continue
      config[_AGGREGATED_SEND_RECV_CONFIG]["status"] = status
      inst.backend_config = json.dumps(config, separators=(",", ":")).encode()
  return proto.SerializeToString()


def max_async_overlap_transform(serialized_hlo: bytes) -> bytes:
  """Rewrite schedules and restamp send/recv aggregation for XLA."""
  from_proto = getattr(_hlo.HloModule, "from_serialized_hlo_module_proto")
  module: _hlo.HloModule = from_proto(serialized_hlo)
  results = transform_module(module)
  return restamp_send_recv_aggregation(
      module.as_serialized_hlo_module_proto(),
      [name for name, _, _ in results],
  )


def _blocker_label(graph: ScheduleGraph, blocker: str | None) -> str:
  """Render a blocking instruction for the debug report."""
  if blocker is None:
    return "-"
  return f"{blocker} ({graph.opcodes.get(blocker, '?')})"


def debug_report(serialized_hlo: bytes, max_bundle_members: int = 6) -> str:
  """Return a human-readable trace of the transform."""
  from_proto = getattr(_hlo.HloModule, "from_serialized_hlo_module_proto")
  module: _hlo.HloModule = from_proto(serialized_hlo)
  results = transform_module(module)
  lines: list[str] = []
  for name, graph, result in results:
    original = graph.positions()
    final = {n: i for i, n in enumerate(result.order)}
    lines.append("=" * 100)
    lines.append(f"COMPUTATION {name}: {len(graph.order)} instructions," f" {len(result.pairs)} async pairs")
    lines.append("=" * 100)
    lines.append("-- ASYNC STARTS (bubble order; final order is the reverse)")
    lines.append(f"{'#':<4}{'start':<40}{'orig':>7}{'probe':>7}{'final':>7}" f"  {'blocker':<34}bundle")
    for i, start in enumerate(result.start_bubble_order):
      probe = result.start_probes[start]
      members = sorted(result.start_bundles[start] - {start})
      shown = ", ".join(members[:max_bundle_members])
      if len(members) > max_bundle_members:
        shown += f", (+{len(members) - max_bundle_members} more)"
      lines.append(
          f"{i:<4}{start:<40}{original[start]:>7}{probe.slot:>7}{final[start]:>7}"
          f"  {_blocker_label(graph, probe.blocker):<40}{shown}"
      )
    lines.append("")
    lines.append("-- ASYNC DONES (final order; probed in the hoisted order)")
    lines.append(f"{'#':<4}{'done':<40}{'orig':>7}{'probe':>7}{'final':>7}" f"  {'blocker':<34}bundle")
    for i, done in enumerate(result.done_bubble_order):
      probe = result.done_probes[done]
      members = sorted(result.done_bundles[done] - {done})
      shown = ", ".join(members[:max_bundle_members])
      if len(members) > max_bundle_members:
        shown += f", (+{len(members) - max_bundle_members} more)"
      lines.append(
          f"{i:<4}{done:<40}{original[done]:>7}{probe.slot:>7}"
          f"{final[done]:>7}  {_blocker_label(graph, probe.blocker):<40}{shown}"
      )
    lines.append("")
  return "\n".join(lines)


def register_transform(name: str = _DEFAULT_TRANSFORM_NAME, platforms: Sequence[str] | str | None = None) -> None:
  """Register the post-scheduler transformation with JAX/XLA."""
  unregister_transform(name=name, platforms=platforms)
  jex_xla.register_hlo_module_transformation(
      max_async_overlap_transform, name=name, stage=jex_xla.PipelineStage.POST_SCHEDULER, platforms=platforms,
  )


def unregister_transform(name: str = _DEFAULT_TRANSFORM_NAME, platforms: Sequence[str] | str | None = None) -> bool:
  """Unregister the post-scheduler transformation from JAX/XLA."""
  return jex_xla.clear_hlo_module_transformation(
      name=name, stage=jex_xla.PipelineStage.POST_SCHEDULER, platforms=platforms,
  )
