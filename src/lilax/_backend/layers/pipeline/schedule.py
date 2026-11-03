# Copyright 2023-2026 Google LLC
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

"""Stage schedule of a (circular) pipeline: which microbatch and repeat each stage runs on each loop iteration."""

import dataclasses

from jax import numpy as jnp

from lilax._backend.common.common_types import Config


@dataclasses.dataclass(frozen=True)
class PipelineSchedule:
  """Static mapping from pipeline loop iterations to per-stage microbatches, repeats and buffer slots."""

  num_stages: int
  forwarding_delay: int
  num_microbatches: int
  num_repeats: int

  @classmethod
  def from_config(cls, config: Config) -> "PipelineSchedule":
    return cls(
        num_stages=config.ici_pipeline_parallelism * config.dcn_pipeline_parallelism,
        forwarding_delay=2 if config.pipeline_delay_activation_forwarding else 1,
        num_microbatches=config.num_pipeline_microbatches,
        num_repeats=config.num_pipeline_repeats,
    )

  @property
  def microbatches_per_stage(self):
    return self.num_microbatches // self.num_stages

  @property
  def use_circ_storage(self):
    return self.num_repeats > 1 and self.num_microbatches > self.num_stages * self.forwarding_delay

  @property
  def bubble_iterations(self):
    return self.forwarding_delay * (self.num_stages - 1)

  @property
  def total_iterations(self):
    # Each microbatch should go through each stage (with repeats) - so there is num_micro * (num_stages * repeats)
    # compute to perform
    # Each iteration is vmapped by num_stages, so the number of iterations should be
    # num_micro * num_stages * repeats / num_stages = num_micro * repeats
    # However due to the pipeline bubble some iterations process less than num_stages microbatches. It takes
    # num_micro * repeat iterations for the last microbatch to start the final repeat, then an additional
    # num_stages - 1 to finish the final repeat.
    # Thus the total iterations is num_micro * repeat + num_stages - 1, & we may consider the num_stages - 1 as bubble.
    # The bubble doubles when we use forwarding delay.
    return self.num_microbatches * self.num_repeats + self.bubble_iterations

  def iterations_to_complete_first_microbatch_one_repeat(self):
    # Return the number of iterations it takes for microbatch 0 to finish a repeat
    return self.forwarding_delay * (self.num_stages - 1)

  def iterations_to_complete_first_microbatch(self):
    # Return the number of iterations it takes for microbatch 0 to finish the last stage of the last repeat
    return self.num_microbatches * (self.num_repeats - 1) + self.iterations_to_complete_first_microbatch_one_repeat()

  def output_microbatch_0_index(self):
    # The first real output (microbatch 0) takes a certain amount of loop iterations to finish and be pushed to
    # state_io - it will land on a different index of state_io depending on the number of iterations.
    return self.iterations_to_complete_first_microbatch() % self.microbatches_per_stage

  def stage_microbatches_processed(self, loop_iteration):
    # Stage 0 has processed one microbatch every loop_iter, but Stage 1 is 1 behind due to bubble, etc for other stages
    return jnp.maximum(loop_iteration - self.forwarding_delay * jnp.arange(self.num_stages), 0)

  def microbatch_and_repeat_ids(self, microbatches_processed):
    """Splits per-stage processed-microbatch counts into microbatch ids and repeat ids."""
    microbatch_ids = microbatches_processed % self.num_microbatches
    repeat_ids = microbatches_processed // self.num_microbatches
    return microbatch_ids, repeat_ids

  def state_io_index(self, loop_iteration):
    # state_io has a rotating microbatch index (size of microbatches_per_stage)
    return loop_iteration % self.microbatches_per_stage

  def circ_storage_read_index(self, loop_iteration):
    # circ_storage also has a rotating index for microbatch, size of num_microbatches
    return loop_iteration % self.num_microbatches

  def circ_storage_write_index(self, loop_iteration):
    # We rotate the pushing index into circ storage, and ensure that microbatch 0 lands in index 0
    return (loop_iteration - self.iterations_to_complete_first_microbatch_one_repeat() - 1) % self.num_microbatches
