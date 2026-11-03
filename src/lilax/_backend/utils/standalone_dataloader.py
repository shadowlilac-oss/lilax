"""Load training batches without running training steps."""

import datetime
import os

import jax
import numpy as np

from lilax._backend.trainers.pre_train.train import get_first_step
from lilax._backend.utils import max_logging
from lilax._backend.utils.train_utils import setup_train_loop, validate_train_config


def data_load_loop(config, state=None):
  """Load batches for the configured training steps and return the training state."""
  _, _, _, model, _, _, _, data_loader, _, _, state = setup_train_loop(config, recorder=None)

  start = datetime.datetime.now()
  start_step = get_first_step(model, state)
  example_batch = data_loader.load_next_batch()
  jax.block_until_ready(example_batch)
  first_end = datetime.datetime.now()
  time_to_load_first_batch = first_end - start
  if jax.process_index() == 0:
    max_logging.log(
        f"STANDALONE DATALOADER : First step completed in {time_to_load_first_batch.seconds} seconds, on host 0"
    )

  for _ in np.arange(start_step + 1, config.steps):
    example_batch = data_loader.load_next_batch()

  jax.block_until_ready(example_batch)
  end = datetime.datetime.now()
  if jax.process_index() == 0:
    max_logging.log(f"STANDALONE DATALOADER : {config.steps} batches loaded in {(end-start).seconds} seconds, on host 0")
  return state


def run(config):
  """Validate a training configuration and load its batches."""
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  validate_train_config(config)
  max_logging.log(f"Found {jax.device_count()} devices.")
  max_logging.log(f"Found {jax.process_count()} processes.")
  max_logging.log(f"Found {jax.devices()} devices.")
  if config.dataset_type in ("tfds", "c4_mlperf"):
    os.environ["TFDS_DATA_DIR"] = config.dataset_path
  return data_load_loop(config)
