"""Generate and save sparse teacher logits for offline distillation."""

from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import os
import re
import subprocess
import time

from array_record.python import array_record_module
from etils import epath
from flax import nnx
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from lilax._backend.utils import max_logging


def get_top_k_logits(logits: jax.Array, k: int):
  """Extract top-k logit values and vocabulary indices."""
  top_k_values, top_k_indices = jax.lax.top_k(logits, k)
  return top_k_values, top_k_indices


def get_start_step(gcs_upload_path, local_tmp_dir, steps_per_file):
  """Determine the generation start step from existing files."""
  if jax.process_index() != 0:
    return 0
  output_dir = gcs_upload_path if gcs_upload_path else local_tmp_dir
  output_path = epath.Path(output_dir)
  if not output_path.exists():
    output_path.mkdir(parents=True, exist_ok=True)
    return 0
  existing_files = list(output_path.glob("teacher_top_k_part_*.array_record"))
  if not existing_files:
    return 0
  max_part_num = max(
      (int(match.group(1)) for file in existing_files if (match := re.search(r"part_(\d+)_host", file.name))),
      default=-1,
  )
  if max_part_num == -1:
    return 0
  start_step = max_part_num * steps_per_file
  max_logging.log(f"Found existing data, resuming from step {start_step}")
  return start_step


def create_tf_example(example_dict):
  """Serialize one example as a TensorFlow Example."""
  features = {}
  for key, value in example_dict.items():
    if key == "sequence_hash":
      features[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
      continue
    flat_val = np.asarray(value).ravel()
    if flat_val.dtype in [np.float32, np.float64, np.float16, jnp.bfloat16]:
      if flat_val.dtype != np.float32:
        flat_val = flat_val.astype(np.float32)
      features[key] = tf.train.Feature(float_list=tf.train.FloatList(value=flat_val.tolist()))
    elif flat_val.dtype in [np.int32, np.int64]:
      if flat_val.dtype != np.int64:
        flat_val = flat_val.astype(np.int64)
      features[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=flat_val.tolist()))
    else:
      raise ValueError(f"Unsupported dtype {flat_val.dtype} for key {key}")
  return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()


def background_process_and_write(writer, tokens, vals, idx, opt_data, serialization_executor):
  """Serialize batches in parallel and write records in order."""
  tokens_np = np.asarray(tokens)
  vals_np = np.asarray(vals)
  idx_np = np.asarray(idx)
  opt_data_np = {key: np.asarray(value) for key, value in opt_data.items()}
  batch_size = tokens_np.shape[0]
  example_dicts = []
  for index in range(batch_size):
    seq_bytes = tokens_np[index].tobytes()
    example_dict = {
        "inputs": tokens_np[index], "top_k_logits": vals_np[index], "top_k_indices": idx_np[index],
        "sequence_hash": hash(seq_bytes),
    }
    for key, val_np in opt_data_np.items():
      example_dict[key] = val_np[index]
    example_dicts.append(example_dict)
  serialized_records = list(serialization_executor.map(create_tf_example, example_dicts))
  for record in serialized_records:
    writer.write(record)


def background_upload(local_path, gcs_path, process_index):
  """Upload a completed local shard to GCS."""
  try:
    subprocess.run(["gcloud", "storage", "cp", local_path, gcs_path], check=True, capture_output=True)
    os.remove(local_path)
    if process_index == 0:
      max_logging.log(f"Background upload complete: {gcs_path}")
  except subprocess.CalledProcessError as error:
    if process_index == 0:
      max_logging.log(f"Upload failed for {local_path}: {error.stderr.decode()}")


@nnx.jit(static_argnames=("k",))
def teacher_step(model, batch, k):
  """Run the teacher forward pass and extract top-k logits."""
  logits = model(
      decoder_input_tokens=batch["inputs"], decoder_positions=batch["inputs_position"],
      decoder_segment_ids=batch.get("inputs_segmentation"), decoder_target_tokens=batch.get("targets"),
      decoder_target_mask=batch.get("targets_segmentation"), enable_dropout=False,
  )
  return get_top_k_logits(logits, k=k)


def generate_and_save_data(
    config, teacher_model, mesh, train_iterator, *, top_k=128,
    optional_keys=("inputs_position", "inputs_segmentation", "targets_segmentation", "targets"),
    gcs_upload_path=None, local_tmp_dir="/tmp", steps_per_file=50, upload_workers=4, serialization_workers=16,
):
  """Generate teacher logits and write local or GCS ArrayRecord shards."""
  writer = None
  local_output_path = None
  if not os.path.exists(local_tmp_dir):
    os.makedirs(local_tmp_dir, exist_ok=True)
  upload_executor = ThreadPoolExecutor(max_workers=upload_workers)
  write_executor = ThreadPoolExecutor(max_workers=1)
  serialization_executor = ThreadPoolExecutor(max_workers=serialization_workers)
  start_step = get_start_step(gcs_upload_path, local_tmp_dir, steps_per_file)
  start_step = int(multihost_utils.broadcast_one_to_all(jnp.array(start_step)))
  multihost_utils.sync_global_devices("start_generation_loop")
  with jax.set_mesh(mesh):
    if jax.process_index() == 0:
      max_logging.log(f"Starting Distributed Top-K generation loop for {config.steps - start_step} steps...")
    loop_start = time.time()
    for step, batch in enumerate(islice(train_iterator, start_step, config.steps), start=start_step):
      step_start = time.time()
      if step % steps_per_file == 0:
        if writer:
          write_executor.shutdown(wait=True)
          writer.close()
          if gcs_upload_path:
            gcs_file_path = os.path.join(gcs_upload_path, os.path.basename(local_output_path))
            if jax.process_index() == 0:
              max_logging.log(f"Queueing distributed background uploads for Step {step}...")
            upload_executor.submit(background_upload, local_output_path, gcs_file_path, jax.process_index())
          write_executor = ThreadPoolExecutor(max_workers=1)
        file_index = step // steps_per_file
        filename = f"teacher_top_k_part_{file_index:05d}_host_{jax.process_index():03d}.array_record"
        local_output_path = os.path.join(local_tmp_dir, filename)
        writer = array_record_module.ArrayRecordWriter(local_output_path, "group_size:1")
      tokens = batch["inputs"]
      top_k_vals, top_k_idx = teacher_step(teacher_model, batch, top_k)
      global_tokens = multihost_utils.process_allgather(tokens, tiled=True)
      global_vals = multihost_utils.process_allgather(top_k_vals, tiled=True)
      global_idx = multihost_utils.process_allgather(top_k_idx, tiled=True)
      optional_data = {}
      for key in optional_keys:
        if key in batch:
          optional_data[key] = multihost_utils.process_allgather(batch[key], tiled=True)
      if writer:
        global_tokens_np = np.array(global_tokens)
        global_vals_np = np.array(global_vals)
        global_idx_np = np.array(global_idx)
        optional_data_np = {key: np.array(value) for key, value in optional_data.items()}
        global_batch_size = global_tokens_np.shape[0]
        local_batch_size = global_batch_size // jax.process_count()
        start_idx = jax.process_index() * local_batch_size
        end_idx = start_idx + local_batch_size
        local_tokens_np = global_tokens_np[start_idx:end_idx]
        local_vals_np = global_vals_np[start_idx:end_idx]
        local_idx_np = global_idx_np[start_idx:end_idx]
        local_opt_data_np = {key: value[start_idx:end_idx] for key, value in optional_data_np.items()}
        write_executor.submit(
            background_process_and_write, writer, local_tokens_np, local_vals_np, local_idx_np, local_opt_data_np,
            serialization_executor,
        )
      if step % 50 == 0 and jax.process_index() == 0:
        max_logging.log(f"Successfully processed step {step} in {time.time() - step_start:.4f}s")
      multihost_utils.sync_global_devices(f"step_{step}_complete")
    if jax.process_index() == 0:
      max_logging.log(f"Generation loop finished in {time.time() - loop_start:.2f}s")
  multihost_utils.sync_global_devices("loop_finished")
  if writer:
    if write_executor:
      write_executor.shutdown(wait=True)
    if serialization_executor:
      serialization_executor.shutdown(wait=True)
    writer.close()
    if gcs_upload_path:
      gcs_file_path = os.path.join(gcs_upload_path, os.path.basename(local_output_path))
      upload_executor.submit(background_upload, local_output_path, gcs_file_path, jax.process_index())
    if upload_executor:
      if jax.process_index() == 0:
        max_logging.log("Waiting for all background uploads to finish across all hosts...")
      upload_executor.shutdown(wait=True)
      if jax.process_index() == 0:
        max_logging.log("All GCS uploads complete.")
  multihost_utils.sync_global_devices("upload_complete")
