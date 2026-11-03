"""Input pipeline for C4 MLPerf datasets."""

from collections.abc import Sequence
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as P

try:
  import tensorflow as tf
  import tensorflow_datasets as tfds
except ImportError as error:
  raise ImportError(
      "TensorFlow and tensorflow-datasets are required. Run `pip install tensorflow tensorflow-datasets`"
  ) from error

from lilax._backend.input_pipeline import multihost_dataloading
import datasets
import grain.python as grain
from grain._src.python.dataset.sources import tfrecord_dataset
from grain.experimental import TFRecordIterDataset
from lilax._backend.diffusion.block_diffusion import corruption as block_diffusion_corruption
from lilax._backend.input_pipeline import encoding_dsv4
from lilax._backend.input_pipeline.tfds.tfds_operations import get_tokenizer, TokenizeOp
from lilax._backend.input_pipeline.protos import example_pb2
from lilax._backend.multimodal import processor as mm_processor
from lilax._backend.input_pipeline.packing import sequence_packing
from lilax._backend.utils import max_logging
from lilax._backend.utils.sharding import remove_size_one_mesh_axis

AUTOTUNE = tf.data.experimental.AUTOTUNE


def _shift_left_and_pad(tensor, pad_val):
  v = tf.expand_dims(tensor, 0)
  if isinstance(v, tf.Tensor):
    v = tf.RaggedTensor.from_tensor(v)
  pad_shape = tf.concat([v.bounding_shape()[:-2], [1, 1]], axis=0)
  pad_tensor = tf.broadcast_to(pad_val, pad_shape)
  last_in_sequence = tf.concat([v[..., -1:, 1:], pad_tensor], axis=-1)
  v = tf.concat([v[..., :-1, :], last_in_sequence], axis=-2)
  return v[0]


def rekey(ds, key_map=None):
  def _rekey(x, key_map=None):
    if key_map:
      return {new_key: x[old_key] for new_key, old_key in key_map.items() if old_key}
    return x

  return ds.map(functools.partial(_rekey, key_map=key_map), num_parallel_calls=AUTOTUNE)


def reduce_concat_tokens(dataset, feature_key="targets", batch_size=128):
  dataset = dataset.map(lambda x: {feature_key: x[feature_key]}, num_parallel_calls=AUTOTUNE)
  dataset = dataset.padded_batch(batch_size, padded_shapes={feature_key: [-1]})

  def _my_fn(x):
    tokens = tf.reshape(x[feature_key], [-1])
    tokens = tf.boolean_mask(tokens, tf.cast(tokens, tf.bool))
    return {feature_key: tokens}

  return dataset.map(_my_fn, num_parallel_calls=AUTOTUNE)


def split_tokens(dataset, max_tokens_per_segment=128, feature_key="targets"):
  def _split_tokens(x):
    tokens = x[feature_key]
    n_tokens = tf.size(tokens)
    length = max_tokens_per_segment
    num_segments = tf.cast(tf.math.ceil(tf.cast(n_tokens, tf.float32) / tf.cast(length, tf.float32)), tf.int32)
    padding = num_segments * length - tf.size(tokens)
    tokens = tf.pad(tokens, [[0, padding]])
    return tf.reshape(tokens, [-1, length])

  def _strip_padding(x):
    return {feature_key: tf.boolean_mask(x, tf.cast(x, tf.bool))}

  dataset = dataset.filter(lambda x: tf.not_equal(tf.size(x[feature_key]), 0))
  dataset = dataset.map(_split_tokens, num_parallel_calls=AUTOTUNE)
  dataset = dataset.unbatch()
  return dataset.map(_strip_padding, num_parallel_calls=AUTOTUNE)


def split_tokens_to_targets_length(dataset, sequence_length):
  return split_tokens(dataset, max_tokens_per_segment=sequence_length)


def _pad_to_batch_size(ds, batch_size, num_examples=0, num_batches=0, dataloading_host_count=0):
  if num_examples > 0:
    local_num = num_examples
  else:
    local_num = 0
    for _ in ds:
      local_num += 1
  local_num_batches = (local_num + batch_size - 1) // batch_size
  if dataloading_host_count <= 0:
    dataloading_host_count = jax.process_count()
  if num_batches <= 0:
    if dataloading_host_count < jax.process_count():
      max_logging.log(
          f"Only {dataloading_host_count} of {jax.process_count()} processes load eval data, "
          "skipping the cross-host batch count allgather."
      )
      num_batches = local_num_batches
    else:
      num_batches_all = multihost_utils.process_allgather(jnp.array([local_num_batches]), tiled=False)
      num_batches = int(np.max(num_batches_all))
  pad_num = num_batches * batch_size - local_num
  assert pad_num >= 0
  max_logging.log(f"Eval data has {local_num} local entries, padding now with {pad_num} extra entries to get {num_batches} batches.")

  def _add_pad(x):
    x["targets_segmentation"] *= 0
    return x

  pad_ds = ds.take(1).map(_add_pad).repeat(pad_num)
  return ds.concatenate(pad_ds)


def get_local_rows_loading_real_data(data_sharding, global_batch_size_to_load, global_batch_size_to_train_on,
                                     max_target_length, mesh):
  local_batch_size = global_batch_size_to_load // jax.process_count()
  num_local_devices = len(mesh.local_devices)
  if num_local_devices == 0 or local_batch_size % num_local_devices != 0:
    return list(range(local_batch_size))
  rows_per_device = local_batch_size // num_local_devices
  data_sharding_pspec = remove_size_one_mesh_axis(P(*data_sharding), mesh)
  sharding = jax.sharding.NamedSharding(mesh, data_sharding_pspec)
  devices_indices_map = sharding.devices_indices_map((global_batch_size_to_load, max_target_length))
  local_rows = []
  for device_position, device in enumerate(mesh.local_devices):
    indices = devices_indices_map[device]
    if not indices[0].stop or indices[0].stop <= global_batch_size_to_train_on:
      local_rows.extend(range(device_position * rows_per_device, (device_position + 1) * rows_per_device))
  return local_rows


def _place_real_rows_in_local_batch(batch, local_batch_size: int, real_data_rows: Sequence[int]):
  rows = tf.constant(list(real_data_rows), dtype=tf.int32)
  output = {}
  for key, value in batch.items():
    indices = rows[: tf.shape(value)[0], tf.newaxis]
    shape = tf.concat([[local_batch_size], tf.shape(value)[1:]], axis=0)
    scattered = tf.scatter_nd(indices, value, shape)
    output[key] = tf.ensure_shape(scattered, [local_batch_size] + value.shape.as_list()[1:])
  return output


def get_dataset(dataset_name: str, split: str, dataloading_host_index: int, dataloading_host_count: int,
                data_dir: str | None = None, enable_data_shuffling: bool = False, data_shuffle_seed: int = 0,
                shard_in_read: bool = False, interleave_cycle_length: int = -1) -> tf.data.Dataset:
  extra_read_config = {}
  if interleave_cycle_length > 0:
    extra_read_config["interleave_cycle_length"] = interleave_cycle_length
  if shard_in_read:
    read_config = tfds.ReadConfig(
        shuffle_seed=data_shuffle_seed,
        input_context=tf.distribute.InputContext(
            input_pipeline_id=dataloading_host_index, num_input_pipelines=dataloading_host_count,
        ),
        **extra_read_config,
    )
    ds_builder = tfds.builder(dataset_name, data_dir=data_dir)
    ds_builder.download_and_prepare()
    ds = ds_builder.as_dataset(split=split, read_config=read_config, shuffle_files=enable_data_shuffling)
  else:
    read_config = tfds.ReadConfig(shuffle_seed=data_shuffle_seed, **extra_read_config)
    ds_builder = tfds.builder(dataset_name, data_dir=data_dir)
    ds = ds_builder.as_dataset(split=split, read_config=read_config, shuffle_files=enable_data_shuffling)
    ds = ds.shard(num_shards=dataloading_host_count, index=dataloading_host_index)
  return ds


def _files_hold_one_example_each(dataset_name: str, split: str, data_dir: str | None, dataloading_host_count: int) -> bool:
  file_instructions = tfds.builder(dataset_name, data_dir=data_dir).info.splits[split].file_instructions
  return len(file_instructions) >= dataloading_host_count and all(f.skip == 0 and f.take == 1 for f in file_instructions)


def _sequences_per_eval_file(dataset_name: str, split: str, data_dir: str | None, feature_key: str,
                             max_target_length: int) -> int:
  ds = tfds.builder(dataset_name, data_dir=data_dir).as_dataset(split=f"{split}[0:1]", shuffle_files=False)
  for example in ds.take(1):
    return (int(tf.size(example[feature_key])) + 1) // max_target_length
  return 0


def _shared_eval_file_assignment(host_index: int, local_num_examples: int, sequences_per_file: int) -> tuple[int, int] | None:
  if local_num_examples <= 0 or sequences_per_file <= local_num_examples:
    return None
  if sequences_per_file % local_num_examples != 0:
    return None
  start = host_index * local_num_examples
  return start // sequences_per_file, start % sequences_per_file


def format_fn(x, eos_id: int = 1, pad_id: int = 0):
  x["inputs"] = x["targets"]
  x["inputs_position"] = x["targets_position"]
  x["targets"] = _shift_left_and_pad(x["targets"], eos_id)
  x["inputs_segmentation"] = tf.where(
      tf.logical_and(x["targets"] != eos_id, x["targets"] != pad_id), x["targets_segmentation"], 0
  )
  x["targets_segmentation"] = x["inputs_segmentation"]
  return x


def chunk_token_stream(dataset, feature_key="targets", sequence_length=4096, eod_id: int | None = None):
  if eod_id is not None:
    eod_tensor = tf.constant([eod_id], dtype=tf.int32)
    ds = dataset.map(lambda x: tf.concat([tf.cast(x[feature_key], tf.int32), eod_tensor], axis=0), num_parallel_calls=AUTOTUNE)
  else:
    ds = dataset.map(lambda x: tf.cast(x[feature_key], tf.int32), num_parallel_calls=AUTOTUNE)
  ds = ds.unbatch()
  ds = ds.batch(sequence_length, drop_remainder=True)
  return ds.map(lambda tokens: {feature_key: tokens}, num_parallel_calls=AUTOTUNE)


def format_continuous_stream_fn(x, max_target_length: int, eod_id: int = 1):
  targets_raw = tf.cast(x["targets"], tf.int32)
  inputs = targets_raw
  targets = tf.concat([targets_raw[1:], [eod_id]], axis=0)
  inputs_position = tf.range(max_target_length, dtype=tf.int32)
  targets_position = inputs_position
  inputs_segmentation = tf.ones([max_target_length], dtype=tf.int32)
  targets_segmentation = tf.ones([max_target_length], dtype=tf.int32)
  return {
      "inputs": inputs, "targets": targets, "inputs_position": inputs_position, "targets_position": targets_position,
      "inputs_segmentation": inputs_segmentation, "targets_segmentation": targets_segmentation,
  }


def preprocess_train_dataset(train_ds: tf.data.Dataset, sp_tokenizer, train_global_batch_size_to_load: int,
                             max_target_length: int, shuffle_buffer_size: int, data_shuffle_seed: int,
                             is_tokenized_dataset: bool = False, use_stream_chunking: bool | None = None) -> tf.data.Dataset:
  pad_id = sp_tokenizer.pad_id
  eod_id = sp_tokenizer.eos_id
  if use_stream_chunking is None:
    use_stream_chunking = is_tokenized_dataset
  if not is_tokenized_dataset:
    train_ds = train_ds.map(
        lambda x: TokenizeOp(tokenizer_model=sp_tokenizer, features=x, data_keys=("targets",)), num_parallel_calls=AUTOTUNE
    )
  if use_stream_chunking:
    train_ds = train_ds.shuffle(shuffle_buffer_size, seed=data_shuffle_seed)
    train_ds = chunk_token_stream(train_ds, feature_key="targets", sequence_length=max_target_length, eod_id=eod_id)
    train_ds = train_ds.map(
        lambda x: format_continuous_stream_fn(x, max_target_length=max_target_length, eod_id=eod_id),
        num_parallel_calls=AUTOTUNE,
    )
  else:
    train_ds = reduce_concat_tokens(train_ds, feature_key="targets", batch_size=4096)
    train_ds = split_tokens_to_targets_length(train_ds, max_target_length)
    train_ds = train_ds.shuffle(shuffle_buffer_size, seed=data_shuffle_seed)
    train_ds = sequence_packing.pack_dataset(train_ds, max_target_length, pad_id=pad_id)
    train_ds = train_ds.map(lambda x: format_fn(x, pad_id=pad_id), num_parallel_calls=AUTOTUNE)
  train_ds = train_ds.batch(train_global_batch_size_to_load // jax.process_count(), drop_remainder=True)
  return train_ds.prefetch(AUTOTUNE)


def preprocess_eval_dataset(eval_ds: tf.data.Dataset, sp_tokenizer, eval_global_batch_size_to_load: int,
                            max_target_length: int, num_examples: int = 0, is_tokenized_dataset: bool = True,
                            use_stream_chunking: bool | None = None, real_data_rows: Sequence[int] | None = None,
                            num_batches: int = 0, dataloading_host_count: int = 0,
                            skip_examples: int = 0) -> tf.data.Dataset:
  pad_id = sp_tokenizer.pad_id
  eod_id = sp_tokenizer.eos_id
  if use_stream_chunking is None:
    use_stream_chunking = is_tokenized_dataset
  if not is_tokenized_dataset:
    eval_ds = eval_ds.map(
        lambda x: TokenizeOp(tokenizer_model=sp_tokenizer, features=x, data_keys=("targets",)), num_parallel_calls=AUTOTUNE
    )
  if use_stream_chunking:
    eval_ds = chunk_token_stream(eval_ds, feature_key="targets", sequence_length=max_target_length, eod_id=eod_id)
    eval_ds = eval_ds.map(
        lambda x: format_continuous_stream_fn(x, max_target_length=max_target_length, eod_id=eod_id),
        num_parallel_calls=AUTOTUNE,
    )
  else:
    eval_ds = reduce_concat_tokens(eval_ds, feature_key="targets", batch_size=24567)
    eval_ds = split_tokens_to_targets_length(eval_ds, max_target_length)
    eval_ds = sequence_packing.pack_dataset(eval_ds, max_target_length, pad_id=pad_id)
    eval_ds = eval_ds.map(lambda x: format_fn(x, pad_id=pad_id), num_parallel_calls=AUTOTUNE)
  if skip_examples > 0:
    eval_ds = eval_ds.skip(skip_examples)
  local_batch_size = eval_global_batch_size_to_load // jax.process_count()
  if real_data_rows is None:
    real_data_rows = range(local_batch_size)
  real_data_rows = list(real_data_rows)
  real_batch_size = len(real_data_rows)
  eval_ds = _pad_to_batch_size(eval_ds, real_batch_size, num_examples, num_batches, dataloading_host_count)
  eval_ds = eval_ds.batch(real_batch_size, drop_remainder=False)
  if real_data_rows != list(range(local_batch_size)):
    eval_ds = eval_ds.map(
        functools.partial(_place_real_rows_in_local_batch, local_batch_size=local_batch_size, real_data_rows=real_data_rows),
        num_parallel_calls=AUTOTUNE,
    )
  if num_batches > 0:
    eval_ds = eval_ds.take(num_batches)
  eval_ds = eval_ds.cache()
  return eval_ds.prefetch(AUTOTUNE)


def make_c4_mlperf_train_iterator(config, global_mesh, process_indices):
  train_col = config.train_data_columns[0]
  is_tokenized_dataset = not config.tokenize_train_data
  train_split = getattr(config, "train_split", "train2")
  train_data_dir = getattr(config, "dataset_path", None)
  train_ds = get_dataset(
      dataset_name=config.dataset_name, split=train_split,
      dataloading_host_index=process_indices.index(jax.process_index()), dataloading_host_count=len(process_indices),
      data_dir=train_data_dir, enable_data_shuffling=config.enable_data_shuffling, data_shuffle_seed=config.data_shuffle_seed,
      shard_in_read=config.train_shard_in_read, interleave_cycle_length=config.train_interleave_cycle_length,
  )
  train_ds = rekey(train_ds, {"inputs": None, "targets": train_col})
  sp_tokenizer = get_tokenizer(
      config.tokenizer_path, config.tokenizer_type, config.add_bos, config.add_eos, config.hf_access_token
  )
  use_stream_chunking = getattr(config, "use_stream_chunking", None)
  train_ds = preprocess_train_dataset(
      train_ds, sp_tokenizer=sp_tokenizer, train_global_batch_size_to_load=config.global_batch_size_to_load,
      max_target_length=config.max_target_length, shuffle_buffer_size=128, data_shuffle_seed=config.data_shuffle_seed,
      is_tokenized_dataset=is_tokenized_dataset, use_stream_chunking=use_stream_chunking,
  )
  return multihost_dataloading.MultiHostDataLoadIterator(train_ds, global_mesh)


def make_c4_mlperf_eval_iterator(config, global_mesh, process_indices, num_examples: int = 0):
  eval_col = config.eval_data_columns[0]
  is_tokenized_dataset = not config.tokenize_eval_data
  eval_split = getattr(config, "eval_split", None)
  if eval_split is None:
    if config.eval_dataset_name == "c4/en:3.0.4":
      eval_split = "validation_24567exp"
    elif config.eval_dataset_name in ["c4/en:3.0.1", "c4/en:3.0.5", "c4/en:3.0.8", "c4/en:3.0.9"]:
      eval_split = "validation"
    else:
      raise ValueError(
          f"{config.eval_dataset_name=} should be one of "
          "('c4/en:3.0.1', 'c4/en:3.0.4', 'c4/en:3.0.5', 'c4/en:3.0.8', 'c4/en:3.0.9')"
      )
  eval_data_dir = getattr(config, "dataset_path", None)
  eval_host_index = process_indices.index(jax.process_index())
  eval_shard_in_read = _files_hold_one_example_each(
      config.eval_dataset_name, eval_split, eval_data_dir, len(process_indices)
  )
  eval_ds = get_dataset(
      dataset_name=config.eval_dataset_name, split=eval_split, dataloading_host_index=eval_host_index,
      dataloading_host_count=len(process_indices), data_dir=eval_data_dir, enable_data_shuffling=False,
      shard_in_read=eval_shard_in_read,
  )
  eval_ds = rekey(eval_ds, {"inputs": None, "targets": eval_col})
  sp_tokenizer = get_tokenizer(
      config.tokenizer_path, config.tokenizer_type, config.add_bos, config.add_eos, config.hf_access_token
  )
  global_batch_size_to_load_eval = config.global_batch_size_to_load_eval
  global_batch_size_to_eval_on = getattr(config, "global_batch_size_to_eval_on", 0) or global_batch_size_to_load_eval
  local_batch_size = global_batch_size_to_load_eval // jax.process_count()
  real_data_rows = list(range(local_batch_size))
  if global_batch_size_to_eval_on < global_batch_size_to_load_eval:
    rows = get_local_rows_loading_real_data(
        config.data_sharding, global_batch_size_to_load_eval, global_batch_size_to_eval_on,
        config.max_target_length, global_mesh,
    )
    if rows:
      real_data_rows = rows
    else:
      max_logging.log(
          f"Host {jax.process_index()} loads real eval data but holds no evaluated row, "
          "filling the whole local batch with real examples."
      )
  max_logging.log(
      f"Eval data loading host {jax.process_index()} fills {len(real_data_rows)} of "
      f"{local_batch_size} local rows with real examples."
  )
  if num_examples <= 0:
    eval_steps = getattr(config, "eval_steps", -1)
    eval_batch_size = global_batch_size_to_eval_on
    if eval_steps > 0 and eval_batch_size > 0:
      num_examples = int(eval_steps * eval_batch_size)
  local_num_examples = 0
  num_batches = 0
  if num_examples > 0:
    host_idx = process_indices.index(jax.process_index()) if jax.process_index() in process_indices else 0
    per_host, remainder = divmod(num_examples, len(process_indices))
    local_num_examples = per_host + (1 if host_idx < remainder else 0)
    max_local_num_examples = per_host + (1 if remainder else 0)
    num_batches = (max_local_num_examples + len(real_data_rows) - 1) // len(real_data_rows)
  use_stream_chunking = getattr(config, "use_stream_chunking", None)
  effective_stream_chunking = is_tokenized_dataset if use_stream_chunking is None else use_stream_chunking
  skip_examples = 0
  if eval_shard_in_read and effective_stream_chunking and num_examples > 0 and num_examples % len(process_indices) == 0:
    sequences_per_file = _sequences_per_eval_file(
        config.eval_dataset_name, eval_split, eval_data_dir, eval_col, config.max_target_length
    )
    assignment = _shared_eval_file_assignment(eval_host_index, local_num_examples, sequences_per_file)
    if assignment is not None:
      file_index, skip_examples = assignment
      eval_ds = tfds.builder(config.eval_dataset_name, data_dir=eval_data_dir).as_dataset(
          split=f"{eval_split}[{file_index}:{file_index + 1}]", shuffle_files=False
      )
      eval_ds = rekey(eval_ds, {"inputs": None, "targets": eval_col})
      max_logging.log(
          f"Eval data loading host {jax.process_index()} (loading index {eval_host_index} of "
          f"{len(process_indices)}) reads eval file {file_index}, sequences "
          f"[{skip_examples}, {skip_examples + local_num_examples}) of {sequences_per_file}."
      )
  eval_ds = preprocess_eval_dataset(
      eval_ds, sp_tokenizer=sp_tokenizer, eval_global_batch_size_to_load=global_batch_size_to_load_eval,
      max_target_length=config.max_target_length, num_examples=local_num_examples,
      is_tokenized_dataset=is_tokenized_dataset, use_stream_chunking=use_stream_chunking,
      real_data_rows=real_data_rows, num_batches=num_batches, dataloading_host_count=len(process_indices),
      skip_examples=skip_examples,
  )
  return multihost_dataloading.MultiHostDataLoadIterator(eval_ds, global_mesh)
