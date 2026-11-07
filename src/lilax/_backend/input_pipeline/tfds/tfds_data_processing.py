"""Input pipeline for TFDS datasets."""

import functools
import math

import jax

try:
  import tensorflow as tf
  import tensorflow_datasets as tfds
except ImportError as error:
  raise ImportError(
      "The deprecated TFDS pipeline requires TensorFlow and tensorflow-datasets.\n\n"
      "Recommended: migrate to Grain by setting dataset_type=grain and grain_file_type=tfrecord. Existing "
      "dataset_path, dataset_name, train_split, eval_dataset_name, and eval_split settings will be used to "
      "automatically construct file paths when grain_train_files or grain_eval_files are not provided. Set the "
      "grain_train_files or grain_eval_files explicitly for a custom path.\n\n"
      "To continue using the deprecated TFDS pipeline, run: `pip install tensorflow tensorflow-datasets`."
  ) from error

import datasets
from lilax._backend.input_pipeline.sources import host_sharding
from grain._src.python.dataset.sources import tfrecord_dataset
from grain.experimental import TFRecordIterDataset
from lilax._backend.diffusion.block_diffusion import corruption as block_diffusion_corruption
from lilax._backend.input_pipeline import encoding_dsv4
from lilax._backend.input_pipeline.tfds import tfds_operations
from lilax._backend.input_pipeline.protos import example_pb2
from lilax._backend.multimodal import processor as mm_processor
from lilax._backend.input_pipeline import multihost_dataloading
from lilax._backend.input_pipeline.packing import sequence_packing
from lilax._backend.observability import max_logging

AUTOTUNE = tf.data.experimental.AUTOTUNE

try:
  tf.config.experimental.set_visible_devices([], "GPU")
except tf.errors.NotFoundError:
  pass


def get_datasets(dataset_name, data_split, shuffle_files, shuffle_seed, dataloading_host_index,
                 dataloading_host_count, dataset_path=None):
  ds_builder = tfds.builder(dataset_name, data_dir=dataset_path)
  read_config = tfds.ReadConfig(shuffle_seed=shuffle_seed) if shuffle_files else tfds.ReadConfig()
  num_shards = ds_builder.info.splits[data_split].num_shards
  _, _, row_shard = host_sharding.compute_file_sharding(num_shards, dataloading_host_index, dataloading_host_count)
  if num_shards >= dataloading_host_count:
    input_pipeline_id, num_input_pipelines = dataloading_host_index, dataloading_host_count
  else:
    input_pipeline_id, num_input_pipelines = dataloading_host_index % num_shards, num_shards
    max_logging.warning(
        f"{dataset_name}: shard count ({num_shards}) < host count ({dataloading_host_count}). "
        f"Each shard will be read by at most {math.ceil(dataloading_host_count / num_shards)} hosts. "
        "Concurrent reading by multiple hosts may cause slow down."
    )
  read_config.input_context = tf.distribute.InputContext(
      input_pipeline_id=input_pipeline_id, num_input_pipelines=num_input_pipelines
  )
  ds = ds_builder.as_dataset(split=data_split, read_config=read_config, shuffle_files=shuffle_files)
  if row_shard is not None:
    ds = ds.shard(num_shards=row_shard[1], index=row_shard[0])
  return ds


def preprocessing_pipeline(
    dataset, tokenizer_path, tokenizer_type: str, global_batch_size: int, max_target_length: int,
    data_column_names, shuffle: bool = False, data_shuffle_seed=0, tokenize: bool = True,
    add_bos: bool = True, add_eos: bool = True, num_epochs: None | int = 1, pack_examples: bool = True,
    shuffle_buffer_size: int = 1024, shift: bool = True, drop_remainder: bool = True,
    prefetch_size=tf.data.experimental.AUTOTUNE, hf_access_token: str = "",
):
  missing = [c for c in data_column_names if c not in dataset.element_spec]
  if missing:
    raise ValueError(
        f"Column {missing} not found in dataset. Available columns: {sorted(dataset.element_spec.keys())}. "
        "Please set train_data_columns or eval_data_columns accordingly."
    )
  for col in data_column_names:
    col_dtype = dataset.element_spec[col].dtype
    if tokenize and col_dtype != tf.string:
      raise ValueError(
          f"tokenize_data=True but column '{col}' has dtype {col_dtype} (expected tf.string). "
          "Set tokenize_train_data or tokenize_eval_data to False if your dataset is already tokenized."
      )
    if not tokenize and col_dtype == tf.string:
      raise ValueError(
          f"tokenize_data=False but column '{col}' has dtype tf.string (expected integer). "
          "Set tokenize_train_data or tokenize_eval_data to True if your dataset needs tokenization."
      )
  assert len(data_column_names) == 1
  dataset = dataset.map(lambda x: tfds_operations.normalize_features(x, data_column_names[0]), num_parallel_calls=AUTOTUNE)
  data_column_names = ("inputs", "targets")
  tokenizer_model = tfds_operations.get_tokenizer(tokenizer_path, tokenizer_type, add_bos, add_eos, hf_access_token)
  if tokenizer_model.pad_id is not None:
    pad_id = tokenizer_model.pad_id
  elif tokenizer_model.unk_id is not None:
    pad_id = tokenizer_model.unk_id
  else:
    pad_id = -1
  if tokenize:
    dataset = dataset.map(
        lambda x: tfds_operations.TokenizeOp(tokenizer_model=tokenizer_model, features=x, data_keys=data_column_names),
        num_parallel_calls=AUTOTUNE,
    )
  if max_target_length > 0:
    dataset = dataset.map(
        lambda x: tfds_operations.truncate_to_max_allowable_length(x, max_target_length + 1),
        num_parallel_calls=AUTOTUNE,
    )
  if shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size, seed=data_shuffle_seed)
  dataset = dataset.repeat(num_epochs)
  if shift:
    dataset = dataset.map(tfds_operations.shift_data_by_truncation, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
  if pack_examples:
    dataset = sequence_packing.pack_dataset(dataset, max_target_length, pad_id)
    dataset = dataset.batch(global_batch_size // jax.process_count(), drop_remainder=drop_remainder)
  else:
    dataset = dataset.padded_batch(
        global_batch_size // jax.process_count(), padded_shapes={k: max_target_length for k in data_column_names},
        padding_values={k: pad_id for k in data_column_names}, drop_remainder=drop_remainder,
    )
    dataset = dataset.map(
        lambda x: tfds_operations.add_segmentation_and_position(x, data_column_names, padding_token=pad_id),
        num_parallel_calls=tf.data.AUTOTUNE, deterministic=True,
    )
  if prefetch_size:
    dataset = dataset.prefetch(prefetch_size)
  return dataset


def make_tfds_train_iterator(config, global_mesh, process_indices_train):
  assert config.global_batch_size_to_load % global_mesh.size == 0, "Batch size should be divisible by number of global devices."
  get_datasets_kwargs = {
      "dataset_name": config.dataset_name, "dataset_path": config.dataset_path, "data_split": config.train_split,
      "shuffle_files": config.enable_data_shuffling, "shuffle_seed": config.data_shuffle_seed,
  }
  preprocessing_fn = functools.partial(
      preprocessing_pipeline, tokenizer_path=config.tokenizer_path, tokenizer_type=config.tokenizer_type,
      global_batch_size=config.global_batch_size_to_load, max_target_length=config.max_target_length,
      data_column_names=config.train_data_columns, shuffle=config.enable_data_shuffling,
      data_shuffle_seed=config.data_shuffle_seed, tokenize=config.tokenize_train_data, add_bos=config.add_bos,
      add_eos=config.add_eos, num_epochs=config.num_epoch, pack_examples=config.packing, hf_access_token=config.hf_access_token,
  )
  if not config.colocated_python_data_input:
    train_ds = get_datasets(
        dataloading_host_index=process_indices_train.index(jax.process_index()),
        dataloading_host_count=len(process_indices_train), **get_datasets_kwargs,
    )
    return multihost_dataloading.MultiHostDataLoadIterator(
        preprocessing_fn(dataset=train_ds), global_mesh, config.generate_padding_batch_train
    )
  return multihost_dataloading.RemoteIteratorWrapper(
      functools.partial(get_datasets, **get_datasets_kwargs), preprocessing_fn, global_mesh,
      (config.global_batch_size_to_load, config.max_target_length), checkpoint_path=config.checkpoint_dir,
  )


def make_tfds_eval_iterator(config, global_mesh, process_indices_eval):
  assert config.global_batch_size_to_load_eval % global_mesh.size == 0, "Batch size should be divisible by number of global devices."
  get_ds_fn = functools.partial(
      get_datasets, dataset_name=config.eval_dataset_name, dataset_path=config.dataset_path, data_split=config.eval_split,
      shuffle_files=False, shuffle_seed=config.data_shuffle_seed,
  )
  preprocessing_fn = functools.partial(
      preprocessing_pipeline, tokenizer_path=config.tokenizer_path, tokenizer_type=config.tokenizer_type,
      global_batch_size=config.global_batch_size_to_load_eval, max_target_length=config.max_target_length,
      data_column_names=config.eval_data_columns, shuffle=False, data_shuffle_seed=config.data_shuffle_seed,
      tokenize=config.tokenize_eval_data, add_bos=config.add_bos, add_eos=config.add_eos,
      pack_examples=config.packing, hf_access_token=config.hf_access_token,
  )
  if not config.colocated_python_data_input:
    eval_ds = get_ds_fn(
        dataloading_host_index=process_indices_eval.index(jax.process_index()), dataloading_host_count=len(process_indices_eval)
    )
    return multihost_dataloading.MultiHostDataLoadIterator(
        preprocessing_fn(dataset=eval_ds), global_mesh, config.generate_padding_batch_eval
    )
  return multihost_dataloading.RemoteIteratorWrapper(
      get_ds_fn, preprocessing_fn, global_mesh, (config.global_batch_size_to_load_eval, config.max_target_length),
      checkpoint_path=config.checkpoint_dir,
  )
