"""Input pipeline."""

import functools

import jax
from jax.sharding import PartitionSpec as P

from lilax._backend.configs import pyconfig
from lilax._backend.input_pipeline.grain_data_processing import make_grain_train_iterator, make_grain_eval_iterator
from lilax._backend.input_pipeline.hf_data_processing import make_hf_train_iterator, make_hf_eval_iterator
from lilax._backend.input_pipeline.olmo.olmo_grain_data_processing import (
    make_olmo_grain_train_iterator,
    make_olmo_grain_eval_iterator,
)
from lilax._backend.input_pipeline.synthetic_data_processing import SyntheticDataIterator, PlaceHolderDataIterator
from lilax._backend.utils import max_logging
from lilax._backend.distributed.sharding import remove_size_one_mesh_axis


def get_process_loading_real_data(
    data_sharding, global_batch_size_to_load, global_batch_size_to_train_on, max_target_length, mesh
):
  """Get processes loading real data."""
  data_sharding_pspec = remove_size_one_mesh_axis(P(*data_sharding), mesh)
  sharding = jax.sharding.NamedSharding(mesh, data_sharding_pspec)
  devices_indices_map = sharding.devices_indices_map((global_batch_size_to_load, max_target_length))
  batch_cutoff = global_batch_size_to_train_on
  process_loading_real_data = set()
  for device, indices in devices_indices_map.items():
    if not indices[0].stop or indices[0].stop <= batch_cutoff:
      process_loading_real_data.add(device.process_index)
  return list(process_loading_real_data)


def create_process_specific_iterator(
    config: pyconfig.HyperParameters, mesh, process_indices, input_iterator, is_training: bool = True
):
  """Create a real iterator on loading processes and a placeholder elsewhere."""
  if jax.process_index() in process_indices:
    iterator_fn = functools.partial(input_iterator, config, mesh, process_indices)
    output_iterator = iterator_fn()
  else:
    output_iterator = PlaceHolderDataIterator(config, mesh, is_training=is_training)
  return output_iterator


def create_data_iterator(
    config: pyconfig.HyperParameters, mesh, *, train_dataset=None, eval_dataset=None, processing_class=None
):
  """Create train and eval data iterators given configs and mesh."""
  if config.dataset_type == "synthetic":
    eval_iterator = SyntheticDataIterator(config, mesh) if config.eval_interval > 0 else None
    return SyntheticDataIterator(config, mesh), eval_iterator

  dataset_type_to_train_eval_iterator = {
      "grain": (make_grain_train_iterator, make_grain_eval_iterator),
      "hf": (make_hf_train_iterator, make_hf_eval_iterator),
      "olmo_grain": (make_olmo_grain_train_iterator, make_olmo_grain_eval_iterator),
  }
  if config.dataset_type in ("tfds", "c4_mlperf"):
    from lilax._backend.input_pipeline.tfds.tfds_data_processing import make_tfds_train_iterator, make_tfds_eval_iterator  # pylint: disable=import-outside-toplevel
    from lilax._backend.input_pipeline.tfds.tfds_data_processing_c4_mlperf import make_c4_mlperf_train_iterator, make_c4_mlperf_eval_iterator  # pylint: disable=import-outside-toplevel

    dataset_type_to_train_eval_iterator["tfds"] = (make_tfds_train_iterator, make_tfds_eval_iterator)
    dataset_type_to_train_eval_iterator["c4_mlperf"] = (make_c4_mlperf_train_iterator, make_c4_mlperf_eval_iterator)
  if config.dataset_type in ["tfds", "grain", "hf", "c4_mlperf", "olmo_grain"]:
    if config.dataset_type == "c4_mlperf":
      assert config.packing, "c4_mlperf dataloader only works with packing. For padded version, use tfds dataloader"
    train_iterator, eval_iterator = dataset_type_to_train_eval_iterator[config.dataset_type]
    if config.dataset_type == "hf":
      train_iterator = functools.partial(train_iterator, dataset=train_dataset, processing_class=processing_class)
      eval_iterator = functools.partial(eval_iterator, dataset=eval_dataset, processing_class=processing_class)
  else:
    max_logging.log(
        f"WARNING: '{config.dataset_type}' is not a supported dataset type."
        "Using synthetic data. Please choose from 'tfds', 'grain', 'hf', or 'c4_mlperf'."
    )
    output_train_iterator, output_eval_iterator = SyntheticDataIterator(config, mesh), None
    return output_train_iterator, output_eval_iterator

  process_indices_train = get_process_loading_real_data(
      config.data_sharding,
      config.global_batch_size_to_load,
      config.global_batch_size_to_train_on,
      config.max_target_length,
      mesh,
  )
  output_train_iterator = create_process_specific_iterator(config, mesh, process_indices_train, train_iterator)
  if config.expansion_factor_real_data > 1:
    assert len(process_indices_train) == jax.process_count() // config.expansion_factor_real_data

  output_eval_iterator = None
  if config.eval_interval > 0:
    process_indices_eval = get_process_loading_real_data(
        config.data_sharding,
        config.global_batch_size_to_load_eval,
        config.global_batch_size_to_eval_on,
        config.max_target_length,
        mesh,
    )
    if config.expansion_factor_real_data > 1:
      assert len(process_indices_eval) == jax.process_count() // config.expansion_factor_real_data
    output_eval_iterator = create_process_specific_iterator(
        config, mesh, process_indices_eval, eval_iterator, is_training=False
    )
  return output_train_iterator, output_eval_iterator
