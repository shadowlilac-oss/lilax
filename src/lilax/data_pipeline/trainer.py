"""Resolve Trainer dataset descriptors into the existing sharded input pipelines."""

from lilax._backend.configs.pyconfig import HyperParameters
from lilax._backend.configs.dataset_types import DatasetType
from lilax._backend.input_pipeline.input_pipeline_interface import create_data_iterator
from lilax.data import SyntheticDataset, TeacherLogitsDataset


def validate_eval_dataset(dataset):
  sources = dataset.values() if isinstance(dataset, dict) else (dataset,)
  if any(isinstance(source, SyntheticDataset) for source in sources):
    raise ValueError("SyntheticDataset is unbounded and supports training only; supply finite evaluation data.")


def dataset_overrides(dataset):
  if isinstance(dataset, SyntheticDataset):
    return {"dataset_type": DatasetType.SYNTHETIC, "reuse_example_batch": dataset.reuse_example_batch,
            "max_segments_per_seq": dataset.max_segments_per_seq}
  if isinstance(dataset, TeacherLogitsDataset):
    return {"dataset_type": DatasetType.GRAIN, "grain_train_files": dataset.files, "grain_eval_files": dataset.files,
            "grain_file_type": "arrayrecord", "train_data_columns": ["inputs"], "eval_data_columns": ["inputs"],
            "tokenize_train_data": False, "tokenize_eval_data": False, "is_offline_distillation": True}
  return {"dataset_type": DatasetType.HF} if dataset is not None else {}


def create_data_iterators(
    config, mesh, *, train_dataset=None, eval_dataset=None, processing_class=None, data_collator=None,
    formatting_func=None, dataset_text_field="text", evaluation_only=False,
):
  validate_eval_dataset(eval_dataset)
  if isinstance(eval_dataset, TeacherLogitsDataset) and config.colocated_python_data_input:
    raise NotImplementedError("Finite TeacherLogitsDataset evaluation requires colocated_python_data_input=False; "
                              "the remote iterator has no synchronized exhaustion protocol.")
  options = dict(processing_class=processing_class, data_collator=data_collator, formatting_func=formatting_func,
                 dataset_text_field=dataset_text_field)
  train_iterator = eval_iterator = None
  if not evaluation_only:
    train_config = HyperParameters(config._pydantic_config.model_copy(
        update=dataset_overrides(train_dataset) | {"eval_interval": 0},
    ))
    source = None if isinstance(train_dataset, (SyntheticDataset, TeacherLogitsDataset)) else train_dataset
    train_iterator, _ = create_data_iterator(train_config, mesh, train_dataset=source, **options)
  if eval_dataset is not None:
    updates = dataset_overrides(eval_dataset) | {"eval_interval": 1, "eval_steps": -1,
                                              "generate_padding_batch_eval": False}
    eval_config = HyperParameters(config._pydantic_config.model_copy(update=updates))
    source = None if isinstance(eval_dataset, (SyntheticDataset, TeacherLogitsDataset)) else eval_dataset
    _, eval_iterator = create_data_iterator(
        eval_config, mesh, eval_dataset=source, evaluation_only=True, finite_evaluation=True, **options,
    )
  return train_iterator, eval_iterator
