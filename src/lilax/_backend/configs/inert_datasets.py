from typing import Literal

from pydantic import BaseModel, Field

from lilax._backend.configs.dataset_types import DatasetType


class DatasetGeneral(BaseModel):
  """General configuration for dataset and data loading."""

  dataset_type: DatasetType = Field(DatasetType.SYNTHETIC, description="The type of the data loading pipeline.")
  per_device_batch_size: int | float = Field(12, description="The batch size per device.")
  eval_per_device_batch_size: int | float = Field(0.0, description="The batch size per device for evaluation. Defaults to per_device_batch_size.")
  max_corpus_chars: int = Field(10_000_000, description="Maximum number of characters to use from the corpus.")
  train_data_columns: list[str] = Field(["text"], description="Column(s) to use from the training data.")
  train_image_column: str | list[str] = Field("image", description="Column name(s) for images in the training data.")
  eval_data_columns: list[str] = Field(["text"], description="Column(s) to use from the evaluation data.")
  eval_image_column: str | list[str] = Field("image", description="Column name(s) for images in evaluation data.")
  default_prompt: str = Field("", description="Default prompt injected into the dataset when the prompt column is missing.")
  packing: bool = Field(True, description="Whether to pack multiple short examples into a single sequence.")
  grain_packing_type: Literal["first_fit", "best_fit", "concat_then_split"] = Field(
      "first_fit", description="Packing type when using Grain pipeline. 'first_fit', 'best_fit' or 'concat_then_split'."
  )
  max_segments_per_seq: int = Field(-1, description="Maximum number of segments that can be packed into a single sequence. -1 or None for no limit.")
  use_stream_chunking: bool | None = Field(None, description=(
      "Whether to use continuous stream chunking (unpacked monolithic token stream with 0% padding, "
      "monotonic positions, and uniform cross-document attention) for c4_mlperf datasets. "
      "If None, defaults to True for pre-tokenized datasets and False for raw text."
  ))
  num_epoch: int = Field(1, description="Number of epochs to train for.")
  expansion_factor_real_data: float = Field(-1.0, description="Factor for partial data loading on hosts.")
  reuse_example_batch: int = Field(0, description="For performance testing, repeatedly uses the same batch.")
  generate_padding_batch_train: bool = Field(False, description="Whether to generate a padding batch for training to ensure divisibility.")
  generate_padding_batch_eval: bool = Field(False, description="Whether to generate a padding batch for evaluation to ensure divisibility.")
  enable_rampup_batch_size: bool = Field(False, description="Enable rampup batch size.")
  per_device_batch_size_start: float = Field(4.0, description="Start per device batch size for rampup.")
  per_device_batch_size_increment: float = Field(2.0, description="Increment for per device batch size for rampup.")
  global_rampup_samples: int = Field(500, description="Target number of training samples for rampup.")
  colocated_python_data_input: bool = Field(False, description="Experimental feature for Pathways.")


class TfdsDataset(BaseModel):
  """Configuration specific to TFDS datasets."""

  dataset_path: str = Field("", description="Path to the TFDS dataset.")
  dataset_name: str = Field("c4/en:3.1.0", description="Name of the TFDS dataset.")
  eval_dataset_name: str = Field("c4/en:3.1.0", description="Name of the TFDS eval dataset.")
  train_split: str = Field("train", description="Dataset split for training.")
  eval_split: str = Field("validation", description="Dataset split for evaluation.")


class HfDataset(BaseModel):
  """Configuration specific to HuggingFace datasets."""

  hf_path: str = Field("", description="Path of the Hugging Face dataset.")
  hf_name: str | None = Field(None, description="Name of the Hugging Face dataset.")
  hf_data_dir: str | None = Field(None, description="Data directory for the HF dataset.")
  hf_train_files: str | None = Field(None, description="Files for the HF training split.")
  hf_eval_split: str | None = Field(None, description="Name of the HF evaluation split.")
  hf_eval_files: str | None = Field(None, description="Files for the HF evaluation split.")
  hf_access_token: str | None = Field(None, description="Hugging Face API access token.")


class GrainDataset(BaseModel):
  """Configuration specific to Grain datasets."""

  grain_train_files: str = Field("", description="Training source specification for the selected Grain file type.")
  grain_eval_files: str = Field("", description="Evaluation source specification for the selected Grain file type.")
  grain_train_mixture_config_path: str = Field("", description="Path to an ArrayRecord JSON file specifying Grain training mixture weights.")
  grain_file_type: str = Field("arrayrecord", description="File type for Grain data. Supported: arrayrecord, tfrecord, parquet, mmap, mmap_npy.")
  grain_use_elastic_iterator: bool = Field(False, description=(
      "Whether to use grain's `ElasticIterator` for data loading. When True, the iterator"
      "checkpoint can be restored after a change in the number of data-loading shards."
  ))
  grain_worker_count: int = Field(1, description="Number of workers for Grain data loading.")
  grain_per_worker_buffer_size: int = Field(1, description="Per-worker buffer size for Grain train data loading.")
  grain_worker_count_eval: int = Field(1, description="Number of workers for Grain eval data loading.")
  grain_per_worker_buffer_size_eval: int = Field(1, description="Per-worker buffer size for Grain eval data loading.")
  grain_ram_budget_mb: int = Field(1024, description="RAM budget (MB) for auto-tuning worker count.")
  grain_num_threads: int = Field(16, description="Number of threads for Grain ReadOptions during training.")
  grain_prefetch_buffer_size: int = Field(500, description="Prefetch buffer size for Grain ReadOptions during training.")
  grain_num_threads_eval: int = Field(16, description="Number of threads for Grain ReadOptions during evaluation.")
  grain_prefetch_buffer_size_eval: int = Field(500, description="Prefetch buffer size for Grain ReadOptions during evaluation.")
  grain_data_source_max_workers: int = Field(16, description="Max workers for ThreadPoolExecutor when mixing multiple Grain data sources.")
  grain_index_storage_option: Literal["in_memory", "offloaded"] | None = Field(None, description=(
      "ArrayRecord reader index storage. None uses the ArrayRecord reader default. Do not use 'offloaded' with "
      "direct gs:// paths because it can significantly degrade input performance. For Cloud Storage, use a "
      "filesystem with metadata caching, such as GCSFUSE."
  ))
  grain_shuffle_buffer_size: int = Field(100, description="Shuffle buffer size when using Parquet or TFRecord.")


class MMapDataset(BaseModel):
  """Configuration for ``grain_file_type='mmap'`` and ``'mmap_npy'``."""

  mmap_eod_id: int = Field(0, description="EOD token ID already present in the preprocessed Megatron data.")
  blend_cache_dir: str = Field("", description="Optional runtime cache for multi-dataset mmap_npy global blend dispatch indices.")
  blend_index_dir: str = Field("", description="Optional directory containing pre-generated dataset_index.npy and dataset_sample_index.npy.")
  reset_attention_mask: bool = Field(True, description="Start a new attention segment and reset positions after every retained EOD boundary.")
  eod_mask_loss: bool = Field(False, description="Exclude positions whose input token is EOD from the loss.")
  packing_max_segments_per_sample: int = Field(25, description=(
      "Short-segment merge divisor: max_target_length // value. Only used with reset_attention_mask=True; "
      "set <=0 to retain every EOD boundary."
  ))
  mmap_split_sentences: bool = Field(False, description="Whether preprocessing used --split-sentences; enables document-level indexing when true.")
  mmap_npy_split: str = Field("", description="mmap_npy split ratio, e.g. '99,1'; training uses split 0 and evaluation uses split 1.")


class OlmoGrainDataset(BaseModel):
  """Configuration for the OLMo numpy fixed-seq-length input pipeline (dataset_type=olmo_grain).

  Separate from the standard grain config because this pipeline reads
  pre-tokenized fixed-length sequences from raw npy files (one ``int32``
  token per element, ``sequence_length`` from an index JSON), not
  arrayrecord/tfds shards — so flags like ``grain_train_files`` /
  ``packing`` don't apply.

  Worker count, per-worker buffer size, and shuffle seed reuse the standard
  grain flags (``grain_worker_count``, ``grain_per_worker_buffer_size``,
  ``data_shuffle_seed``); only OLMo-specific fields are listed here.
  """

  olmo_index_path: str = Field("", description="Path or gs:// URI to the JSON index from build_olmo_npy_index.py.")
  olmo_path_remap_from: str = Field("", description="If set, rewrite index file paths starting with this prefix to olmo_path_remap_to.")
  olmo_path_remap_to: str = Field("", description="Replacement prefix used together with olmo_path_remap_from (e.g. /mnt/disks/.../).")
  olmo_apply_ngram_filter: bool = Field(True, description="Mask repetitive instances per OLMo-core's repetition filter.")
