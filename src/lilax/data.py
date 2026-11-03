"""Public data pipeline APIs."""

try:
  import triton  # pylint: disable=unused-import
except ImportError:
  pass

from datasets import load_dataset

from lilax._backend.input_pipeline.distillation_generation import generate_distillation_data
from lilax._backend.input_pipeline.distillation_data_processing import (
    InputRequest,
    filter_dataset as filter_distillation_dataset,
    load_dataset as load_distillation_dataset,
    process_dataset as process_distillation_dataset,
)
from lilax._backend.input_pipeline.input_pipeline_interface import create_data_iterator
from lilax._backend.input_pipeline.tokenizer import (
    HFTokenizer,
    SentencePieceTokenizer,
    TikTokenTokenizer,
    build_tokenizer,
)
from lilax._backend.utils.mmap_index_builder import (
    build_blend_indices,
    build_document_index,
    build_indices,
    build_sample_index,
    build_shuffle_index,
    compute_index_hash,
    compute_num_epochs,
    convert,
    convert_blend,
    discover_shards,
    get_document_sizes,
    resolve_shard_prefixes,
    should_separate_last_epoch,
)
from lilax._backend.utils.standalone_dataloader import data_load_loop
