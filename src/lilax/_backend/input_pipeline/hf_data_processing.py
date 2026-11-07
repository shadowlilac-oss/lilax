"""Input pipeline using Huggingface datasets."""

from typing import Optional

import datasets
import ml_collections
import jax
import transformers
import grain.python as grain

from lilax._backend.input_pipeline import data_processing_utils
from lilax._backend.configs.batch_sizes import uses_tunix_gradient_accumulation
from lilax._backend.input_pipeline import dpo_utils
from lilax._backend.input_pipeline.sources import hf_data_source
from lilax._backend.input_pipeline.transforms import block_diffusion_transform
from lilax._backend.input_pipeline.conversation import chat_template_formatting
from lilax._backend.input_pipeline.conversation import conversational_columns
from lilax._backend.input_pipeline.transforms import hf_tokenization
from lilax._backend.input_pipeline.sft import pad_token_remap
from lilax._backend.input_pipeline.sft import sft_masking
from lilax._backend.input_pipeline.sft import formatting
from lilax._backend.input_pipeline.transforms.collation import BatchCollator, PreferenceBatchCollator
from lilax._backend.input_pipeline.multimodal import sft_preprocessing
from lilax._backend.input_pipeline.multimodal import batch_transforms
from lilax._backend.input_pipeline.multimodal import qwen3_omni_position_transform
from lilax._backend.input_pipeline.transforms import feature_transforms
from lilax._backend.input_pipeline.transforms import pad_or_trim
from lilax._backend.input_pipeline.transforms import target_shift
from lilax._backend.input_pipeline.packing import reformat_packing
from lilax._backend.input_pipeline import instruction_data_processing
from lilax._backend.input_pipeline import multihost_dataloading
from lilax._backend.distributed import elastic


def _get_pad_id(tokenizer):
  if tokenizer.pad_token_id is not None:
    pad_id = tokenizer.pad_token_id
  elif tokenizer.unk_token_id is not None:
    pad_id = tokenizer.unk_token_id
  else:
    pad_id = -1
  return pad_id


def _dataset_columns(dataset):
  features = dataset.features or dataset.column_names
  return next(iter(dataset), {}) if features is None else features


def _language_model_columns(dataset, columns, tokenize, text_field):
  features = _dataset_columns(dataset)
  if set(columns).issubset(features):
    return columns
  return [text_field if tokenize else "input_ids"]


def _sft_data_columns(dataset, config, column_name, text_field="text"):
  columns = getattr(config, column_name)
  if column_name in config._pydantic_config.model_fields_set:
    return columns
  features = _dataset_columns(dataset)
  supported = (["messages"], ["prompt", "completion"], ["question", "answer"], [text_field])
  matches = [candidate for candidate in supported if set(candidate).issubset(features)]
  if len(matches) != 1:
    raise ValueError(f"Select one SFT column layout on the dataset or provide formatting_func; found {matches}.")
  return matches[0]


def _dpo_data_columns(dataset, config, column_name):
  if column_name in config._pydantic_config.model_fields_set:
    return getattr(config, column_name)
  features = _dataset_columns(dataset)
  if all(name in features for name in ("prompt", "chosen", "rejected")):
    return ["prompt", "chosen", "rejected"]
  if all(name in features for name in ("chosen", "rejected")):
    return ["chosen", "rejected"]
  raise ValueError("DPO requires chosen/rejected columns and an optional prompt column.")


def _get_training_objective_transform(
    config: ml_collections.ConfigDict,
    *,
    shift: bool,
    use_dpo: bool,
    use_sft: bool,
    packing: bool,
    pad_id: int,
    bos_token_id: int | None,
) -> target_shift.ShiftData | block_diffusion_transform.BlockDiffusionCorruption | None:
  """Selects target preparation for causal or block-diffusion training."""
  objective = getattr(config, "training_objective", "causal_lm")
  if objective == "block_diffusion":
    completion_only = bool(use_sft and getattr(config, "sft_train_on_completion_only", False))
    if use_dpo:
      raise ValueError("Block-diffusion training is not compatible with DPO.")
    if packing:
      raise ValueError("Block-diffusion training requires packing=False.")
    return block_diffusion_transform.BlockDiffusionCorruption(
        block_size=config.causal_block_size,
        mask_id=config.block_diffusion_mask_id,
        min_noise=config.block_diffusion_min_noise,
        logit_alignment=config.block_diffusion_logit_alignment,
        canvas_policy=config.block_diffusion_canvas_policy,
        axis=1,
        completion_only=completion_only,
    )
  if objective != "causal_lm":
    raise ValueError(f"Unsupported training objective: {objective}")
  if shift and not use_dpo:
    return target_shift.ShiftData(ignored_ids=[pad_id, bos_token_id], axis=1)
  return None


def _vision_sft_columns(dataset, text_columns):
  features = _dataset_columns(dataset)
  if len(text_columns) == 2 and set(text_columns).issubset(features):
    return text_columns
  matches = [columns for columns in (["prompt", "completion"], ["query", "response"], ["question", "answer"])
             if set(columns).issubset(features)]
  if len(matches) != 1:
    raise ValueError("Multimodal SFT requires prompt/completion, query/response or question/answer columns. "
                     "Rename or transform your dataset to supply the prompt and response explicitly.")
  return matches[0]


def format_sharegpt(example):
  """Convert a ShareGPT conversation into query and response fields."""
  query = ""
  response = ""
  conversations = example.get("conversations", [])
  if not isinstance(conversations, list):
    raise TypeError(f"Expected 'conversations' to be a list, got {type(conversations).__name__}.")
  for turn in conversations:
    if not isinstance(turn, dict):
      continue
    if turn.get("from") == "human":
      query = turn.get("value", "")
    elif turn.get("from") == "gpt":
      response = turn.get("value", "")
      break
  example["query"] = query
  example["response"] = [response]
  return example


def vision_sft_preprocessing_pipeline(
    dataset,
    config,
    dataloading_host_index,
    dataloading_host_count,
    global_mesh,
    text_columns,
    image_column,
    global_batch_size,
    processing_class=None,
    data_collator=None,
    is_training=True,
    tokenize=True,
):
  """Pipeline for multimodal SFT with HF dataset."""
  if (
      image_column == "video"
      and "conversations" in _dataset_columns(dataset)
      and len(text_columns) == 2
      and text_columns[0] not in _dataset_columns(dataset)
  ):
    dataset = dataset.map(format_sharegpt)

  text_columns = _vision_sft_columns(dataset, text_columns)
  if config.elastic_enabled and is_training:
    local_batch_size = elastic.get_local_batch_size(config)
  else:
    local_batch_size = global_batch_size // jax.process_count()
  if is_training and uses_tunix_gradient_accumulation(config):
    batch_size = local_batch_size // config.gradient_accumulation_steps
  else:
    batch_size = local_batch_size

  if is_training and config.enable_data_shuffling and config.num_epoch > 1:
    epoch_datasets = [dataset.shuffle(seed=config.data_shuffle_seed + i) for i in range(config.num_epoch)]
    dataset = datasets.concatenate_datasets(epoch_datasets)
  elif is_training and config.enable_data_shuffling:
    dataset = dataset.shuffle(seed=config.data_shuffle_seed)
  elif is_training and config.num_epoch > 1:
    dataset = (
        datasets.concatenate_datasets([dataset] * config.num_epoch)
        if isinstance(dataset, datasets.Dataset)
        else dataset.repeat(config.num_epoch)
    )

  if isinstance(image_column, list):
    dataset = dataset.map(
        sft_preprocessing.merge_image_columns,
        fn_kwargs={
            "image_columns": image_column,
            "max_num_images_per_example": config.max_num_images_per_example,
        },
        remove_columns=image_column,
    )
    image_column = "images"

  dataset = dataset.select_columns(text_columns + [image_column])
  if image_column != "images":
    dataset = dataset.rename_column(image_column, "images")

  if tokenize:
    dataset = dataset.map(
        sft_preprocessing.reformat_prompt,
        fn_kwargs={
            "column": text_columns[0], "image_placeholder": config.image_placeholder,
            "model_name": config.model_name, "video_placeholder": getattr(config, "video_placeholder", "<|video|>"),
        },
    )
    dataset = dataset.map(
        sft_preprocessing.reformat_response,
        fn_kwargs={"column": text_columns[1], "model_name": config.model_name},
    )
  if processing_class is None:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.tokenizer_path,
        add_bos_token=False,
        add_eos_token=False,
        legacy=False,
        token=config.hf_access_token,
        extra_special_tokens={},
    )
  else:
    tokenizer = processing_class.tokenizer if isinstance(processing_class, transformers.ProcessorMixin) else processing_class
  pad_id = _get_pad_id(tokenizer)
  if tokenize:
    dataset = dataset.map(
        hf_tokenization.tokenization, batched=True, batch_size=global_batch_size,
        fn_kwargs={"hf_tokenizer": tokenizer, "truncation": False, "max_length": config.max_target_length,
                   "column_names": text_columns, "add_special_tokens": False},
    )
  dataset = hf_data_source.HFDataSource(
      dataset=dataset,
      dataloading_host_index=dataloading_host_index,
      dataloading_host_count=dataloading_host_count,
      num_threads=1,
      max_target_length=config.max_target_length,
      data_column_names=text_columns,
  )
  operations = [sft_preprocessing.PrepareImageFusion(text_columns[0], config)]
  operations.append(
      sft_masking.SFTPromptMaskingVision(
          query_column=text_columns[0],
          response_column=text_columns[1],
          max_target_length=config.max_target_length,
          pad_id=pad_id,
          completion_only=config.sft_train_on_completion_only,
      )
  )
  operations.append(
      pad_or_trim.PadOrTrimToMaxLength(
          config.max_target_length,
          pad_id,
          config=config,
          max_num_images_per_example=config.max_num_images_per_example,
      )
  )
  operations.append(batch_transforms.ExtractImagesAndMasks())
  operations.append(grain.Batch(batch_size=batch_size, drop_remainder=is_training))
  if not is_training:
    operations.append(pad_or_trim.PadBatchToSize(batch_size, pad_id))
  operations.append(batch_transforms.FoldImagesIntoBatch(model_name=config.model_name))
  if config.use_mrope:
    operations.append(
        qwen3_omni_position_transform.ComputeQwen3OmniPositions(
            data_column="inputs",
            spatial_merge_size=config.spatial_merge_size_for_vit,
            position_id_per_seconds=config.position_id_per_seconds,
            use_audio_in_video=getattr(config, "use_audio_in_video", False),
            config=config,
            keep_aux_fields=False,
        )
    )
  operations.append(target_shift.ShiftData(ignored_ids=[pad_id], axis=1) if data_collator is None
                    else BatchCollator(data_collator, [pad_id]))
  dummy_index_sampler = grain.IndexSampler(
      num_records=len(dataset),
      num_epochs=1,
      shard_options=grain.ShardOptions(
          shard_index=dataloading_host_index, shard_count=dataloading_host_count, drop_remainder=False
      ),
      shuffle=False,
      seed=0,
  )
  dataloader = grain.DataLoader(
      data_source=dataset,
      operations=operations,
      sampler=dummy_index_sampler,
      worker_count=config.grain_worker_count,
      worker_buffer_size=1,
      read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=batch_size * 4),
  )
  multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataloader, global_mesh)
  return multihost_gen


def preprocessing_pipeline(
    dataloading_host_index,
    dataloading_host_count,
    global_mesh,
    dataset,
    config,
    data_column_names,
    tokenize,
    tokenizer_path,
    hf_access_token,
    global_batch_size,
    max_target_length,
    shuffle,
    data_shuffle_seed,
    chat_template_path="",
    add_bos=True,
    add_eos=True,
    packing=True,
    shift=True,
    num_threads=1,
    drop_remainder=True,
    generate_padding_batch=False,
    use_dpo=False,
    use_sft=None,
    use_tunix_gradient_accumulation=False,
    num_microbatches=1,
    sft_train_on_completion_only=True,
    grain_worker_count=1,
    max_segments_per_seq=None,
    num_epoch=1,
    chat_template: Optional[str] = None,
    formatting_func_path: Optional[str] = None,
    formatting_func_kwargs: Optional[dict] = None,
    processing_class=None,
    data_collator=None,
    is_training=True,
):
  """Pipeline for preprocessing HF dataset."""
  assert global_batch_size % global_mesh.size == 0, "Batch size should be divisible by number of global devices."
  if config.elastic_enabled and is_training:
    local_batch_size = elastic.get_local_batch_size(config)
  else:
    local_batch_size = global_batch_size // jax.process_count()
  if use_tunix_gradient_accumulation:
    batch_size = local_batch_size // num_microbatches
  else:
    batch_size = local_batch_size

  if shuffle and num_epoch > 1:
    epoch_datasets = [dataset.shuffle(seed=data_shuffle_seed + i) for i in range(num_epoch)]
    dataset = datasets.concatenate_datasets(epoch_datasets)
  elif shuffle:
    dataset = dataset.shuffle(seed=data_shuffle_seed)
  elif num_epoch > 1:
    dataset = (
        datasets.concatenate_datasets([dataset] * num_epoch)
        if isinstance(dataset, datasets.Dataset)
        else dataset.repeat(num_epoch)
    )

  is_offline = getattr(config, "is_offline_distillation", False)
  if processing_class is None:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        add_bos_token=add_bos if not use_sft else False,
        add_eos_token=add_eos if not use_sft else False,
        legacy=False,
        token=hf_access_token,
        extra_special_tokens={},
    )
  else:
    tokenizer = processing_class.tokenizer if isinstance(processing_class, transformers.ProcessorMixin) else processing_class
  if is_offline:
    data_column_names = ["inputs", "top_k_logits", "top_k_indices"]
  dataset = dataset.select_columns(data_column_names)
  if use_sft and not is_offline:
    pad_token_remap.fix_pad_token_if_needed(tokenizer)
  plain_text = use_sft and len(data_column_names) == 1 and data_column_names[0] not in ("messages", "conversations")
  plain_pairs = (use_sft and data_column_names == ["prompt", "completion"]
                 and not conversational_columns.is_conversational(_dataset_columns(dataset), data_column_names))
  if plain_text and not is_offline:
    dataset = dataset.map(formatting.text_segments, fn_kwargs={"text_field": data_column_names[0]})
  elif plain_pairs and not is_offline:
    dataset = dataset.map(formatting.paired_segments,
                          fn_kwargs={"prompt_field": data_column_names[0], "completion_field": data_column_names[1],
                                     "tokenizer": tokenizer if tokenize else None},
                          remove_columns=[data_column_names[1]])
    data_column_names = [data_column_names[0]]
  elif use_sft and not is_offline:
    if not chat_template:
      chat_template = instruction_data_processing.load_chat_template_from_file(chat_template_path)
    data_processing_utils.validate_and_configure_sft_columns(data_column_names, tokenizer, chat_template)
    dataset, data_column_names = instruction_data_processing.convert_to_conversational_format(
        dataset=dataset,
        data_columns=data_column_names,
        formatting_func_path=formatting_func_path,
        formatting_func_kwargs=formatting_func_kwargs,
    )
    assert conversational_columns.is_conversational(
        _dataset_columns(dataset), data_column_names
    ), "Dataset is not in conversational format."
    if len(data_column_names) > 1:
      combined_column_name = "messages"
      dataset_features = datasets.Features(
          {combined_column_name: [{"content": datasets.Value(dtype="string"), "role": datasets.Value(dtype="string")}]}
      )
      dataset = dataset.map(
          conversational_columns.combine_columns,
          fn_kwargs={"columns": data_column_names, "data_column": combined_column_name},
          remove_columns=data_column_names,
          features=dataset_features,
      )
      data_column_names = [combined_column_name]
    dataset = dataset.map(
        chat_template_formatting.apply_chat_template,
        fn_kwargs={
            "tokenizer_model": tokenizer,
            "data_column_name": data_column_names[0],
            "model_name": getattr(config, "model_name", None),
        },
    )

  pad_id = _get_pad_id(tokenizer)
  if is_offline and pad_id == -1:
    pad_id = 0
  if tokenize and not is_offline and not use_sft and not use_dpo and not config.use_truncation:
    if len(data_column_names) != 1:
      raise ValueError("Tokenize-and-chunk preprocessing requires one text column.")
    dataset = dataset.map(hf_tokenization.chunk_tokenization, batched=True,
                          fn_kwargs={"hf_tokenizer": tokenizer, "max_length": max_target_length,
                                     "column_name": data_column_names[0]})
  elif tokenize and not is_offline and not plain_pairs:
    dataset = dataset.map(
        hf_tokenization.tokenization,
        batched=True,
        fn_kwargs={
            "hf_tokenizer": tokenizer,
            "truncation": config.use_truncation and not use_sft,
            "max_length": max_target_length,
            "column_names": data_column_names,
            "add_special_tokens": not use_sft or plain_text,
        },
    )
  if is_offline and isinstance(dataset, datasets.IterableDataset):
    num_threads = 1
    grain_worker_count = 0
  dataset = hf_data_source.HFDataSource(
      dataset,
      dataloading_host_index,
      dataloading_host_count,
      num_threads,
      max_target_length,
      data_column_names,
  )
  operations = []
  if is_offline:
    operations.append(feature_transforms.HFNormalizeFeatures("inputs", is_offline_distillation=True))
    data_column_names = ("inputs", "targets", "top_k_logits", "top_k_indices")
  elif use_sft:
    operations.append(
        sft_masking.SFTPromptMasking(
            text_column_name=data_column_names[0],
            completion_only=sft_train_on_completion_only,
            max_target_length=max_target_length,
            unk_id=pad_id,
            training_objective=getattr(config, "training_objective", "causal_lm"),
        )
    )
    data_column_names = ("inputs", "targets")
  elif not use_dpo:
    assert len(data_column_names) == 1
    operations.append(feature_transforms.HFNormalizeFeatures(data_column_names[0]))
    data_column_names = ("inputs", "targets")

  if packing:
    length_struct = {col: max_target_length for col in data_column_names}
    max_segments = max_segments_per_seq
    if max_segments is not None and max_segments <= 0:
      max_segments = None
    operations.append(
        grain.experimental.PackAndBatchOperation(
            batch_size=batch_size,
            length_struct=length_struct,
            max_sequences_per_bin=max_segments,
        )
    )
    operations.append(reformat_packing.ReformatPacking(data_column_names))
  else:
    if use_dpo and not is_offline:
      max_prompt_length = config.dpo.max_prompt_length
      operations.append(dpo_utils.DPODataFormatting(pad_id, max_target_length, data_column_names, max_prompt_length))
    else:
      operations.append(pad_or_trim.PadOrTrimToMaxLength(max_target_length, pad_id, config=config))
    operations.append(grain.Batch(batch_size=batch_size, drop_remainder=drop_remainder))
    if not drop_remainder:
      operations.append(pad_or_trim.PadBatchToSize(batch_size, pad_id))

  if is_offline:
    target_transform = target_shift.ShiftData(ignored_ids=[pad_id], axis=1) if shift else None
  else:
    target_transform = _get_training_objective_transform(
        config,
        shift=shift,
        use_dpo=use_dpo,
        use_sft=use_sft,
        packing=packing,
        pad_id=pad_id,
        bos_token_id=tokenizer.bos_token_id,
    )
  if data_collator is not None:
    if use_dpo:
      operations.append(PreferenceBatchCollator(data_collator, pad_id, max_target_length, config.dpo.max_prompt_length))
    elif getattr(config, "training_objective", "causal_lm") == "block_diffusion":
      operations.append(BatchCollator(data_collator, [None], shift=False))
      operations.append(target_transform)
    else:
      operations.append(BatchCollator(data_collator, [pad_id] if is_offline else [pad_id, tokenizer.bos_token_id]))
  elif target_transform is not None:
    operations.append(target_transform)
  dummy_index_sampler = grain.IndexSampler(
      num_records=len(dataset),
      num_epochs=1,
      shard_options=grain.ShardOptions(
          shard_index=dataloading_host_index, shard_count=dataloading_host_count, drop_remainder=False
      ),
      shuffle=False,
      seed=0,
  )
  dataloader = grain.DataLoader(
      data_source=dataset,
      operations=operations,
      sampler=dummy_index_sampler,
      worker_count=grain_worker_count,
      worker_buffer_size=1,
      read_options=grain.ReadOptions(num_threads=num_threads, prefetch_buffer_size=128),
  )
  multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataloader, global_mesh, generate_padding_batch)
  return multihost_gen


def make_hf_train_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices_train,
    *,
    dataset=None,
    processing_class=None,
    formatting_func=None,
    dataset_text_field="text",
    data_collator=None,
):
  """Load, preprocess dataset and return iterators."""
  if dataset is None:
    train_ds = datasets.load_dataset(
        config.hf_path,
        name=config.hf_name,
        data_dir=config.hf_data_dir,
        data_files=config.hf_train_files,
        split=config.train_split,
        streaming=True,
        token=config.hf_access_token,
    )
  else:
    train_ds = dataset
  if formatting_func is not None:
    train_ds = formatting.format_dataset(train_ds, formatting_func, dataset_text_field)
    dataset = train_ds
  is_offline = getattr(config, "is_offline_distillation", False)
  data_columns = config.train_data_columns
  if dataset is not None and not (config.use_sft or config.use_dpo or is_offline):
    data_columns = _language_model_columns(dataset, data_columns, config.tokenize_train_data, dataset_text_field)
  if dataset is not None and config.use_dpo:
    data_columns = _dpo_data_columns(dataset, config, "train_data_columns")
  if dataset is not None and config.use_sft and not config.use_multimodal and not is_offline:
    data_columns = _sft_data_columns(dataset, config, "train_data_columns", dataset_text_field)
  if config.use_sft and config.use_multimodal and not is_offline:
    train_iter = vision_sft_preprocessing_pipeline(
        dataset=train_ds,
        config=config,
        dataloading_host_index=process_indices_train.index(jax.process_index()),
        dataloading_host_count=len(process_indices_train),
        global_mesh=global_mesh,
        text_columns=config.train_data_columns,
        image_column=config.train_image_column,
        global_batch_size=config.global_batch_size_to_load,
        tokenize=config.tokenize_train_data,
        processing_class=processing_class,
        data_collator=data_collator,
    )
  else:
    train_iter = preprocessing_pipeline(
        dataloading_host_index=process_indices_train.index(jax.process_index()),
        dataloading_host_count=len(process_indices_train),
        global_mesh=global_mesh,
        dataset=train_ds,
        config=config,
        data_column_names=data_columns,
        tokenize=config.tokenize_train_data,
        tokenizer_path=config.tokenizer_path,
        hf_access_token=config.hf_access_token,
        global_batch_size=config.global_batch_size_to_load,
        max_target_length=config.max_target_length,
        shuffle=config.enable_data_shuffling,
        data_shuffle_seed=config.data_shuffle_seed,
        add_bos=config.add_bos,
        add_eos=config.add_eos,
        packing=config.packing,
        generate_padding_batch=config.generate_padding_batch_train,
        use_dpo=config.use_dpo,
        use_sft=config.use_sft,
        use_tunix_gradient_accumulation=uses_tunix_gradient_accumulation(config),
        num_microbatches=config.gradient_accumulation_steps,
        sft_train_on_completion_only=config.sft_train_on_completion_only,
        chat_template_path=config.chat_template_path,
        max_segments_per_seq=config.max_segments_per_seq,
        num_epoch=config.num_epoch,
        chat_template=config.chat_template,
        formatting_func_path=config.formatting_func_path,
        formatting_func_kwargs=config.formatting_func_kwargs,
        grain_worker_count=config.grain_worker_count,
        processing_class=processing_class,
        data_collator=data_collator,
    )
  return train_iter


def make_hf_eval_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices_eval,
    *,
    dataset=None,
    processing_class=None,
    formatting_func=None,
    dataset_text_field="text",
    data_collator=None,
):
  """Make Hugging Face evaluation iterator. Load and preprocess eval dataset: and return iterator."""
  if dataset is None:
    eval_ds = datasets.load_dataset(
        config.hf_path,
        name=config.hf_name,
        data_dir=config.hf_data_dir,
        data_files=config.hf_eval_files,
        split=config.hf_eval_split,
        streaming=True,
        token=config.hf_access_token,
    )
  else:
    eval_ds = dataset
  if formatting_func is not None:
    eval_ds = formatting.format_dataset(eval_ds, formatting_func, dataset_text_field)
    dataset = eval_ds
  is_offline = getattr(config, "is_offline_distillation", False)
  data_columns = config.eval_data_columns
  if dataset is not None and not (config.use_sft or config.use_dpo or is_offline):
    data_columns = _language_model_columns(dataset, data_columns, config.tokenize_eval_data, dataset_text_field)
  if dataset is not None and config.use_dpo:
    data_columns = _dpo_data_columns(dataset, config, "eval_data_columns")
  if dataset is not None and config.use_sft and not config.use_multimodal and not is_offline:
    data_columns = _sft_data_columns(dataset, config, "eval_data_columns", dataset_text_field)
  if config.use_sft and config.use_multimodal and not is_offline:
    eval_iter = vision_sft_preprocessing_pipeline(
        dataset=eval_ds,
        config=config,
        dataloading_host_index=process_indices_eval.index(jax.process_index()),
        dataloading_host_count=len(process_indices_eval),
        global_mesh=global_mesh,
        text_columns=config.eval_data_columns,
        image_column=config.eval_image_column,
        global_batch_size=config.global_batch_size_to_load_eval,
        tokenize=config.tokenize_eval_data,
        processing_class=processing_class,
        data_collator=data_collator,
        is_training=False,
    )
  else:
    eval_iter = preprocessing_pipeline(
        dataloading_host_index=process_indices_eval.index(jax.process_index()),
        dataloading_host_count=len(process_indices_eval),
        global_mesh=global_mesh,
        dataset=eval_ds,
        config=config,
        data_column_names=data_columns,
        tokenize=config.tokenize_eval_data,
        tokenizer_path=config.tokenizer_path,
        hf_access_token=config.hf_access_token,
        global_batch_size=config.global_batch_size_to_load_eval,
        max_target_length=config.max_target_length,
        is_training=False,
        shuffle=False,
        drop_remainder=False,
        data_shuffle_seed=config.data_shuffle_seed,
        add_bos=config.add_bos,
        add_eos=config.add_eos,
        packing=config.packing,
        generate_padding_batch=config.generate_padding_batch_eval,
        use_dpo=config.use_dpo,
        use_sft=config.use_sft,
        num_microbatches=config.gradient_accumulation_steps,
        sft_train_on_completion_only=config.sft_train_on_completion_only,
        chat_template_path=config.chat_template_path,
        max_segments_per_seq=config.max_segments_per_seq,
        chat_template=config.chat_template,
        formatting_func_path=config.formatting_func_path,
        formatting_func_kwargs=config.formatting_func_kwargs,
        grain_worker_count=config.grain_worker_count,
        processing_class=processing_class,
        data_collator=data_collator,
    )
  return eval_iter
