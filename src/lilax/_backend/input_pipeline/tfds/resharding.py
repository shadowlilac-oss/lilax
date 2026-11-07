"""Reshard TFDS TFRecord datasets without decoding their records."""

import json
import multiprocessing
import os
import queue
import threading

import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm


def _reshard_worker(worker_id, num_workers, src_files, dst_dir, dataset_name, split, total_shards, buffer_size, progress):
  writers = {}
  shard_lengths = {}
  dataset = tf.data.TFRecordDataset(src_files, compression_type=None, buffer_size=buffer_size)
  dataset = dataset.shard(num_workers, worker_id)

  i = -1
  for i, record in enumerate(dataset):
    shard_index = (i * num_workers + worker_id) % total_shards
    if shard_index not in writers:
      filename = f"{dataset_name}-{split}.tfrecord-{shard_index:05d}-of-{total_shards:05d}"
      writers[shard_index] = tf.io.TFRecordWriter(os.path.join(dst_dir, filename))
      shard_lengths[shard_index] = 0
    writers[shard_index].write(record.numpy())
    shard_lengths[shard_index] += 1
    if (i + 1) % 1000 == 0:
      progress.put(1000)

  remainder = (i + 1) % 1000
  if remainder > 0:
    progress.put(remainder)
  for writer in writers.values():
    writer.close()
  return shard_lengths


def _progress_listener(progress, total_examples):
  pbar = tqdm(total=total_examples, desc="Resharding Progress", unit=" records", unit_scale=True)
  while True:
    try:
      update = progress.get(timeout=0.1)
      if update == "DONE":
        break
      pbar.update(update)
    except queue.Empty:
      continue
  pbar.close()


def reshard_tfds(src_dir, dst_dir, num_shards=2048, split="train", dataset_name=None, num_workers=16,
                 buffer_size=32 * 1024 * 1024):
  """Reshard comma-separated TFDS splits into raw TFRecord files and update their metadata."""
  if not tf.io.gfile.exists(dst_dir):
    tf.io.gfile.makedirs(dst_dir)

  target_splits = [s.strip() for s in split.split(",") if s.strip()]
  print(f"Loading metadata from {src_dir}...")
  info_path = os.path.join(src_dir, "dataset_info.json")
  if not tf.io.gfile.exists(info_path):
    raise FileNotFoundError(f"Required metadata file not found: {info_path}")
  with tf.io.gfile.GFile(info_path, "r") as f:
    info_json = json.load(f)

  dataset_name = dataset_name or info_json.get("name")
  if not dataset_name:
    try:
      dataset_name = tfds.builder_from_directory(src_dir).name
    except ImportError:
      raise
    except Exception as error:
      print(f"Warning: Could not load metadata via tfds.builder_from_directory: {error}")
      print("Warning: Dataset name could not be determined, and output filenames will use 'unknown'.")
      dataset_name = "unknown"

  with multiprocessing.Manager() as manager:
    for split_name in target_splits:
      print(f"\n--- Processing split: {split_name} ---")
      num_examples = 0
      splits_meta = info_json.get("splits", {})
      if isinstance(splits_meta, list):
        split_item = next((s for s in splits_meta if s["name"] == split_name), None)
      else:
        split_item = splits_meta.get(split_name)
      if split_item:
        num_examples = int(split_item.get("numExamples", split_item.get("num_examples", 0)))

      pattern = os.path.join(src_dir, f"{dataset_name}-{split_name}.tfrecord*")
      src_files = sorted(tf.io.gfile.glob(pattern))
      if not src_files:
        pattern = os.path.join(src_dir, f"{split_name}.tfrecord*")
        src_files = sorted(tf.io.gfile.glob(pattern))
      if not src_files:
        raise FileNotFoundError(f"Could not find TFRecord files for split '{split_name}' in {src_dir}")

      print(f"Found {len(src_files)} source files for split '{split_name}' ({num_examples} examples).")
      split_workers = num_workers
      if num_shards % split_workers != 0:
        for i in range(split_workers, 0, -1):
          if num_shards % i == 0:
            split_workers = i
            break
        print(f"Adjusted num_workers to {split_workers} to be a factor of {num_shards}")
      print(f"Resharding into {num_shards} shards using {split_workers} workers...")

      progress = manager.Queue()
      listener = threading.Thread(target=_progress_listener, args=(progress, num_examples if num_examples > 0 else None))
      listener.start()
      tasks = [
          (i, split_workers, src_files, dst_dir, dataset_name, split_name, num_shards, buffer_size, progress)
          for i in range(split_workers)
      ]
      with multiprocessing.Pool(processes=split_workers) as pool:
        results = pool.starmap(_reshard_worker, tasks)
      progress.put("DONE")
      listener.join()

      all_shard_lengths = {}
      for result in results:
        all_shard_lengths.update(result)
      total_count = sum(all_shard_lengths.values())
      print(f"Successfully resharded {total_count} examples for '{split_name}'.")
      if num_examples > 0 and total_count != num_examples:
        print(f"Warning: Total examples {total_count} does not match original {num_examples} for split '{split_name}'.")
      shard_lengths = [str(all_shard_lengths.get(i, 0)) for i in range(num_shards)]

      if "splits" not in info_json:
        info_json["splits"] = {}
      splits_meta = info_json["splits"]
      if isinstance(splits_meta, list):
        found = False
        for split_item in splits_meta:
          if split_item.get("name") == split_name:
            split_item["shardLengths"] = shard_lengths
            split_item["numShards"] = str(num_shards)
            split_item["numExamples"] = str(total_count)
            found = True
            break
        if not found:
          splits_meta.append({"name": split_name, "shardLengths": shard_lengths,
                              "numShards": str(num_shards), "numExamples": str(total_count)})
      elif split_name in splits_meta:
        split_item = splits_meta[split_name]
        split_item["shardLengths"] = shard_lengths
        split_item["numShards"] = str(num_shards)
        if "numExamples" in split_item:
          split_item["numExamples"] = str(total_count)
        else:
          split_item["num_examples"] = str(total_count)
      else:
        splits_meta[split_name] = {"shardLengths": shard_lengths, "numShards": str(num_shards),
                                   "numExamples": str(total_count)}

  print("\nCreating new dataset_info.json...")
  with tf.io.gfile.GFile(os.path.join(dst_dir, "dataset_info.json"), "w") as f:
    json.dump(info_json, f, indent=4)
  features_path = os.path.join(src_dir, "features.json")
  if tf.io.gfile.exists(features_path):
    tf.io.gfile.copy(features_path, os.path.join(dst_dir, "features.json"), overwrite=True)
  print(f"Done! Resharded dataset available at {dst_dir}")
