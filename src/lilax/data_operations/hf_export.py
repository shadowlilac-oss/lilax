# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export streaming Hugging Face datasets to ArrayRecord or Parquet."""

import json
import multiprocessing
import os
import shutil
import time

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from array_record.python.array_record_module import ArrayRecordWriter
from datasets import load_dataset
from huggingface_hub import HfApi
from pyarrow.fs import GcsFileSystem, LocalFileSystem

from lilax._backend.input_pipeline.protos import example_pb2, feature_pb2

MAX_RETRIES = 10
RETRY_WAIT_SECONDS = 30


def _to_feature(value):
  if isinstance(value, bool):
    return feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=[int(value)]))
  if isinstance(value, int):
    return feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=[value]))
  if isinstance(value, float):
    return feature_pb2.Feature(float_list=feature_pb2.FloatList(value=[value]))
  if isinstance(value, (str, bytes)):
    v = value.encode("utf-8") if isinstance(value, str) else value
    return feature_pb2.Feature(bytes_list=feature_pb2.BytesList(value=[v]))
  if isinstance(value, list):
    if not value:
      return feature_pb2.Feature(bytes_list=feature_pb2.BytesList(value=[]))
    first = value[0]
    if isinstance(first, (bool, int)):
      return feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=[int(v) for v in value]))
    if isinstance(first, float):
      return feature_pb2.Feature(float_list=feature_pb2.FloatList(value=value))
    if isinstance(first, (str, bytes)):
      v = [x.encode("utf-8") if isinstance(x, str) else x for x in value]
      return feature_pb2.Feature(bytes_list=feature_pb2.BytesList(value=v))
  json_v = json.dumps(value, ensure_ascii=False).encode("utf-8") if value is not None else b""
  return feature_pb2.Feature(bytes_list=feature_pb2.BytesList(value=[json_v] if json_v else []))


def _serialize_example(example):
  features = {k: _to_feature(v) for k, v in example.items()}
  return example_pb2.Example(features=feature_pb2.Features(feature=features)).SerializeToString()


def _arrayrecord_shard(task):
  worker_id = task["worker_id"]
  output_dir = task["output"]
  file_size_bytes = task["file_size_bytes"]
  checkpoint_path = os.path.join(output_dir, f".checkpoint-worker-{worker_id:04d}.json")
  kwargs = {"path": task["dataset"], "streaming": True, "split": task["split"], "token": task["token"]}
  if task["config"]:
    kwargs.update(json.loads(task["config"]))

  def load_and_restore():
    ds = load_dataset(**kwargs).shard(num_shards=task["num_workers"], index=worker_id)
    if os.path.exists(checkpoint_path):
      with open(checkpoint_path, "r", encoding="utf-8") as f:
        ckpt = json.load(f)
      ds.load_state_dict(ckpt["ds_state"])
      return ds, ckpt
    return ds, {"local_file_idx": 0, "total_bytes": 0, "total_written": 0, "filenames": [],
                "adjusted_file_size_bytes": file_size_bytes}

  ds, state = load_and_restore()
  if state["filenames"]:
    print(f"  [Worker {worker_id}] Resumed: {state['total_written']:,} records, {len(state['filenames'])} files written")
  local_file_idx, total_bytes, total_written, filenames, adjusted_file_size_bytes = (
      state[k] for k in ["local_file_idx", "total_bytes", "total_written", "filenames", "adjusted_file_size_bytes"]
  )
  file_bytes, writer, t0 = 0, None, time.time()

  def open_writer():
    nonlocal writer
    fname = f"worker-{worker_id:04d}-{local_file_idx:06d}.arrayrecord"
    writer = ArrayRecordWriter(os.path.join(output_dir, fname), f"group_size:{task['group_size']}")
    filenames.append(fname)

  def close_writer():
    nonlocal writer
    if writer:
      writer.close()
      writer = None

  def save_checkpoint():
    ckpt = {"worker_id": worker_id, "local_file_idx": local_file_idx, "total_bytes": total_bytes,
            "total_written": total_written, "filenames": filenames, "adjusted_file_size_bytes": adjusted_file_size_bytes,
            "ds_state": ds.state_dict()}
    with open(checkpoint_path, "w", encoding="utf-8") as f:
      json.dump(ckpt, f)

  open_writer()
  for attempt in range(MAX_RETRIES):
    try:
      for example in ds:
        record = _serialize_example(example)
        record_size = len(record)
        writer.write(record)
        file_bytes += record_size
        total_bytes += record_size
        total_written += 1
        if file_bytes >= adjusted_file_size_bytes:
          close_writer()
          actual_size = os.path.getsize(os.path.join(output_dir, filenames[-1]))
          if actual_size > 0:
            adjusted_file_size_bytes = int(file_size_bytes / (actual_size / file_bytes))
          elapsed = time.time() - t0
          print(f"  [Worker {worker_id}] File {local_file_idx} written | "
                f"{actual_size/1e6:.0f} MB | {total_written/elapsed:,.0f} rec/sec")
          local_file_idx += 1
          file_bytes = 0
          save_checkpoint()
          open_writer()
      break
    except (requests.exceptions.RequestException, ConnectionError) as e:
      close_writer()
      if attempt >= MAX_RETRIES - 1:
        print(f"  [Worker {worker_id}] Max retries reached. Re-run the script to resume from checkpoint.")
        raise
      print(f"  [Worker {worker_id}] Connection error: {e}. Rolling back to last checkpoint."
            f"Retrying in {RETRY_WAIT_SECONDS}s, ({attempt+1}/{MAX_RETRIES})...")
      time.sleep(RETRY_WAIT_SECONDS)
      ds, state = load_and_restore()
      local_file_idx, total_bytes, total_written, filenames, adjusted_file_size_bytes = (
          state[k] for k in ["local_file_idx", "total_bytes", "total_written", "filenames", "adjusted_file_size_bytes"]
      )
      file_bytes = 0
      open_writer()
  close_writer()
  elapsed = time.time() - t0
  print(f"  [Worker {worker_id}] Done: {total_written:,} records, {len(filenames)} files in {elapsed:.1f}s")
  return {"worker_id": worker_id, "total_records": total_written, "total_bytes": total_bytes, "filenames": filenames}


def _filesystem_and_path(output_path):
  if output_path.startswith("gs://"):
    fs, base_path = GcsFileSystem(), output_path[len("gs://"):]
  else:
    fs, base_path = LocalFileSystem(), os.path.abspath(output_path)
    os.makedirs(base_path, exist_ok=True)
  return fs, base_path if base_path.endswith("/") else base_path + "/"


def _parquet_shard(task):
  worker_id = task["worker_id"]
  output_path = task["output"]
  local_checkpoint_dir = (os.path.abspath(output_path) if not output_path.startswith("gs://")
                          else os.path.join("/tmp", output_path.replace("gs://", "gs_")))
  os.makedirs(local_checkpoint_dir, exist_ok=True)
  checkpoint_path = os.path.join(local_checkpoint_dir, f".checkpoint-worker-{worker_id:04d}.json")
  kwargs = {"path": task["dataset"], "streaming": True, "split": task["split"], "token": task["token"]}
  if task["config"]:
    kwargs.update(json.loads(task["config"]))

  def load_and_restore():
    ds = load_dataset(**kwargs).shard(num_shards=task["num_workers"], index=worker_id)
    if os.path.exists(checkpoint_path):
      with open(checkpoint_path, "r", encoding="utf-8") as f:
        ckpt = json.load(f)
      ds.load_state_dict(ckpt["ds_state"])
      return ds, ckpt
    return ds, {"local_file_idx": 0, "total_jsonl_bytes": 0, "total_written": 0, "filenames": []}

  ds, state = load_and_restore()
  local_file_idx, total_jsonl_bytes, total_written, filenames = (
      state[k] for k in ["local_file_idx", "total_jsonl_bytes", "total_written", "filenames"]
  )
  fs, base_path = _filesystem_and_path(output_path)
  comp = None if task["compression"] == "none" else task["compression"]
  file_examples, file_jsonl_bytes, t0 = [], 0, time.time()

  def write_file():
    nonlocal local_file_idx, total_written
    fname = f"worker-{worker_id:04d}-{local_file_idx:06d}.parquet"
    pq.write_table(pa.Table.from_pylist(file_examples), base_path + fname, filesystem=fs,
                   compression=comp, row_group_size=task["row_group_size"])
    filenames.append(fname)
    total_written += len(file_examples)
    elapsed = time.time() - t0
    print(f"  [Worker {worker_id}] File {local_file_idx} written | {total_written/elapsed:,.0f} rows/sec")
    local_file_idx += 1
    ckpt = {"worker_id": worker_id, "local_file_idx": local_file_idx, "total_jsonl_bytes": total_jsonl_bytes,
            "total_written": total_written, "filenames": filenames, "ds_state": ds.state_dict()}
    with open(checkpoint_path, "w", encoding="utf-8") as f:
      json.dump(ckpt, f)

  for attempt in range(MAX_RETRIES):
    try:
      for example in ds:
        size = len(json.dumps(example, ensure_ascii=False).encode("utf-8"))
        file_examples.append(example)
        file_jsonl_bytes += size
        total_jsonl_bytes += size
        if file_jsonl_bytes >= task["jsonl_bytes_per_file"]:
          write_file()
          file_examples, file_jsonl_bytes = [], 0
      break
    except (requests.exceptions.RequestException, ConnectionError) as e:
      if attempt >= MAX_RETRIES - 1:
        print(f"  [Worker {worker_id}] Max retries reached. Re-run the script to resume from checkpoint.")
        raise
      print(f"  [Worker {worker_id}] Connection error: {e}. Rolling back to last checkpoint."
            f"Retrying in {RETRY_WAIT_SECONDS}s, ({attempt+1}/{MAX_RETRIES})...")
      time.sleep(RETRY_WAIT_SECONDS)
      ds, state = load_and_restore()
      local_file_idx, total_jsonl_bytes, total_written, filenames = (
          state[k] for k in ["local_file_idx", "total_jsonl_bytes", "total_written", "filenames"]
      )
      file_examples, file_jsonl_bytes = [], 0
  if file_examples:
    write_file()
  elapsed = time.time() - t0
  print(f"  [Worker {worker_id}] Done: {total_written:,} records, {len(filenames)} files in {elapsed:.1f}s")
  return {"worker_id": worker_id, "total_rows": total_written, "total_bytes": total_jsonl_bytes, "filenames": filenames}


def _workers(dataset, config, split, token, workers):
  num_workers = workers or os.cpu_count()
  kwargs = {"path": dataset, "streaming": True, "split": split, "token": token}
  if config:
    kwargs.update(json.loads(config))
  ds = load_dataset(**kwargs)
  if hasattr(ds, "n_shards") and ds.n_shards and ds.n_shards < num_workers:
    print(f"{num_workers=} exceeds available data shard: {ds.n_shards}. Setting workers to {ds.n_shards}")
    num_workers = ds.n_shards
  return num_workers


def export_arrayrecord(dataset, output, config=None, split="train", name_prefix=None, file_size_mb=500,
                       group_size=1, workers=None, token=None):
  """Stream a dataset into local or GCSFUSE-mounted ArrayRecord files."""
  if output.startswith("gs://"):
    raise ValueError("gs:// paths are not supported. Mount the bucket with GCSFUSE and pass the mount path instead.")
  num_workers = _workers(dataset, config, split, token, workers)
  prefix = name_prefix or dataset.split("/")[-1].lower().replace(" ", "-")
  output_dir = os.path.abspath(output)
  os.makedirs(output_dir, exist_ok=True)
  tasks = [{"worker_id": i, "num_workers": num_workers, "dataset": dataset, "config": config, "split": split,
            "token": token, "output": output_dir, "file_size_bytes": file_size_mb * 1_000_000,
            "group_size": group_size} for i in range(num_workers)]
  print(f"Dataset: {dataset} (split: {split}) | Target: {file_size_mb} MB | Workers: {num_workers}\n")
  t0 = time.time()
  with multiprocessing.get_context("spawn").Pool(processes=num_workers) as pool:
    results = pool.map(_arrayrecord_shard, tasks)
  write_elapsed = time.time() - t0
  all_filenames = []
  for result in sorted(results, key=lambda x: x["worker_id"]):
    all_filenames.extend(result["filenames"])
  print("\nVerifying downloaded files...")
  existing_files = []
  for fname in all_filenames:
    fpath = os.path.join(output_dir, fname)
    if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
      existing_files.append(fname)
    else:
      print(f"  [Warning] File not found or empty, skipping: {fpath}")
  total_files = len(existing_files)
  if total_files:
    print(f"\nRenaming {total_files} verified files...")
    existing_files.sort()
    width = max(5, len(str(total_files)))
    for idx, old_name in enumerate(existing_files):
      new_name = f"{prefix}-{idx:0{width}d}-of-{total_files:0{width}d}.arrayrecord"
      if old_name != new_name:
        os.rename(os.path.join(output_dir, old_name), os.path.join(output_dir, new_name))
    print(f"  Renamed to: {prefix}-00000-of-{total_files:0{width}d}.arrayrecord ... "
          f"{prefix}-{total_files - 1:0{width}d}-of-{total_files:0{width}d}.arrayrecord")
  else:
    print("No valid files to rename.")
  for i in range(num_workers):
    path = os.path.join(output_dir, f".checkpoint-worker-{i:04d}.json")
    if os.path.exists(path):
      os.remove(path)
  total_elapsed, total_records = time.time() - t0, sum(r["total_records"] for r in results)
  print(f"\nDone! Wrote {total_records:,} records in {total_elapsed:.1f}s ({total_records/write_elapsed:,.0f} rec/sec)")


def export_parquet(dataset, output, num_files, config=None, split="train", name_prefix=None, compression="zstd",
                   row_group_size=None, total_size_gb=None, workers=None, token=None):
  """Stream a dataset into local, GCSFUSE-mounted, or GCS Parquet files."""
  if compression not in ("zstd", "snappy", "gzip", "none"):
    raise ValueError(f"Invalid compression: {compression!r}. Choose from 'zstd', 'snappy', 'gzip', or 'none'.")
  num_workers = _workers(dataset, config, split, token, workers)
  if total_size_gb:
    total_bytes = total_size_gb * 1e9
  else:
    entries = HfApi().list_repo_tree(dataset, repo_type="dataset", token=token, recursive=True)
    total_bytes = sum(e.size for e in entries if hasattr(e, "size") and e.size)
  jsonl_bytes_per_file = total_bytes / num_files
  tasks = [{"worker_id": i, "num_workers": num_workers, "dataset": dataset, "config": config, "split": split,
            "token": token, "output": output, "row_group_size": row_group_size, "compression": compression,
            "jsonl_bytes_per_file": jsonl_bytes_per_file} for i in range(num_workers)]
  print(f"Plan: {total_bytes/1e9:.1f} GB -> ~{num_files} files | Workers: {num_workers}\n")
  t0 = time.time()
  with multiprocessing.get_context("spawn").Pool(processes=num_workers) as pool:
    results = pool.map(_parquet_shard, tasks)
  write_elapsed = time.time() - t0
  all_filenames = []
  for result in sorted(results, key=lambda x: x["worker_id"]):
    all_filenames.extend(result["filenames"])
  fs, base_path = _filesystem_and_path(output)
  if len(all_filenames) > num_files:
    print(f"\nMerging {len(all_filenames) - num_files + 1} smallest files into one...")
    infos = sorted([(f, fs.get_file_info(base_path + f).size) for f in all_filenames], key=lambda x: x[1])
    num_merge = len(all_filenames) - num_files + 1
    to_merge, remain = [x[0] for x in infos[:num_merge]], [x[0] for x in infos[num_merge:]]
    merged_name = f"merged-{int(time.time())}.parquet"
    pq.write_table(pa.concat_tables([pq.read_table(base_path + f, filesystem=fs) for f in to_merge]),
                   base_path + merged_name, filesystem=fs, compression=None if compression == "none" else compression,
                   row_group_size=row_group_size)
    for f in to_merge:
      fs.delete_file(base_path + f)
    all_filenames = remain + [merged_name]
  prefix = name_prefix or dataset.split("/")[-1].lower().replace(" ", "-")
  print("\nVerifying downloaded files...")
  existing_files = []
  for fname in all_filenames:
    fpath = base_path + fname
    try:
      if fs.get_file_info(fpath).size > 0:
        existing_files.append(fname)
      else:
        print(f"  [Warning] File is empty, skipping: {fpath}")
    except FileNotFoundError:
      print(f"  [Warning] File not found, skipping: {fpath}")
  total_files = len(existing_files)
  if total_files:
    print(f"\nRenaming {total_files} verified files to HuggingFace convention...")
    existing_files.sort()
    width = max(5, len(str(total_files)))
    for idx, old_name in enumerate(existing_files):
      new_name = f"{prefix}-{idx:0{width}d}-of-{total_files:0{width}d}.parquet"
      if old_name != new_name:
        fs.move(base_path + old_name, base_path + new_name)
    print(f"  Renamed to: {prefix}-00000-of-{total_files:0{width}d}.parquet ...")
  else:
    print("No valid files to rename.")
  is_gcs = output.startswith("gs://")
  local_checkpoint_dir = os.path.join("/tmp", output.replace("gs://", "gs_")) if is_gcs else os.path.abspath(output)
  for i in range(num_workers):
    path = os.path.join(local_checkpoint_dir, f".checkpoint-worker-{i:04d}.json")
    if os.path.exists(path):
      os.remove(path)
  if is_gcs and os.path.isdir(local_checkpoint_dir):
    shutil.rmtree(local_checkpoint_dir)
  total_rows = sum(r["total_rows"] for r in results)
  print(f"\nDone! Wrote {total_rows:,} rows in {time.time()-t0:.1f}s ({total_rows/write_elapsed:,.0f} rows/sec)")
