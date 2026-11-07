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

"""Download OLMo mix files and build their numpy index."""

import io
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

import numpy as np
import requests
from google.cloud import storage
from requests.exceptions import RequestException

from lilax._backend.input_pipeline.olmo.olmo_data import (
    build_index,
    has_npy_magic,
    parse_npy_header,
    read_raw_metadata_from_path,
)


def parse_mix_file(mix_path: str, tokenizer: str, *, warning_file=None) -> list[tuple[str, str]]:
  """Read labels and relative paths, substituting the tokenizer in each path."""
  entries = []
  with open(mix_path, encoding="utf-8") as f:
    for line_num, line in enumerate(f, start=1):
      line = line.strip()
      if not line or line.startswith("#"):
        continue
      parts = line.split(",", maxsplit=1)
      if len(parts) != 2:
        print(f"WARNING: {mix_path}:{line_num}: skipping malformed line: {line!r}", file=warning_file)
        continue
      label, rel_path = parts
      entries.append((label, rel_path.replace("{TOKENIZER}", tokenizer)))
  return entries


def _total_size_from_response(resp, bytes_have):
  cr = resp.headers.get("Content-Range")
  if cr and "/" in cr:
    tail = cr.split("/")[-1].strip()
    if tail.isdigit():
      return int(tail)
  cl = resp.headers.get("Content-Length")
  if cl is not None:
    return int(cl) + (bytes_have if resp.status_code == 206 else 0)
  return None


def http_resumable_to_local(
    url: str,
    local_path: str,
    max_retries: int = 20,
    chunk_size: int = 8 * 1024 * 1024,
    timeout: int = 300,
    progress_every_s: float = 30.0,
) -> tuple[int, int | None]:
  """Download to local storage, resuming partial files with HTTP Range."""
  os.makedirs(os.path.dirname(local_path), exist_ok=True)
  session = requests.Session()
  total_size = None
  bytes_have = os.path.getsize(local_path) if os.path.exists(local_path) else 0
  if bytes_have:
    print(f"    resuming {os.path.basename(local_path)} from {bytes_have / 1e9:.2f} GB", flush=True)

  for attempt in range(1, max_retries + 1):
    try:
      headers = {}
      if bytes_have > 0:
        headers["Range"] = f"bytes={bytes_have}-"
      with session.get(url, stream=True, headers=headers, timeout=timeout) as resp:
        resp.raise_for_status()
        if bytes_have > 0 and resp.status_code == 200:
          print(f"    server returned 200 to Range request; restarting {os.path.basename(local_path)} from 0", flush=True)
          bytes_have = 0
          if os.path.exists(local_path):
            os.remove(local_path)
        new_total = _total_size_from_response(resp, bytes_have)
        if new_total is not None:
          if total_size is not None and new_total != total_size:
            print(f"    WARNING: total size changed mid-download ({total_size} → {new_total}); using new value", flush=True)
          total_size = new_total
        mode = "ab" if bytes_have > 0 else "wb"
        last_log = attempt_start = time.time()
        attempt_start_bytes = bytes_have
        with open(local_path, mode) as fp:
          for chunk in resp.iter_content(chunk_size=chunk_size):
            if not chunk:
              continue
            fp.write(chunk)
            bytes_have += len(chunk)
            now = time.time()
            if now - last_log >= progress_every_s:
              dt = max(now - attempt_start, 1e-6)
              dbytes = bytes_have - attempt_start_bytes
              rate_mb = (dbytes / 1024 / 1024) / dt
              if total_size:
                pct = 100.0 * bytes_have / total_size
                eta_s = (total_size - bytes_have) / max(dbytes / dt, 1)
                print(f"    [{os.path.basename(local_path)}] {bytes_have/1e9:.2f}/{total_size/1e9:.2f} GB "
                      f"({pct:.1f}%) @ {rate_mb:.0f} MB/s; ETA {eta_s/60:.0f} min", flush=True)
              else:
                print(f"    [{os.path.basename(local_path)}] {bytes_have/1e9:.2f} GB @ {rate_mb:.0f} MB/s", flush=True)
              last_log = now
      if total_size is not None and bytes_have != total_size:
        raise RuntimeError(f"truncated: have {bytes_have} bytes, expected {total_size}")
      return bytes_have, total_size
    except (RequestException, RuntimeError, ConnectionError) as exc:
      wait = min(2**attempt, 60)
      print(f"    [retry {attempt}/{max_retries}] @{bytes_have/1e9:.2f} GB: "
            f"{type(exc).__name__}: {str(exc)[:200]}; sleeping {wait}s", flush=True)
      time.sleep(wait)
      bytes_have = os.path.getsize(local_path) if os.path.exists(local_path) else 0
  raise RuntimeError(f"exhausted {max_retries} retries; got {bytes_have} bytes"
                     f"{'/' + str(total_size) if total_size else ''}")


def download_one(rel_path: str, http_base_url: str, gcs_dest_prefix: str, staging_dir: str,
                 skip_existing: bool, max_retries: int) -> tuple[str, str, int]:
  """Download one entry to local staging and upload it to GCS."""
  http_url = urljoin(http_base_url.rstrip("/") + "/", rel_path)
  gcs_dest = gcs_dest_prefix.rstrip("/") + "/" + rel_path.lstrip("/")
  bucket_name, _, blob_name = gcs_dest.replace("gs://", "").partition("/")
  if skip_existing:
    try:
      if storage.Client().bucket(bucket_name).blob(blob_name).exists():
        return rel_path, "skipped", 0
    except ValueError as exc:
      print(f"Error while accessing {gcs_dest} from GCE: {str(exc)}")
  local_path = os.path.join(staging_dir, rel_path)
  try:
    bytes_written, _ = http_resumable_to_local(http_url, local_path, max_retries=max_retries)
    blob = storage.Client().bucket(bucket_name).blob(blob_name)
    blob.upload_from_filename(local_path)
    blob.reload()
    if blob.size != bytes_written:
      return rel_path, f"error:size_mismatch local={bytes_written} gcs={blob.size}", bytes_written
    os.remove(local_path)
    return rel_path, "ok", bytes_written
  except Exception:  # pylint: disable=broad-except
    return rel_path, f"error:{traceback.format_exc()[-800:]}", 0


def download_olmo_data_to_gcs(
    mix_file: str,
    gcs_dest: str,
    *,
    tokenizer: str = "allenai/dolma3-tokenizer",
    http_base_url: str = "http://olmo-data.org/",
    workers: int = 2,
    staging_dir: str = "/tmp/olmo-staging",
    max_retries: int = 20,
    skip_existing: bool = True,
    dry_run: bool = False,
) -> None:
  """Download a mix over HTTP and upload its files under a GCS prefix."""
  print(f"Parsing mix file: {mix_file}")
  entries = parse_mix_file(mix_file, tokenizer)
  print(f"  {len(entries)} files found  (tokenizer={tokenizer!r})")
  gcs_dest_prefix = gcs_dest.rstrip("/")
  http_base = http_base_url.rstrip("/") + "/"
  if dry_run:
    for label, rel_path in entries:
      print(f"  [{label}]  {urljoin(http_base, rel_path)}  →  {gcs_dest_prefix + '/' + rel_path.lstrip('/')}")
    print(f"\nDry run complete. {len(entries)} files listed.")
    return

  os.makedirs(staging_dir, exist_ok=True)
  if skip_existing:
    print("Existing GCS files will be skipped.")
  print(f"Staging to: {staging_dir}")
  print(f"Workers: {workers}  Max retries per file: {max_retries}")
  rel_paths = [r for _, r in entries]
  n_ok = n_skipped = n_error = total_bytes = 0
  errors = []
  start = time.time()
  with ThreadPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(download_one, path, http_base_url, gcs_dest_prefix, staging_dir,
                           skip_existing, max_retries): path for path in rel_paths}
    for i, future in enumerate(as_completed(futures), start=1):
      rel_path, status, nbytes = future.result()
      total_bytes += nbytes
      if status == "ok":
        n_ok += 1
      elif status == "skipped":
        n_skipped += 1
      else:
        n_error += 1
        errors.append((rel_path, status))
      elapsed = time.time() - start
      tag = status.split(":", 1)[0]
      print(f"  [{i}/{len(rel_paths)}] ok={n_ok} skipped={n_skipped} err={n_error}"
            f" | {total_bytes/1e9:.2f} GB | {elapsed:.0f}s | {tag} {rel_path}", flush=True)
  elapsed = time.time() - start
  print(f"\nDone in {elapsed:.1f}s. ok={n_ok} skipped={n_skipped} errors={n_error}"
        f" | {total_bytes/1e9:.2f} GB transferred")
  if errors:
    print(f"\nFailed files ({len(errors)}):")
    for rel_path, status in errors:
      print(f"  {rel_path}\n    {status}")
    raise RuntimeError(f"Failed to download {len(errors)} files")


def _split_gs_uri(uri: str) -> tuple[str, str]:
  if not uri.startswith("gs://"):
    raise ValueError(f"Not a gs:// URI: {uri}")
  bucket, _, key = uri[len("gs://"):].partition("/")
  if not bucket or not key:
    raise ValueError(f"Malformed gs:// URI: {uri}")
  return bucket, key


def make_header_reader(dtype_for_raw: str):
  """Create a reader for local or GCS NPY headers and headerless streams."""
  client_holder = {"client": None}

  def reader(path: str):
    if path.startswith("gs://"):
      if client_holder["client"] is None:
        client_holder["client"] = storage.Client()
      bucket, key = _split_gs_uri(path)
      blob = client_holder["client"].bucket(bucket).blob(key)
      head = blob.download_as_bytes(start=0, end=7)
      if has_npy_magic(head):
        return parse_npy_header(io.BytesIO(blob.download_as_bytes(start=0, end=4095)))
      blob.reload()
      if blob.size is None:
        raise RuntimeError(f"GCS blob {path} has unknown size")
      size = int(blob.size)
      itemsize = np.dtype(dtype_for_raw).itemsize
      if size % itemsize != 0:
        raise ValueError(f"GCS blob {path} size {size} is not a multiple of dtype "
                         f"{dtype_for_raw} itemsize ({itemsize}).")
      return dtype_for_raw, (size // itemsize,)
    with open(path, "rb") as fh:
      head = fh.read(8)
    if has_npy_magic(head):
      with open(path, "rb") as fh:
        return parse_npy_header(io.BytesIO(fh.read()))
    return read_raw_metadata_from_path(path, dtype_for_raw)

  return reader


def scan_headers_parallel(paths_and_labels: list[tuple[str, str]], *, dtype_for_raw: str,
                          workers: int = 32, progress_every: int = 50):
  """Scan headers concurrently and return results in mix-file order."""
  reader = make_header_reader(dtype_for_raw)
  results = [None] * len(paths_and_labels)
  start = time.time()
  with ThreadPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(reader, path): (i, label, path)
               for i, (label, path) in enumerate(paths_and_labels)}
    for done, future in enumerate(as_completed(futures), start=1):
      i, label, path = futures[future]
      dtype, shape = future.result()
      results[i] = (label, path, dtype, shape)
      if done % progress_every == 0 or done == len(paths_and_labels):
        print(f"  scanned {done}/{len(paths_and_labels)} headers ({time.time() - start:.0f}s)",
              file=sys.stderr, flush=True)
  return results


def build_olmo_npy_index(
    mix_file: str,
    gcs_base: str,
    sequence_length: int,
    output: str,
    *,
    tokenizer: str = "allenai/dolma3-tokenizer",
    dtype: str = "uint32",
    workers: int = 32,
):
  """Build and save the original OLMo index, returning the index object."""
  print(f"Parsing mix file: {mix_file}", file=sys.stderr)
  entries = parse_mix_file(mix_file, tokenizer, warning_file=sys.stderr)
  print(f"  {len(entries)} entries", file=sys.stderr)
  base = gcs_base.rstrip("/") + "/"
  resolved = [(label, base + rel.lstrip("/")) for label, rel in entries]
  print(f"Scanning {len(resolved)} .npy headers ({workers} threads, raw dtype={dtype})...", file=sys.stderr)
  headers = scan_headers_parallel(resolved, dtype_for_raw=dtype, workers=workers)
  header_cache = {path: (header_dtype, shape) for _, path, header_dtype, shape in headers}
  index = build_index([(path, label) for label, path, _, _ in headers], sequence_length=sequence_length,
                      tokenizer=tokenizer, header_reader=header_cache.__getitem__)
  print(f"Total tokens: {index.total_tokens:,}  |  instances at SEQ={sequence_length}: "
        f"{index.total_instances:,}  |  fingerprint: {index.fingerprint}", file=sys.stderr)
  index.save(output)
  print(f"Wrote index to {output}", file=sys.stderr)
  return index
