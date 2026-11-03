# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare the LLaVA-Video-178K dataset."""

import os
import shutil
import tarfile

import datasets
from google.cloud import storage
from huggingface_hub import hf_hub_download


def prepare_llava_video_178k(
    output_dir, fold="0_30_s_academic_v0_1", download_videos=False, video_subset=None, local_temp_dir=None
):
  """Download metadata and optionally extract video archives into local storage or GCS."""
  is_gcs = output_dir.startswith("gs://")
  if is_gcs:
    if local_temp_dir is None:
      local_temp_dir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "llava_video_178k_temp")
    local_output_dir = local_temp_dir
  else:
    local_output_dir = output_dir

  fold_dir = os.path.join(local_output_dir, fold)
  os.makedirs(fold_dir, exist_ok=True)
  repo_id = "lmms-lab/LLaVA-Video-178K"

  print("Downloading metadata...")
  metadata_file = f"{fold}_cap_processed.json"
  local_metadata_path = hf_hub_download(
      repo_id=repo_id,
      filename=f"{fold}/{metadata_file}",
      repo_type="dataset",
      local_dir=local_output_dir,
      local_dir_use_symlinks=False,
  )

  print("Converting metadata to Parquet...")
  dataset = datasets.load_dataset("json", data_files=local_metadata_path, split="train")
  parquet_path = os.path.join(fold_dir, "llava-video-178k-caption-00000-of-00001.parquet")
  dataset.to_parquet(parquet_path)
  print(f"Saved parquet to {parquet_path}")

  if download_videos:
    print("Downloading videos...")
    if video_subset is not None:
      tar_files = [f"{fold}/{fold}_videos_{i}.tar.gz" for i in range(1, video_subset + 1)]
    else:
      tar_files = [f"{fold}/{fold}_videos_{i}.tar.gz" for i in range(1, 9)]

    for tar_file in tar_files:
      print(f"Downloading {tar_file}...")
      local_tar_path = hf_hub_download(
          repo_id=repo_id,
          filename=tar_file,
          repo_type="dataset",
          local_dir=local_output_dir,
          local_dir_use_symlinks=False,
      )
      print(f"Extracting {local_tar_path} to {fold_dir}...")
      with tarfile.open(local_tar_path, "r:gz") as tar:
        tar.extractall(path=fold_dir)
    print("Videos preparation complete.")
  else:
    print("Skipping videos download. Use --download_videos to download them.")

  if is_gcs:
    bucket_name, _, prefix = output_dir[5:].partition("/")
    bucket = storage.Client().bucket(bucket_name)

    def raise_walk_error(error):
      raise error

    for root, _, files in os.walk(fold_dir, onerror=raise_walk_error):
      for filename in files:
        local_path = os.path.join(root, filename)
        relative_path = os.path.relpath(local_path, os.path.dirname(fold_dir)).replace(os.sep, "/")
        object_name = f"{prefix.rstrip('/')}/{relative_path}" if prefix.rstrip("/") else relative_path
        bucket.blob(object_name).upload_from_filename(local_path)
    shutil.rmtree(fold_dir)
