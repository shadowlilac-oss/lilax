"""Common GCS utilities needed by multiple modules."""

import json
import os
import shutil
import socket
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from etils import epath
from google.cloud import storage
import jax
import yaml

from lilax._backend.observability import max_logging


def write_config_raw_keys_for_gcs(raw_keys):
  """Writes config raw keys to GCS when enabled on the primary process."""
  if not raw_keys.get("save_config_to_gcs") or jax.process_index() != 0:
    return
  max_logging.log("Writing config to GCS...")
  raw_keys_dict = dict(raw_keys)
  filename = "config.yml"
  with open(filename, "wt", encoding="utf8") as config_for_gcs:
    yaml.dump(raw_keys_dict, config_for_gcs)
  config_for_gcs.close()
  gcs_filename = os.path.join(raw_keys["base_output_directory"], raw_keys["run_name"], filename)
  max_logging.log(f"Moving file {filename} to GCS...")
  upload_blob(gcs_filename, filename)
  max_logging.log(f"File {filename} moved successfully!")


def parse_gcs_bucket_and_prefix(destination_gcs_name):
  path_parts = destination_gcs_name.replace("gs://", "").split("/")
  bucket = path_parts.pop(0)
  key = "/".join(path_parts)
  return bucket, key


def add_trailing_slash(path):
  if not path.endswith("/"):
    return path + "/"
  return path


def upload_blob(destination_gcs_name, source_file_name):
  """Uploads a file to a GCS location."""
  bucket_name, prefix_name = parse_gcs_bucket_and_prefix(destination_gcs_name)
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(prefix_name)
  blob.upload_from_filename(source_file_name)


def upload_dump(local_dir, target_dir, module_name=None, delete_local_after=True, all_host_upload=False):
  """Uploads a directory to a GCS location, with an optional filter."""
  if not all_host_upload and jax.process_index() != 0:
    return
  storage_client = storage.Client()
  bucket_name, prefix_name = parse_gcs_bucket_and_prefix(target_dir)
  bucket = storage_client.get_bucket(bucket_name)
  if all_host_upload:
    hostname = socket.gethostname()
    prefix_name = os.path.join(prefix_name, hostname)
    target_dir = os.path.join(target_dir, hostname)
  max_logging.log(f"Uploading Dump to {target_dir}...")
  for root, _, files in os.walk(local_dir):
    for file in files:
      if module_name and module_name not in file:
        continue
      else:
        max_logging.log(f"Uploading {file}")
      local_path = os.path.join(root, file)
      relative_path = os.path.relpath(local_path, local_dir)
      blob_name = os.path.join(prefix_name, relative_path)
      blob = bucket.blob(blob_name)
      blob.upload_from_filename(local_path)
  max_logging.log(f"Dump Uploaded to {target_dir}!")
  if delete_local_after:
    shutil.rmtree(local_dir)


def gcs_path_exists(file_path):
  """Checks if a GCS file_path exists."""
  try:
    storage_client = storage.Client()
    bucket_name, file_name = parse_gcs_bucket_and_prefix(file_path)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    return blob.exists()
  except ValueError as e:
    print(f"Error while accessing {file_path} from GCE: {str(e)}")
    return False


def gcs_list_directories(directory_path):
  """Lists directory prefixes one level down within a GCS directory."""
  storage_client = storage.Client()
  bucket_name, directory_prefix = parse_gcs_bucket_and_prefix(directory_path)
  bucket = storage_client.bucket(bucket_name)
  if not directory_prefix.endswith("/"):
    directory_prefix += "/"
  delimiter = "/"
  blobs = bucket.list_blobs(prefix=directory_prefix, delimiter=delimiter)
  directories = []
  for page in blobs.pages:
    for prefix in page.prefixes:
      path_obj = Path(prefix)
      directory = path_obj.name
      directories.append(directory)
  return directories


def gcs_delete_directory(directory_path: str):
  """Deletes all blobs under a GCS directory prefix."""
  storage_client = storage.Client()
  bucket_name, directory_prefix = parse_gcs_bucket_and_prefix(directory_path)
  bucket = storage_client.bucket(bucket_name)
  if not directory_prefix.endswith("/"):
    directory_prefix += "/"
  blobs = list(bucket.list_blobs(prefix=directory_prefix))
  if blobs:
    def _delete_blob(blob):
      try:
        blob.delete()
      except Exception as e:
        max_logging.log(f"Error deleting blob {blob.name}: {e}")

    with ThreadPoolExecutor(max_workers=32) as executor:
      executor.map(_delete_blob, blobs)


def gcs_glob_pattern(pattern):
  """Globs GCS files and returns a list of full GCS paths."""
  storage_client = storage.Client()
  bucket_name, glob_pattern = parse_gcs_bucket_and_prefix(pattern)
  blobs = storage_client.list_blobs(bucket_name, match_glob=glob_pattern)
  data_files = [f"gs://{bucket_name}/{blob.name}" for blob in blobs]
  return data_files


def read_bytes_from_gcs(file_path):
  """Reads raw bytes from a GCS file, or returns None if unavailable."""
  try:
    storage_client = storage.Client()
    bucket_name, file_prefix = parse_gcs_bucket_and_prefix(file_path)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_prefix)
    return blob.download_as_bytes()
  except Exception as e:
    max_logging.log(f"Error reading bytes from GCS path {file_path}: {e}")
    return None


def read_json_from_gcs(file_path):
  """Reads a JSON file from a GCS bucket."""
  try:
    raw = read_bytes_from_gcs(file_path)
    if raw is None:
      return None
    return json.loads(raw)
  except (ValueError, TypeError, json.JSONDecodeError) as e:
    max_logging.log(f"Error reading JSON file from GCS: {str(e)}")
    return None


def write_dict_to_gcs_json(data_dict, file_path):
  """Writes a Python dictionary to a JSON file in GCS."""
  try:
    storage_client = storage.Client()
    bucket_name, file_prefix = parse_gcs_bucket_and_prefix(file_path)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_prefix)
    json_string = json.dumps(data_dict, indent=4)
    blob.upload_from_string(json_string, content_type="application/json")
  except (ValueError, TypeError, RecursionError) as e:
    print(f"Failed to write json file at {file_path} with error: {str(e)}")


def mkdir_and_check_permissions(path: str | epath.Path) -> epath.Path:
  """Creates a directory if it doesn't exist and verifies write permissions."""
  if isinstance(path, str):
    path = epath.Path(path)
  is_gcs = path.as_posix().startswith("gs://")
  if is_gcs:
    if len(path.parts) < 3:
      raise ValueError(f"Invalid GCS path (missing bucket name): '{path}'")
    bucket_name = path.parts[2]
    try:
      storage_client = storage.Client()
      storage_client.get_bucket(bucket_name)
    except Exception as e:
      raise FileNotFoundError(f"GCS bucket 'gs://{bucket_name}' not found or accessible.") from e
  path.mkdir(exist_ok=True, parents=True)
  if not is_gcs and not path.exists():
    raise PermissionError(f"Failed to create the directory '{path}'. Please check that you have write access.")
  temp_file_path = path / f".write_test_{uuid.uuid4()}"
  try:
    temp_file_path.write_text("test")
  except Exception as e:
    raise PermissionError(f"Directory '{path}' exists, but is not writable. Please check permissions.") from e
  finally:
    try:
      temp_file_path.unlink()
    except Exception:
      pass
  return path
