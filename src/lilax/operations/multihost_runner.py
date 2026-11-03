"""Run a command on workers of existing TPU slices."""

from collections import namedtuple
from datetime import datetime
from tempfile import gettempdir
import os
import re
import subprocess
import sys
import time


Slice = namedtuple("Slice", ["name", "slice_num", "num_workers", "version"])


def get_project():
  result = subprocess.run(["gcloud", "config", "get", "project"], check=True, capture_output=True)
  project = result.stdout.decode().strip().split("\n")[-1]
  if not project:
    sys.exit("You must specify the project in the PROJECT flag or set it with 'gcloud config set project <project>'")
  return project


def get_zone():
  result = subprocess.run(["gcloud", "config", "get", "compute/zone"], check=True, capture_output=True)
  zone = result.stdout.decode().strip().split("\n")[-1]
  if not zone:
    sys.exit("You must specify the zone in the ZONE flag or set it with 'gcloud config set compute/zone <zone>'")
  return zone


def filter_instances(instance_list, tpu_prefix):
  for instance in instance_list:
    if instance.split(",")[0] == tpu_prefix:
      return [instance]
  return [instance for instance in instance_list if re.fullmatch(tpu_prefix + "-[0-9]+", instance.split(",")[0])]


def get_slices(tpu_prefix, project, zone):
  """Discover matching slices and their worker count using gcloud."""
  command = [
      "gcloud", "alpha", "compute", "tpus", "tpu-vm", "list", f"--filter=name~{tpu_prefix}",
      "--format=csv(name,accelerator_type)", f"--project={project}", f"--zone={zone}",
  ]
  try:
    result = subprocess.run(command, capture_output=True, check=True)
  except subprocess.CalledProcessError as error:
    print(
        f"Error occurred trying to find TPU slices named {tpu_prefix} or matching regex \n {tpu_prefix}-[0-9]+ "
        f"in project {project} zone {zone}"
    )
    print(f"Error is:\n {error.stderr}")
    return []

  instances = filter_instances(result.stdout.decode().strip().split("\n")[1:], tpu_prefix)
  if not instances:
    print(f"No TPUs found with name {tpu_prefix} or matching regex {tpu_prefix}-[0-9]+ in project {project} and zone {zone}.")
    return []
  print(f"{len(instances)} slices found.", flush=True)
  names = [instance.split(",")[0] for instance in instances]
  versions = [instance.split(",")[1] for instance in instances]
  command = [
      "gcloud", "compute", "tpus", "tpu-vm", "describe", names[0], "--flatten=networkEndpoints[]",
      "--format=csv[no-heading](networkEndpoints.ipAddress)", f"--project={project}", f"--zone={zone}",
  ]
  result = subprocess.run(command, capture_output=True, check=True)
  num_workers = len(result.stdout.decode().strip().split("\n"))
  slices = [None for _ in instances]
  for name, version in zip(names, versions):
    slice_num = int(name.split("-")[-1]) if len(instances) > 1 else 0
    slices[slice_num] = Slice(name, slice_num, num_workers, version)
  return slices


def kill_existing_processes_str():
  return """#!/bin/bash
_TPU_VERSION_NAME="${1}"
device_name="accel"
if [[ "${_TPU_VERSION_NAME}" =~ ^v[56].* ]]; then
  device_name="vfio/"
fi
echo "Searching for existing processes on device ${device_name}..."
pids=$(sudo lsof -t /dev/${device_name}* | sort -u)
if [[ -n "${pids}" ]]; then
  echo "Existing processes found with pid ${pids}"
  for pid in ${pids}; do
    echo "Killing process ${pid}..."
    kill -9 "${pid}"
    tail --pid="${pid}" -f /dev/null
    echo "Existing process ${pid} on your TPU was killed successfully!"
  done
  echo "All existing processes killed, so your TPU is ready to use!"
else
  echo "No existing processes found, so your TPU is ready to use!"
fi
sudo rm -f /tmp/libtpu_lockfile"""


class Tee:
  """Write subprocess output to both the terminal and a persistent log."""

  def __init__(self, *files, bufsize=1):
    files = [file.fileno() if hasattr(file, "fileno") else file for file in files]
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid:
      os.close(read_fd)
      self._fileno = write_fd
      self.child_pid = pid
      return
    os.close(write_fd)
    while data := os.read(read_fd, bufsize):
      for file in files:
        os.write(file, data)
    os._exit(0)

  def fileno(self):
    return self._fileno

  def close(self):
    os.close(self._fileno)
    os.waitpid(self.child_pid, 0)


def run_commands(commands, id_to_print, jobname, worker_list, *, scp_timeout_secs=600,
                 is_shell=False, output_logs=None, fail_fast=True):
  """Run worker commands concurrently, optionally saving each worker's output."""
  children = []
  opened_logs = []
  tees = []
  start_time = datetime.now()
  try:
    for index, command in enumerate(commands):
      if output_logs:
        output_log = open(output_logs[index], "wt", encoding="utf-8")
        opened_logs.append(output_log)
        if index == id_to_print:
          output_log = Tee(sys.stdout, output_log)
          tees.append(output_log)
      elif index == id_to_print:
        output_log = None
      else:
        output_log = subprocess.DEVNULL
      children.append(subprocess.Popen(command, stdout=output_log, stderr=output_log, shell=is_shell))

    while True:
      returncodes = [child.poll() for child in children]
      max_returncode = max([0] + [code for code in returncodes if code is not None])
      completed = len([code for code in returncodes if code is not None])
      seconds_elapsed = (datetime.now() - start_time).total_seconds()
      if completed < len(returncodes):
        slow_worker = worker_list[returncodes.index(None)]
        slow_str = f", slice {slow_worker[0]} worker {slow_worker[1]} still working"
      else:
        slow_str = ""
      print(f"[t={seconds_elapsed:.2f}, {jobname}] Completed {completed}/{len(returncodes)}{slow_str}...")

      if seconds_elapsed >= scp_timeout_secs and 0 not in returncodes and jobname == "SCP":
        print(
            f"SCP operation timed out after SCP_TIMEOUT_SECS={scp_timeout_secs} seconds - terminating all processes. "
            "If progress was being made, increase the timeout or decrease the number of files in SCRIPT_DIR. "
            "Otherwise check that INTERNAL_IP is set correctly for the runner and worker projects."
        )
        for child in children:
          child.terminate()
        max_returncode = 255
        break
      if fail_fast and max_returncode > 0:
        print(f"Terminating all {jobname} processes since at least one failed.")
        for child in children:
          child.terminate()
        break
      if completed == len(returncodes):
        break
      time.sleep(1)
    return max_returncode, returncodes
  except BaseException:
    for child in children:
      if child.poll() is None:
        child.terminate()
    raise
  finally:
    for child in children:
      child.wait()
    for tee in tees:
      tee.close()
    for output_log in opened_logs:
      output_log.close()


def scps(slices, run_name_dir, zip_name, *, script_dir, project, zone, internal_ip=False, scp_timeout_secs=600):
  """Archive the script directory and copy it to every TPU worker."""
  os.makedirs(run_name_dir, exist_ok=True)
  zip_path = os.path.join(run_name_dir, zip_name)
  subprocess.run(["tar", "--exclude=tmp", "-czf", zip_path, "./"], check=True, cwd=script_dir)
  commands = []
  worker_list = []
  for current_slice in slices:
    for worker_num in range(current_slice.num_workers):
      command = [
          "gcloud", "compute", "tpus", "tpu-vm", "scp", f"--worker={worker_num}", zip_path,
          f"{current_slice.name}:~/", "--strict-host-key-checking=no", f"--project={project}", f"--zone={zone}",
      ]
      if internal_ip:
        command.append("--internal-ip")
      commands.append(command)
      worker_list.append([current_slice.slice_num, worker_num])
  return_code, _ = run_commands(commands, 0, "SCP", worker_list, scp_timeout_secs=scp_timeout_secs)
  if return_code != 0:
    print("Failed to scp zipped code directory with error code ", return_code)
  return return_code


def execute_main_command(main_command, slices, local_log_dir, zip_name, *, run_name, project, zone,
                         use_existing_folder=False, internal_ip=False, scp_timeout_secs=600):
  """Run the command on every worker, with one log file per worker."""
  commands = []
  output_logs = []
  worker_list = []
  os.makedirs(local_log_dir, exist_ok=True)
  for slice_num, current_slice in enumerate(slices):
    for worker_num in range(current_slice.num_workers):
      output_logs.append(os.path.join(
          local_log_dir, f"output_slice_{current_slice.slice_num:04d}_worker_{worker_num:04d}.txt"
      ))
      remote_commands = []
      if not use_existing_folder:
        remote_commands.extend([f"mkdir -p {run_name}", f"mv {zip_name} {run_name}"])
      remote_commands.append(f"cd {run_name}")
      if not use_existing_folder:
        remote_commands.append(f"tar xzf {zip_name}")
      remote_commands.extend([
          f"echo '{kill_existing_processes_str()}' > kill_existing_processes.sh",
          f"bash kill_existing_processes.sh {current_slice.version}",
          main_command,
      ])
      command = [
          "gcloud", "alpha", "compute", "tpus", "tpu-vm", "ssh", current_slice.name,
          f"--worker={worker_num}", "--command", " && ".join(remote_commands), "--strict-host-key-checking=no",
          f"--project={project}", f"--zone={zone}",
      ]
      if internal_ip:
        command.append("--internal-ip")
      commands.append(command)
      worker_list.append([slice_num, worker_num])
  return_code, return_codes = run_commands(
      commands, 0, "MAIN COMMAND", worker_list, scp_timeout_secs=scp_timeout_secs, output_logs=output_logs
  )
  if return_code > 0:
    failure_index = next((index for index, code in enumerate(return_codes) if code), None)
    print(
        f"Main command failed on slice {worker_list[failure_index][0]} worker"
        f" {worker_list[failure_index][1]} with error code {return_codes[failure_index]}, see logs for details",
        flush=True,
    )
  return return_code


def run_multihost(tpu_prefix, command, *, project=None, zone=None, script_dir=None, run_name=None,
                  use_existing_folder=False, internal_ip=False, scp_timeout_secs=600):
  """Copy a directory and run a command on existing TPU slices.

  Logs and the archive are saved under the system temporary directory in ``run_name``.
  Existing TPU processes are killed before the command runs, as in the original runner.
  Returns the source runner's status code.
  """
  if not tpu_prefix:
    raise ValueError("--TPU_PREFIX must be a non-empty string specifying your TPU slice names.")
  if run_name is None:
    run_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  if use_existing_folder and not run_name:
    raise ValueError("When USE_EXISTING_FOLDER is true, RUN_NAME must be specified.")
  if script_dir is None:
    script_dir = os.getcwd()
  print("Starting multihost runner...", flush=True)
  if not project:
    project = get_project()
  if not zone:
    zone = get_zone()
  if not os.path.isdir(script_dir):
    sys.exit(f"No directory named {script_dir} found.")

  slices = get_slices(tpu_prefix, project, zone)
  if not slices:
    print(f"Failed to retrieve slices {tpu_prefix} in project {project} zone {zone}", flush=True)
    return 1
  local_log_dir = os.path.join(gettempdir(), run_name, "")
  zip_name = f"script_dir_zip_{run_name}.tar.gz"
  if not use_existing_folder:
    return_code = scps(
        slices, local_log_dir, zip_name, script_dir=script_dir, project=project, zone=zone,
        internal_ip=internal_ip, scp_timeout_secs=scp_timeout_secs,
    )
    if return_code > 0:
      print(f"Moving the directory {script_dir} to the VMs failed with error code {return_code}")
      return return_code

  print(f"Running main command, logs located in: {local_log_dir}", flush=True)
  return_code = execute_main_command(
      command, slices, local_log_dir, zip_name, run_name=run_name, project=project, zone=zone,
      use_existing_folder=use_existing_folder, internal_ip=internal_ip, scp_timeout_secs=scp_timeout_secs,
  )
  if return_code == 0:
    print(f"Main command completed successfully, logs located in: {local_log_dir}", flush=True)
    print("Multihost runner finished successfully!", flush=True)
  else:
    print(f"Main command finished with errors, check the logs located in: {local_log_dir}", flush=True)
  return return_code
