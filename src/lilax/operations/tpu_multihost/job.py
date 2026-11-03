"""Submit a queued TPU job using the original multihost job workflow."""

from datetime import datetime
import os
import shutil
import subprocess
import sys


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


def move_script_dir_to_gcs(script_dir, tmp_dir, zip_name, bucket_path):
  """Archive the script directory and move it to GCS."""
  os.makedirs(os.path.join(script_dir, tmp_dir), exist_ok=True)
  zip_path = os.path.join(tmp_dir, zip_name)
  subprocess.run(["tar", "--exclude=tmp", "-czf", zip_path, "./"], check=True, cwd=script_dir)
  return subprocess.run(
      ["gcloud", "storage", "mv", os.path.join(script_dir, zip_path), "/".join((bucket_path, zip_name))],
      check=True, capture_output=True,
  )


def get_env_command_str(num_slices):
  env_str = """curl -s 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/tpu-env' -H 'Metadata-Flavor: Google' > /tmp/tpu-env
NODE_ID=$(grep '^NODE_ID' /tmp/tpu-env | cut -d "'" -f 2)
WORKER_ID=$(grep '^WORKER_ID' /tmp/tpu-env | cut -d "'" -f 2)
ZONE=$(grep '^ZONE' /tmp/tpu-env | cut -d "'" -f 2)
PROJECT=$(grep '^CONSUMER_PROJECT_ID' /tmp/tpu-env | cut -d "'" -f 2)"""
  if num_slices == 1:
    slice_assignment = "SLICE_ID=0"
  else:
    slice_assignment = """SLICE_ID=$(grep '^MEGASCALE_SLICE_ID' /tmp/tpu-env | cut -d "'" -f 2)"""
  return env_str + "\n" + slice_assignment


def create_ops_config_str(run_name, log_name):
  return f"""logging:
  receivers:
    {run_name}_log:
      type: files
      include_paths:
      - /{run_name}/{log_name}
      record_log_file_path: true
  service:
    pipelines:
      default_pipeline:
        receivers: [{run_name}_log]"""


def install_ops_script_str(run_name, log_name):
  return f"""OPS_FILE=/etc/google-cloud-ops-agent/config.yaml
  if ! test -f \$OPS_FILE;
  then
    curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
    downloaded=0
    while [ \$downloaded -eq 0 ]
    do
      pid=\$(sudo lsof /var/lib/dpkg/lock-frontend | awk \\"END{{print \$2}}\\")
      if [[ ! -z \"\${{pid}}\" ]]
      then
        sleep 10
      else
        sudo bash add-google-cloud-ops-agent-repo.sh --also-install
        downloaded=1
      fi
    done
  fi
  sudo chmod 777 /etc/google-cloud-ops-agent/config.yaml
  sudo echo \\"{create_ops_config_str(run_name, log_name)}\\" >> /etc/google-cloud-ops-agent/config.yaml
  sudo service google-cloud-ops-agent restart
"""


def setup_ops_str(run_name, log_name):
  return f"""
    echo "{install_ops_script_str(run_name, log_name)}" > install_ops_wait_dpkg.sh
    bash install_ops_wait_dpkg.sh &
  """


def write_download_from_gcs_sh(zip_gcs_path):
  return f"""GCS_READ_SUCCESS=0
while [ \$GCS_READ_SUCCESS -eq 0 ]
do
  {{
      gcloud storage cp {zip_gcs_path} . &&
      echo 'Code download from GCS successful!' && GCS_READ_SUCCESS=1
  }} || {{
      echo 'Failed to read GCS via gcloud storage, trying again'
      sleep 10
  }}
done"""


def download_from_gcs(zip_gcs_path):
  return f"""
    echo "{write_download_from_gcs_sh(zip_gcs_path)}" > download_from_gcs.sh
    bash download_from_gcs.sh
  """


def create_kill_command_str(run_name, project, zone):
  return f"""if [[ $SLICE_ID -eq 0 && $WORKER_ID -eq 0 ]]; then
  echo "This worker (slice 0 worker 0) will wait 10 minutes before tearing down the job to allow other workers to gracefully exit."
  sleep 600
  gcloud alpha compute tpus queued-resources delete {run_name} --force --quiet --project={project} --zone={zone}
  fi"""


def write_startup_script(zip_gcs_path, zip_name, log_name, bucket_path, startup_script_file, *,
                         run_name, num_slices, command, project, zone):
  """Write the startup script executed on each provisioned TPU worker."""
  startup_script = f"""#!/bin/bash
mkdir -p {run_name}
cd {run_name}
{get_env_command_str(num_slices)}
{setup_ops_str(run_name, log_name)}
sudo python3 -m virtualenv venv
source venv/bin/activate
ulimit -n 100000
(({download_from_gcs(zip_gcs_path)}
tar xzf {zip_name}
{command}) 2>&1) >> {log_name}
(echo "multihost_job finished main command on slice $SLICE_ID worker $WORKER_ID at $(date "+%Y-%m-%d %H:%M:%S") UTC with exit status $?.
This worker will immediately send its logs to GCS.") >> {log_name}
gcloud storage cp {log_name} "{bucket_path}/"
(({create_kill_command_str(run_name, project, zone)}) 2>&1 ) >> {log_name}"""
  with open(startup_script_file, "wt", encoding="utf-8") as startup_file:
    startup_file.write(startup_script)
  return startup_script


def run_create_resources(startup_script_file, *, run_name, tpu_type, version, project, zone,
                         num_slices, cqr_extra_args=None, enable_autocheckpoint=False):
  """Submit the queued-resource request through gcloud."""
  command = (
      f"gcloud alpha compute tpus queued-resources create {run_name} --accelerator-type={tpu_type} "
      f"--runtime-version={version} --project={project} --zone={zone}"
  )
  if num_slices > 1:
    command += f" --node-prefix={run_name} --node-count={num_slices}"
  else:
    command += f" --node-id={run_name}"
  if cqr_extra_args:
    command += " " + cqr_extra_args
  if enable_autocheckpoint:
    command += " --autocheckpoint-enabled"
  command += f" --metadata-from-file=startup-script={startup_script_file}"
  return subprocess.run(command, check=False, shell=True, capture_output=True)


def google_cloud_logging_url(run_name, project):
  return f"https://console.cloud.google.com/logs/query;query=resource.type%3D%22gce_instance%22%20AND%0Alog_id%2528%22{run_name}_log%22%2529;?project={project}"


def google_cloud_logging_single_host_url(run_name, project):
  return f"https://console.cloud.google.com/logs/query;query=resource.type%3D%22gce_instance%22%20AND%0Alog_id%2528%22{run_name}_log%22%2529%20AND%0Alabels.%22agent.googleapis.com%2Flog_file_path%22%3D%20%22%2F{run_name}%2Fmain_command_log_slice_0_worker_0%22;?project={project}"


def gcs_bucket_url(bucket_name, bucket_dir, project):
  return f"https://console.cloud.google.com/storage/browser/{bucket_name}/{bucket_dir}?project={project}"


def run_multihost_job(command, bucket_name, *, tpu_type="v4-8", version="tpu-ubuntu2204-base", num_slices=2,
                      script_dir=None, bucket_dir="", project=None, zone=None, run_name=None,
                      cqr_extra_args=None, enable_autocheckpoint=False):
  """Submit a new queued TPU job and return the source workflow's status code.

  This provisions new TPU resources and uploads an archive of ``script_dir`` to
  ``gs://<bucket_name>/<bucket_dir>/<run_name>/``. Each worker installs and
  configures the Cloud Ops Agent, runs ``command`` in a new virtualenv, and
  copies its log to that GCS location. Slice 0 worker 0 waits ten minutes after
  its command ends, then force-deletes the queued resource and its TPUs. The
  uploaded archive and logs remain in GCS; local temporary files are removed
  only after a successful resource request. The command must include any
  required dependency installation. This returns after request submission,
  not after the workers complete.

  The original workflow passes resource request arguments through a shell and
  embeds the command and other values in worker shell scripts; use trusted input.
  """
  print("\nStarting multihost_job...\n", flush=True)
  if not project:
    project = get_project()
  if not zone:
    zone = get_zone()
  if not run_name:
    run_name = os.getlogin() + "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  if script_dir is None:
    script_dir = os.getcwd()
  if len(bucket_name) > 5 and bucket_name[:5] == "gs://":
    bucket_name = bucket_name[5:]

  print("Running multihost_job with the following configuration:")
  print(f"Project               (--PROJECT)               = {project}")
  print(f"Zone                  (--ZONE)                  = {zone}")
  print(f"TPU type              (--TPU_TYPE)              = {tpu_type}")
  print(f"TPU runtime version   (--VERSION)               = {version}")
  print(f"Number of slices      (--NUM_SLICES)            = {num_slices}")
  print(f"Script dir            (--SCRIPT_DIR)            = {script_dir}")
  print(f"Bucket name           (--BUCKET_NAME)           = {bucket_name}")
  print(f"Bucket dir            (--BUCKET_DIR)            = {bucket_dir}")
  print(f"Run name              (--RUN_NAME)              = {run_name}")
  print(f"Extra CQR args        (--CQR_EXTRA_ARGS)        = {cqr_extra_args}")
  print(f"Command to run        (--COMMAND)               = {command}")
  print(f"Enable Autocheckpoint (--ENABLE_AUTOCHECKPOINT) = {enable_autocheckpoint}\n")

  tmp_dir_relative_to_script = os.path.join("tmp", run_name, "")
  tmp_dir = os.path.join(script_dir, tmp_dir_relative_to_script)
  zip_name = f"script_dir_zip_{run_name}.tar.gz"
  bucket_dir = os.path.join(bucket_dir, run_name)
  bucket_path = f"gs://{bucket_name}/{bucket_dir}"
  startup_script_file = os.path.join(tmp_dir, "startup_script.txt")

  print(f"Moving {script_dir} to {bucket_path}...")
  result = move_script_dir_to_gcs(script_dir, tmp_dir_relative_to_script, zip_name, bucket_path)
  if result.returncode != 0:
    print("\n\n Moving code to GCS failed")
    print(f"Running 'gcloud storage mv zip {bucket_path}' failed with error: ")
    print(result.stderr.decode())
    print("\nYou may need to run 'gcloud auth login'")
    return -1
  print("Move successful!\n")

  log_name = "main_command_log_slice_${SLICE_ID}_worker_${WORKER_ID}"
  write_startup_script(
      os.path.join(bucket_path, zip_name), zip_name, log_name, bucket_path, startup_script_file,
      run_name=run_name, num_slices=num_slices, command=command, project=project, zone=zone,
  )
  print("Running CQR command...")
  result = run_create_resources(
      startup_script_file, run_name=run_name, tpu_type=tpu_type, version=version, project=project, zone=zone,
      num_slices=num_slices, cqr_extra_args=cqr_extra_args, enable_autocheckpoint=enable_autocheckpoint,
  )
  if result.returncode != 0:
    print(f"\n\nCreate resource request returned with ERROR returncode {result.returncode}.\n")
    print("Create resource error:\n" + result.stderr.decode())
    return 1
  print("CQR command received! TPUs are being created.\n")

  shutil.rmtree(tmp_dir)
  print("------------------------------------ \n")
  print("multihost_job finished running, TPUs are firing up now to run your job remotely.\n")
  print(f"Your job is being logged, follow it here:\n{google_cloud_logging_url(run_name, project)}\n")
  print(
      "To see the output of a single host, you may edit the slice and worker number in the log_file_path property here:"
      f"\n{google_cloud_logging_single_host_url(run_name, project)}\n"
  )
  print(f"When your job is finished, the main command log is in the GCS bucket here:"
        f"\n{gcs_bucket_url(bucket_name, bucket_dir, project)}\n")
  print("View the status of the created TPUs via: ")
  print(f"gcloud alpha compute tpus queued-resources list --filter={run_name} --zone={zone} --project={project}\n")
  return 0
