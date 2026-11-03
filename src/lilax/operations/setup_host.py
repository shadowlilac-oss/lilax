"""Explicit host provisioning and virtual-environment selection for MaxText setup."""

import os
from pathlib import Path
import shutil
import subprocess

from lilax.operations import setup_extras


def select_python(python):
  """Select an existing virtual-environment interpreter running Python 3.12 or newer."""
  python = setup_extras._target_python(python)
  result = subprocess.run(
      [python, "-c", "import sys; print(sys.version_info.major, sys.version_info.minor)"],
      check=True, capture_output=True, text=True,
  )
  major, minor = (int(part) for part in result.stdout.split())
  if (major, minor) < (3, 12):
    raise ValueError(f"MaxText requires Python 3.12 or higher; selected {major}.{minor}")
  return python


def create_python_312_environment(bootstrap_python, environment_path):
  """Create the source's seeded Python 3.12 environment using uv in a selected venv."""
  bootstrap_python = setup_extras._target_python(bootstrap_python)
  environment_path = Path(environment_path).expanduser()
  subprocess.run([bootstrap_python, "-m", "uv", "venv", "--python", "3.12", str(environment_path), "--seed"],
                 check=True)
  with open("/tmp/venv_created", "a", encoding="utf-8") as marker:
    marker.write(f"{environment_path.resolve()}\n")
  return str(environment_path.resolve() / "bin" / "python")


def provision_host():
  """Provision apt packages and GCSFuse on the host; requires appropriate privileges."""
  env = os.environ.copy()
  env.update(DEBIAN_FRONTEND="noninteractive", NEEDRESTART_SUSPEND="1", NEEDRESTART_MODE="l")
  sudo = shutil.which("sudo")
  needrestart = Path("/etc/needrestart/needrestart.conf")
  if sudo and needrestart.is_file():
    subprocess.run([sudo, "sed", "-i", "s/#\\$nrconf{restart} = 'i';/\\$nrconf{restart} = 'a';/",
                    str(needrestart)], check=True, env=env)
  else:
    print("Skipping editing needrestart.conf")
  for command in (["apt-get", "update"], ["apt-get", "install", "-y", "--no-install-recommends", "sudo"]):
    try:
      subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError:
      if not sudo:
        raise
      subprocess.run([sudo, *command], check=True, env=env)
  sudo = shutil.which("sudo")
  prefix = [sudo, "env", "DEBIAN_FRONTEND=noninteractive", "NEEDRESTART_SUSPEND=1", "NEEDRESTART_MODE=l"] if sudo else []
  subprocess.run([*prefix, "apt-get", "update"], check=True, env=env)
  subprocess.run([*prefix, "apt-get", "upgrade", "-y"], check=True, env=env)
  subprocess.run([*prefix, "apt-get", "install", "-y", "--no-install-recommends", "numactl", "lsb-release", "gnupg",
                  "curl", "net-tools", "iproute2", "procps", "lsof", "git", "ethtool", "ca-certificates"],
                 check=True, env=env)
  release = subprocess.run(["lsb_release", "-c", "-s"], check=True, capture_output=True, text=True, env=env).stdout.strip()
  repo = "gcsfuse-bookworm" if release == "trixie" else f"gcsfuse-{release}"
  key_path = Path("/tmp/apt-key.gpg")
  with key_path.open("wb") as key_file:
    subprocess.run(["curl", "-fsSL", "https://packages.cloud.google.com/apt/doc/apt-key.gpg"],
                   check=True, stdout=key_file, env=env)
  subprocess.run([*prefix, "gpg", "--dearmor", "--yes", "-o", "/usr/share/keyrings/cloud.google.gpg", str(key_path)],
                 check=True, env=env)
  key_path.unlink()
  repo_line = f"deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt {repo} main\n"
  subprocess.run([*prefix, "tee", "/etc/apt/sources.list.d/gcsfuse.list"], input=repo_line, text=True, check=True, env=env)
  subprocess.run([*prefix, "apt-get", "update", "-y"], check=True, env=env)
  subprocess.run([*prefix, "apt-get", "-y", "install", "gcsfuse"], check=True, env=env)
  subprocess.run([*prefix, "rm", "-f", "/usr/sbin/mount.gcsfuse", "/usr/sbin/mount.fuse.gcsfuse",
                  "/sbin/mount.gcsfuse", "/sbin/mount.fuse.gcsfuse"], check=True, env=env)
  try:
    subprocess.run([*prefix, "objcopy", "--remove-section=.go.buildinfo", "/usr/bin/gcsfuse"],
                   check=False, stderr=subprocess.DEVNULL, env=env)
  except FileNotFoundError:
    pass
  apt_lists = Path("/var/lib/apt/lists")
  paths = [str(path) for path in apt_lists.glob("*") if not path.name.startswith(".")]
  if paths:
    subprocess.run([*prefix, "rm", "-rf", *paths], check=True, env=env)
