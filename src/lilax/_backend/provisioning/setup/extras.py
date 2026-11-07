"""Install the optional pre-training and TPU post-training dependencies."""

from contextlib import ExitStack
from importlib import resources
import os
from pathlib import Path
import shutil
import subprocess
import sys


def _target_python(python):
  python = Path(python)
  if not python.is_absolute() or not (python.parent.parent / "pyvenv.cfg").is_file():
    raise ValueError("python must be an absolute path to an interpreter in a virtual environment")
  if not python.is_file():
    raise FileNotFoundError(python)
  return str(python)


def _install_uv(python, env):
  subprocess.run([python, "-m", "pip", "install", "uv"], check=True, capture_output=True, env=env)
  subprocess.run([python, "-m", "uv", "--version"], check=True, capture_output=True, env=env)


def _compiler_environment(env):
  if sys.platform != "linux":
    return
  if shutil.which("gcc", path=env.get("PATH")):
    try:
      result = subprocess.run(["gcc", "-dumpversion"], capture_output=True, text=True, check=False, env=env)
      if int(result.stdout.strip().split(".")[0]) >= 11:
        return
    except (OSError, ValueError):
      pass
  for version in ("12", "11"):
    if shutil.which(f"gcc-{version}", path=env.get("PATH")) and shutil.which(f"g++-{version}", path=env.get("PATH")):
      env["CC"] = f"gcc-{version}"
      env["CXX"] = f"g++-{version}"
      return
  raise RuntimeError("A C++20 compatible compiler (GCC/G++ >= 11) is required to build vLLM")


def install_pretrain_extras(python, github_requirements=None, *, with_tf=None, tf_requirements=None):
  """Install pre-training GitHub dependencies and, optionally, TensorFlow dependencies."""
  python = _target_python(python)
  env = os.environ.copy()
  if with_tf is None:
    with_tf = env.get("WITH_TF", "").lower() in ("true", "1", "yes")
  assets = resources.files("lilax._backend").joinpath("assets", "dependencies", "extra_deps")
  with ExitStack() as stack:
    if github_requirements is None:
      github_requirements = stack.enter_context(resources.as_file(assets.joinpath("pre_train_github_deps.txt")))
    github_requirements = Path(github_requirements)
    if not github_requirements.is_file():
      raise FileNotFoundError(github_requirements)
    _install_uv(python, env)
    subprocess.run([python, "-m", "uv", "pip", "install", "--python", python, "-r", str(github_requirements),
                    "--no-deps"], check=True, env=env)
    if with_tf:
      if tf_requirements is None:
        tf_requirements = stack.enter_context(resources.as_file(assets.joinpath("tf_requirements.txt")))
      tf_requirements = Path(tf_requirements)
      if not tf_requirements.is_file():
        raise FileNotFoundError(tf_requirements)
      subprocess.run([python, "-m", "uv", "pip", "install", "--python", python, "-r", str(tf_requirements),
                      "--no-deps"], check=True, env=env)


def install_tpu_posttrain_extras(python, github_requirements=None, vllm_adapter=None):
  """Install TPU post-training GitHub dependencies and the migrated vLLM adapter."""
  python = _target_python(python)
  env = os.environ.copy()
  env["VLLM_TARGET_DEVICE"] = "tpu"
  env["UV_TORCH_BACKEND"] = "cpu"
  env.setdefault("VLLM_VERSION_OVERRIDE", "0.30.0+tpu")
  _compiler_environment(env)
  assets = resources.files("lilax._backend").joinpath("assets")
  with ExitStack() as stack:
    if github_requirements is None:
      github_requirements = stack.enter_context(resources.as_file(
          assets.joinpath("dependencies", "extra_deps", "post_train_github_deps.txt")))
    github_requirements = Path(github_requirements)
    if not github_requirements.is_file():
      raise FileNotFoundError(github_requirements)
    _install_uv(python, env)
    subprocess.run([python, "-m", "uv", "pip", "install", "--python", python, "-r", str(github_requirements),
                    "--no-deps", "--no-build-isolation"], check=True, capture_output=True, text=True, env=env)
    if vllm_adapter is None:
      vllm_adapter = stack.enter_context(resources.as_file(assets.joinpath("integration", "vllm")))
    subprocess.run([python, "-m", "uv", "pip", "install", "--python", python, str(vllm_adapter), "--no-deps"],
                   check=True, capture_output=True, text=True, env=env)
