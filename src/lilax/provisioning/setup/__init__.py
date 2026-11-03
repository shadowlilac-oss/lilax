"""Install MaxText source dependencies into an explicitly selected virtual environment."""

from contextlib import ExitStack
from importlib import resources
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import warnings

from lilax.provisioning.setup import extras, host


_JAX_INDEX = "https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/"
_LIBTPU_INDEX = "https://storage.googleapis.com/jax-releases/libtpu_releases.html"


def _run(command, *, env=None, check=True):
  return subprocess.run(command, check=check, env=env)


def _pip(python, *arguments, env=None, check=True):
  return _run([python, "-m", "uv", "pip", *arguments, "--python", python], env=env, check=check)


def _cleanup(python, *, allow_system_changes=False):
  result = subprocess.run(
      [python, "-c", "import json, sysconfig; print(json.dumps([sysconfig.get_path('purelib'), "
       "sysconfig.get_path('platlib')]))"], check=True, capture_output=True, text=True)
  for directory in set(json.loads(result.stdout)) - {None, ""}:
    site = Path(directory)
    shutil.rmtree(site / "flaxlib_src", ignore_errors=True)
    (site / "ray/jars/ray_dist.jar").unlink(missing_ok=True)
    for path in (site / "ray/_private/runtime_env/agent/thirdparty_files").glob("*.dist-info"):
      shutil.rmtree(path, ignore_errors=True)
  if not allow_system_changes:
    return
  for root in (Path("/usr/local"), Path("/root")):
    for path in root.glob("**/virtualenv/seed/wheels/embed/*.whl"):
      path.unlink(missing_ok=True)
  for path in ("/usr/sbin/mount.gcsfuse", "/usr/sbin/mount.fuse.gcsfuse", "/sbin/mount.gcsfuse",
               "/sbin/mount.fuse.gcsfuse", "/usr/bin/gcloud-crc32c",
               "/usr/lib/google-cloud-sdk/bin/gcloud-crc32c", "/usr/local/google-cloud-sdk/bin/gcloud-crc32c"):
    Path(path).unlink(missing_ok=True)
  try:
    _run(["objcopy", "--remove-section=.go.buildinfo", "/usr/bin/gcsfuse"], check=False)
  except FileNotFoundError:
    pass
  for root in (Path("/usr/lib/google-cloud-sdk"), Path("/usr/local/google-cloud-sdk")):
    for path in root.glob("platform/bundledpythonunix/lib/python*/site-packages/*.dist-info"):
      shutil.rmtree(path, ignore_errors=True)


def _custom_libtpu(python, gcs_path):
  warnings.warn("Installing a custom libtpu may be incompatible with pinned dependencies", stacklevel=2)
  destination = Path.home() / "custom_libtpu/libtpu.so"
  destination.unlink(missing_ok=True)
  _pip(python, "install", "-U", "crcmod")
  _run(["gcloud", "storage", "cp", gcs_path, str(destination)])


def _system_run(command):
  try:
    _run(command)
  except subprocess.CalledProcessError:
    sudo = shutil.which("sudo")
    if not sudo:
      raise
    _run([sudo, *command])


def _posttrain_toolchain():
  release = Path("/etc/os-release")
  if release.is_file() and "bullseye" in release.read_text():
    source = "/etc/apt/sources.list.d/bookworm.list"
    content = "deb http://deb.debian.org/debian bookworm main\n"
    try:
      Path(source).write_text(content)
    except PermissionError:
      sudo = shutil.which("sudo")
      if not sudo:
        raise
      subprocess.run([sudo, "tee", source], input=content, text=True, check=True)
    _system_run(["apt-get", "update", "-y"])
    _system_run(["apt-get", "install", "-y", "--no-install-recommends", "-t", "bookworm", "gcc-12", "g++-12",
                 "build-essential", "cmake", "ninja-build"])
    return {"CC": "gcc-12", "CXX": "g++-12"}
  try:
    _system_run(["apt-get", "update", "-y"])
    _system_run(["apt-get", "install", "-y", "--no-install-recommends", "gcc-12", "g++-12", "build-essential",
                 "cmake", "ninja-build"])
  except subprocess.CalledProcessError:
    _system_run(["apt-get", "install", "-y", "--no-install-recommends", "build-essential", "cmake", "ninja-build"])
  return {}


def _install(python, source, device, mode, workflow, with_tf, jax_version, libtpu_version, libtpu_gcs_path,
             vllm_adapter, allow_system_changes, grain_wheels):
  if device == "gpu":
    if mode != "stable" or workflow != "pre-training":
      raise ValueError("GPU NGC setup supports only stable mode and the pre-training workflow")
    if jax_version or libtpu_version or libtpu_gcs_path or vllm_adapter or with_tf is False or allow_system_changes:
      raise ValueError("GPU NGC setup does not support custom JAX/TPU, vLLM, TensorFlow exclusion, or system changes")
    if grain_wheels is None:
      raise ValueError("GPU NGC setup requires grain_wheels pointing to the built Grain wheel directory")
    installer = resources.files("lilax.provisioning.setup").joinpath("install_ngc.py")
    with resources.as_file(installer) as path:
      _run([python, str(path), str(source), str(grain_wheels)])
    return
  _run([python, "-m", "pip", "install", "--no-cache-dir", "-U", "pip", "setuptools", "wheel", "uv"])
  if workflow == "post-training" and device != "tpu":
    raise ValueError("post-training requires device='tpu'")
  if workflow == "pre-training" and mode == "libtpu-only":
    if device != "tpu" or not libtpu_gcs_path:
      raise ValueError("libtpu-only requires device='tpu' and libtpu_gcs_path")
    _custom_libtpu(python, libtpu_gcs_path)
    return
  if device not in ("tpu", "gpu"):
    raise ValueError("device must be 'tpu' or 'gpu'")
  if workflow == "pre-training" and mode not in ("stable", "nightly"):
    raise ValueError("mode must be 'stable', 'nightly', or 'libtpu-only'")

  assets = resources.files("lilax._backend").joinpath("assets", "dependencies")
  with ExitStack() as stack:
    def asset(*parts):
      return stack.enter_context(resources.as_file(assets.joinpath(*parts)))

    if workflow == "post-training":
      compiler_env = _posttrain_toolchain() if allow_system_changes else {}
      original_env = {key: os.environ.get(key) for key in compiler_env}
      os.environ.update(compiler_env)
      try:
        env = os.environ.copy()
        env["UV_TORCH_BACKEND"] = "cpu"
        requirements = asset("requirements", "generated_requirements", "tpu-post-train-requirements.txt")
        _pip(python, "install", "--resolution=lowest", "-r", str(requirements), env=env)
        github = asset("extra_deps", "post_train_github_deps.txt")
        extras.install_tpu_posttrain_extras(python, github, vllm_adapter)
        _pip(python, "install", "--no-deps", "setuptools>=84.0.0")
        _cleanup(python, allow_system_changes=allow_system_changes)
      finally:
        for key, value in original_env.items():
          if value is None:
            os.environ.pop(key, None)
          else:
            os.environ[key] = value
      if (source / "pyproject.toml").is_file():
        _pip(python, "install", "--no-deps", "-e", str(source))
      return

    requirements = asset("requirements", "generated_requirements", "tpu-requirements.txt")
    _pip(python, "install", "--resolution=lowest", "-r", str(requirements))
    github = asset("extra_deps", "pre_train_github_deps.txt")
    install_tf = (
        os.environ.get("TF", os.environ.get("tf", "false")) == "true"
        or os.environ.get("WITH_TF", "").lower() in ("true", "1", "yes")
    ) if with_tf is None else with_tf
    tf = asset("extra_deps", "tf_requirements.txt") if install_tf else None
    extras.install_pretrain_extras(python, github, with_tf=install_tf, tf_requirements=tf)
    if (source / "pyproject.toml").is_file():
      _pip(python, "install", "--no-deps", "-e", str(source))
    _cleanup(python, allow_system_changes=allow_system_changes)

  if mode == "nightly":
    _pip(python, "install", "-U", "--no-deps", "git+https://github.com/google/flax.git")
    _cleanup(python, allow_system_changes=allow_system_changes)
    for package in ("jax", "jaxlib", "libtpu"):
      if _pip(python, "show", package, check=False).returncode == 0:
        _pip(python, "uninstall", package)
    packages = ["jax", "jaxlib"]
    if jax_version:
      packages = [f"{package}=={jax_version}" for package in packages]
    _pip(python, "install", "-U", "--pre", "--no-deps", *packages, "-i", _JAX_INDEX)
  elif jax_version:
    warnings.warn("Installing a custom jax version may be incompatible with pinned dependencies", stacklevel=2)
    _pip(python, "install", "-U", f"jax[tpu]=={jax_version}")

  if device == "tpu":
    if libtpu_gcs_path:
      _custom_libtpu(python, libtpu_gcs_path)
    elif libtpu_version:
      warnings.warn("Installing a custom libtpu version may be incompatible with pinned dependencies", stacklevel=2)
      _pip(python, "install", "-U", "--no-deps", f"libtpu=={libtpu_version}", "-f", _LIBTPU_INDEX)
    elif mode == "nightly":
      _pip(python, "install", "-U", "--pre", "--no-deps", "libtpu", "-f", _LIBTPU_INDEX)


def install_overlay(python, source_project, *, workflow="pre-training", vllm_adapter=None):
  """Install editable code and optional adapter metadata without changing image files."""
  python = extras._target_python(python)
  source = Path(source_project)
  if not source.is_absolute() or not source.is_dir():
    raise ValueError("source_project must be an absolute path to an existing source directory")
  if workflow not in ("pre-training", "post-training"):
    raise ValueError("workflow must be 'pre-training' or 'post-training'")
  if (source / "pyproject.toml").is_file():
    _pip(python, "install", "--no-deps", "-e", str(source))
  if workflow == "post-training":
    adapter = vllm_adapter or source / "src/lilax/_backend/assets/integration/vllm"
    _pip(python, "install", "--no-deps", str(adapter))


def setup(python, source_project, *, device=None, mode=None, workflow=None, with_tf=None,
          jax_version=None, libtpu_version=None, libtpu_gcs_path=None, vllm_adapter=None,
          max_attempts=1, retry_delay=5, allow_system_changes=False, grain_wheels=None):
  """Install the selected workflow into a venv; system changes require explicit opt-in."""
  python = host.select_python(python)
  source = Path(source_project)
  if not source.is_absolute() or not source.is_dir():
    raise ValueError("source_project must be an absolute path to an existing source directory")
  device = device or os.environ.get("DEVICE") or "tpu"
  mode = mode or os.environ.get("MODE") or "stable"
  workflow = workflow or os.environ.get("WORKFLOW") or "pre-training"
  jax_version = jax_version if jax_version is not None else os.environ.get("JAX_VERSION")
  libtpu_version = libtpu_version if libtpu_version is not None else os.environ.get("LIBTPU_VERSION")
  libtpu_gcs_path = libtpu_gcs_path if libtpu_gcs_path is not None else os.environ.get("LIBTPU_GCS_PATH")
  if workflow not in ("pre-training", "post-training"):
    raise ValueError("workflow must be 'pre-training' or 'post-training'")
  if max_attempts < 1:
    raise ValueError("max_attempts must be at least 1")
  jax_version = None if jax_version == "NONE" else jax_version
  libtpu_version = None if libtpu_version == "NONE" else libtpu_version
  libtpu_gcs_path = None if libtpu_gcs_path == "NONE" else libtpu_gcs_path
  for attempt in range(max_attempts):
    try:
      _install(python, source, device, mode, workflow, with_tf, jax_version, libtpu_version, libtpu_gcs_path,
               vllm_adapter, allow_system_changes, grain_wheels)
      return
    except subprocess.CalledProcessError:
      if attempt + 1 == max_attempts:
        raise
      time.sleep(retry_delay)
