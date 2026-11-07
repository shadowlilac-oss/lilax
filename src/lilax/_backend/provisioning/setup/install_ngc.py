"""Install the NGC profile while preserving the image's JAX and TE artifacts."""

import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import tempfile
import tomllib

from packaging.tags import sys_tags
from packaging.requirements import Requirement
from packaging.utils import parse_wheel_filename
from packaging.version import Version


PROTECTED = {
    "jax": "0.11.2.dev20260904+1b907ec45a",
    "jaxlib": "0.11.2.dev20261013",
    "jax-cuda13-plugin": "0.11.2.dev20261013",
    "jax-cuda13-pjrt": "0.11.2.dev20261013",
    "transformer-engine": "2.20.2+6ea2a74a9",
    "nvidia-cutlass-dsl-libs-cu12": "4.8.0",
}
MODULES = ("jax", "jaxlib", "jax_cuda13_plugin", "jax_plugins.xla_cuda13", "transformer_engine",
           "transformer_engine.jax")
ROOTS = (Path("/opt/jax"), Path("/opt/xla"), Path("/opt/jaxlibs"), Path("/opt/transformer-engine"))
CUDA_LINK = Path("/usr/local/cuda")
CUDA_ROOT = Path("/usr/local/cuda-13.4")
NATIVE_FILES = (
    "/usr/local/cuda/lib64/libcudart.so",
    "/usr/local/cuda/lib64/libcudart.so.13",
    "/usr/local/cuda/lib64/libcudart.so.13.4.92",
    "/usr/lib/x86_64-linux-gnu/libcudnn.so",
    "/usr/lib/x86_64-linux-gnu/libcudnn.so.9",
    "/usr/lib/x86_64-linux-gnu/libcudnn.so.9.27.0",
    "/opt/jaxlibs/jaxlib/jaxlib/_jax.so",
    "/opt/jaxlibs/jaxlib/jaxlib/libjax_common.so",
    "/opt/jaxlibs/jax_cuda13_plugin/jax_cuda13_plugin/cuda_plugin_extension.so",
    "/opt/jaxlibs/jax_cuda13_pjrt/jax_plugins/xla_cuda13/xla_cuda_plugin.so",
    "/usr/local/lib/python3.12/dist-packages/transformer_engine/transformer_engine_jax.cpython-312-x86_64-linux-gnu.so",
    "/usr/local/lib/python3.12/dist-packages/transformer_engine/libtransformer_engine.so",
    "/usr/local/lib/python3.12/dist-packages/transformer_engine/libnccl_ep.so",
)
FORBIDDEN = {"jax-cuda12-plugin", "jax-cuda12-pjrt", "transformer-engine-jax", "transformer-engine-cu12",
             "tensorflow", "nvidia-cudnn-cu13"}
TF_CHAIN = {"tensorflow-text", "seqio", "google-jetstream"}
TUNIX_URL = "https://github.com/google/tunix/archive/385551df4643ee3ac3205e813a93c2ff217dffa5.zip"
TORCH_URL = ("https://download-r2.pytorch.org/whl/cu134/"
             "torch-2.15.0%2Bcu134-cp312-cp312-manylinux_2_28_x86_64.whl")


def _file_state(path):
  info = path.lstat()
  mode = stat.S_IMODE(info.st_mode)
  if path.is_symlink():
    target = os.readlink(path)
    resolved = path.resolve(strict=True)
    if resolved.is_file():
      return {"mode": mode, "link": target, "target": _file_state(resolved)}
    raise RuntimeError(f"Cannot protect directory symlink: {path}")
  if not path.is_file():
    raise RuntimeError(f"Cannot protect non-file: {path}")
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return {"mode": mode, "sha256": digest.hexdigest()}


def _snapshot():
  if not CUDA_LINK.is_symlink() or CUDA_LINK.resolve(strict=True) != CUDA_ROOT:
    raise RuntimeError(f"Expected {CUDA_LINK} to resolve to {CUDA_ROOT}")
  cuda_link = {"target": os.readlink(CUDA_LINK), "mode": stat.S_IMODE(CUDA_LINK.lstat().st_mode)}
  distributions = {}
  files = {Path(path) for path in NATIVE_FILES}
  for name, expected_version in PROTECTED.items():
    matches = [dist for dist in metadata.distributions() if
               dist.metadata["Name"].lower().replace("_", "-") == name]
    if len(matches) != 1 or matches[0].version != expected_version:
      raise RuntimeError(f"Expected exactly one {name}=={expected_version}")
    dist = matches[0]
    if dist.files is None:
      raise RuntimeError(f"No installed file record for {name}")
    distributions[name] = {
        "version": dist.version,
        "direct_url": dist.read_text("direct_url.json"),
        "metadata": dist.read_text("METADATA"),
        "entry_points": dist.read_text("entry_points.txt"),
    }
    files.update(Path(dist.locate_file(path)).absolute() for path in dist.files)

  installed = {dist.metadata["Name"].lower().replace("_", "-") for dist in metadata.distributions()}
  if installed & FORBIDDEN:
    raise RuntimeError(f"Prohibited distributions present: {sorted(installed & FORBIDDEN)}")

  for root in ROOTS:
    if not root.is_dir() or root.is_symlink():
      raise RuntimeError(f"Missing or symlinked protected root: {root}")
    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
      for name in subdirectories:
        path = Path(directory, name)
        if path.is_symlink():
          files.add(path)
      files.update(Path(directory, name) for name in filenames)

  modules = {}
  for name in MODULES:
    spec = util.find_spec(name)
    if spec is None:
      raise RuntimeError(f"Protected module is not importable: {name}")
    modules[name] = {"origin": spec.origin, "locations": list(spec.submodule_search_locations or [])}

  return {"flax_version": metadata.version("flax"), "cuda_link": cuda_link,
          "distributions": distributions, "modules": modules,
          "files": {str(path): _file_state(path) for path in sorted(files)}}


def _read_snapshot(env):
  result = subprocess.run([sys.executable, __file__, "--snapshot"], env=env, check=True, capture_output=True, text=True)
  return json.loads(result.stdout)


def _grain_snapshot(wheel):
  matches = [dist for dist in metadata.distributions() if dist.metadata["Name"].lower().replace("_", "-") == "grain"]
  if len(matches) != 1 or matches[0].version != "0.2.19" or matches[0].files is None:
    raise RuntimeError("Expected exactly one recorded grain==0.2.19 installation")
  dist = matches[0]
  direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
  expected_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
  if direct_url.get("url") != wheel.as_uri() or direct_url.get("archive_info", {}).get("hashes", {}).get("sha256") != expected_hash:
    raise RuntimeError("Installed Grain does not originate from the built wheel")
  paths = {str(path): Path(dist.locate_file(path)).absolute() for path in dist.files}
  required = {"grain/proto/execution_summary_pb2.py", "grain/_src/python/experimental/index_shuffle/libindex_shuffle.so",
              "grain/_src/python/experimental/index_shuffle/python/index_shuffle_module.so"}
  if not required <= paths.keys():
    raise RuntimeError("Built Grain installation is missing generated artifacts")
  spec = util.find_spec("grain")
  if spec is None or spec.origin != str(paths.get("grain/__init__.py")):
    raise RuntimeError("Grain import does not resolve to the built wheel")
  return {"metadata": dist.read_text("METADATA"), "record": dist.read_text("RECORD"), "direct_url": direct_url,
          "origin": spec.origin, "locations": list(spec.submodule_search_locations or []),
          "files": {name: _file_state(path) for name, path in sorted(paths.items())}}


def _install(arguments, report, env):
  subprocess.run([sys.executable, "-m", "pip", "install", "--report", str(report), *arguments], env=env, check=True)
  installed = json.loads(report.read_text())["install"]
  replaced = {item["metadata"]["name"].lower().replace("_", "-") for item in installed} & PROTECTED.keys()
  if replaced:
    raise RuntimeError(f"The installation replaced protected distributions: {sorted(replaced)}")
  forbidden = {item["metadata"]["name"].lower().replace("_", "-") for item in installed} & FORBIDDEN
  if forbidden:
    raise RuntimeError(f"The installation selected prohibited distributions: {sorted(forbidden)}")
  return installed


def _install_image_dependencies(source, report, env):
  project = tomllib.loads((source / "pyproject.toml").read_text())
  files = project["tool"]["hatch"]["metadata"]["hooks"]["requirements_txt"]["optional-dependencies"]["ngc-image"]
  ordinary, chain, tensorflow = list(project["project"]["dependencies"]), [], []
  for filename in files:
    for line in (source / filename).read_text().splitlines():
      line = line.strip()
      if not line or line.startswith("#"):
        continue
      requirement = Requirement(line)
      name = requirement.name.lower().replace("_", "-")
      if name == "tensorflow":
        tensorflow.append(requirement)
      elif name in TF_CHAIN:
        chain.append(line)
      else:
        ordinary.append(line)
  if len(tensorflow) != 1 or str(tensorflow[0].specifier) != "==2.20.0":
    raise RuntimeError("Expected the pinned TensorFlow 2.20.0 requirement")
  if {Requirement(item).name.lower().replace("_", "-") for item in chain} != TF_CHAIN:
    raise RuntimeError("Expected the pinned TensorFlow Text, SeqIO and JetStream requirements")

  _install(["--no-deps", f"{source}[ngc-image]", *chain], report, env)
  for name in TF_CHAIN:
    for item in metadata.distribution(name).requires or []:
      requirement = Requirement(item)
      dependency = requirement.name.lower().replace("_", "-")
      if dependency not in TF_CHAIN | {"tensorflow"} and (
          requirement.marker is None or requirement.marker.evaluate({"extra": ""})):
        ordinary.append(item)
  ordinary.extend(["tensorflow-cpu==2.20.0", "flax==0.12.12", "nvidia-nvshmem-cu13==3.7.2"])
  constraints = report.with_name("constraints.txt")
  constraints.write_text("".join(f"{name}<0\n" for name in sorted(FORBIDDEN)) +
                         "".join(f"{name}=={version}\n" for name, version in sorted(PROTECTED.items())))
  arguments = ["--constraint", str(constraints), *ordinary]
  _install(["--dry-run", *arguments], report, env)
  _install(arguments, report, env)

  for name in TF_CHAIN:
    for item in metadata.distribution(name).requires or []:
      requirement = Requirement(item)
      if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
        continue
      dependency = requirement.name.lower().replace("_", "-")
      if dependency == "tensorflow":
        dependency = "tensorflow-cpu"
      version = metadata.version(dependency)
      if not requirement.specifier.contains(version, prereleases=True):
        raise RuntimeError(f"Installed {dependency}=={version} does not satisfy {requirement}")


def _torch_snapshot():
  matches = [dist for dist in metadata.distributions() if dist.metadata["Name"].lower().replace("_", "-") == "torch"]
  if len(matches) != 1 or matches[0].files is None:
    raise RuntimeError("Expected exactly one recorded GPU Torch installation")
  dist = matches[0]
  if Version(dist.version) != Version("2.15.0+cu134"):
    raise RuntimeError(f"Expected torch==2.15.0+cu134, got {dist.version}")
  direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
  if direct_url.get("url") != TORCH_URL:
    raise RuntimeError("Installed Torch does not originate from the selected official wheel")
  paths = {str(path): Path(dist.locate_file(path)).absolute() for path in dist.files}
  spec = util.find_spec("torch")
  if spec is None or spec.origin != str(paths.get("torch/__init__.py")):
    raise RuntimeError("Torch import does not resolve to the installed distribution")
  return {"version": dist.version, "metadata": dist.read_text("METADATA"), "record": dist.read_text("RECORD"),
          "direct_url": dist.read_text("direct_url.json"), "origin": spec.origin,
          "locations": list(spec.submodule_search_locations or []),
          "files": {name: _file_state(path) for name, path in sorted(paths.items())}}


def _verify_tunix():
  dist = metadata.distribution("google-tunix")
  direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
  if direct_url.get("url") != TUNIX_URL:
    raise RuntimeError("Installed Tunix does not originate from the pinned source")
  jax_requirement = None
  for value in dist.requires or []:
    requirement = Requirement(value)
    if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
      continue
    if requirement.name.lower().replace("_", "-") == "jax":
      jax_requirement = requirement
      continue
    installed = metadata.version(requirement.name)
    if not requirement.specifier.contains(installed, prereleases=True):
      raise RuntimeError(f"Installed {requirement.name}=={installed} does not satisfy Tunix requirement {requirement}")
  if jax_requirement is None or jax_requirement.specifier.contains(metadata.version("jax"), prereleases=True):
    raise RuntimeError("Expected the pinned Tunix JAX metadata incompatibility")
  print(f"Pinned Tunix declares {jax_requirement}; installed JAX is {metadata.version('jax')}", file=sys.stderr)


def main():
  if sys.argv[1:] == ["--snapshot"]:
    print(json.dumps(_snapshot(), sort_keys=True))
    return
  if len(sys.argv) != 3:
    raise SystemExit("Usage: python install_ngc.py /path/to/lilax /path/to/grain-wheels")
  source = Path(sys.argv[1]).resolve(strict=True)
  if not (source / "pyproject.toml").is_file():
    raise SystemExit(f"Not a project directory: {source}")
  if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12) or platform.machine() != "x86_64":
    raise RuntimeError("The NGC Grain wheel requires CPython 3.12 on x86_64")
  wheels = list(Path(sys.argv[2]).resolve(strict=True).glob("grain-*.whl"))
  if len(wheels) != 1:
    raise RuntimeError("Expected exactly one built Grain wheel")
  wheel = wheels[0].resolve(strict=True)
  name, version, _, tags = parse_wheel_filename(wheel.name)
  if name != "grain" or str(version) != "0.2.19" or not tags.intersection(sys_tags()):
    raise RuntimeError(f"Incompatible Grain wheel: {wheel.name}")
  env = os.environ.copy()
  env["PYTHONDONTWRITEBYTECODE"] = "1"
  before = _read_snapshot(env)
  with tempfile.TemporaryDirectory() as temporary:
    report = Path(temporary, "report.json")
    installed = _install([f"{wheel}[parquet]"], report, env)
    if not any(item["metadata"]["name"].lower() == "grain" for item in installed):
      raise RuntimeError("Pip did not install the built Grain wheel")
    if _read_snapshot(env) != before:
      raise RuntimeError("Installing Grain changed protected image artifacts or import origins")
    grain_before = _grain_snapshot(wheel)
    _install_image_dependencies(source, report, env)
    if _grain_snapshot(wheel) != grain_before:
      raise RuntimeError("The installation changed the built Grain wheel or its import origin")
    _install(["--no-deps", f"torch @ {TORCH_URL}"], report, env)
    torch_before = _torch_snapshot()
    installed = _install(["--no-deps", f"google-tunix @ {TUNIX_URL}"], report, env)
    if len(installed) != 1 or installed[0]["metadata"]["name"].lower().replace("_", "-") != "google-tunix":
      raise RuntimeError("Pip did not install only the pinned Tunix distribution")
    _verify_tunix()
    if _grain_snapshot(wheel) != grain_before or _torch_snapshot() != torch_before:
      raise RuntimeError("Installing Tunix changed Grain or Torch artifacts or import origins")
    after = _read_snapshot(env)
  before.pop("flax_version")
  if after.pop("flax_version") != "0.12.12":
    raise RuntimeError("The installation did not install flax==0.12.12")
  if before != after:
    raise RuntimeError("The installation changed protected image artifacts or import origins")


if __name__ == "__main__":
  main()
