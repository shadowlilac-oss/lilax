
"""Proto compilation (PGLE profile + XSpace) into a cached generated-code directory."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

from lilax._backend.utils.gpu_collective_overlap_pass.logger import _logger

# ---------------------------------------------------------------------------
# Proto compilation (PGLE profile + XSpace), all in one directory to avoid
# descriptor pool conflicts. TODO tixxx, remove this once jax exposes
# protobuf utilities.
# ---------------------------------------------------------------------------
_XLA_SRC = os.environ.get("COLLECTIVE_OVERLAP_XLA_PATH", "")
_TSL_SRC = f"{_XLA_SRC}/third_party/tsl"
_PROTO_OUT_DIR_BASE = "/tmp/_gpu_collective_overlap_pass_proto"

_PROTO_SOURCES = [
    "tsl/profiler/protobuf/profiled_instructions.proto",
    "tsl/profiler/protobuf/xplane.proto",
]

_PROTO_INIT_DIRS = [
    "",
    "tsl",
    "tsl/profiler",
    "tsl/profiler/protobuf",
]


def _proto_out_dir() -> str:
  """Generated-code cache dir, keyed by protobuf runtime version and user so a
  directory produced by another container's protoc (or another user) is never
  reused."""
  from google.protobuf import __version__ as pb_version  # pylint: disable=import-outside-toplevel

  return f"{_PROTO_OUT_DIR_BASE}_{pb_version}_{os.getuid()}"


_GENCODE_CHECK_RE = re.compile(
    r"_runtime_version\.ValidateProtobufRuntimeVersion\(\s*_runtime_version\.Domain\.\w+,"
    r"\s*(\d+),\s*(\d+),\s*(\d+),[^)]*\)",
    re.S,
)


def _relax_gencode_version(root: str) -> int:
  """Drop the gencode-vs-runtime version check from generated modules whose
  gencode is newer than the Python protobuf runtime; returns how many files.

  The `protoc` on PATH can be newer than the installed runtime (e.g. gencode
  7.36.2 vs runtime 6.33.6), which makes every import raise VersionError. These
  are plain messages with no newer-protoc-only features, so they load fine on an
  older runtime once the check is gone.
  """
  from google.protobuf import __version__ as pb_version  # pylint: disable=import-outside-toplevel

  runtime = tuple(int(x) for x in re.findall(r"\d+", pb_version)[:3])
  patched = 0
  for dirpath, _, files in os.walk(root):
    for name in files:
      if not name.endswith("_pb2.py"):
        continue
      path = os.path.join(dirpath, name)
      with open(path, encoding="utf-8") as f:
        text = f.read()
      new = _GENCODE_CHECK_RE.sub(
          lambda m: "pass" if tuple(int(m.group(i)) for i in (1, 2, 3)) > runtime else m.group(0),
          text,
      )
      if new != text:
        with open(path, "w", encoding="utf-8") as f:
          f.write(new)
        patched += 1
  return patched


def _ensure_protos():
  """Compile all needed proto files once into a single output directory."""
  out = _proto_out_dir()
  marker = os.path.join(out, "tsl", "profiler", "protobuf", "xplane_pb2.py")
  if not os.path.exists(marker):
    # Generate into a private directory and rename it into place, so ranks
    # racing on the same node never read a half-written tree.
    tmp = f"{out}.tmp.{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    for rel in _PROTO_INIT_DIRS:
      d = os.path.join(tmp, rel)
      os.makedirs(d, exist_ok=True)
      with open(os.path.join(d, "__init__.py"), "a", encoding="utf-8"):
        pass
    try:
      subprocess.run(
          [
              "protoc",
              f"--proto_path={_TSL_SRC}",
              "--proto_path=/usr/local/include",
              f"--python_out={tmp}",
              *_PROTO_SOURCES,
          ],
          check=True,
          capture_output=True,
      )
    except subprocess.CalledProcessError as exc:
      _logger.warning(
          "gpu_collective_overlap_pass: protoc failed: %s",
          exc.stderr.decode("utf-8", errors="replace")[-1000:],
      )
      shutil.rmtree(tmp, ignore_errors=True)
      raise
    patched = _relax_gencode_version(tmp)
    if patched:
      _logger.warning(
          "gpu_collective_overlap_pass: protoc is newer than the Python protobuf runtime; "
          "removed the gencode version check from %d generated module(s).",
          patched,
      )
    try:
      os.rename(tmp, out)
    except OSError:  # another process finished first
      shutil.rmtree(tmp, ignore_errors=True)
  if out not in sys.path:
    sys.path.insert(0, out)
