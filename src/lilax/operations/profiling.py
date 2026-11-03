"""Native profiling operations."""

import contextlib
import os
from pathlib import Path
import tempfile

from etils import epath

from lilax._backend.common.profiler import Profiler, nnx_named_scope_context


@contextlib.contextmanager
def stage_xprof_logdir(logdir, step):
  """Yield a local copy of a GCS logdir, or pass a local logdir through.

  The caller must consume or persist the files before leaving the context.
  The original logdir and stop step should be retained as artifact metadata.
  """
  if not os.fspath(logdir).startswith("gs://"):
    yield logdir
    return

  source = epath.Path(logdir)
  temporary_root = Path.home() / "tmp" / "lilax"
  temporary_root.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(prefix=f"xprof-step-{step}-", dir=temporary_root) as temporary_dir:
    local_root = Path(temporary_dir)
    directories = [source]
    while directories:
      directory = directories.pop()
      local_directory = local_root / str(directory.relative_to(source))
      local_directory.mkdir(parents=True, exist_ok=True)
      for entry in directory.iterdir():
        if entry.is_dir():
          directories.append(entry)
        else:
          entry.copy(local_directory / entry.name)
    yield temporary_dir
