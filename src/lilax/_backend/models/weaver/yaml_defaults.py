# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loading of isolated Weaver-specific defaults from ``weaver.yml``."""

from __future__ import annotations

import functools
import pathlib
from typing import Any

import yaml


_WEAVER_YAML_PATH = pathlib.Path(__file__).resolve().parents[2] / "configs" / "models" / "weaver.yml"


@functools.lru_cache(maxsize=1)
def _load_weaver_yaml_defaults() -> dict[str, Any]:
  """Loads isolated Weaver-specific defaults from ``src/maxtext/configs/models/weaver.yml``.

  Keeping Weaver-specific diffusion and dual-pathway parameters in ``weaver.yml``
  avoids polluting core MaxText ``base.yml`` and ``types.py`` while providing a
  single source of truth for ``WeaverConfig`` defaults.
  """
  if _WEAVER_YAML_PATH.is_file():
    with _WEAVER_YAML_PATH.open("r", encoding="utf-8") as f:
      loaded = yaml.safe_load(f)
      if isinstance(loaded, dict):
        return loaded
  return {}
