"""Parsing of LIBTPU/XLA flag strings into compiler options."""

import re


def parse_libtpu_flags_to_dict(flags_str: str) -> dict:
  """Parses a string of XLA flags into a dictionary of compilation options.
  This function is only for compilation usage.
  """
  if not flags_str or not flags_str.strip():
    return {}
  cleaned_str = flags_str.replace("\\", " ")
  tokens = cleaned_str.split()
  options_dict = {}
  token_pattern = re.compile(r"^--([a-zA-Z0-9_]+)=(.+)$")
  for token in tokens:
    match = token_pattern.match(token)
    if not match:
      raise ValueError(f"Invalid flag format detected: '{token}'. Expected format: '--key=value'")
    key, value = match.groups()
    if key in options_dict:
      raise ValueError(f"Duplicate flag detected: '--{key}'")
    options_dict[key] = value
  return options_dict
