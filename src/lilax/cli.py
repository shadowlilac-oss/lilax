"""Command-line entry point for lilax."""

import argparse
import os
import sys

import yaml

from lilax.launch._command import add_launch_command
from lilax.launch._wizard import add_config_command


def main():
  if len(sys.argv) > 1 and sys.argv[1] == "convert":
    os.execv(sys.executable, [sys.executable, "-m", "lilax.checkpoints.portable_conversion", *sys.argv[2:]])
  parser = argparse.ArgumentParser(prog="lilax")
  commands = parser.add_subparsers(dest="command", required=True)
  add_config_command(commands)
  add_launch_command(commands)
  commands.add_parser("convert", help="convert a weight-only checkpoint")
  args = parser.parse_args()
  try:
    args.handler(args)
  except (OSError, ValueError, yaml.YAMLError) as error:
    parser.error(str(error))
