"""Standalone checkpoint save and restore operations."""

from lilax.checkpointing import add_entropy_to_checkpoint, checkpoint_loop


def run(config):
  return checkpoint_loop(config)
