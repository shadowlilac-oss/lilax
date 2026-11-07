"""DCN traffic control bandwidth throttling."""

import subprocess

from lilax._backend.observability import max_logging


def maybe_apply_dcn_throttling(config):
  """Apply a configured traffic control bandwidth limit."""
  if not config.dcn_bandwidth_limit:
    return
  interface = config.dcn_bandwidth_interface
  try:
    subprocess.run(
        ["tc", "qdisc", "del", "dev", interface, "root"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
  except Exception as error:
    max_logging.error(f"Failed to clean up existing traffic control on {interface}: {error}")
  rate = config.dcn_bandwidth_limit
  burst = config.dcn_bandwidth_burst
  latency = config.dcn_bandwidth_latency
  max_logging.log(f"Applying tc egress limit of {rate} (burst: {burst}, latency: {latency}) on {interface}...")
  try:
    cmd = ["tc", "qdisc", "add", "dev", interface, "root", "tbf", "rate", rate, "burst", burst, "latency", latency]
    subprocess.run(cmd, check=True)
    max_logging.log("DCN Bandwidth throttling applied successfully.")
  except Exception as error:
    max_logging.error(f"Failed to apply DCN bandwidth throttling: {error}")


def maybe_cleanup_dcn_throttling(config):
  """Clean up traffic control rules."""
  if not config.dcn_bandwidth_limit:
    return
  interface = config.dcn_bandwidth_interface
  max_logging.log(f"Cleaning up tc egress limit on {interface}...")
  try:
    subprocess.run(
        ["tc", "qdisc", "del", "dev", interface, "root"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    max_logging.log("DCN Bandwidth throttling cleaned up successfully.")
  except Exception as error:
    max_logging.error(f"Failed to clean up DCN bandwidth throttling: {error}")
