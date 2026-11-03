"""Explicit host network tuning from the MaxText preflight workflow."""

from pathlib import Path
import re
import shutil
import subprocess


def _apply_rto_settings(prefix):
  print("Running rto_setup.sh")
  print("Adjust Network settings and apply non cache copy")
  subprocess.run([*prefix, "sysctl", "net.ipv4.tcp_slow_start_after_idle=0"], check=True)
  subprocess.run([*prefix, "sysctl", "net.ipv4.tcp_no_metrics_save=1"], check=True)

  routes = subprocess.run([*prefix, "ip", "route", "show"], check=True, capture_output=True, text=True).stdout.splitlines()
  route = routes[0] if routes else ""
  routes = subprocess.run([*prefix, "ip", "route", "show"], check=True, capture_output=True, text=True).stdout.splitlines()
  second_route = routes[1] if len(routes) > 1 else ""
  if second_route.startswith("default"):
    subprocess.run([*prefix, "ip", "route", "delete", *route.replace(" lock", "").split()], check=True)

  routes = subprocess.run([*prefix, "ip", "route", "show"], check=True, capture_output=True, text=True).stdout.splitlines()
  route = routes[0] if routes else ""
  print(f"route rto before change: {route}")
  if "lock" in route and "5ms" in route.split("lock", 1)[1]:
    print(route)
  else:
    subprocess.run([*prefix, "ip", "route", "change", *route.split(), "rto_min", "5ms"], check=True)
  routes = subprocess.run([*prefix, "ip", "route", "show"], check=True, capture_output=True, text=True).stdout.splitlines()
  print(f"route rto after change: {routes[0] if routes else ''}")

  for path, value in (
      ("/sys/module/tcp_cubic/parameters/hystart_detect", "2"),
      ("/proc/sys/net/core/somaxconn", "4096"),
      ("/proc/sys/net/ipv4/tcp_max_syn_backlog", "4096"),
      ("/proc/sys/net/ipv4/tcp_mtu_probing", "0"),
  ):
    if prefix:
      subprocess.run([*prefix, "tee", path], input=f"{value}\n", text=True, stdout=subprocess.DEVNULL, check=True)
    else:
      Path(path).write_text(f"{value}\n", encoding="utf-8")
  subprocess.run([*prefix, "sysctl", "-w", "net.core.optmem_max=131072"], check=True)

  print("\nPrinting output of 'ip route show':")
  subprocess.run([*prefix, "ip", "route", "show"], check=True)
  routes = subprocess.run([*prefix, "ip", "route", "show"], check=True, capture_output=True, text=True).stdout.splitlines()
  fields = re.split(r"\s", routes[0]) if routes else []
  device_name = fields[4] if len(fields) >= 5 else ""
  print(f"dev_name={device_name}")
  subprocess.run([*prefix, "ethtool", "-K", device_name, "tx-nocache-copy", "on"], check=True)
  print("rto_setup.sh finished")


def apply_rto_settings():
  """Apply the RTO settings directly; the caller must have host privileges."""
  _apply_rto_settings([])


def run_preflight():
  """Apply host network settings, using sudo when it is available."""
  sudo = shutil.which("sudo")
  print("Running preflight.sh")
  print("running rto_setup.sh with sudo" if sudo else "running rto_setup.sh without sudo")
  _apply_rto_settings([sudo] if sudo else [])
