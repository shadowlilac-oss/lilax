"""Run a command on a GPU node with the source GPUDirect configuration."""

import glob
import os
import socket
import subprocess
import time


def run_gpu_multi_process(command):
  """Run one node's command using the distributed settings in the environment."""
  env = os.environ.copy()
  for name in ("NNODES", "NODE_RANK", "JAX_COORDINATOR_PORT", "JAX_COORDINATOR_ADDRESS", "GPUS_PER_NODE"):
    if not env.get(name):
      raise ValueError(f"{name} must be set")

  direct = env["USE_GPUDIRECT"]
  if direct in ("tcpx", "fastrak"):
    env.update({
        "CUDA_DEVICE_MAX_CONNECTIONS": "1", "NCCL_CROSS_NIC": "0", "NCCL_DEBUG": "INFO",
        "NCCL_DYNAMIC_CHUNK_SIZE": "524288", "NCCL_NET_GDR_LEVEL": "PIX", "NCCL_NVLS_ENABLE": "0",
        "NCCL_P2P_NET_CHUNKSIZE": "524288", "NCCL_P2P_NVL_CHUNKSIZE": "1048576",
        "NCCL_P2P_PCI_CHUNKSIZE": "524288", "NCCL_PROTO": "Simple", "NCCL_SOCKET_IFNAME": "eth0",
        "NVTE_FUSED_ATTN": "1", "TF_CPP_MAX_LOG_LEVEL": "100",
        "TF_CPP_VMODULE": "profile_guided_latency_estimator=10", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85",
    })
    compat = sorted(glob.glob("/usr/local/cuda-*/compat"))
    compat_path = ":" + ":".join(compat) if compat else ""
    env["LD_LIBRARY_PATH"] = f"{compat_path}:{env['LD_LIBRARY_PATH']}:/usr/local/tcpx/lib64"
    if direct == "tcpx":
      env.update({
          "NCCL_ALGO": "Ring", "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,ENV,TUNING,NET,VERSION",
          "NCCL_GPUDIRECTTCPX_CTRL_DEV": "eth0", "NCCL_GPUDIRECTTCPX_FORCE_ACK": "0",
          "NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS": "1000000",
          "NCCL_GPUDIRECTTCPX_RX_BINDINGS": "eth1:22-35,124-139;eth2:22-35,124-139;eth3:74-87,178-191;eth4:74-87,178-191",
          "NCCL_GPUDIRECTTCPX_SOCKET_IFNAME": "eth1,eth2,eth3,eth4",
          "NCCL_GPUDIRECTTCPX_TX_BINDINGS": "eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177",
          "NCCL_GPUDIRECTTCPX_TX_COMPLETION_NANOSLEEP": "1000", "NCCL_MAX_NCHANNELS": "12",
          "NCCL_MIN_NCHANNELS": "12", "NCCL_NSOCKS_PERTHREAD": "4", "NCCL_P2P_PXN_LEVEL": "0",
          "NCCL_SOCKET_NTHREADS": "1",
      })
    else:
      env.update({
          "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "NCCL_ALGO": "Ring,Tree", "NCCL_BUFFSIZE": "8388608",
          "NCCL_FASTRAK_CTRL_DEV": "eth0", "NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL": "0",
          "NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING": "0", "NCCL_FASTRAK_IFNAME": "eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8",
          "NCCL_FASTRAK_NUM_FLOWS": "2", "NCCL_FASTRAK_USE_LLCM": "1", "NCCL_FASTRAK_USE_SNAP": "1",
          "NCCL_MIN_NCHANNELS": "4", "NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE": "/usr/local/nvidia/lib64/a3plus_guest_config.textproto",
          "NCCL_TUNER_CONFIG_PATH": "/usr/local/nvidia/lib64/a3plus_tuner_config.textproto",
          "NCCL_TUNER_PLUGIN": "libnccl-tuner.so",
      })

  for attempt in range(500):
    try:
      env["JAX_COORDINATOR_IP"] = socket.gethostbyname(env["JAX_COORDINATOR_ADDRESS"])
      break
    except socket.gaierror:
      print(f"Failed to resolve coordinator on attempt {attempt + 1}, retrying...")
      time.sleep(1)
  return subprocess.run(command, env=env, check=False).returncode
