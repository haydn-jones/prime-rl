import os

import pynvml

from prime_rl.utils.logger import get_logger


def disable_nccl_p2p_if_unavailable() -> None:
    """Disable NCCL P2P/SHM transports when GPUs lack NVLink interconnect.

    In environments without NVLink (e.g. PCIe-only or VMs with GPU passthrough),
    NCCL's P2P and SHM transports can fail because they rely on CUDA IPC which
    requires peer access. Disabling them forces NCCL to use socket-based transport.

    Uses pynvml to check physical GPU topology, which works regardless of
    CUDA_VISIBLE_DEVICES restrictions on the current process.
    """
    pynvml.nvmlInit()
    try:
        n = pynvml.nvmlDeviceGetCount()
        if n < 2:
            return

        for i in range(n):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            for link in range(18):
                try:
                    if pynvml.nvmlDeviceGetNvLinkState(handle, link):
                        return  # NVLink found, P2P should work
                except pynvml.NVMLError:
                    break

        # No NVLink found between any GPUs — disable P2P/SHM
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_SHM_DISABLE", "1")
        get_logger().info("No NVLink detected, disabling NCCL P2P and SHM transports")
    finally:
        pynvml.nvmlShutdown()
