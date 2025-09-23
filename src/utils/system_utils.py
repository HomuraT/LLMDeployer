import platform
import socket
import os
import time

try:
    import psutil
except ImportError:
    psutil = None


def get_system_stats() -> dict:
    """
    返回基础系统信息与负载：CPU、内存、磁盘、网络等。
    """
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pid": os.getpid(),
        "time": int(time.time()),
    }

    if not psutil:
        info.update({
            "psutil_available": False,
        })
        return info

    info["psutil_available"] = True
    try:
        info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        vm = psutil.virtual_memory()
        info["memory_total_mb"] = int(vm.total // (1024 * 1024))
        info["memory_used_mb"] = int((vm.total - vm.available) // (1024 * 1024))
        info["memory_available_mb"] = int(vm.available // (1024 * 1024))

        disk = psutil.disk_usage("/")
        info["disk_total_gb"] = round(disk.total / (1024 ** 3), 2)
        info["disk_used_gb"] = round(disk.used / (1024 ** 3), 2)
        info["disk_free_gb"] = round(disk.free / (1024 ** 3), 2)

        net = psutil.net_io_counters()
        info["net_bytes_sent_mb"] = round(net.bytes_sent / (1024 ** 2), 2)
        info["net_bytes_recv_mb"] = round(net.bytes_recv / (1024 ** 2), 2)
    except Exception:
        pass

    return info


