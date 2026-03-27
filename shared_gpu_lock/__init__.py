"""Общий GPU: in-process очередь + кросс-процессный file-lock (portalocker)."""

from shared_gpu_lock.file_lock import acquire_gpu_lock, get_gpu_lock_file_path
from shared_gpu_lock.gpu_manager import (
    GPUResourceManager,
    GPUPriority,
    get_gpu_manager,
    reset_gpu_manager_singleton,
)

__all__ = [
    "acquire_gpu_lock",
    "get_gpu_lock_file_path",
    "GPUResourceManager",
    "GPUPriority",
    "get_gpu_manager",
    "reset_gpu_manager_singleton",
]
