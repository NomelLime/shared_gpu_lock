"""
Кросс-процессная блокировка GPU через portalocker (тот же файл, что у ShortsProject/Orchestrator).

Путь по умолчанию: родитель каталога пакета shared_gpu_lock трижды вверх → GitHub/.gpu_lock,
либо переменная окружения SHARED_GPU_LOCK_PATH.

Реентерабельность: вложенные acquire_gpu_lock / gpu_manager.acquire в одном потоке
не дублируют файловый lock.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

_thread_local = threading.local()


def get_gpu_lock_file_path() -> Path:
    """
    Единый путь к lock-файлу для всех локальных проектов под GitHub.
    Переопределение: SHARED_GPU_LOCK_PATH (абсолютный или относительный путь).
    Если пакет не в editable-дереве GitHub — fallback на temp (задайте env).
    """
    env = os.environ.get("SHARED_GPU_LOCK_PATH")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    if len(here.parents) >= 3:
        cand = here.parents[2]
        if cand.name == "GitHub" or (cand / "ShortsProject").is_dir() or (cand / "Orchestrator").is_dir():
            return cand / ".gpu_lock"
    return Path(tempfile.gettempdir()) / "shared_gpu_lock" / "gpu.lock"


@contextmanager
def _acquire_gpu_lock_impl(consumer: str, timeout: float) -> Iterator[None]:
    try:
        import portalocker
    except ImportError:
        logger.debug("[GPU-Lock] portalocker не установлен — lock пропущен для %s", consumer)
        yield
        return

    path = get_gpu_lock_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    lock_file = None
    acquired = False
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        try:
            lock_file = open(str(path), "w")
            portalocker.lock(lock_file, portalocker.LOCK_EX | portalocker.LOCK_NB)
            acquired = True
            break
        except portalocker.AlreadyLocked:
            if lock_file:
                try:
                    lock_file.close()
                except Exception:
                    pass
                lock_file = None
            time.sleep(2.0)
        except Exception as e:
            logger.warning("[GPU-Lock] Ошибка lock для %s: %s", consumer, e)
            if lock_file:
                try:
                    lock_file.close()
                except Exception:
                    pass
                lock_file = None
            break

    if not acquired:
        logger.warning(
            "[GPU-Lock] Timeout (%ds) для %s — GPU занят",
            int(timeout), consumer,
        )
        raise TimeoutError(f"GPU lock timeout ({int(timeout)}s) for {consumer}")

    try:
        yield
    finally:
        if lock_file:
            try:
                import portalocker as _pl

                _pl.unlock(lock_file)
                lock_file.close()
            except Exception:
                pass


@contextmanager
def acquire_gpu_lock(consumer: str = "unknown", timeout: float = 120.0) -> Iterator[None]:
    """
    Файловый lock между процессами. Реентерабелен в одном потоке.
    """
    depth = getattr(_thread_local, "gpu_lock_depth", 0)
    if depth > 0:
        _thread_local.gpu_lock_depth = depth + 1
        try:
            yield
        finally:
            _thread_local.gpu_lock_depth -= 1
        return

    _thread_local.gpu_lock_depth = 1
    try:
        with _acquire_gpu_lock_impl(consumer, timeout):
            yield
    finally:
        _thread_local.gpu_lock_depth = 0


def cross_process_gpu_section(consumer: str, timeout: float) -> "contextmanager":
    """
    Та же реентерабельная семантика, что у acquire_gpu_lock — для GPUResourceManager.acquire.
    """
    return acquire_gpu_lock(consumer=consumer, timeout=timeout)
