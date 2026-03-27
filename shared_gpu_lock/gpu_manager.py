"""
Менеджер GPU-ресурсов: in-process PriorityQueue + семафор и кросс-процессный file-lock
(portalocker), чтобы несколько локальных процессов не грузили VRAM одновременно.

Приоритеты (чем ниже число — тем выше приоритет):
  0 — CRITICAL (антибан, CAPTCHA)
  1 — LLM / Ollama
  2 — TTS / Kokoro
  3 — VideoGen / AnimateDiff
  4 — ffmpeg encode
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from shared_gpu_lock.file_lock import acquire_gpu_lock

logger = logging.getLogger(__name__)


class GPUPriority:
    CRITICAL = 0
    LLM = 1
    TTS = 2
    VIDEO_GEN = 3
    ENCODE = 4


_GPU_TASK_MAX_RETRIES = 3


@dataclass(order=True)
class _GPUTask:
    priority: int
    seq: int = field(compare=False)
    consumer: str = field(compare=False)
    event: threading.Event = field(compare=False, default_factory=threading.Event)
    retries: int = field(compare=False, default=0)


class GPUResourceManager:
    """
    max_concurrent: сколько задач могут использовать GPU одновременно в этом процессе.
    Между процессами — общий файловый lock (см. shared_gpu_lock.file_lock).
    """

    def __init__(
        self,
        max_concurrent: int = 1,
        *,
        use_cross_process_lock: bool = True,
        cross_process_lock_timeout: float = 360.0,
    ) -> None:
        self._max = max_concurrent
        self._semaphore = threading.Semaphore(max_concurrent)
        self._lock = threading.Lock()
        self._task_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=100)
        self._active: Dict[str, float] = {}
        self._seq = 0
        self._stats: Dict[str, Dict] = {}
        self._use_cross_process_lock = use_cross_process_lock
        self._cross_process_lock_timeout = cross_process_lock_timeout
        self._dispatcher_thread = threading.Thread(
            target=self._dispatcher, daemon=True, name="gpu-dispatcher"
        )
        self._running = False

    def start(self) -> None:
        if not self._running:
            self._running = True
            self._dispatcher_thread = threading.Thread(
                target=self._dispatcher, daemon=True, name="gpu-dispatcher"
            )
            self._dispatcher_thread.start()
            logger.info("[GPUManager] Запущен (max_concurrent=%d)", self._max)

    def stop(self) -> None:
        self._running = False
        logger.info("[GPUManager] Остановлен.")

    @contextmanager
    def acquire(self, consumer: str, priority: int = GPUPriority.ENCODE):
        task = self._enqueue(consumer, priority)
        logger.debug("[GPUManager] [%s] Ожидаем GPU (приоритет %d)...", consumer, priority)

        task.event.wait(timeout=360)
        if not task.event.is_set():
            raise TimeoutError(f"[GPUManager] GPU не получен за 360с для '{consumer}'")

        start_time = time.monotonic()
        with self._lock:
            self._active[consumer] = start_time

        logger.info("[GPUManager] [%s] GPU захвачен (приоритет %d)", consumer, priority)

        lock_cm = (
            acquire_gpu_lock(consumer=consumer, timeout=self._cross_process_lock_timeout)
            if self._use_cross_process_lock
            else nullcontext()
        )

        try:
            with lock_cm:
                try:
                    yield
                finally:
                    elapsed = time.monotonic() - start_time
                    with self._lock:
                        self._active.pop(consumer, None)
                        self._update_stats(consumer, elapsed)
                    self._semaphore.release()
                    logger.info(
                        "[GPUManager] [%s] GPU освобождён (занято %.1f сек.)",
                        consumer,
                        elapsed,
                    )
        except TimeoutError:
            elapsed = time.monotonic() - start_time
            with self._lock:
                self._active.pop(consumer, None)
            self._semaphore.release()
            logger.error("[GPUManager] [%s] Таймаут кросс-процессного GPU-lock", consumer)
            raise

    def gpu_task(self, priority: int = GPUPriority.ENCODE):
        def decorator(fn: Callable):
            def wrapper(*args, **kwargs):
                consumer = fn.__qualname__
                with self.acquire(consumer, priority):
                    return fn(*args, **kwargs)

            wrapper.__name__ = fn.__name__
            return wrapper

        return decorator

    def status(self) -> Dict:
        with self._lock:
            return {
                "active": dict(self._active),
                "queue_size": self._task_queue.qsize(),
                "max_concurrent": self._max,
                "stats": dict(self._stats),
            }

    def _enqueue(self, consumer: str, priority: int) -> _GPUTask:
        with self._lock:
            seq = self._seq
            self._seq += 1
        task = _GPUTask(priority=priority, seq=seq, consumer=consumer)
        self._task_queue.put(task)
        return task

    def _dispatcher(self) -> None:
        while self._running:
            try:
                task = self._task_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            acquired = self._semaphore.acquire(timeout=300)
            if not acquired:
                task.retries += 1
                if task.retries >= _GPU_TASK_MAX_RETRIES:
                    logger.error(
                        "[GPUManager] Задача [%s] отменена после %d попыток — GPU не освобождается",
                        task.consumer,
                        task.retries,
                    )
                    continue
                logger.warning(
                    "[GPUManager] Timeout ожидания слота для [%s] (попытка %d/%d)",
                    task.consumer,
                    task.retries,
                    _GPU_TASK_MAX_RETRIES,
                )
                self._task_queue.put(task)
                continue

            task.event.set()

    def _update_stats(self, consumer: str, elapsed: float) -> None:
        if consumer not in self._stats:
            self._stats[consumer] = {"calls": 0, "total_sec": 0.0, "avg_sec": 0.0}
        s = self._stats[consumer]
        s["calls"] += 1
        s["total_sec"] += elapsed
        s["avg_sec"] = s["total_sec"] / s["calls"]


_gpu_manager: Optional[GPUResourceManager] = None


def get_gpu_manager() -> GPUResourceManager:
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUResourceManager(max_concurrent=1)
        _gpu_manager.start()
    return _gpu_manager
