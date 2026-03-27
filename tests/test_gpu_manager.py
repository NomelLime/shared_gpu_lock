"""Тесты GPUResourceManager и get_gpu_manager (пакет shared_gpu_lock)."""

from __future__ import annotations

import threading
import time
from typing import List

import pytest

from shared_gpu_lock.gpu_manager import (
    GPUPriority,
    GPUResourceManager,
    get_gpu_manager,
    reset_gpu_manager_singleton,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_gpu_manager_singleton()
    yield
    reset_gpu_manager_singleton()


class TestGPUResourceManager:
    def setup_method(self):
        self.gpu = GPUResourceManager(max_concurrent=1, use_cross_process_lock=False)
        self.gpu.start()

    def teardown_method(self):
        self.gpu.stop()

    def test_priority_order(self):
        acquired_order: List[str] = []
        barrier = threading.Barrier(3)

        def worker(priority, label):
            barrier.wait()
            if label == "llm":
                time.sleep(0.08)
            with self.gpu.acquire(label, priority):
                acquired_order.append(label)

        t_llm = threading.Thread(target=worker, args=(GPUPriority.LLM, "llm"))
        t_crit = threading.Thread(target=worker, args=(GPUPriority.CRITICAL, "critical"))
        t_llm.start()
        t_crit.start()
        barrier.wait()
        t_llm.join(timeout=5)
        t_crit.join(timeout=5)

        assert acquired_order == ["critical", "llm"]

    def test_concurrent_limit_one(self):
        concurrent = [0]
        max_seen = [0]
        lock = threading.Lock()

        def worker():
            with self.gpu.acquire("w", GPUPriority.ENCODE):
                with lock:
                    concurrent[0] += 1
                    max_seen[0] = max(max_seen[0], concurrent[0])
                time.sleep(0.05)
                with lock:
                    concurrent[0] -= 1

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert max_seen[0] == 1

    def test_dispatcher_abort_sets_aborted_fast(self):
        """После исчерпания ретраев диспетчера — event + aborted, без долгого wait(360)."""
        g = GPUResourceManager(
            max_concurrent=1,
            use_cross_process_lock=False,
            dispatcher_semaphore_acquire_timeout=0.12,
        )
        g.start()
        g._semaphore.acquire(blocking=False)

        err: List[Exception] = []

        def waiter():
            try:
                with g.acquire("blocked", GPUPriority.LLM):
                    pass
            except RuntimeError as e:
                err.append(e)

        t = threading.Thread(target=waiter)
        t.start()
        t.join(timeout=15)
        g._semaphore.release()
        t.join(timeout=5)
        assert err and "отменена" in str(err[0])
        g.stop()


class TestGetGpuManagerSingleton:
    def test_same_instance_after_concurrent_first_calls(self):
        ids: List[int] = []

        def one():
            ids.append(id(get_gpu_manager()))

        threads = [threading.Thread(target=one) for _ in range(32)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(set(ids)) == 1
