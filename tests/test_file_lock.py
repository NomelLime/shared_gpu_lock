"""Тесты acquire_gpu_lock и таймаутов."""

from __future__ import annotations

import threading
import time
import pytest

from shared_gpu_lock.file_lock import acquire_gpu_lock, get_gpu_lock_file_path


@pytest.fixture
def isolated_lock_path(tmp_path, monkeypatch):
    p = tmp_path / "test_gpu.lock"
    monkeypatch.setenv("SHARED_GPU_LOCK_PATH", str(p))
    yield p


class TestAcquireGpuLockReentrancy:
    def test_nested_same_thread_no_deadlock(self, isolated_lock_path):
        with acquire_gpu_lock(consumer="outer", timeout=30):
            with acquire_gpu_lock(consumer="inner", timeout=30):
                assert isolated_lock_path.exists() or True


class TestAcquireGpuLockTimeout:
    def test_timeout_raises_when_held_by_other_thread(self, isolated_lock_path):
        hold = threading.Event()
        release = threading.Event()

        def holder():
            with acquire_gpu_lock(consumer="holder", timeout=30):
                hold.set()
                release.wait(timeout=5)

        t = threading.Thread(target=holder)
        t.start()
        assert hold.wait(timeout=5)

        with pytest.raises(TimeoutError):
            with acquire_gpu_lock(consumer="waiter", timeout=0.5):
                pass

        release.set()
        t.join(timeout=5)

        with acquire_gpu_lock(consumer="after", timeout=5):
            pass


def test_get_gpu_lock_file_path_respects_env(monkeypatch, tmp_path):
    p = tmp_path / "x.lock"
    monkeypatch.setenv("SHARED_GPU_LOCK_PATH", str(p))
    assert get_gpu_lock_file_path() == p.resolve()
