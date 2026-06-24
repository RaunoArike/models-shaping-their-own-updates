"""
Code execution for MBPP-Honeypot.

Provides parallel and synchronous code execution with timeout and memory limits.
"""

import atexit
import contextlib
import multiprocessing as mp
import os
import queue
import resource
import signal
import time

from tqdm.auto import tqdm

from .worker import _persistent_worker


class PersistentExecutor:
    """Global worker pool that lives for the entire training run."""

    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.ctx = mp.get_context("spawn")
        self.task_queue = self.ctx.Queue(maxsize=10000)
        self.result_queue = self.ctx.Queue(maxsize=10000)
        self.workers = []
        self.task_counter = 0
        self.started = False

    def start(self):
        if self.started:
            return
        for _ in range(self.num_workers):
            p = self.ctx.Process(target=_persistent_worker, args=(self.task_queue, self.result_queue), daemon=True)
            p.start()
            self.workers.append(p)
        self.started = True
        tqdm.write(f"[CodeExecutor] Started {self.num_workers} workers")

    def _check_and_restart_workers(self):
        dead_count = 0
        for i, worker in enumerate(self.workers):
            if not worker.is_alive():
                dead_count += 1
                worker.terminate()
                new_worker = self.ctx.Process(
                    target=_persistent_worker, args=(self.task_queue, self.result_queue), daemon=True
                )
                new_worker.start()
                self.workers[i] = new_worker
        if dead_count > 0:
            tqdm.write(f"[CodeExecutor] Restarted {dead_count} dead workers")

    def execute_batch(self, code_strings: list[str], timeout_seconds: int = 5) -> list[bool]:
        """Submit all code strings at once and collect results in parallel."""
        if not self.started:
            self.start()
        if not code_strings:
            return []

        self._check_and_restart_workers()

        drained = 0
        while True:
            try:
                self.result_queue.get(block=False)
                drained += 1
            except (queue.Empty, Exception):
                break
        if drained > 0:
            print(f"[CodeExecutor] Drained {drained} stale results from queue")

        task_ids = []
        for code_string in code_strings:
            task_id = self.task_counter
            self.task_counter += 1
            task_ids.append(task_id)
            self.task_queue.put((task_id, code_string, timeout_seconds))

        results: dict[int, bool] = {}
        deadline = time.time() + 30.0
        last_result_time = time.time()

        while len(results) < len(task_ids) and time.time() < deadline:
            if time.time() - last_result_time > 15.0:
                tqdm.write(f"[CodeExecutor] Stalled at {len(results)}/{len(task_ids)} results, checking workers...")
                self._check_and_restart_workers()
                time.sleep(1.0)
                last_result_time = time.time()
            try:
                result_id, success = self.result_queue.get(timeout=0.1)
                results[result_id] = success
                last_result_time = time.time()
            except Exception:
                continue

        if len(results) < len(task_ids):
            tqdm.write(f"[CodeExecutor] Failed to get results for {len(task_ids) - len(results)} tasks")

        return [results.get(task_id, False) for task_id in task_ids]

    def shutdown(self):
        if not self.started:
            return
        for _ in range(self.num_workers):
            try:
                self.task_queue.put(None)
            except Exception:
                pass
        for p in self.workers:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()
        self.started = False


class SynchronousExecutor:
    """
    Synchronous executor for single-worker scenarios (avoids daemon process issues).

    Includes memory limits on Linux. NOT safe for memory-intensive code on macOS.
    """

    def __init__(self):
        import sys
        self.started = True
        self._memory_limit_set = False
        self._old_limit = None
        self._is_linux = sys.platform.startswith("linux")

    def start(self):
        pass

    def _set_memory_limit(self):
        if self._memory_limit_set or not self._is_linux:
            return
        import psutil
        try:
            process = psutil.Process()
            task_allowance = 4 * 1024 * 1024 * 1024
            total_limit = int(process.memory_info().vms + task_allowance)
            self._old_limit, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (total_limit, hard))
            self._memory_limit_set = True
        except (ValueError, OSError, AttributeError):
            pass

    def _restore_memory_limit(self):
        if not self._memory_limit_set or self._old_limit is None:
            return
        try:
            _, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (self._old_limit, hard))
            self._memory_limit_set = False
        except (ValueError, OSError):
            pass

    def execute_batch(self, code_strings: list[str], timeout_seconds: int = 5) -> list[bool]:
        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timed out")

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        self._set_memory_limit()
        results = []
        try:
            for code_string in code_strings:
                success = False
                try:
                    signal.alarm(timeout_seconds)
                    try:
                        with open(os.devnull, "w") as devnull:
                            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                                exec(code_string, {})
                        success = True
                    except (MemoryError, Exception):
                        success = False
                    finally:
                        signal.alarm(0)
                except (TimeoutError, MemoryError, Exception):
                    success = False
                results.append(success)
        finally:
            signal.signal(signal.SIGALRM, old_handler)
            self._restore_memory_limit()
        return results

    def shutdown(self):
        self._restore_memory_limit()


_global_executor = None


def get_executor(num_workers: int = 4) -> PersistentExecutor | SynchronousExecutor:
    global _global_executor
    if num_workers == 1:
        return SynchronousExecutor()
    if _global_executor is None:
        _global_executor = PersistentExecutor(num_workers)
        _global_executor.start()
        atexit.register(_global_executor.shutdown)
    return _global_executor
