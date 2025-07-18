import threading
import time
import psutil
import os
from typing import Dict, Optional
from contextlib import contextmanager


class MemoryMonitor:
    def __init__(self, sampling_interval: float = 0.01):
        self.process = psutil.Process(os.getpid())
        self.sampling_interval = sampling_interval
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._peak_memory = 0.0
        self._baseline_memory = 0.0
        self._lock = threading.Lock()

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def _monitor_loop(self):
        while self._monitoring:
            try:
                current_memory = self._get_memory_mb()
                with self._lock:
                    if current_memory > self._peak_memory:
                        self._peak_memory = current_memory
                time.sleep(self.sampling_interval)
            except Exception:
                time.sleep(self.sampling_interval)

    def start_monitoring(self):
        if self._monitoring:
            return

        with self._lock:
            self._baseline_memory = self._get_memory_mb()
            self._peak_memory = self._baseline_memory
            self._monitoring = True

        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> float:
        if not self._monitoring:
            return 0.0

        self._monitoring = False

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)

        with self._lock:
            peak_increase = max(0.0, self._peak_memory - self._baseline_memory)
            return peak_increase

    def get_current_peak(self) -> float:
        with self._lock:
            return max(0.0, self._peak_memory - self._baseline_memory)

    @contextmanager
    def monitor_step(self, step_name: str = ""):
        self.start_monitoring()
        try:
            yield self
        finally:
            self.peak_usage = self.stop_monitoring()


class ProcessStepMemoryTracker:

    def __init__(self, implementation_name: str, sampling_interval: float = 0.01):
        self.implementation_name = implementation_name
        self.memory_usage: Dict[str, float] = {}
        self.monitor = MemoryMonitor(sampling_interval)

    @contextmanager
    def track_step(self, step_name: str):
        self.monitor.start_monitoring()
        try:
            yield
        finally:
            peak_memory = self.monitor.stop_monitoring()
            self.memory_usage[step_name] = peak_memory

    def get_memory_usage(self) -> Dict[str, Dict[str, float]]:
        return {self.implementation_name: self.memory_usage.copy()}

    def get_step_memory(self, step_name: str) -> float:
        return self.memory_usage.get(step_name, 0.0)

    def clear(self):
        self.memory_usage.clear()
