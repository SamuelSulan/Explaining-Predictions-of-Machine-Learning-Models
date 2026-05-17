"""Lightweight runtime/resource measurement helpers."""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


def _rss_mb(process) -> float | None:
    if process is None:
        return None
    try:
        return float(process.memory_info().rss / (1024 * 1024))
    except Exception:
        return None


def _cuda_snapshot(reset_peak: bool = False) -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()
        return {
            "cuda_available": True,
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_allocated_mb": float(torch.cuda.memory_allocated() / (1024 * 1024)),
            "cuda_reserved_mb": float(torch.cuda.memory_reserved() / (1024 * 1024)),
            "cuda_peak_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
            "cuda_peak_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024 * 1024)),
        }
    except Exception as exc:
        return {"cuda_available": False, "cuda_error": f"{type(exc).__name__}: {exc}"}


@dataclass
class PerfConfig:
    enabled: bool = True
    sample_interval_seconds: float = 0.02


class ResourceMeasurement:
    """Measure wall time, process CPU time, RSS memory, and optional CUDA memory."""

    def __init__(self, name: str, config: PerfConfig | None = None) -> None:
        self.name = name
        self.config = config or PerfConfig()
        self.metrics: dict = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss_mb: float | None = None
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
        except Exception:
            self._process = None

    def __enter__(self) -> "ResourceMeasurement":
        if not self.config.enabled:
            self._start_wall = time.perf_counter()
            self._start_cpu = time.process_time()
            return self

        self._start_wall = time.perf_counter()
        self._start_cpu = time.process_time()
        self._start_rss_mb = _rss_mb(self._process)
        self._peak_rss_mb = self._start_rss_mb
        self._cuda_start = _cuda_snapshot(reset_peak=True)

        if self._process is not None:
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        return self

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            rss = _rss_mb(self._process)
            if rss is not None and (self._peak_rss_mb is None or rss > self._peak_rss_mb):
                self._peak_rss_mb = rss
            time.sleep(max(float(self.config.sample_interval_seconds), 0.001))

    def __exit__(self, exc_type, exc, tb) -> None:
        end_wall = time.perf_counter()
        end_cpu = time.process_time()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        wall = max(end_wall - self._start_wall, 1e-12)
        cpu = max(end_cpu - self._start_cpu, 0.0)
        end_rss = _rss_mb(self._process)
        metrics = {
            "name": self.name,
            "wall_time_s": float(wall),
            "process_cpu_time_s": float(cpu),
            "process_cpu_util_percent": float((cpu / wall) * 100.0),
        }
        if self.config.enabled:
            metrics.update(
                {
                    "rss_start_mb": self._start_rss_mb,
                    "rss_end_mb": end_rss,
                    "rss_delta_mb": None
                    if self._start_rss_mb is None or end_rss is None
                    else float(end_rss - self._start_rss_mb),
                    "rss_peak_mb": self._peak_rss_mb,
                    "cuda_start": self._cuda_start,
                    "cuda_end": _cuda_snapshot(reset_peak=False),
                }
            )
        self.metrics = metrics


@contextmanager
def measured(name: str, config: PerfConfig | None = None) -> Iterator[ResourceMeasurement]:
    measurement = ResourceMeasurement(name, config=config)
    with measurement:
        yield measurement


def summarize_measurements(measurements: list[dict]) -> dict:
    if not measurements:
        return {}
    wall_times = [m.get("wall_time_s", 0.0) for m in measurements]
    cpu_times = [m.get("process_cpu_time_s", 0.0) for m in measurements]
    return {
        "count": len(measurements),
        "total_wall_time_s": float(sum(wall_times)),
        "mean_wall_time_s": float(sum(wall_times) / len(wall_times)),
        "total_process_cpu_time_s": float(sum(cpu_times)),
        "mean_process_cpu_util_percent": float(
            sum(m.get("process_cpu_util_percent", 0.0) for m in measurements) / len(measurements)
        ),
        "max_rss_peak_mb": max(
            [m.get("rss_peak_mb") for m in measurements if m.get("rss_peak_mb") is not None],
            default=None,
        ),
    }
