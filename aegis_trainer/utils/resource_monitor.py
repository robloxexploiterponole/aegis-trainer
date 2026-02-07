"""
ResourceMonitor — CPU, RAM, and VRAM usage tracking for AEGIS AI Trainer.

Designed for Intel Arc B580 (Vulkan, NOT CUDA). VRAM detection priority:
  1. torch.xpu.mem_get_info() via Intel Extension for PyTorch (IPEX)
  2. sysfs /sys/class/drm/card*/device/mem_info_vram_used (Intel i915/xe driver)
  3. Fallback to 0/0 (no VRAM monitoring)

Never calls torch.cuda.* — it will fail on this hardware.

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import psutil

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResourceLimits:
    """Thresholds for resource usage. Exceeding any triggers throttling.

    Defaults calibrated for the target system:
      - 8 CPU cores
      - 120 GB RAM
      - 11 GB VRAM (Intel Arc B580)
    """

    max_cpu_percent: float = 85.0
    max_ram_percent: float = 90.0
    max_vram_percent: float = 85.0
    max_ram_bytes: int = 120 * 1024**3   # 120 GB
    max_vram_bytes: int = 11 * 1024**3   # 11 GB


@dataclass(frozen=True)
class ResourceSnapshot:
    """Point-in-time snapshot of system resource usage."""

    cpu_percent: float
    ram_used_bytes: int
    ram_total_bytes: int
    vram_used_bytes: int
    vram_total_bytes: int
    timestamp: float

    @property
    def ram_percent(self) -> float:
        if self.ram_total_bytes == 0:
            return 0.0
        return (self.ram_used_bytes / self.ram_total_bytes) * 100.0

    @property
    def vram_percent(self) -> float:
        if self.vram_total_bytes == 0:
            return 0.0
        return (self.vram_used_bytes / self.vram_total_bytes) * 100.0

    def __repr__(self) -> str:
        return (
            f"ResourceSnapshot("
            f"cpu={self.cpu_percent:.1f}%, "
            f"ram={self.ram_used_bytes / 1024**3:.1f}/{self.ram_total_bytes / 1024**3:.1f}GB "
            f"({self.ram_percent:.1f}%), "
            f"vram={self.vram_used_bytes / 1024**3:.1f}/{self.vram_total_bytes / 1024**3:.1f}GB "
            f"({self.vram_percent:.1f}%))"
        )


class ResourceMonitor:
    """Monitors CPU, RAM, and VRAM usage with throttling support.

    Usage::

        monitor = ResourceMonitor()
        snap = monitor.get_snapshot()
        print(snap)

        if monitor.is_over_threshold():
            monitor.check_and_throttle()  # Blocks until resources free up
    """

    def __init__(
        self,
        limits: Optional[ResourceLimits] = None,
        throttle_sleep_seconds: float = 2.0,
        throttle_max_wait_seconds: float = 300.0,
    ):
        """Initialize the resource monitor.

        Args:
            limits: Resource thresholds. Uses defaults if None.
            throttle_sleep_seconds: Seconds to sleep between throttle checks.
            throttle_max_wait_seconds: Maximum total wait time before giving up.
        """
        self.limits = limits or ResourceLimits()
        self.throttle_sleep_seconds = throttle_sleep_seconds
        self.throttle_max_wait_seconds = throttle_max_wait_seconds

        # Detect VRAM backend
        self._vram_backend = self._detect_vram_backend()
        if self._vram_backend == "xpu":
            logger.info("VRAM monitoring via torch.xpu (Intel IPEX)")
        elif self._vram_backend == "sysfs":
            logger.info("VRAM monitoring via sysfs (Intel DRM driver)")
        else:
            logger.warning(
                "No VRAM monitoring available. VRAM usage will report 0/0."
            )

    @staticmethod
    def _detect_vram_backend() -> str:
        """Detect the best available VRAM monitoring backend.

        Returns:
            "xpu" if torch.xpu is available, "sysfs" if Intel DRM sysfs nodes
            exist, or "none" if no backend is available.
        """
        # 1. Try torch.xpu (Intel Extension for PyTorch / IPEX)
        try:
            import torch
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                # Verify mem_get_info works
                torch.xpu.mem_get_info()
                return "xpu"
        except Exception:
            pass

        # 2. Try sysfs for Intel DRM devices
        drm_path = Path("/sys/class/drm")
        if drm_path.exists():
            for card_dir in sorted(drm_path.glob("card*")):
                vram_used = card_dir / "device" / "mem_info_vram_used"
                vram_total = card_dir / "device" / "mem_info_vram_total"
                if vram_used.exists() and vram_total.exists():
                    return "sysfs"

        return "none"

    def get_cpu_usage(self) -> float:
        """Get current CPU usage as a percentage (0-100).

        Returns:
            CPU usage percentage averaged across all cores.
        """
        return psutil.cpu_percent(interval=0.1)

    def get_ram_usage(self) -> Tuple[int, int]:
        """Get current RAM usage.

        Returns:
            Tuple of (used_bytes, total_bytes).
        """
        mem = psutil.virtual_memory()
        return mem.used, mem.total

    def get_vram_usage(self) -> Tuple[int, int]:
        """Get current VRAM usage from the best available backend.

        Returns:
            Tuple of (used_bytes, total_bytes). Returns (0, 0) if no
            VRAM monitoring is available.
        """
        if self._vram_backend == "xpu":
            return self._get_vram_xpu()
        elif self._vram_backend == "sysfs":
            return self._get_vram_sysfs()
        return 0, 0

    @staticmethod
    def _get_vram_xpu() -> Tuple[int, int]:
        """Read VRAM via torch.xpu.mem_get_info()."""
        try:
            import torch
            free, total = torch.xpu.mem_get_info()
            used = total - free
            return used, total
        except Exception as exc:
            logger.debug("torch.xpu.mem_get_info() failed: %s", exc)
            return 0, 0

    @staticmethod
    def _get_vram_sysfs() -> Tuple[int, int]:
        """Read VRAM from Intel DRM sysfs nodes.

        Scans /sys/class/drm/card*/device/ for mem_info_vram_used and
        mem_info_vram_total. Returns the first device found (typically
        the discrete GPU).
        """
        drm_path = Path("/sys/class/drm")
        try:
            for card_dir in sorted(drm_path.glob("card*")):
                device_dir = card_dir / "device"
                vram_used_path = device_dir / "mem_info_vram_used"
                vram_total_path = device_dir / "mem_info_vram_total"

                if vram_used_path.exists() and vram_total_path.exists():
                    used = int(vram_used_path.read_text().strip())
                    total = int(vram_total_path.read_text().strip())
                    if total > 0:
                        return used, total
        except (OSError, ValueError) as exc:
            logger.debug("sysfs VRAM read failed: %s", exc)

        return 0, 0

    def get_snapshot(self) -> ResourceSnapshot:
        """Capture a point-in-time snapshot of all resource usage.

        Returns:
            ResourceSnapshot with current CPU, RAM, and VRAM metrics.
        """
        cpu = self.get_cpu_usage()
        ram_used, ram_total = self.get_ram_usage()
        vram_used, vram_total = self.get_vram_usage()

        return ResourceSnapshot(
            cpu_percent=cpu,
            ram_used_bytes=ram_used,
            ram_total_bytes=ram_total,
            vram_used_bytes=vram_used,
            vram_total_bytes=vram_total,
            timestamp=time.time(),
        )

    def is_over_threshold(self) -> bool:
        """Check if any resource exceeds its configured threshold.

        Returns:
            True if CPU, RAM, or VRAM usage exceeds limits.
        """
        snap = self.get_snapshot()

        # CPU check
        if snap.cpu_percent > self.limits.max_cpu_percent:
            return True

        # RAM checks: percentage and absolute
        if snap.ram_percent > self.limits.max_ram_percent:
            return True
        if snap.ram_used_bytes > self.limits.max_ram_bytes:
            return True

        # VRAM checks (only if monitoring is available)
        if snap.vram_total_bytes > 0:
            if snap.vram_percent > self.limits.max_vram_percent:
                return True
            if snap.vram_used_bytes > self.limits.max_vram_bytes:
                return True

        return False

    def check_and_throttle(self) -> None:
        """Block execution while resources are over threshold.

        Sleeps in a loop until resources drop below threshold or
        the maximum wait time is exceeded. Logs status periodically.

        Raises:
            TimeoutError: If resources remain over threshold for longer
                than throttle_max_wait_seconds.
        """
        if not self.is_over_threshold():
            return

        logger.warning("Resource threshold exceeded — throttling...")
        start = time.time()
        waited = 0.0

        while self.is_over_threshold():
            waited = time.time() - start
            if waited > self.throttle_max_wait_seconds:
                snap = self.get_snapshot()
                raise TimeoutError(
                    f"Resource throttle timeout after {waited:.0f}s. "
                    f"Current usage: {snap}"
                )

            if int(waited) % 30 == 0 and waited > 0:
                snap = self.get_snapshot()
                logger.info(
                    "Still throttled (%.0fs elapsed). %s", waited, snap
                )

            time.sleep(self.throttle_sleep_seconds)

        elapsed = time.time() - start
        logger.info("Throttle released after %.1fs", elapsed)
