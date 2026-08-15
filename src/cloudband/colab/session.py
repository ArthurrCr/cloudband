"""Colab session setup that runs after the repository is on sys.path.

The clone itself must stay in the notebook, since the package does not exist
before it. Everything after the clone lives here.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REQUIRED_PACKAGES: tuple[str, ...] = ("omnicloudmask", "rasterio", "pandas", "numpy")


@dataclass(frozen=True)
class Session:
    """Where the session reads and writes."""

    project_dir: Path
    data_dir: Path
    results_dir: Path
    device: str
    package_versions: dict[str, str]


def package_versions(names: tuple[str, ...] = REQUIRED_PACKAGES) -> dict[str, str]:
    """Return installed versions, marking absent packages explicitly."""
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "absent"
    return versions


def detect_device() -> str:
    """Return the accelerator a runtime offers, without importing torch eagerly."""
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        name: str = torch.cuda.get_device_name(0)
        return f"cuda:{name}"
    return "cpu"


def require_gpu() -> str:
    """Raise when no CUDA device is present, before a long run starts."""
    device = detect_device()
    if not device.startswith("cuda"):
        raise RuntimeError("no CUDA device; switch the Colab runtime to GPU")
    return device


def free_disk_gb(path: Path) -> float:
    """Return free space in gibibytes at a path."""
    return shutil.disk_usage(path).free / 1024**3


def require_disk(minimum_gb: float, path: Path) -> float:
    """Raise when free space is below a threshold, before a download starts."""
    available = free_disk_gb(path)
    if available < minimum_gb:
        raise RuntimeError(f"{available:.1f} GiB free at {path}, {minimum_gb} GiB required")
    return available


def install(specifiers: tuple[str, ...]) -> None:
    """Install pinned requirements into the running kernel."""
    if not specifiers:
        return
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", *specifiers],
        check=True,
    )


def reload_package(prefix: str = "cloudband") -> None:
    """Drop cached modules so a fresh pull is picked up without restarting."""
    for name in [key for key in sys.modules if key == prefix or key.startswith(f"{prefix}.")]:
        del sys.modules[name]


def start(
    project_dir: Path,
    data_subdir: str = "data",
    results_subdir: str = "results",
    require_accelerator: bool = False,
) -> Session:
    """Resolve directories, check the runtime and report what was found."""
    project_dir = Path(project_dir)
    if not project_dir.is_dir():
        raise RuntimeError(f"{project_dir} does not exist; clone the repository first")

    data_dir = project_dir / data_subdir
    results_dir = project_dir / results_subdir
    for directory in (data_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)

    device = require_gpu() if require_accelerator else detect_device()
    session = Session(
        project_dir=project_dir,
        data_dir=data_dir,
        results_dir=results_dir,
        device=device,
        package_versions=package_versions(),
    )
    print(f"project: {session.project_dir}")
    print(f"device: {session.device}")
    print(f"free disk: {free_disk_gb(project_dir):.1f} GiB")
    for name, version in session.package_versions.items():
        print(f"{name}: {version}")
    return session