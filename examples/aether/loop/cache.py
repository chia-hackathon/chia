"""On-disk cache for the simulator build and the baseline measurement.

Both are expensive and depend only on things that rarely change within a run
(the hardware config and the pristine kernel), so they are cached in
``OUT_DIR/_cache`` keyed by a hash of everything that can invalidate them.
Cache misses (missing file, stale content, unpicklable garbage) fall back to
a normal rebuild/remeasure — this is a speedup, never a correctness
dependency.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import subprocess
from pathlib import Path

from chia.chipyard.state_def import BuildArtifact

from constants import AETHER_ROOT, CHIPYARD_PATH, CONFIG_PACKAGE, OUT_DIR
from context import CONFIG
from nodes import Measurement

logger = logging.getLogger("aether.cache")

CACHE_DIR = OUT_DIR / "_cache"


def _saturn_commit() -> str:
    """HEAD of the local Saturn checkout the sealed collateral comes from —
    part of the build's identity even though it isn't a chisel_build input."""
    try:
        out = subprocess.run(
            ["git", "-C", str(AETHER_ROOT / "repos" / "saturn"), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def sim_key() -> str:
    """Cache key for the built simulator: everything that changes the bitstream.
    VLEN/DLEN are baked into the config name (e.g. GENV256D128ShuttleConfig)."""
    parts = "|".join([CONFIG(), CONFIG_PACKAGE, CHIPYARD_PATH, _saturn_commit()])
    return hashlib.sha256(parts.encode()).hexdigest()[:16]


def baseline_key(skey: str, kernel: bytes, kernel_name: str) -> str:
    """Cache key for the baseline measurement: the pristine kernel content
    plus the simulator it was measured on. `kernel_name` is mixed in because
    two different kernels can share byte-identical agent-owned file content
    (e.g. both start from an untouched include/gemmini.h) — without the name,
    they would collide on the same cached baseline."""
    digest = hashlib.sha256(kernel_name.encode() + b"\0" + kernel).hexdigest()[:16]
    return f"{digest}-{skey}"


def _load(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except Exception as exc:
        logger.warning("cache: failed to load %s (%s) — ignoring", path, exc)
        return None


def _save(path: Path, obj) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(obj, fh)
    tmp.replace(path)


def load_simulator(key: str) -> BuildArtifact | None:
    art = _load(CACHE_DIR / f"sim-{key}.pkl")
    if art is None:
        return None
    if not (isinstance(art, BuildArtifact) and art.success
            and art.simulator_binary_content
            and art.config == CONFIG() and art.config_package == CONFIG_PACKAGE):
        logger.warning("cache: sim-%s.pkl is stale/invalid — rebuilding", key)
        return None
    return art


def save_simulator(key: str, art: BuildArtifact) -> None:
    _save(CACHE_DIR / f"sim-{key}.pkl", art)


def load_baseline(key: str) -> Measurement | None:
    m = _load(CACHE_DIR / f"baseline-{key}.pkl")
    if m is not None and not (isinstance(m, Measurement) and m.ok):
        logger.warning("cache: baseline-%s.pkl is stale/invalid — remeasuring", key)
        return None
    return m


def save_baseline(key: str, m: Measurement) -> None:
    _save(CACHE_DIR / f"baseline-{key}.pkl", m)
