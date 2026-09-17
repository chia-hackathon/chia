"""Driver-side selection of "which kernel, which simulator config".

`loop.py` calls :func:`select` once from its CLI args; every other module asks
here instead of importing a hard-wired constant. The selection lives only on
the driver — every Ray remote in this loop takes its benchmark-specific inputs
as explicit arguments (file bytes, a compile command, an ELF name), so nothing
remote ever has to know what was selected.
"""

from __future__ import annotations

from constants import DEFAULT_KERNEL, SIM_CONFIG as DEFAULT_SIM_CONFIG
from kernels import Kernel, get_kernel

_kernel: Kernel = get_kernel(DEFAULT_KERNEL)
_config: str = DEFAULT_SIM_CONFIG


def select(kernel_name: str | None = None, config: str | None = None) -> Kernel:
    """Pick the kernel entry and simulator config for this run."""
    global _kernel, _config
    if kernel_name:
        _kernel = get_kernel(kernel_name)
    if config:
        _config = config
    return _kernel


def K() -> Kernel:
    """The selected kernel entry."""
    return _kernel


def CONFIG() -> str:
    """The selected simulator config (e.g. GENV256D128ShuttleConfig)."""
    return _config
