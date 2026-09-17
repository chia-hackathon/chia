"""Staging of the benchmark tree: what the agent may edit vs. what is sealed.

The loop reads the pristine benchmark sources from the local Saturn checkout at
call time. Every build is assembled from *these* bytes plus the single kernel
file the agent produced — so an agent edit anywhere else simply never reaches
the compiler. This is the sealing mechanism; the prompt only documents it.

Which files those are comes from the selected `Kernel` entry (`kernels.py`).
"""

from __future__ import annotations

import glob
import os

from constants import BENCHMARKS_DIR
from context import K


def _read(rel: str) -> bytes:
    return (BENCHMARKS_DIR / rel).read_bytes()


def sealed_files() -> dict[str, bytes]:
    """Every file the compiler needs EXCEPT the kernel, keyed by relative path.

    A glob may legitimately match the kernel file too (e.g. `vec-sgemv/*.S`);
    `build_inputs` overwrites it with the agent's bytes afterwards.
    """
    out: dict[str, bytes] = {}
    for pattern in K().sealed_globs:
        matches = glob.glob(str(BENCHMARKS_DIR / pattern))
        if not matches:
            raise FileNotFoundError(
                f"no files matched {pattern!r} under {BENCHMARKS_DIR}. "
                "Is benchmarks/env initialised? "
                "(git submodule update --init --depth 1 benchmarks/env)"
            )
        for path in matches:
            rel = os.path.relpath(path, BENCHMARKS_DIR)
            out[rel] = open(path, "rb").read()
    return out


def pristine_kernel() -> bytes:
    """The unmodified kernel — iteration 0's baseline and the agent's starting point."""
    return _read(K().kernel_rel)


def build_inputs(kernel: bytes) -> dict[str, bytes]:
    """Sealed collateral + *kernel*, ready for RiscvBuildNode.build_program."""
    files = sealed_files()
    files[K().kernel_rel] = kernel
    return files
