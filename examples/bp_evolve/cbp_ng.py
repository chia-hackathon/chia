"""``CbpNgNode``: build and run a HARCOM predictor under CBP-NG.

Tier 0 of the cascade, and the primitive CHIA does not have yet.  It follows
:class:`chia.simulators.champsim.ChampSimNode` deliberately: the same
placement-group lifecycle, the same "build returns bytes, run takes bytes"
split, so a build can happen once and the 168 trace runs can land anywhere in
the pool.  A CBP-NG binary is ~400 KB, well inside what the object store should
carry.

How a variant is selected is worth knowing, because it is the only part that is
not obvious from ``cbp-ng``'s own scripts.  ``branch_predictor.hpp`` ends with::

    #ifdef PREDICTOR
    using branch_predictor = PREDICTOR;
    #else
    using branch_predictor = tage<>;
    #endif

so a variant needs no edit to any tracked file: it is written to
``predictors/evolved_<id>.hpp`` (in ``predictors/`` so its own ``#include
"../cbp.hpp"`` resolves), pulled in with ``-include``, and named with
``-DPREDICTOR='<struct><>'``.  Two variants can therefore build concurrently in
one checkout without racing on a shared file, which they could not do if the
predictor were selected by editing ``branch_predictor.hpp``.

The compile flags are copied from ``cbp-ng/compile`` rather than referenced,
because ``-Werror`` is load-bearing here: the proposal scores designs under it,
so the loop must not quietly build with anything laxer.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass

import ray
from ray.util.placement_group import (
    placement_group as _placement_group,
    remove_placement_group as _remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from chia.base.ChiaFunction import ChiaFunction

from vfs import TraceCounters

# Verbatim from cbp-ng/compile.  -Werror is the point, not an accident.
CBP_CXX = os.environ.get("BPE_CBP_CXX", "g++")
CBP_STD = "-std=c++20"
CBP_OPT = "-O3"
CBP_WARNINGS = (
    "-Wall", "-Wextra", "-pedantic", "-Wold-style-cast", "-Werror",
    "-Wno-deprecated-declarations", "-Wno-mismatched-tags",
)

_VARIANT_ID = re.compile(r"\A[A-Za-z_][A-Za-z0-9_]*\Z")


@dataclass
class CbpBuildResult:
    """A compiled CBP-NG binary, or the diagnostics explaining why there isn't one.

    Attributes:
        binary: Raw executable bytes; empty on failure.
        struct_name: The predictor struct that was compiled in.
        variant_id: Identifier used for the header filename and the binary hash.
        header_path: Where the variant header was written on the build worker.
        success: True iff the compiler exited 0 within the timeout.
        returncode: Compiler exit code.
        build_duration_s: Wall-clock seconds.
        diagnostics: Filtered compiler errors on failure, empty on success.
            This is what a repair round is handed, so it is trimmed to the
            lines that name a file and a reason rather than the full template
            backtrace, which is mostly noise about HARCOM's internals.
    """
    binary: bytes
    struct_name: str
    variant_id: str
    header_path: str
    success: bool
    returncode: int
    build_duration_s: float
    diagnostics: str = ""


@dataclass
class CbpRunResult:
    """One trace's counters, or why they are missing.

    ``counters`` is None unless ``success``; every caller must check, because a
    timeout and a clean run differ only in this field.
    """
    trace_name: str
    success: bool
    returncode: int
    wall_s: float
    counters: TraceCounters | None = None
    timed_out: bool = False
    stderr_tail: str = ""


def _filter_cxx_diagnostics(output: str, max_bytes: int = 4000) -> str:
    """Keep the lines a repair round can act on.

    A HARCOM template error runs to hundreds of lines, almost all of them
    ``required from here`` frames inside ``harcom.hpp``.  The useful content is
    the lines naming the *predictor* file and the ``error:`` text.  Everything
    else is dropped, and the result truncated, so a repair prompt carries the
    mistake rather than the library's call stack.
    """
    keep = []
    for line in output.splitlines():
        if "error:" in line or "warning:" in line:
            keep.append(line)
        elif "required from" in line and "harcom.hpp" not in line:
            keep.append(line)
        elif line.strip().startswith("note:") and "harcom.hpp" not in line:
            keep.append(line)
    if not keep:
        keep = output.splitlines()[-40:]
    text = "\n".join(keep)
    return text if len(text) <= max_bytes else text[:max_bytes] + "\n... (truncated)"


class _PinnedChiaFn:
    """Bind a class-level ``@ChiaFunction`` to this node's placement.

    Mirrors ``ChampSimNode._PinnedChiaFn`` so the two nodes behave the same way
    under ``.options(...)``.
    """

    def __init__(self, fn, scheduling_opts: dict):
        self._fn = fn
        self._opts = dict(scheduling_opts)

    def options(self, **overrides):
        return self._fn.options(**{**self._opts, **overrides})

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def chia_remote(self, *args, **kwargs):
        return self._fn.options(**self._opts).chia_remote(*args, **kwargs)

    def chia_remote_blocking(self, *args, **kwargs):
        return self._fn.options(**self._opts).chia_remote_blocking(*args, **kwargs)


class CbpNgNode:
    """CBP-NG build/run primitives sharing one placement.

    Build and run do **not** need co-location -- the binary travels as bytes --
    but a placement group is still the natural handle for "the pool that has a
    cbp-ng checkout on disk", and builds do need that checkout.
    """

    _MEMBER_FNS = ("build_cbp", "run_cbp")
    _DEFAULT_BUNDLE = {"CPU": 1, "cbp_ng": 1.0}

    def __init__(
        self,
        placement_group=None,
        require_colocated: bool = False,
        *,
        bundle_index: int = 0,
        reserve_bundle: dict | None = None,
        pg_strategy: str = "STRICT_PACK",
        wait_for_pg: bool = True,
        pg_ready_timeout_s: float | None = None,
    ):
        self._owns_pg = False
        self._bundle_index = bundle_index

        if placement_group is not None:
            self._pg = placement_group
        elif require_colocated:
            bundle = reserve_bundle or dict(self._DEFAULT_BUNDLE)
            if bundle.get("cbp_ng", 0) < 1.0:
                raise ValueError(
                    f"reserve_bundle must provide cbp_ng>=1.0; got {bundle!r}")
            self._pg = _placement_group([bundle], strategy=pg_strategy)
            self._owns_pg = True
            self._bundle_index = 0
            if wait_for_pg:
                ray.get(self._pg.ready(), timeout=pg_ready_timeout_s)
        else:
            self._pg = None

        self._sched_opts = (
            {"scheduling_strategy": PlacementGroupSchedulingStrategy(
                placement_group=self._pg,
                placement_group_bundle_index=self._bundle_index)}
            if self._pg is not None else {}
        )

        for name in self._MEMBER_FNS:
            setattr(self, name,
                    _PinnedChiaFn(getattr(type(self), name), self._sched_opts))

    @property
    def placement_group(self):
        return self._pg

    @property
    def owns_placement_group(self) -> bool:
        return self._owns_pg

    @property
    def task_options(self) -> dict:
        return dict(self._sched_opts)

    def close(self) -> None:
        if self._owns_pg and self._pg is not None:
            _remove_placement_group(self._pg)
            self._pg = None
            self._owns_pg = False

    def __enter__(self) -> "CbpNgNode":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- build --------------------------------------------------------------

    @staticmethod
    @ChiaFunction(resources={"cbp_ng": 1.0})
    def build_cbp(
        cbp_root: str,
        source: str,
        struct_name: str,
        variant_id: str,
        *,
        template_args: str = "",
        timeout_s: int = 300,
        extra_flags: tuple[str, ...] = (),
    ) -> CbpBuildResult:
        """Compile ``source`` as the active predictor and return the binary.

        Args:
            cbp_root: Path to the cbp-ng checkout on this worker.
            source: Complete header defining ``struct_name``.
            struct_name: The predictor struct; must match what ``source``
                declares, since it becomes ``-DPREDICTOR``.
            variant_id: Filename-safe identifier, also used to keep concurrent
                builds in one checkout from colliding.
            template_args: Contents of the angle brackets in ``-DPREDICTOR``.
                Empty means ``<>``, i.e. every template parameter defaulted --
                which is why a generated predictor must give defaults for all
                of them.
            timeout_s: Compiler wall-clock limit.
            extra_flags: Appended after the standard flags. Cannot be used to
                relax warnings: they are re-appended last (see below).
        """
        if not _VARIANT_ID.match(variant_id):
            raise ValueError(f"variant_id must be a C identifier; got {variant_id!r}")
        if not _VARIANT_ID.match(struct_name):
            raise ValueError(f"struct_name must be a C identifier; got {struct_name!r}")

        header_rel = os.path.join("predictors", f"evolved_{variant_id}.hpp")
        header_path = os.path.join(cbp_root, header_rel)
        os.makedirs(os.path.dirname(header_path), exist_ok=True)
        with open(header_path, "w") as f:
            f.write(source)

        out_path = os.path.join(
            tempfile.gettempdir(), f"cbp_{variant_id}_{os.getpid()}")

        # Warning flags go last so an extra_flags entry cannot turn -Werror off
        # by preceding it; gcc takes the last of conflicting -W options.
        cmd = [
            CBP_CXX, CBP_STD, "-o", out_path, "cbp.cpp", "-lz", CBP_OPT,
            *extra_flags,
            *CBP_WARNINGS,
            "-include", header_rel,
            f"-DPREDICTOR={struct_name}<{template_args}>",
        ]

        started = time.time()
        timed_out = False
        try:
            proc = subprocess.run(
                cmd, cwd=cbp_root, capture_output=True, text=True,
                timeout=timeout_s)
            rc, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as e:
            timed_out = True
            rc = -1
            out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        wall = time.time() - started

        binary = b""
        success = (rc == 0) and not timed_out and os.path.exists(out_path)
        if success:
            with open(out_path, "rb") as f:
                binary = f.read()
            os.unlink(out_path)

        if timed_out:
            diagnostics = (f"compile timed out after {wall:.0f}s "
                           f"(limit {timeout_s}s); the design is probably "
                           f"instantiating far more hardware than it looks like")
        elif success:
            diagnostics = ""
        else:
            diagnostics = _filter_cxx_diagnostics(out + "\n" + err)

        # The header stays on disk only as long as the build needs it: leaving
        # one per variant would grow predictors/ without bound across a sweep,
        # and a stale evolved_*.hpp is a confusing thing to find in a checkout.
        try:
            os.unlink(header_path)
        except OSError:
            pass

        return CbpBuildResult(
            binary=binary, struct_name=struct_name, variant_id=variant_id,
            header_path=header_path, success=success, returncode=rc,
            build_duration_s=wall, diagnostics=diagnostics)

    # -- run ----------------------------------------------------------------

    @staticmethod
    @ChiaFunction(resources={"cbp_ng": 1.0})
    def run_cbp(
        binary: bytes,
        trace: str,
        *,
        trace_name: str | None = None,
        warmup_instructions: int = 1_000_000,
        simulation_instructions: int = 40_000_000,
        timeout_s: int = 1800,
    ) -> CbpRunResult:
        """Run one trace and parse the counter line CBP-NG prints to stdout.

        The binary is written to a content-hashed path so concurrent runs of
        the same variant on one worker share a single file rather than racing.
        """
        # A compression suffix, not split("."): trace basenames carry dots
        # inside the workload name -- gmsh-5.4132_0_trace.gz,
        # java16-specjbb-64k-ir1000.2_trace.gz -- and cutting at the first one
        # collapsed the official 168 into 155 labels.  The label is what the
        # counter line is keyed by, so a collision there makes two different
        # workloads indistinguishable to anything reading results back by name.
        base = os.path.basename(trace)
        for suffix in (".gz", ".xz", ".bz2", ".zst"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        name = trace_name or base

        digest = hashlib.sha256(binary).hexdigest()[:12]
        binary_path = os.path.join(tempfile.gettempdir(), f"cbp_bin_{digest}")
        if not os.path.exists(binary_path):
            tmp = f"{binary_path}.{os.getpid()}.tmp"
            with open(tmp, "wb") as f:
                f.write(binary)
            os.chmod(tmp, 0o755)
            os.replace(tmp, binary_path)   # atomic: no reader sees a partial file

        cmd = [binary_path, trace, name,
               str(warmup_instructions), str(simulation_instructions)]

        started = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout_s)
            rc, out, err, timed_out = proc.returncode, proc.stdout, proc.stderr, False
        except subprocess.TimeoutExpired:
            rc, out, err, timed_out = -1, "", "", True
        wall = time.time() - started

        counters = None
        parse_error = ""
        if rc == 0 and not timed_out:
            first = next((ln for ln in out.splitlines() if ln.strip()), "")
            try:
                counters = TraceCounters.from_out_line(first)
            except ValueError as e:
                parse_error = str(e)

        return CbpRunResult(
            trace_name=name,
            success=counters is not None,
            returncode=rc, wall_s=wall, counters=counters, timed_out=timed_out,
            stderr_tail=(parse_error or err[-2000:]))


def find_traces(trace_dir: str) -> list[str]:
    """Every CBP-NG trace under ``trace_dir``, sorted for reproducibility.

    Sorted because the inner-loop subset is a deterministic sample of this
    list: an unsorted directory listing would silently change which 24 traces
    every variant is scored on between runs, and cross-generation comparisons
    would stop meaning anything.
    """
    if not os.path.isdir(trace_dir):
        return []
    return sorted(
        os.path.join(trace_dir, f)
        for f in os.listdir(trace_dir)
        if f.endswith((".gz", ".trace", ".xz", ".zst"))
    )
