#!/usr/bin/env python3
"""Hardware design-point generator + legality pre-check for the AETHER OUTER loop.

The inner (kernel) loop keeps the hardware fixed.  The *outer* loop searches
hardware: Saturn's (VLEN, DLEN) and Gemmini's systolic-array / scratchpad shape.
This module is the piece that

  1. names a design point           -> ``HwPoint.name()``
  2. rejects illegal ones for free  -> ``legal(point)``   (no Chisel, no sbt)
  3. renders its Chisel config      -> ``render_config(point)``
  4. splices it into GemminiConfigs.scala -> ``inject(point, path)``
  5. proposes candidates            -> ``default_grid()`` / ``neighbors(point)``

Every legality rule below is transcribed from a ``require``/``assert`` in the
vendored RTL, cited by absolute-ish path under ``repos/`` + line number, so a
point that passes ``legal()`` should not blow up during Chisel elaboration.
A point that fails ``legal()`` definitely would -- so never spend a build on it.


ARTEFACT INVALIDATION (what a parameter change forces the loop to rebuild)
-------------------------------------------------------------------------
* ``dim`` / ``sp_capacity_kb`` / ``acc_capacity_kb`` / ``sp_banks`` /
  ``acc_banks`` / ``dataflow`` -- any Gemmini change rewrites
  ``gemmini_params.h`` (DIM, BANK_NUM, BANK_ROWS, ACC_ROWS are emitted straight
  from the elaborated config, see repos/gemmini/src/main/scala/gemmini/
  Controller.scala:29 + GemminiConfigs.scala:282-509).  That INVALIDATES:
    - ``libgemmini.so`` (spike's Gemmini golden model, built from a *static*
      gemmini_params.h) -> cospike lockstep will structurally disagree until it
      is rebuilt.  Task #4 owns that rebuild.
    - every Gemmini binary in gemmini-rocc-tests (tiling constants are baked in
      at compile time) -> the benchmark ELFs must be recompiled.
    - the Verilator simulator, obviously.
  Setting ``GEMMINI_ONLY_GENERATE_GEMMINI_H=1`` makes elaboration dump the
  header and exit immediately -- the cheap way to refresh it (Controller.scala:30).
* ``vlen`` / ``dlen`` -- pure Saturn/RVV change.  libgemmini and gemmini_params.h
  are UNAFFECTED, but:
    - the RVV kernel roofline moves (VLEN sets vector-register bytes and hence
      the achievable LMUL/tiling; DLEN sets peak lanes = bytes/cycle), so
      inner-loop kernel sources tuned for one (VLEN,DLEN) are usually not
      optimal for another and must be re-optimized;
    - the SoC bus width / tile beat bytes are derived from DLEN (see
      ``sbus_width``/``beat_bytes`` below), so DLEN changes also change the
      memory system, not just the vector unit;
    - and because Gemmini's ``dma_buswidth`` follows the system bus, a DLEN of
      256 or more DOES perturb gemmini_params.h (MAX_BLOCK_LEN) after all --
      i.e. dlen in (256, 512) invalidates libgemmini.so too.
* ``saturn_params`` (genParams/refParams/...) -- microarchitecture preset only;
  no software artefact changes, RTL rebuild only.
"""

from __future__ import annotations

import importlib.util

import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

# --- reuse the docker-side baseline literal (single source of truth) --------
# docker/add_coexist_config.py must stay standalone (it is COPYed into the image
# with no loop/ around it), so IT keeps the literal baseline config text and we
# import it here rather than re-typing it.  self_test() asserts that our generic
# template reproduces that literal byte-for-byte.
_AETHER_ROOT = Path(os.environ.get("AETHER_ROOT", Path(__file__).resolve().parent.parent))
_ADD_COEXIST = _AETHER_ROOT / "docker" / "add_coexist_config.py"


def _load_add_coexist():
    spec = importlib.util.spec_from_file_location("add_coexist_config", _ADD_COEXIST)
    if spec is None or spec.loader is None:  # pragma: no cover
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    _coexist = _load_add_coexist()
except Exception:  # pragma: no cover - only if docker/ is missing
    _coexist = None


# --- fixed (not searched) parts of the design point -------------------------
# Gemmini element types.  defaultConfig is int8 in/weight, int32 acc
# (repos/gemmini/src/main/scala/gemmini/Configs.scala:20-167).  libgemmini.so is
# built for exactly these, so the search does NOT touch them.
INPUT_WIDTH = 8
ACC_WIDTH = 32

SATURN_PRESETS = ("minParams", "refParams", "dspParams", "genParams")
_PRESET_PREFIX = {"minParams": "MIN", "refParams": "REF", "dspParams": "DSP", "genParams": "GEN"}
DATAFLOWS = ("BOTH", "WS", "OS")

# Search-space policy caps (NOT RTL requires -- they bound build time / area).
MAX_VLEN = 1024          # README.md:15 documents Zvl64/128/256/512/1024
MAX_DLEN = 512           # mLen (= dLen here) must be <= 512, Parameters.scala:365
MAX_DIM = 32             # largeChipConfig's mesh is 32x32; bigger is unbuildable here
MIN_DIM = 2              # GemminiConfigs.scala:196


def _is_pow2(n: int) -> bool:
    return isinstance(n, int) and n >= 1 and (n & (n - 1)) == 0


# ---------------------------------------------------------------------------
# The design point
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HwPoint:
    """One hardware design point (Saturn + Gemmini on a Shuttle tile).

    Defaults == the already-built AETHER baseline ``GENV256D128GemminiShuttleConfig``
    (Saturn VLEN=256/DLEN=128 genParams + gemmini.DefaultGemminiConfig).
    """

    # --- Saturn (the proposal searches only these two) ---
    vlen: int = 256
    dlen: int = 128
    saturn_params: str = "genParams"      # microarch preset, held fixed by default

    # --- Gemmini (fields that GemminiArrayConfig plainly exposes) ---
    dim: int = 16                          # meshRows == meshColumns, tile{Rows,Columns}=1
    sp_capacity_kb: int = 256              # CapacityInKilobytes(sp_capacity)
    acc_capacity_kb: int = 64              # CapacityInKilobytes(acc_capacity)
    dataflow: str = "BOTH"                 # gemmini.Dataflow.{BOTH,WS,OS}
    sp_banks: int = 4
    acc_banks: int = 2

    # ---- derived SoC knobs (not searched; follow the Saturn config idiom in
    # repos/saturn/chipyard/SaturnConfigs.scala: DLEN<=128 -> 128b sbus/16B beats,
    # DLEN=256 -> 256b/32B, DLEN=512 -> 256b/64B) ----
    @property
    def sbus_width(self) -> int:
        return min(max(self.dlen, 128), 256)

    @property
    def beat_bytes(self) -> int:
        return max(self.dlen, 128) // 8

    @property
    def dma_buswidth(self) -> int:
        """Gemmini dma_buswidth follows the system bus (default is 128)."""
        return self.sbus_width

    # ---- Gemmini derived geometry (GemminiConfigs.scala:106-114, 224-231) ----
    @property
    def sp_width(self) -> int:
        return self.dim * INPUT_WIDTH

    @property
    def sp_bank_entries(self) -> float:
        # kb * 1024 * 8 / (sp_banks * sp_width)   -- kept exact (may be fractional)
        return self.sp_capacity_kb * 1024 * 8 / (self.sp_banks * self.sp_width)

    @property
    def acc_bank_entries(self) -> float:
        # kb * 1024 * 8 / (acc_banks * DIM * accType.getWidth)
        return self.acc_capacity_kb * 1024 * 8 / (self.acc_banks * self.dim * ACC_WIDTH)

    @property
    def sp_rows(self) -> float:
        return self.sp_banks * self.sp_bank_entries

    @property
    def acc_rows(self) -> float:
        return self.acc_banks * self.acc_bank_entries

    # ---- identity ----
    def is_baseline(self) -> bool:
        return self == BASELINE

    def name(self) -> str:
        """Stable id, e.g. ``GENV256D128_G16SP256ACC64``.

        Non-default bank counts / dataflow are appended so the id stays unique:
        ``..._WS``, ``..._B4x2`` (only when they differ from the defaults).
        """
        pre = _PRESET_PREFIX.get(self.saturn_params, self.saturn_params)
        s = f"{pre}V{self.vlen}D{self.dlen}_G{self.dim}SP{self.sp_capacity_kb}ACC{self.acc_capacity_kb}"
        if (self.sp_banks, self.acc_banks) != (4, 2):
            s += f"_B{self.sp_banks}x{self.acc_banks}"
        if self.dataflow != "BOTH":
            s += f"_{self.dataflow}"
        return s

    def scala_base(self) -> str:
        """Base of the emitted Scala class names (``<base>Config`` / ``<base>CosimConfig``).

        The baseline keeps its historical name so the already-built docker image
        and docker/AetherCosimDockerfile's grep checks keep working.
        """
        if self.is_baseline():
            return "GENV256D128GemminiShuttle"
        return self.name().replace("_", "") + "GemminiShuttle"

    def config_class(self) -> str:
        return self.scala_base() + "Config"

    def cosim_class(self) -> str:
        return self.scala_base() + "CosimConfig"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["name"] = self.name()
        d["config_class"] = self.config_class()
        d["cosim_class"] = self.cosim_class()
        d["sbus_width"] = self.sbus_width
        d["beat_bytes"] = self.beat_bytes
        return d

    @staticmethod
    def from_dict(d: dict) -> "HwPoint":
        fields = HwPoint.__dataclass_fields__  # type: ignore[attr-defined]
        return HwPoint(**{k: v for k, v in d.items() if k in fields})

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.name()


BASELINE = HwPoint()


# ---------------------------------------------------------------------------
# Legality pre-check (cheap; runs before any Chisel build)
# ---------------------------------------------------------------------------
def legal(point: HwPoint) -> tuple[bool, str]:
    """Return ``(ok, reason)``.  ``reason`` is "" when ok.

    Sources (all paths relative to ``repos/``):
      saturn/src/main/scala/common/Parameters.scala:363  require(dLen >= 64)
      saturn/src/main/scala/common/Parameters.scala:364  require(isPow2(dLen))
      saturn/src/main/scala/common/Parameters.scala:365  require(64 <= mLen <= 512)
      saturn/src/main/scala/common/Parameters.scala:366  require(isPow2(mLen))
          (WithShuttleVectorUnit sets mLen = dLen unless overridden --
           saturn/src/main/scala/shuttle/Configs.scala:24-26, so the mLen bounds
           become DLEN bounds here.)
      saturn/src/main/scala/backend/Backend.scala:39     require(vLen >= 64)
      saturn/src/main/scala/backend/Backend.scala:41     require(vLen >= dLen)
      saturn/src/main/scala/backend/Backend.scala:42     require(vLen % dLen == 0)
      saturn/README.md:15                                Zvl64..Zvl1024 -> VLEN <= 1024
      gemmini/src/main/scala/gemmini/GemminiConfigs.scala:196  DIM >= 2
      gemmini/src/main/scala/gemmini/GemminiConfigs.scala:275  square mesh
      gemmini/src/main/scala/gemmini/GemminiConfigs.scala:277  isPow2(DIM)
      gemmini/src/main/scala/gemmini/GemminiConfigs.scala:273  isPow2(sp_bank_entries)
      gemmini/src/main/scala/gemmini/GemminiConfigs.scala:274  sp_bank_entries % DIM == 0
      gemmini/src/main/scala/gemmini/GemminiConfigs.scala:278  acc_bank_entries % DIM == 0
      gemmini/src/main/scala/gemmini/GemminiConfigs.scala:246-249 I/O_TILE_BYTE_WIDTH pow2
      gemmini/src/main/scala/gemmini/GemminiConfigs.scala:257  USABLE_SP_TILES >= TOTAL_ACC_TILES
    """
    p = point

    # ---------------- Saturn ----------------
    if p.saturn_params not in SATURN_PRESETS:
        return False, f"saturn_params {p.saturn_params!r} not in {SATURN_PRESETS}"
    if not _is_pow2(p.vlen):
        return False, f"vlen={p.vlen} not a power of two"
    if not _is_pow2(p.dlen):
        return False, f"dlen={p.dlen} not a power of two (Parameters.scala:364)"
    if p.dlen < 64:
        return False, f"dlen={p.dlen} < 64 (Parameters.scala:363)"
    if p.dlen > MAX_DLEN:
        return False, f"dlen={p.dlen} > {MAX_DLEN}; mLen(=dLen) must be <= 512 (Parameters.scala:365)"
    if p.vlen < 64:
        return False, f"vlen={p.vlen} < 64 (Backend.scala:39)"
    if p.vlen > MAX_VLEN:
        return False, f"vlen={p.vlen} > {MAX_VLEN} (only Zvl64..Zvl1024 exist, saturn/README.md:15)"
    if p.vlen < p.dlen:
        return False, f"vlen={p.vlen} < dlen={p.dlen} (Backend.scala:41)"
    if p.vlen % p.dlen != 0:
        return False, f"vlen={p.vlen} not a multiple of dlen={p.dlen} (Backend.scala:42)"

    # ---------------- Gemmini ----------------
    if p.dataflow not in DATAFLOWS:
        return False, f"dataflow {p.dataflow!r} not in {DATAFLOWS}"
    if not _is_pow2(p.dim):
        return False, f"dim={p.dim} not a power of two (GemminiConfigs.scala:277)"
    if p.dim < MIN_DIM:
        return False, f"dim={p.dim} < {MIN_DIM} (GemminiConfigs.scala:196)"
    if p.dim > MAX_DIM:
        return False, f"dim={p.dim} > {MAX_DIM} (search-space cap: area/build time)"
    if p.sp_banks < 1 or not _is_pow2(p.sp_banks):
        return False, f"sp_banks={p.sp_banks} must be a power of two >= 1"
    if p.acc_banks < 1 or not _is_pow2(p.acc_banks):
        return False, f"acc_banks={p.acc_banks} must be a power of two >= 1"
    if p.sp_capacity_kb < 1 or p.acc_capacity_kb < 1:
        return False, "capacities must be >= 1 KiB"

    spe = p.sp_bank_entries
    if spe != int(spe):
        return False, (f"sp {p.sp_capacity_kb}KB / {p.sp_banks} banks / DIM {p.dim} "
                       f"gives fractional sp_bank_entries={spe}")
    spe = int(spe)
    if not _is_pow2(spe):
        return False, f"sp_bank_entries={spe} not a power of two (GemminiConfigs.scala:273)"
    if spe % p.dim != 0:
        return False, f"sp_bank_entries={spe} not a multiple of DIM={p.dim} (GemminiConfigs.scala:274)"

    ace = p.acc_bank_entries
    if ace != int(ace):
        return False, (f"acc {p.acc_capacity_kb}KB / {p.acc_banks} banks / DIM {p.dim} "
                       f"gives fractional acc_bank_entries={ace}")
    ace = int(ace)
    if ace % p.dim != 0:
        return False, f"acc_bank_entries={ace} not a multiple of DIM={p.dim} (GemminiConfigs.scala:278)"

    # cisc-gemmini tile byte widths must be powers of two (GemminiConfigs.scala:242-249)
    cisc_dim = p.dim // 2
    if cisc_dim < 1:
        return False, f"dim={p.dim} too small: cisc_dim = DIM/2 == 0 (GemminiConfigs.scala:213)"
    i_tile = p.dim * ((INPUT_WIDTH + cisc_dim - 1) // cisc_dim)
    o_tile = p.dim * ((ACC_WIDTH + cisc_dim - 1) // cisc_dim)
    if not _is_pow2(i_tile):
        return False, f"I_TILE_BYTE_WIDTH={i_tile} not a power of two (GemminiConfigs.scala:246)"
    if not _is_pow2(o_tile):
        return False, f"O_TILE_BYTE_WIDTH={o_tile} not a power of two (GemminiConfigs.scala:248)"

    # scratchpad must hold at least as many DIM-tiles as the accumulator, +2
    usable_sp_tiles = int(p.sp_rows) // p.dim - 2
    total_acc_tiles = int(p.acc_rows) // p.dim
    if usable_sp_tiles < total_acc_tiles:
        return False, (f"USABLE_SP_TILES={usable_sp_tiles} < TOTAL_ACC_TILES={total_acc_tiles} "
                       f"(GemminiConfigs.scala:257); grow sp_capacity_kb or shrink acc_capacity_kb")

    return True, ""


# ---------------------------------------------------------------------------
# Chisel config rendering
# ---------------------------------------------------------------------------
_HEADER = """

// ------------------------------------------------------------------
// AETHER: Saturn RVV 1.0 (vector port) + Gemmini (RoCC) on one Shuttle tile
// ------------------------------------------------------------------"""

_COSIM_COMMENT = """// Same, plus cospike lockstep co-simulation (run Gemmini workloads with
// +cospike-extension=gemmini; plain RVV workloads need no extra plusarg)."""


def _gemmini_fragment(p: HwPoint) -> str:
    """The ``new gemmini.DefaultGemminiConfig...`` fragment (no trailing ' ++')."""
    if (p.dim, p.sp_capacity_kb, p.acc_capacity_kb, p.dataflow,
            p.sp_banks, p.acc_banks, p.dma_buswidth) == (16, 256, 64, "BOTH", 4, 2, 128):
        # exactly GemminiConfigs.defaultConfig -- the one libgemmini.so matches
        return "  new gemmini.DefaultGemminiConfig"
    return (
        "  new gemmini.DefaultGemminiConfig(gemminiConfig = gemmini.GemminiConfigs.defaultConfig.copy(\n"
        f"    meshRows = {p.dim}, meshColumns = {p.dim},\n"
        f"    sp_capacity = gemmini.CapacityInKilobytes({p.sp_capacity_kb}),\n"
        f"    acc_capacity = gemmini.CapacityInKilobytes({p.acc_capacity_kb}),\n"
        f"    sp_banks = {p.sp_banks}, acc_banks = {p.acc_banks},\n"
        f"    dataflow = gemmini.Dataflow.{p.dataflow},\n"
        f"    dma_buswidth = {p.dma_buswidth}))"
    )


def render_config(point: HwPoint) -> str:
    """Chisel source text for ``point``'s two config classes (plain + cosim).

    Rendered in exactly the style docker/add_coexist_config.py uses; for
    ``BASELINE`` the output is byte-identical to that script's literal.
    """
    p = point
    vec = (f"  new saturn.shuttle.WithShuttleVectorUnit"
           f"({p.vlen}, {p.dlen}, saturn.common.VectorParams.{p.saturn_params}) ++")
    gem = _gemmini_fragment(p) + " ++"
    sbus = f"  new chipyard.config.WithSystemBusWidth({p.sbus_width}) ++"
    beat = f"  new shuttle.common.WithShuttleTileBeatBytes({p.beat_bytes}) ++"
    return "\n".join([
        _HEADER,
        f"class {p.config_class()} extends Config(",
        vec,
        gem,
        sbus,
        beat,
        "  new shuttle.common.WithNShuttleCores(1) ++",
        "  new chipyard.config.AbstractConfig)",
        "",
        _COSIM_COMMENT,
        f"class {p.cosim_class()} extends Config(",
        "  new chipyard.harness.WithCospike ++",
        "  new chipyard.config.WithTraceIO ++",
        vec,
        gem,
        sbus,
        "  new shuttle.common.WithShuttleDebugROB ++",
        beat,
        "  new shuttle.common.WithNShuttleCores(1) ++",
        "  new chipyard.config.AbstractConfig)",
        "",
    ])


def injection_patch(point: HwPoint) -> tuple[str, str]:
    """``(mark, text)`` -- append ``text`` to GemminiConfigs.scala unless ``mark`` is in it."""
    return f"class {point.config_class()} extends Config(", render_config(point)


def inject(point: HwPoint, gemmini_configs_scala_path) -> bool:
    """Idempotently add ``point``'s config classes to GemminiConfigs.scala.

    Returns True if the file was modified, False if the classes were already
    there.  Delegates to docker/add_coexist_config.py's ``append_config`` so the
    anchor check / append behaviour has exactly one implementation.
    """
    mark, text = injection_patch(point)
    path = str(gemmini_configs_scala_path)
    if _coexist is not None:
        return _coexist.append_config(path, text, mark)
    # fallback (docker/ not reachable): same semantics, no anchor check
    src = Path(path).read_text()
    if mark in src:
        return False
    Path(path).write_text(src.rstrip("\n") + "\n" + text)
    return True


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------
def default_grid() -> list[HwPoint]:
    """A small, all-legal starting grid: 3 Saturn shapes x 3 Gemmini shapes.

    Saturn axis moves the roofline (VLEN = register bytes -> tiling, DLEN = peak
    lanes/cycle).  Gemmini axis trades array size against on-chip capacity.
    """
    saturn = [(256, 128), (512, 128), (512, 256)]
    gemmini = [
        (16, 256, 64),    # == GemminiConfigs.defaultConfig (libgemmini-compatible)
        (8, 256, 64),     # small array, same memory
        (32, 128, 64),    # big array, less scratchpad
    ]
    out = []
    for vlen, dlen in saturn:
        for dim, sp, acc in gemmini:
            out.append(HwPoint(vlen=vlen, dlen=dlen, dim=dim,
                               sp_capacity_kb=sp, acc_capacity_kb=acc))
    return out


_NEIGHBOR_AXES = ("vlen", "dlen", "dim", "sp_capacity_kb", "acc_capacity_kb")


def neighbors(point: HwPoint, include_dataflow: bool = True) -> list[HwPoint]:
    """One-parameter step mutations of ``point`` (halve/double each numeric axis,
    plus the other dataflows).  Only legal points are returned; order is stable."""
    cands: list[HwPoint] = []
    for axis in _NEIGHBOR_AXES:
        cur = getattr(point, axis)
        for nxt in (cur // 2, cur * 2):
            if nxt >= 1 and nxt != cur:
                cands.append(replace(point, **{axis: nxt}))
    if include_dataflow:
        for df in DATAFLOWS:
            if df != point.dataflow:
                cands.append(replace(point, dataflow=df))

    seen, out = set(), []
    for c in cands:
        ok, _ = legal(c)
        if ok and c.name() not in seen:
            seen.add(c.name())
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def self_test() -> int:
    rc = 0

    # 1) the generic template must reproduce the docker script's baseline literal
    if _coexist is None:
        print("WARN: could not import docker/add_coexist_config.py; skipping baseline check")
    else:
        got, want = render_config(BASELINE), _coexist.BASELINE_CONFIG_TEXT
        if got == want:
            print("OK   render_config(BASELINE) == add_coexist_config.BASELINE_CONFIG_TEXT")
        else:
            rc = 1
            print("FAIL baseline render drifted from docker/add_coexist_config.py")
            import difflib
            sys.stdout.writelines(difflib.unified_diff(
                want.splitlines(True), got.splitlines(True), "docker", "hwconfig"))
        if _coexist.MARK != f"class {BASELINE.config_class()}":
            rc = 1
            print("FAIL baseline class name != add_coexist_config.MARK")

    # 2) round-trip
    if HwPoint.from_dict(BASELINE.to_dict()) != BASELINE:
        rc = 1
        print("FAIL to_dict/from_dict round-trip")

    # 3) grid table
    hdr = f"{'name':<34} {'sbus':>5} {'beat':>5} {'spBE':>6} {'accBE':>6}  legal  reason"
    print("\n== default_grid() ==")
    print(hdr)
    print("-" * len(hdr))
    for p in default_grid():
        ok, why = legal(p)
        print(f"{p.name():<34} {p.sbus_width:>5} {p.beat_bytes:>5} "
              f"{p.sp_bank_entries:>6.0f} {p.acc_bank_entries:>6.0f}  "
              f"{'yes' if ok else 'NO ':<5}  {why}")
        if not ok:
            rc = 1

    # 4) known-illegal points must be rejected
    print("\n== illegal-point checks ==")
    bad = [
        (HwPoint(vlen=128, dlen=256), "dlen > vlen"),
        (HwPoint(vlen=256, dlen=32), "dlen < 64"),
        (HwPoint(vlen=256, dlen=96), "dlen not pow2"),
        (HwPoint(vlen=384, dlen=128), "vlen not pow2"),
        (HwPoint(vlen=2048, dlen=128), "vlen > 1024"),
        (HwPoint(dim=24), "dim not pow2"),
        (HwPoint(dim=64), "dim > cap"),
        (HwPoint(sp_capacity_kb=192), "sp_bank_entries not pow2"),
        (HwPoint(sp_capacity_kb=8, acc_capacity_kb=64), "sp too small vs acc"),
        (HwPoint(dataflow="XX"), "bad dataflow"),
    ]
    for p, expect in bad:
        ok, why = legal(p)
        print(f"{'rejected' if not ok else 'ACCEPTED':<9} [{expect:<24}] {why}")
        if ok:
            rc = 1

    # 5) neighbors
    ns = neighbors(BASELINE)
    print(f"\n== neighbors(BASELINE) -> {len(ns)} legal ==")
    for n in ns:
        print("  " + n.name())
    if not ns:
        rc = 1

    # 6) a non-baseline render, eyeballed
    demo = HwPoint(vlen=512, dlen=256, dim=32, sp_capacity_kb=128, acc_capacity_kb=64, dataflow="WS")
    print(f"\n== render_config({demo.name()}) ==")
    print(render_config(demo))
    print("PASS" if rc == 0 else "FAILURES ABOVE")
    return rc


if __name__ == "__main__":
    sys.exit(self_test())
