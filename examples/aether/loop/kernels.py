"""Kernel registry — every benchmark the loop knows how to optimize.

One entry per (kernel, harness). The entry owns EVERYTHING benchmark-specific:
which single file the agent may edit, what else is sealed, how to read the
objective out of the simulator log, what "correct" means, and the prompt text
that only makes sense for that kernel. `constants.py` keeps only global knobs
and `loop.py --kernel NAME` picks the entry.

Adding a kernel = adding a `Kernel(...)` below. No other file changes.

Compilation is generic (it mirrors saturn/benchmarks/Makefile's
compile_template): every `*.c` / `*.S` in the benchmark directory plus the
shared `common/` sources. So an entry never lists source files — only the one
file the agent owns and the ones that are sealed against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# ROOFLINE METHOD — how every `roofline_cycles=` below was derived
# ---------------------------------------------------------------------------
# A roofline is a HARD LOWER BOUND. If a measured run lands below it, the
# NUMBER is wrong, not the measurement. Every `llama-*` entry below was
# recomputed with the one method written out here (2026-09-07, after Round 1
# of the optimization loop falsified several earlier guesses).
#
# ## Machine model — Saturn on GENV256D128GemminiShuttleConfig
#
# VLEN = 256 bit (architectural register width), DLEN = 128 bit (physical
# datapath). Cycles are counted in DLEN, never in VLEN.
#
#   Per cycle, per pipe, the datapath moves 128 bits = 16 B:
#       32-bit elements (fp32 / int32)  ->  4 lanes/cycle
#       16-bit elements (fp16 / int16)  ->  8 lanes/cycle
#        8-bit elements (int8)          -> 16 lanes/cycle
#
#   ARITHMETIC
#     fp32 multiply / add / FMA                        4 elem/cycle
#     fp16 multiply / add / FMA                        8 elem/cycle
#     any int32-destination op (`vwadd.wv`, `vsext.vf4`,
#       `vfcvt.f.x.v`, `vfwcvt.f.x.v`, `vfmacc.vv`)    4 elem/cycle
#     any int16-destination op (`vsext.vf2`, `vwmul.vv`)  8 elem/cycle
#     int8 x int8 widening MAC into int16 (`vwmacc`)   8 MAC/cycle   <-- KEY
#
#     The int8 MAC rate is **8/cycle, not 4/cycle**. Accumulating an int8 dot
#     product straight into int32 is 4/cycle, but you do not have to:
#     accumulate PAIRS of products into an int16 vector (`vwmul.vx` then
#     `vwmacc.vx`, safe because `2*127*128 = 32,512 < 32,767`) and widen into
#     the int32 accumulator once per pair. Round 1's `llama-attn-pv-int8`
#     winner did exactly this and hit 4,853 cycles on a shape whose
#     int32-accumulate "floor" had been claimed to be 8,192. That is how the
#     old numbers broke. So every int8 dot product's COMPUTE floor here is
#     `MACs / 8`.
#     The periodic widen is real work (`MACs/(4*W)` extra cycles when you
#     widen every W products, and W = 2 is the exact-arithmetic limit), but it
#     lands on a different pipe and chains, so it is EXCLUDED from the floor
#     and only discussed per kernel. Excluding it keeps the floor strictly
#     below anything achievable, which is what a floor is for.
#
#   TRANSCENDENTALS — there is NO hardware exp, divide, reciprocal or sqrt
#     unit. Cost them by instruction count of a `vfrec7`/`vfrsqrt7`-seeded
#     sequence. Assumed counts (per element, fp32, enough for a 1e-4 relative
#     gate) — these are ASSUMPTIONS and are restated in each kernel's notes:
#       exp(x)      8 vector ops   (one FMA that fuses `*log2e` with the
#                                   magic-number bias; recover the integer
#                                   part; one `vfnmsac` range reduction; a
#                                   4-term polynomial; build `2^n` by an
#                                   integer shift + add on the bit pattern,
#                                   i.e. NO second multiply)
#       1/x         3 vector ops   (`vfrec7` = 7 bits, one Newton-Raphson
#                                   step = `vfnmsub` + `vfmul` -> ~14 bits)
#       rsqrt(x)    3 vector ops   (`vfrsqrt7` + one NR step)
#     A full-precision `vfdiv`/`vfsqrt` never enters a floor: it is strictly
#     slower than the sequence above, so assuming the sequence stays honest.
#     Round 1 confirmed these are the right shapes — the softmax winner's exp
#     is ~9 ops and the silu winner's is ~18 (it needs a wider polynomial),
#     both above the 8 assumed here, so 8 is a genuine lower bound.
#
#   REDUCTIONS (`vredsum`, `vfredusum`, `vfredmax`) are NOT free and are
#     estimated SEPARATELY from the streaming arithmetic:
#       ~4 cycles of pipe occupancy per reduction when many are in flight and
#       their latencies overlap; ~8-10 if they serialise on a scalar read-back.
#     Floors use the optimistic 4, i.e. a kernel with R reductions carries
#     `+4R` cycles. Round 1 showed the truly fast kernels ELIMINATE the
#     reductions instead (segmented load + transpose + tree-add, or arranging
#     the accumulation to run over the OUTER loop so R = 0), so `+4R` is a
#     soft term, never the binding one.
#
#   MEMORY: the load pipe, the store pipe and each arithmetic pipe are
#     independent DLEN-wide sequencers — **16 B/cycle each, concurrently**.
#     Therefore
#         memory_floor = max(bytes_read, bytes_written) / 16
#     and NOT their sum. (Summing overstates the bound, and an overstated
#     roofline is exactly the bug this method exists to remove. The summed
#     figure is quoted alongside where the two are close.)
#     Bytes counted are MINIMUM traffic: every input byte read once, every
#     output byte written once, plus any re-read the kernel cannot avoid
#     because the data does not fit in the register file (32 architectural
#     registers = 4 groups at LMUL=8 = 1 KiB in total).
#
#   roofline_cycles = max(compute floor, memory floor)
#
#   BOTH numbers are always listed in `notes` so a reader can see which binds.
#
# ## Gemmini entries
#
#   Array DIM = 16 -> `DIM*DIM = 256` MAC/cycle peak. Scratchpad 256 KiB
#   (`BANK_NUM=4` banks), accumulator 64 KiB (2 banks). Gemmini's DMA port and
#   the sbus are 128-bit = 16 B/cycle, but **DRAM is behind the mbus, which is
#   8 B/cycle** (2026-09-09: rocket-chip `BaseSubsystemConfig` sets
#   `MemoryBusKey => MemoryBusParams(beatBytes = 8)`; chipyard's
#   `AbstractConfig` and this SoC's chain in docker/add_coexist_config.py
#   override only the sbus width and the tile beat bytes). Weights are always
#   DRAM traffic, so:
#       compute floor = MACs / 256
#       memory floor  = max(bytes in, bytes out) / 8      <-- 8, not 16
#   The MEASURED cold-DRAM rate is 7.0-7.1 B/cycle and the measured mvin-only
#   ceiling at N=1 is 7.94 B/cycle, i.e. 99% of the 8 B/cycle link. Until
#   2026-09-09 these entries divided by 16 and therefore advertised twice the
#   headroom that exists; three optimization rounds chased it.
#
# ## Round-1 sanity constraint (2026-09-07, model Fable 5.1)
#
#   No `llama-*` entry may carry a roofline above its best MEASURED cycles:
#
#     kernel                    baseline    best   roofline now
#     llama-softmax               21,906   1,752          1,472
#     llama-attn-scores-int8      51,280   8,574          6,144
#     llama-attn-pv-int8          22,856   4,853          4,096
#     llama-silu-mul             361,260  66,811         30,720
#     llama-q8-gemv-gemmini-n1   150,339 132,653         65,536
#
# ---------------------------------------------------------------------------


# Sealed collateral shared by every Saturn benchmark: the runtime, the ara
# helpers, the linker script, the encoding header.
COMMON_SEALED_GLOBS = [
    "common/*.c", "common/*.S", "common/*.h", "common/*.ld",
    "common/ara/*.c", "common/ara/*.h",
    "env/encoding.h",
]

# Per-target hardware briefing spliced into the prompt as ${TARGET_NOTES}.
TARGET_NOTES: dict[str, str] = {
    "saturn": """\
- **Saturn** RVV 1.0 vector unit, `VLEN = 256` bits, `DLEN = 128` bits.
  VLEN is the architectural register width; DLEN is the physical datapath.
  **A full-width (LMUL=1, e32) vector op occupies its execution pipe for
  `VLEN/DLEN = 2` beats; at LMUL=8 it is 16 beats.** Beats, not instructions,
  are what `mcycle` counts. Sizing work in units of DLEN, not VLEN, is what
  separates a good schedule from a bad one.
- Host core is **Shuttle**, a superscalar in-order core. Scalar work is not
  free in principle, but it is usually already hidden behind 16-beat vector ops.
- These facts are from Saturn's own documentation, not guesswork:
  - **The load pipe, the store pipe and each arithmetic pipe are all DLEN bits
    per cycle, independently.** They have separate sequencers with independent
    issue queues that may slip against each other, so a load and an FMA *can*
    be busy in the same cycle. Overlapping loads with arithmetic is the game.
  - **Full chaining, zero dead time**, at DLEN/element-group granularity, and
    it works across unit types (arithmetic can chain off memory and vice-versa).
  - **Chime length = `LMUL * VLEN/DLEN`.** At `LMUL=8, e32` that is 16 cycles.
    The FMA pipe is only 4 stages deep, so at LMUL=8 a *dependent* FMA chain is
    already fully hidden — accumulator serialization is NOT a bottleneck there.
    (It would be at LMUL=1, where chime=2 < 4.)
  - **`vsetvli` on Shuttle is nearly free**: Shuttle bypasses `vtype`/`vl` at
    decode, so a `vset` only costs the following vector instruction its slot in
    the same packet (~1 cycle). It is *Rocket* that pays a 2-cycle bubble.
  - **Strided and indexed (non-segmented) accesses generate only ONE element
    address per cycle** — 4x slower than a unit-stride load at DLEN=128.
- `-ffast-math` is on; float reassociation is permitted, but the benchmark's
  own self-check still has to accept the result.""",

    "gemmini": """\
- **Gemmini** is a systolic-array accelerator attached over RoCC, driven
  entirely from software by the intrinsics in `include/gemmini.h`. There is no
  compiler auto-vectorization here: *every* accelerator instruction is one you
  emitted, so the schedule in that header IS the hardware's performance.
- Default shape (`DefaultGemminiConfig`, and what `gemmini_params.h` says):
  `DIM = 16` (a 16x16 int8 systolic array), `elem_t = int8_t`,
  `acc_t = int32_t`, scratchpad `BANK_NUM = 4` banks x `BANK_ROWS = 4096`
  rows, accumulator `ACC_ROWS = 1024`, `MAX_BYTES = 64` per DMA request.
  **Never hard-code these numbers** — read them from the macros, because the
  header is regenerated from the elaborated hardware config and the loop can
  be pointed at a different Gemmini.
- The units of work are 16x16 tiles. The array retires one 16-wide row per
  cycle once fed, so a `DIM x DIM x DIM` matmul is ~`DIM` cycles of compute
  but needs `2*DIM` rows moved in — **mvin/mvout bandwidth, not the array, is
  the usual bottleneck**, and `MAX_BYTES = 64` means one mvin request carries
  only 64 bytes.
- The three levers that actually move cycles:
  1. **Tiling** (`tiled_matmul_auto` picks `tile_I/J/K` to fit the scratchpad
     and accumulator). Bad tiling spills a tile to DRAM and re-mvins it.
  2. **Loop order / weight stationarity** (`WS`): keeping B resident across
     the I loop is the whole point; anything that re-mvins B per output tile
     loses badly.
  3. **Overlap**: Gemmini's queues let mvin, compute and mvout run
     concurrently. `gemmini_fence()` and over-eager dependency tracking
     serialize them.
- Host core is **Shuttle** (superscalar in-order). RoCC instructions issue
  from it, so scalar address arithmetic in the tiling loops is *not* free the
  way it is behind a 16-beat Saturn vector op — but it is still small next to
  a DMA.
- The cycle count you are scored on is `read_cycles()` taken *around the
  Gemmini matmul only*; the CPU reference matmul in the same program is timed
  separately and is not the objective.""",
}


@dataclass(frozen=True)
class Kernel:
    """One optimizable benchmark."""

    # --- identity ---------------------------------------------------------
    name: str                      # --kernel NAME
    target: str                    # "saturn" | "gemmini"; picks TARGET_NOTES
    bench_dir: str                 # directory under BENCHMARKS_DIR

    # --- what the agent owns ---------------------------------------------
    kernel_rel: str                # THE only file the agent may change
    elf_name: str                  # output binary name

    # --- what is sealed ---------------------------------------------------
    # Globs relative to BENCHMARKS_DIR, in addition to COMMON_SEALED_GLOBS.
    # `kernel_rel` may be matched by these; build_inputs overwrites it with the
    # agent's bytes afterwards, so an over-broad glob is harmless.
    bench_globs: list[str] = field(default_factory=list)

    # --- how the objective is read out of the simulator log ---------------
    cycle_re: str = r"mcycle\s*=\s*(\d+)"
    instret_re: str | None = r"minstret\s*=\s*(\d+)"
    # "first" | "last" | "sum" — how to fold multiple matches into one number.
    metric_agg: str = "first"

    # --- gemmini-only: the tree does not live under BENCHMARKS_DIR --------
    # `main_src` is the ONE benchmark translation unit (it holds `main()`, the
    # CPU golden model and the self-check) — always sealed, never editable.
    # `extra_sealed` are further globs, relative to GEMMINI_TESTS_DIR, pulled
    # off the chipyard container on top of `constants.GEMMINI_SEALED_GLOBS`.
    main_src: str | None = None
    extra_sealed: list[str] = field(default_factory=list)
    extra_cflags: list[str] = field(default_factory=list)

    # --- per-kernel simulator budget (None = the global default) ----------
    sim_timeout_seconds: int | None = None
    sim_timeout_cycles: int | None = None

    # --- reference numbers (None = unknown; the loop measures its own) ----
    reference_cycles: int | None = None
    roofline_cycles: int | None = None

    # --- prompt material --------------------------------------------------
    objective: str = ""            # one line: what to make faster
    check_desc: str = ""           # 自檢方式: how correctness is enforced
    contract: str = ""             # the ABI/semantics the agent must keep
    notes: str = ""                # kernel-specific roofline / ruled-out table

    # --- readiness --------------------------------------------------------
    # False = registered but no working harness yet (loop refuses to run it).
    available: bool = True
    unavailable_reason: str = ""

    @property
    def sealed_globs(self) -> list[str]:
        """Globs `collateral.py` reads out of BENCHMARKS_DIR.

        Gemmini entries have none: their tree is hardware-derived and is
        fetched by `nodes.gemmini_collateral()` off the chipyard container
        instead (see `constants.GEMMINI_SEALED_GLOBS`).
        """
        if self.target == "gemmini":
            return list(self.bench_globs)
        return list(self.bench_globs) + COMMON_SEALED_GLOBS

    @property
    def target_notes(self) -> str:
        return TARGET_NOTES.get(self.target, "")


def _saturn_globs(bench: str) -> list[str]:
    """Everything the compiler/harness needs from a Saturn benchmark dir."""
    return [f"{bench}/*.c", f"{bench}/*.h", f"{bench}/*.S"]


# ===========================================================================
# Registry
# ===========================================================================

VEC_SGEMV = Kernel(
    name="vec-sgemv",
    target="saturn",
    bench_dir="vec-sgemv",
    kernel_rel="vec-sgemv/vec-sgemv.S",
    elf_name="vec-sgemv.riscv",
    bench_globs=_saturn_globs("vec-sgemv"),
    reference_cycles=4277,
    roofline_cycles=4096,
    objective="Make `vec_sgemv` run in fewer **cycles** (`mcycle`), without "
              "changing what it computes.",
    check_desc="`vec-sgemv_main.c` calls `setStats()` around your kernel and "
               "then `verifyFloat()` on the result; a nonzero exit means the "
               "answer was wrong. You get back: pass/fail, `mcycle`, `minstret`.",
    contract="""\
```c
void *vec_sgemv(size_t m, size_t n,
                const float *v,   // m-length vector
                const float *A,   // m * n matrix, ROW-major
                float *c);        // n-length output
```

`c += v^T * A`. Standard RISC-V calling convention: `a0=m, a1=n, a2=v, a3=A,
a4=c`. The caller zeroes `c` beforehand and may call you twice (a warm-up pass
under `PREALLOCATE`), so do not assume cold caches.

Called with `M_DIM = N_DIM = 128`.""",
    notes="""\
# Do the arithmetic before you touch the file — where the floor is

This is a dense 128x128 f32 GEMV. Every element of `A` must be read exactly
once and multiplied exactly once. That is **16384 element-FMAs** and
**16384 element-loads**, and no restructuring can reduce either number.

DLEN = 128 bits = **4 f32 lanes per cycle**. So:

    FMA pipe occupancy    >= 16384 / 4 = 4096 cycles
    A-load unit occupancy >= 16384 / 4 = 4096 cycles   (if the LSU is DLEN-wide)

If (and only if) those two units overlap, the hard floor is **~4096 cycles**.
The pristine kernel measures **4277**. It is therefore already at **~96% of
the roofline**; the entire remaining budget is about **181 cycles (~4.2%)** of
prologue / epilogue / strip-mine overhead. A 1.04x result is a *win* here.

Do not propose anything whose payoff is smaller than the risk of breaking
correctness, and do not propose anything that implicitly claims to beat 4096 —
if your idea seems to, you have made an arithmetic error; recheck it.

Where the 181 cycles plausibly live (this is your actual target):

- `n=128` with `LMUL=8, e32` gives `VLMAX = 8*256/32 = 64`, so the kernel makes
  **2 column passes**. Each pass pays: `vsetvli`, one `vle32.v` of `c`
  (16 beats), two prologue row loads, a scalar-loop drain, the epilogue FMAs,
  and one `vse32.v` of `c` (16 beats). Roughly 90 cycles x 2.
- Pipeline drain at the loop tail and at each pass boundary, where the FMA
  pipe waits on a load that was not issued far enough ahead.

# What has already been ruled out — do not repeat it

Three prior optimization sessions on this exact kernel and config produced:

| attempt | mcycle | minstret | verdict |
|---|---|---|---|
| pristine baseline | **4277** | 1706 | reference |
| cheaper loop-tail branch (`slti/bnez/j` -> countdown `bnez`) | 4270 | 1452 | -0.2%, noise-level |
| tighter tail again (`bge` w/ hoisted constant) | 4277 | 1456 | 0% |
| single-pass row-major sweep, both `c` halves resident | 4487 | 1321 | **+4.9% WORSE** |
| single-pass row-major, software-pipelined one row ahead | 4501 | 1319 | **+5.2% WORSE** |

Two conclusions, both empirical, both firm:

1. **Cutting scalar instructions does not help.** minstret fell 15-23% across
   these attempts and `mcycle` did not move (or got worse). The Shuttle core
   already hides the scalar loop overhead behind the 16-beat vector ops.
   *Stop optimizing minstret. It is not the objective and it does not correlate.*
2. **The single-pass / row-major restructure loses.** At `LMUL=8` there are only
   four register groups (`v0,v8,v16,v24`). Keeping all 128 outputs resident costs
   two of them, leaving one row of `A` (which itself spans two groups) with
   **no room to double-buffer** — so every FMA stalls on its own load. The
   two-pass structure exists precisely so that `v0`/`v8` can run a full row-pair
   ahead of the FMAs. Do not undo it.

# Directions still worth exploring

Pick from these (or something better you can justify), not from the ruled-out list:

- **Deepen the software pipeline** in the existing two-pass structure: issue A
  row loads 3-4 rows ahead instead of 2, so no FMA ever waits. This is the most
  direct attack on drain cycles and keeps the winning structure.
- **Kill one of the two `c` round-trips.** The second pass reloads `c` and
  restores it. Can the `vle32.v`/`vse32.v` of `c` for pass 2 be issued early
  enough to overlap with pass 1's FMAs, instead of sitting in the critical path?
- **Overlap the pass boundary**: start pass 2's first A loads before pass 1's
  final store retires.
- **LMUL choice**: LMUL=8 already maximizes beats-per-instruction. Lowering it
  adds instructions with no beat savings — only try it if you have a concrete
  scheduling reason (e.g. finer-grained double-buffering) and say so.
- **Load form**: `A` is row-major and the pass walks down a column block, so the
  per-row `vle32.v` is already unit-stride within the row, i.e. already at the
  DLEN-per-cycle peak. Any strided/indexed variant drops to 1 element/cycle.
  Don't.
- **Check the store pipe.** `vse32.v` of `c` runs on its own DLEN-wide pipe and
  can overlap with FMAs. If the pass-1 store is currently serialized in front of
  pass-2's first load, moving it earlier is nearly free cycles.""",
)


VEC_SOFTMAX = Kernel(
    name="vec-softmax",
    target="saturn",
    bench_dir="vec-softmax",
    # softmax.c holds BOTH the scalar reference and the vector kernel. The
    # scalar `softmax()` is the golden model the check compares against, so the
    # agent must leave it alone — the prompt says so; the objective (vector
    # cycles) is measured separately, so slowing the scalar side gains nothing.
    kernel_rel="vec-softmax/softmax.c",
    elf_name="vec-softmax.riscv",
    bench_globs=_saturn_globs("vec-softmax"),
    # main.c prints its own cycle line; there is no setStats/mcycle= here.
    cycle_re=r"The vector Softmax execution took (\d+) cycles",
    instret_re=None,
    objective="Make `softmax_vec` run in fewer **cycles**, without changing "
              "what it computes.",
    check_desc="`main.c` runs the scalar `softmax()` and the vector "
               "`softmax_vec()` on the same input and compares every output "
               "element with `similarity_check(..., THRESHOLD=0.1)`. Any "
               "mismatch makes `main` return nonzero and the attempt scores "
               "nothing. It prints `The vector Softmax execution took N "
               "cycles.` — that N is the objective.",
    contract="""\
```c
void softmax_vec(const float *i, const float *o,
                 uint64_t channels, uint64_t innerSize);
```

Row-wise softmax over `channels` x `innerSize` f32, computed *down* the channel
axis for each of `innerSize` columns: for every column j,
`o[c*innerSize+j] = exp(i[c*innerSize+j] - max_c) / sum_c exp(...)`.
Dataset: `channels = 3`, `innerSize = 256`.

`softmax()` (the scalar function in the same file) is the **golden model** the
check compares against. Do not change its behaviour.""",
    notes="""\
# Notes for this kernel

No roofline has been established for this benchmark yet — the loop measures the
pristine baseline at iteration 0 and that number is what you must beat. Nothing
has been ruled out empirically; you are the first session on it.

Useful starting observations:

- The vector path uses `ara/exp.h`'s vectorized `exp`, which is a polynomial
  plus scalb — expect it to dominate the cycle count.
- `channels = 3` is tiny and `innerSize = 256` is the long axis, so the natural
  vectorization is *along innerSize*, stripmined at the largest LMUL that still
  leaves registers for the max/sum accumulators and the exp temporaries.
- The kernel currently makes several passes over the data (max, exp+sum,
  divide). Fusing passes cuts load traffic; check whether that actually helps
  before assuming it does.
- A reciprocal-multiply instead of a vector divide is usually a win if
  `THRESHOLD = 0.1` tolerates it (it is a loose threshold).""",
)


VEC_DOTPROD = Kernel(
    name="vec-dotprod",
    target="saturn",
    bench_dir="vec-dotprod",
    kernel_rel="vec-dotprod/dotproduct.c",
    elf_name="vec-dotprod.riscv",
    bench_globs=_saturn_globs("vec-dotprod"),
    # main.c prints one "Vector cycles: N instructions: M" line per (width,
    # avl) sweep step; the objective is the total across all of them.
    cycle_re=r"Vector cycles:\s*(\d+)",
    instret_re=r"instructions:\s*(\d+)",
    metric_agg="sum",
    objective="Make the four `dotp_v*b` kernels run in fewer **total cycles**, "
              "without changing what they compute.",
    check_desc="`main.c` sweeps avl for 64/32/16/8-bit dot products and prints "
               "`Vector cycles: N instructions: M` for each. The objective is "
               "the SUM of those N. NOTE: this harness's self-check is weak — "
               "`main` returns 0 unconditionally — so a wrong answer is not "
               "reliably caught here. Treat correctness as your own "
               "responsibility.",
    contract="""\
```c
int64_t dotp_v64b(int64_t *a, int64_t *b, uint64_t avl);
int32_t dotp_v32b(int32_t *a, int32_t *b, uint64_t avl);
int16_t dotp_v16b(int16_t *a, int16_t *b, uint64_t avl);
int8_t  dotp_v8b (int8_t  *a, int8_t  *b, uint64_t avl);
```

Each returns the full-width integer dot product of `a[0..avl)` and `b[0..avl)`,
accumulated in the same width as the inputs (wrapping is expected — match the
existing behaviour exactly). The scalar `dotp_s*b` functions in the same file
are the reference; do not change them.""",
    notes="""\
# Notes for this kernel

No roofline established yet — beat the measured baseline. The sweep runs each
width at `avl = 8, 64, 512, ...` up to `vsize`, so short-vector overhead
(`vsetvl`, the reduction tail) is a large fraction of the total at small avl.

Worth looking at:

- The reduction (`vredsum`) at the end of each call is a long-latency serial
  op; at small `avl` it dominates.
- The `if (avl == orig_avl)` branch inside the stripmine loop exists only to
  initialize the accumulator — hoisting that out of the loop is free.
- All four widths share one structure; a win in one usually transfers.""",
)


# ---------------------------------------------------------------------------
# Gemmini (RoCC)
# ---------------------------------------------------------------------------
# These do NOT come from BENCHMARKS_DIR. `nodes.gemmini_collateral()` fetches
# the whole `gemmini-rocc-tests` tree off the chipyard container *after*
# regenerating `include/gemmini_params.h` from the elaborated hardware config
# — see constants.GEMMINI_HEADER_CONFIG_ATTRS for what that header depends on.
# So an entry here only names the pieces:
#   kernel_rel  -> the agent-owned file (the Gemmini software library)
#   main_src    -> the sealed benchmark TU that holds main() + the golden model
# and needs a Gemmini-bearing config: `--config GENV256D128GemminiShuttleConfig`.

_GEMMINI_CONTRACT = """\
The file you own is `include/gemmini.h`, Gemmini's entire software library.
The benchmark calls into it and nothing else:

```c
tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
                  const elem_t* A, const elem_t* B, const void* D, elem_t* C,
                  size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
                  scale_t A_scale, scale_t B_scale, scale_acc_t D_scale,
                  int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
                  bool transpose_A, bool transpose_B,
                  bool full_C, bool low_D, uint8_t weightA,
                  enum tiled_matmul_type_t tiled_matmul_type);
```

computing `C = scale * (A @ B + D)` with `A` `dim_I x dim_K`, `B`
`dim_K x dim_J`, all row-major with the given strides, `elem_t` elements and
`acc_t` accumulation. **Every argument's meaning, and the exact numerical
result, must stay identical** — the benchmark compares against a CPU matmul.

You may restructure anything inside the header: `tiled_matmul_auto`'s tile-size
search, `tiled_matmul_outer`, the `sp_tiled_matmul_ws` inner loop, the
mvin/mvout scheduling, the fences. You may NOT change the macros that come
from `include/gemmini_params.h` (that header is generated by the hardware
elaboration and is overwritten anyway), and you may not touch the benchmark.

Called with `dim_I = dim_J = dim_K = 64`, `WS` (weight-stationary), no bias."""

_GEMMINI_NOTES = """\
# Notes for this kernel

Measured on `GENV256D128GemminiShuttleConfig` (Shuttle + Saturn + a
`DefaultGemminiConfig` Gemmini) during the co-existence bring-up:

| what | cycles |
|---|---|
| Gemmini `tiled_matmul_auto`, 64x64x64 int8 | **2324** |
| the same product on the scalar CPU | 2434665 |

So the accelerator is already ~1000x the scalar core; your job is the 2324.

## Where the floor is

64x64x64 = 64 tiles of 16x16x16. The array consumes one 16-element row per
cycle, so the pure compute time is `64 * 16 = 1024` cycles. Data movement:
A and B are 4096 bytes each, C is 4096 bytes; at `MAX_BYTES = 64` per DMA
request that is 64 + 64 + 64 = 192 requests minimum. **1024 cycles is the hard
floor and 2324 is ~2.3x it** — i.e. roughly 1300 cycles are currently *not*
overlapped compute. That is the budget you are attacking, and it is large,
unlike the Saturn kernels.

## Where those cycles plausibly live

- `tiled_matmul_auto` runs a *search* over tile sizes at run time before it
  issues anything. At 64x64x64 the whole problem fits, so that search is pure
  overhead on the measured region.
- The configuration instructions (`gemmini_config_ex`, `gemmini_config_ld`,
  `gemmini_config_st`) are re-issued per tile in some paths even when nothing
  changed.
- `gemmini_fence()` / `gemmini_flush()` inside the tiling loops drain the
  queues and destroy mvin/compute overlap.
- The first tile's mvin is not overlapped with anything (cold start), and the
  last tile's mvout is not overlapped either — prologue/epilogue.
- Weight-stationary means B should be mvin'd once per (J,K) tile and reused
  across all I. Check whether the current loop nest actually achieves that at
  these dimensions.

## Nothing has been ruled out empirically yet

You are the first optimization session on this kernel. Unlike `vec-sgemv`
there is no table of failed attempts — but also no guarantee that an idea that
looks good is not already what the code does. Read `tiled_matmul_auto` and
`sp_tiled_matmul_ws` before proposing anything.

## Two warnings

- `include/gemmini.h` is ~3600 lines and is used by *every* Gemmini program.
  Only the `tiled_matmul_*` / `sp_tiled_matmul_*` path is exercised here, but
  a change that breaks the macro definitions breaks the build outright.
- The macros in `include/gemmini_params.h` (`DIM`, `BANK_ROWS`, `ACC_ROWS`,
  `MAX_BYTES`, `elem_t`, `acc_t`) are **generated from the RTL**. Read them;
  never replace them with literals."""


def _gemmini_matmul(name: str, *, extra_cflags: list[str], check_desc: str,
                    sim_timeout_seconds: int, reference_cycles: int) -> Kernel:
    """One `bareMetalC/tiled_matmul_ws.c` variant.

    The only thing that varies between the two registered entries is how
    expensive the benchmark's own golden model is (see `check_desc`).
    """
    return Kernel(
        name=name,
        target="gemmini",
        bench_dir="bareMetalC",
        # The agent owns Gemmini's software library, NOT the benchmark: the
        # benchmark file holds main(), the CPU reference matmul and the
        # comparison, so letting the agent edit it would let it edit its own
        # grader. `build_inputs` seals it.
        kernel_rel="include/gemmini.h",
        main_src="bareMetalC/tiled_matmul_ws.c",
        elf_name="tiled_matmul_ws.riscv",
        extra_cflags=extra_cflags,
        sim_timeout_seconds=sim_timeout_seconds,
        sim_timeout_cycles=100_000_000,
        # `tiled_matmul_ws.c` prints "Cycles taken: N" twice — first for the
        # Gemmini matmul, then for the CPU reference. The objective is the
        # FIRST one.
        cycle_re=r"Cycles taken:\s*(\d+)",
        instret_re=None,          # the harness prints no minstret
        metric_agg="first",
        reference_cycles=reference_cycles,
        roofline_cycles=1024,
        objective="Make Gemmini's `tiled_matmul_auto` run in fewer **cycles** "
                  "on a 64x64x64 int8 matmul, without changing what it computes.",
        check_desc=check_desc,
        contract=_GEMMINI_CONTRACT,
        notes=_GEMMINI_NOTES,
    )


# The real entry: random 0/1 inputs and a full scalar reference matmul, i.e.
# the benchmark's own self-check exactly as upstream ships it. The reference
# costs ~2.4M simulated cycles (~35 min on Verilator at the ~1.1k cycles/s this
# config runs at), hence the large sim timeout. This is the one to score an
# optimization run with: the agent owns `gemmini.h`, so only a golden model it
# cannot predict makes "just write the right answer" impossible.
GEMMINI_TILED_MATMUL_WS = _gemmini_matmul(
    "gemmini-tiled-matmul-ws",
    extra_cflags=[],
    # Measured: 2137s wall on GENV256D128GemminiShuttleConfig (the CPU
    # reference matmul is 2.43M of those simulated cycles).
    sim_timeout_seconds=60 * 75,
    reference_cycles=2324,
    check_desc="`bareMetalC/tiled_matmul_ws.c` (SEALED — you cannot edit it, "
               "only `include/gemmini.h`) fills A and B with random 0/1 "
               "values, runs your `tiled_matmul_auto`, then recomputes the "
               "whole product on the scalar CPU and compares element-wise "
               "with `full_is_equal`; any mismatch dumps both matrices and "
               "`exit(1)`, which scores the attempt zero. It prints "
               "`Cycles taken: N` twice — the FIRST (around the Gemmini "
               "matmul) is the objective; the second is the CPU reference and "
               "is ignored.",
)

# Harness smoke test ONLY. `-DFAST` replaces the scalar reference with its
# closed form (inputs become all-ones, so every output must equal K = 64),
# which cuts one run from ~35 minutes to ~1 and is what you want when checking
# that the build/run/metric path works at all.
#
# DO NOT score an LLM optimization run with this entry: the agent owns
# `include/gemmini.h`, and a golden matrix of "64 everywhere" is something it
# could satisfy without doing the matmul. The self-check is real (`exit(1)` on
# mismatch) but it is not adversarially safe.
GEMMINI_TILED_MATMUL_WS_FAST = _gemmini_matmul(
    "gemmini-tiled-matmul-ws-fast",
    extra_cflags=["-DFAST"],
    # Measured: 827s wall (vs 2137s strict) — most of a run on this config is
    # boot/init and UART output, not the reference matmul, so `-DFAST` buys
    # ~2.5x, not the ~35x the cycle counts suggest.
    sim_timeout_seconds=2400,
    reference_cycles=2803,
    check_desc="Same benchmark as `gemmini-tiled-matmul-ws`, built with "
               "`-DFAST`: the inputs are all-ones and the golden matrix is the "
               "closed form rather than a scalar matmul, so the run is ~35x "
               "shorter. Harness smoke test only — the check is too easy to "
               "satisfy without computing anything.",
)


# ---------------------------------------------------------------------------
# Real LLM shapes (int8) — Llama-3.2-1B
# ---------------------------------------------------------------------------
# Both entries keep .data + .bss under 1 KiB and put their matrices at fixed
# DRAM addresses outside the ELF: the Verilator harness loads the image over
# TSI and zeroes .bss at roughly 12 s per KiB, so a 1 MiB `static` array would
# cost hours of wall time before main() runs. Both self-check by SAMPLING
# (8 outputs recomputed scalar) — a full scalar reference is 9.4M cycles for
# the GEMV and 67M MACs for the GEMM.

LLAMA_Q8_GEMV = Kernel(
    name="llama-q8-gemv",
    target="saturn",
    bench_dir="llama-q8-gemv",
    kernel_rel="llama-q8-gemv/llama_q8_gemv.c",
    elf_name="llama-q8-gemv.riscv",
    bench_globs=_saturn_globs("llama-q8-gemv"),
    # Measured on GENV256D128GemminiShuttleConfig: 571 s of Verilator wall for
    # one run (126 s TSI ELF load + boot, 445 s of simulated execution covering
    # the 1 MiB data fill, the GEMV and the 8-row scalar check). 3x that.
    sim_timeout_seconds=1800,
    sim_timeout_cycles=20_000_000,
    reference_cycles=621_724,
    roofline_cycles=131_072,
    objective="Make `llama_q8_gemv` — a 512x2048 int8 GEMV, the decode-step "
              "shape of Llama-3.2-1B — run in fewer **cycles** (`mcycle`), "
              "without changing what it computes.",
    check_desc="`llama-q8-gemv_main.c` (SEALED) generates W and x from a fixed "
               "seed, calls `setStats()` around your `llama_q8_gemv`, then "
               "recomputes **8 sampled rows** with a scalar dot product and "
               "compares them exactly; any mismatch prints `MISMATCH` and "
               "returns 1, which scores the attempt zero. It also prints a "
               "checksum over all 512 outputs. The check is SAMPLED (a full "
               "scalar reference would be 9.4M cycles ~ 90 min of Verilator), "
               "so correctness of the other 504 rows is on you. You get back: "
               "pass/fail, `mcycle`, `minstret`.",
    contract="""\
```c
void llama_q8_gemv(size_t M, size_t K,
                   const int8_t *W,   // M x K, ROW-major, row stride exactly K
                   const int8_t *x,   // K-length activation vector
                   int32_t *y);       // M-length output
```

`y[m] = sum_k W[m*K + k] * x[k]`, exactly. Products are `int8 * int8`;
accumulation and the result are `int32` and must not saturate or round.
`int32` is wide enough: `|sum| <= 2048 * 128 * 128 = 2^25`.

Called once with `M = 512`, `K = 2048`. `W`, `x` and `y` are at fixed DRAM
addresses (`0x81000000`, `0x81200000`, `0x81210000`) *outside* the ELF — see
`llama_q8_gemv.h`, which is SEALED. `y` is zeroed by the caller. There is no
warm-up call, so the first pass over `W` (1 MiB, vs a 512 KiB L2) is cold.

`llama_q8_gemv.h` and `llama-q8-gemv_main.c` are sealed: only
`llama-q8-gemv/llama_q8_gemv.c` reaches the compiler from your workspace.""",
    notes="""\
# Do the arithmetic before you touch the file — where the floor is

`M = 512`, `K = 2048` -> **1,048,576 int8 MACs**, every weight byte read
exactly once. Nothing can reduce either number.

Derived with the ROOFLINE METHOD at the top of this file.

    COMPUTE FLOOR
      int8 x int8 widening MAC into an int16 accumulator (`vwmacc`) retires
      DLEN/16 = 8 MAC/cycle.
      compute floor = 1,048,576 / 8 = 131,072 cycles

    MEMORY FLOOR
      reads  : W = 1 MiB (each byte once) + x = 2 KiB (hoisted, read once)
                                        -> 1,050,624 B / 16 B/c = 65,664 c
      writes : y = 512 * 4 B = 2 KiB     ->      2,048 B / 16 B/c =    128 c
      memory floor = max(65,664, 128) = 65,664 cycles

    roofline = max(131,072, 65,664) = **131,072 cycles — COMPUTE binds**,
    by 2.0x over memory.

**This was 262,144 until 2026-09-07 and that number was WRONG.** The old note
claimed int16 accumulation is impossible here because 2048 terms overflow —
true, but irrelevant: you do not accumulate 2048 terms in int16, you
accumulate TWO (`vwmul.vx` then `vwmacc.vx`, max `2*127*128 = 32,512`) and
widen that pair into the int32 accumulator with one `vwadd.wv`. Round 1's
`llama-attn-pv-int8` winner proved this pattern works and is exact. The
`vwadd.wv` traffic is then only `1,048,576/2` int32-lane ops = 131,072 cycles
on the *other* arithmetic pipe, which chains rather than serialises — so
131,072 is a floor you approach, not one you sit on. A realistic single-pipe
model of the pair pattern is ~196,608 cycles; treat 131,072 as the strict
bound and ~200k as the honest target.

The load pipe is NOT the bottleneck — but only if `x` stays in registers
instead of being re-streamed per row (the pristine kernel re-reads it 512
times, which alone adds 1 MiB and 65,536 cycles of load traffic), and only if
`W`'s 1 MiB stream against a 512 KiB L2 actually sustains 16 B/cycle. The
Gemmini variant of this same 1 MiB weight stream measures only ~7 B/cycle of
cold DRAM bandwidth, so verify before building a plan on 16.

# What the pristine kernel does

One row at a time. Per row: zero an `i32m4` accumulator, then 64 iterations of
`vle8` W + `vle8` x + `vwmul.vv -> i16m2` + `vwadd.wv -> i32m4`, then one
`vredsum`. So:

- `x` is re-loaded from memory for **every one of the 512 rows**.
- LMUL is 1 on the int8 side, i.e. 32 elements per strip: at `e8m1` a vector op
  is only `VLEN/DLEN = 2` beats, far too short to hide the 4-stage FMA/ALU
  latency, so the dependent `vwadd.wv` chain on a single accumulator is very
  likely serializing.
- Exactly one row is in flight, so there is nothing to overlap the reduction
  and the loop-carried accumulate against.
- There is one `vredsum` (a long-latency serial op) per row, 512 of them, and
  each sits on the critical path to `y[m]`.

# Directions worth exploring

- **Raise LMUL.** At `e8m2`/`e8m4` the accumulator becomes `i32m8`, the chime
  grows to 8/16 beats and a dependent accumulate chain is fully hidden. This is
  the single biggest structural lever and is exactly the opposite of the
  `vec-sgemv` situation, where LMUL was already maxed.
- **Process several rows per pass over `x`.** 2, 4 or 8 rows share one set of
  `x` loads and give you that many independent accumulator chains — it kills
  the re-read traffic *and* the dependency stall at once. Watch the register
  budget: at LMUL=8 there are only four groups.
- **Amortize the reductions.** Reducing R rows at the end of one pass lets the
  `vredsum` latencies overlap each other instead of each blocking a row.
- **Pair the products in int16 before widening.** `vwmul.vx acc16, row0, x0`
  then `vwmacc.vx acc16, x1, row1` puts two int8 products in one int16 lane
  (`|2*127*128| = 32,512`, exact), and ONE `vwadd.wv` per pair moves them into
  int32. That halves the number of 4-lane/cycle int32 operations, which is the
  whole reason the floor is 131,072 and not 262,144. Two alternating int32
  accumulator groups keep consecutive `vwadd.wv`s free of a RAW dependency.
  This is the single highest-value structural change and it is exactly what
  Round 1's `llama-attn-pv-int8` winner (4,853 cycles) is built around — read
  `out/loop/20260907-043140-2246/kernel_best.h`.
- **Software-pipeline the W loads** one strip ahead so the arithmetic pipe
  never waits on the 1 MiB stream.

# Nothing has been ruled out empirically yet

You are the first optimization session on this kernel. There is no table of
failed attempts to avoid — but also no guarantee that an idea that looks good
is not already what the code does. Read the file first.

The pristine kernel measures **621,724 cycles** (367,142 instructions), i.e.
**4.74x the 131,072-cycle compute floor** and 9.5x the 65,664-cycle memory
floor. That is a large budget — unlike `vec-sgemv`, which was already at 96%
of its roofline.""",
)


_GEMMINI_DMA_HINTS = """\
# THE ONE NUMBER THAT MATTERS, AND IT IS NOT THE OLD `roofline_cycles`

Until 2026-09-09 the GEMV roofline here was `weight_bytes / 16`, taken from
Gemmini's `dma_buswidth = 128 bit`. **That was wrong by exactly 2x and it sent
three rounds of optimization chasing headroom that does not exist.**

128 bit is the width of Gemmini's own DMA port and of the *system bus*. The
path a weight byte actually travels is

    Gemmini StreamReader --(16 B/cycle TL, Scratchpad.scala:224)--> tile
    master xbar --> sbus (16 B/cycle, WithSystemBusWidth(128))
    --> SiFive InclusiveCache L2, 512 KiB (DTS: cache-controller@2010000)
    --> **mbus, 8 B/cycle** --> AXI4 mem port --> DRAMSim2 model

and the mbus is where it bottlenecks: rocket-chip's `BaseSubsystemConfig` sets
`MemoryBusKey => MemoryBusParams(beatBytes = 8)`, chipyard's `AbstractConfig`
never overrides it, and neither does this SoC's config chain
(`docker/add_coexist_config.py`: only `WithSystemBusWidth(128)` and
`WithShuttleTileBeatBytes(16)`, both sbus/tile-side). B lives at 0x81100000,
inside the cacheable `memory@80000000` region, so every weight byte is a DRAM
byte crossing that 8 B/cycle port.

    TRUE link floor       = weight_bytes / 8 B per cycle
    cold DRAM, MEASURED   = 7.0 - 7.1 B/cycle
    L2 hits, MEASURED     = ~16 B/cycle, but only for bytes already resident.
                            The harness fills B immediately before calling you,
                            so the TAIL of B is still warm: worth ~200-260 KiB.
                            DESCENDING K order is what harvests it. Switching to
                            ascending costs +17,104 cycles (measured). Never
                            "clean that up".

So exactly two levers exist:
  (1) issue the weight stream so the 16-deep DMA request pool is actually full,
  (2) harvest the warm L2 tail (already done: keep K descending).
There is no third lever. The bytes cannot be reduced -- B is read once already
-- and the array is idle either way.

# HOST-ISSUE COST -- the trap that has eaten the most iterations

Gemmini commands issued one at a time by the host cost **~10-22 cycles each**
once the ex queue backs up. Measured: an N=1 hand-issued stream of 9,381
commands ran at 22 cycles per preload/compute pair, 182,726 cycles, +38% vs
the winner. `gemmini_loop_ws` expands the same sequence **in hardware** from
~2 host commands per loop, which is exactly why the N=1 winner uses it.

Before writing a hand-issued stream, count its commands and divide your target
cycle count by that. If you need more than about 1 command per 25 cycles of
runtime, the host is your bottleneck and the DMA is not.

# REQUEST AND COMMAND SHAPE -- where the remaining cycles actually are

    dma_maxbytes            64 B  -> one DMA request = at most 64 contiguous B
    max_in_flight_mem_reqs  16      (StreamReader nXacts)
    ld command slots        nCmds = max_in_flight_mem_reqs / DIM + 1 = **2**

    => requests in flight = 2 x (rows per mvin command), capped at 16.

**2026-09-09, ROUND 4: "more rows per mvin is better" IS WRONG AND IS NOW
MEASURED TO BE WRONG.** This note used to say a 4-row mvin "runs at ~half
rate". It does not; 4 rows is the OPTIMUM. The lmhead kernel (4 MiB of B,
stride 2048) was run end-to-end at five different rows-per-command values,
everything else held fixed:

| rows per mvin | requests in flight | lmhead cycles | B/cycle |
|---|---|---|---|
| 3 (probe, decoded) | 6 | ~649,000 | 6.46 |
| **4** | **8** | **634,507** | **6.61**  <- optimum |
| 5-6 ({6,5,5}) | 10-12 | 682,974 | 6.14 |
| 8 | 16 | 690,466 | 6.07 |
| 16 | 16 (capped) | 695,516 | 6.03 |

The curve has a sharp maximum at 8 requests in flight and gets ~9% WORSE
above it. **Filling the 16-deep request pool is not a goal -- it is the
failure mode.** The RTL says why (verified in
`repos/gemmini/src/main/scala/gemmini/`):

  * `LoadController` has a 3-state FSM that issues **one row per cycle** and
    does not dequeue the next mvin until the current one has issued its LAST
    row (`cmd.ready` only in `sending_rows` with `last_row`;
    LoadController.scala:30-31,74,102-106,164-170). Rows of one command are
    serialized.
  * `dma_maxbytes = 64` and there is NO coalescing across rows (explicit TODO
    at DMA.scala:135), so 1 row = 1 TileLink Get = 1 `XactTracker` slot.
  * `XactTracker` is a flat pool of `nXacts = max_in_flight_mem_reqs = 16`
    slots, **global**, not per command (XactTracker.scala:56-71,
    Scratchpad.scala:208).
  * `DMACommandTracker` allows only `nCmds = max_in_flight_mem_reqs/DIM + 1 =
    2` mvins to be OPEN at once, and a new command cannot issue its first row
    until a tracker slot frees (LoadController.scala:38,84,99).

So the thing that hides DRAM round-trip latency is **overlap between the two
open commands**, and that overlap only exists when `2 x rows` leaves free
slots in the 16-deep pool:

    R = 3-4   ->  6-8 Gets outstanding, 8-10 slots spare. Command B issues its
                  rows while command A's are still returning. Latency hidden.
    R = 8     ->  2 x 8 = exactly 16. B stalls on A's completions the moment
                  it opens. Overlap headroom = 0.
    R = 16    ->  ONE command fills the entire pool. B's first row stalls in
                  `waiting_for_dma_req_ready` until A drains, so a full DRAM
                  round trip is exposed as a bubble every command.

`R x 2 <= ~8` is the design rule. Do not "fix" a 4-row mvin stream by making
its commands bigger.

Two more RTL facts worth knowing before you invent a schedule:
  * The scratchpad is single-ported per bank and **ExecuteController reads
    have PRIORITY over DMA writes** (Scratchpad.scala:510-538, comment at
    :522). Compute can starve a mvin; a mvin never starves compute.
  * `ReservationStation` tracks ld/ex deps by scratchpad address-range
    overlap (fine-grained, :331-378), BUT a new load depends on every older
    un-issued load in the ld queue (:329), so loads DISPATCH in program order
    no matter how independent they are. Reordering loads in software is the
    only way to reorder them at all.

Elaborated `GENV256D128GemminiShuttleConfig` (= `gemmini.DefaultGemminiConfig`
+ `WithSystemBusWidth(128)` + `WithShuttleTileBeatBytes(16)`):
DIM 16, sp 256 KiB / 4 banks, acc 64 KiB / 2 banks, dma_maxbytes 64,
max_in_flight_mem_reqs 16, ld/st/ex queue length 8/2/8,
reservation-station entries ld 8 / st 4 / ex 16.

`sp_tiled_matmul_ws` emits 16-row x 64 B mvins (`B_blocks = MAX_BLOCK_LEN = 4`,
`rows = DIM`; gemmini.h ~line 600). That is NOT a target to match -- it is why
the M=512 sibling looks like 7.92 B/cycle: that number is a 1 MiB stream with
a ~200-260 KiB warm-L2 tail in it, not a better request shape. On a cold
4 MiB stream the same 16-row shape measures 6.03 B/cycle.

# ALREADY MEASURED -- DO NOT REPEAT ANY OF THESE

| tried | result |
|---|---|
| mvin-only probe, N=1/M=512, same shape+order+cache state as the best kernel | **132,109** vs the kernel's 132,424. The stream IS the cost; compute + mvout + fence together are 315 cycles. |
| mvin-only probe, lmhead, v6's 4-row x 64 B shape | ~632,000 (6.6 B/cycle) |
| Ascending K order | +17,104. Never. |
| host RVV / Saturn co-compute of a K or M slice (**5 separate attempts**) | 133,800 / 134,585 / 140,285 / two self-check failures / lmhead 738,754. The core and Gemmini share ONE L2 -> mbus path that is already 99% saturated, so bandwidth is NOT additive. **This direction is CLOSED. Do not propose it again in any form, with any mechanism.** |
| host `lb` touch-prefetch of the next tile | +9,568 |
| Zicbop `prefetch.r` hints | -229 cycles and saturated: budget 2500 == 12000 == best, and a 2-tiles-ahead burst is +723. **Further tuning is worth <0.2%. Do not.** |
| Geometrically shrinking the tail K-tiles (15, 8, 4, 2, 1) | +181 |
| Reading Gemmini perf counters inside the timed region | ~+2,000 |
| Hand-issued per-command stream for N=1 (9,381 commands) | +50,302 |
| TLB pressure (4 entries, stride 512 / 2048) | **NOT a lever.** The bareMetalC harness runs with paging off, and the N=1 kernel sits 1.0% above the link floor while touching 256 pages -- effective cost under 5 cycles per page. Do not restructure a layout "for the TLB". |
| Submitting the best kernel unchanged, or a comment-only diff | **Five iterations were burned this way in rounds 2-3. It is the worst possible use of an iteration.** |

# HOW TO SPEND AN ITERATION

Before editing anything, write exactly these three lines at the top of the
file, with specific numbers:

    HYPOTHESIS: <the mechanism costing cycles, with an arithmetic estimate of
                 how many, derived from B/cycle and the tables above -- not
                 "overhead", not "stalls">
    CHANGE:     <the one structural thing you are changing>
    EXPECTED:   <a single predicted cycle count>

After the measurement, state in one line whether the hypothesis SURVIVED or
DIED and what that retires. ONE mechanism per iteration. A confident
prediction that misses by 10k is worth more than a 200-cycle tweak, because it
closes a branch.

If you conclude the kernel is at its floor, do NOT resubmit it unchanged.
Spend the iteration on a PROBE that produces a number nobody has.

**HARNESS UPDATE (2026-09-11): `printf` probes now work.**
As of 2026-09-11 the simulator log is written to the run directory every
iteration (`out/loop/<run_id>/simlog_NN.txt`), and `llm.format_feedback` now
also appends a "=== benchmark stdout (tail) ===" section to PASSING-run
feedback, not just failing ones. That section is a best-effort filtered tail
of the simulator log (framework/HTIF/Verilator noise stripped), capped at
4 KB. Round 4's `printf` probes (lmhead #2, #7, #8, #11 and n1 #1) were
wasted precisely because that channel did not exist yet -- it does now, so
`printf` is a normal, reliable way to get a number out. To make sure your
probe output survives the noise filter and is easy to spot: prefix every
probe line with `PROBE ` and print one number per line, e.g.
`printf("PROBE cold_bytes=%d\n", n);`. You can still fall back to designing
the probe so the ANSWER IS THE TOTAL if you want a probe to survive even if
the run fails self-check, but for passing runs plain `printf` is fine now."""


# ---------------------------------------------------------------------------
# Round-4 addenda (2026-09-09), appended to the two GEMV entries' notes.
# Both exist because rounds 2 and 3 spent ~30 iterations against a roofline
# that was 2x too optimistic; see the mbus derivation in _GEMMINI_DMA_HINTS.
# ---------------------------------------------------------------------------

_GEMV_N1_ROUND4 = """\


# STATUS OF THIS KERNEL: SOLVED. READ THIS BEFORE PLANNING ANYTHING.

    weight bytes                1,048,576
    TRUE link floor (8 B/c)       131,072
    mvin-only probe               132,109   (1.008x floor)
    current best                  132,424   (1.010x floor, 7.92 B/cycle)

The old `roofline_cycles = 65,536` assumed a 16 B/cycle path to DRAM that this
SoC does not have (mbus is 8 B/cycle -- see the hints below). The real headroom
is **1.0%, about 1,300 cycles**, and the mvin-only probe shows ~1,000 of those
are DMA start-up and drain that no reordering has ever recovered across eleven
attempts (best delta: -229).

Do not spend this round shaving 0.2% here. If this kernel is still in the
queue, the one useful thing left is a probe producing a number nobody has,
printed to stdout with `printf`:

  * mvin-only, ASCENDING order, cold: isolates the warm-tail credit exactly
    (expect ~149k; if it comes out much lower the warm tail is larger than we
    think, and prefetching becomes worth revisiting).
  * mvin-only with the L2 deliberately flushed first: the pure cold rate with
    no benchmark artifact -- the number the whole Llama cost model rests on.

# ROUND 4 (run 20260909-200611-94c0, 2 iterations): closed

Iteration 1 ran exactly the probe suggested above -- best kernel, then a
2 MiB DMA flush of the L2, then a cold re-run, with `printf` of the pure cold
rate, the flush cost and per-`loop_ws` timestamps. It scored 768,849 and
**every printed number was thrown away**: the harness only forwards the
simulator log on a FAILING run (`loop/llm.format_feedback`), so a passing
probe reports nothing but its total. All that can be said from 768,849 is
that (best 132.4k + a 2 MiB DMA flush + a cold 1 MiB re-run + HTIF printf)
sums to it -- three unknowns, one equation. **The cold-DRAM rate is still
unmeasured; the "7.0-7.1 B/cycle" quoted elsewhere in these notes is a FIT,
not a measurement, and the lmhead round measured 6.0-6.6 B/cycle for a cold
4 MiB stream.** Do not repeat this probe until the harness forwards the
simulator log on passing runs; if you must probe, make the ANSWER BE THE
SCORED TOTAL.

Iteration 2 moved the tail `loop_ws` split to the k=0 end: 132,839 (+415).
Retired: the `loop_ws` split position has no independent cost.

**This kernel is finished at 132,424.** Spend iterations on
`llama-q8-gemv-gemmini-n16`, which has never had a single optimization
iteration and sits at 1.28x the same 131,072-cycle floor."""


_GEMV_LMHEAD_ROUND4 = """\


# STATUS OF THIS KERNEL: SOLVED (2026-09-09, after Round 4). READ THIS FIRST.

    weight bytes                4,194,304
    nominal link floor (8 B/c)    524,288    <- NOT REACHABLE, see below
    achievable floor              ~632,000   (mvin-only, best request shape)
    current best                   634,507   (1.004x the mvin-only stream)

**Do not plan an iteration against 524,288.** That number is the mbus width
times the cycle count; it assumes the DRAM side sustains 8 B/cycle to a cold
4 MiB stride-2048 stream, and it does not. What this SoC actually delivers to
Gemmini on this stream is **6.6 B/cycle**, measured five different ways, and
634,507 = 4 MiB / 6.61 B/cycle. The kernel is 0.4% above an mvin-ONLY stream
of the same bytes in the same order (~632,000): there is no compute, no
mvout, no fence and no drain left to hide. Round 4 confirmed this and closed
every remaining branch.

# ROUND 4 (run 20260909-200541-dae1, 12 iterations): everything that was tried

Seed 635,908 -> best 634,507. **-1,401 cycles, -0.22%, for 12 iterations and
$7.29.** The suggested first experiment of Round 3 (`LQ8_B_ROWS 4 -> 16`,
predicted 560-590k) was run as iteration 1 and came out at **695,516** -- 60k
WORSE, and the prediction was off by 20%. Every other lever landed on the
same 634.5k +/- 3k plateau:

| iter | change | cycles |
|---|---|---|
| 1 | `LQ8_B_ROWS` 4 -> 16, `LD_BATCH` 8 -> 2 | 695,516 |
| 2 | printf probe (8 mvin-only shapes) -- OUTPUT LOST | 6,614,734 |
| 3 | `LQ8_B_ROWS` 4 -> 8, `LD_BATCH` 8 -> 4 | 690,466 |
| 4 | chunk-order permutation c_i = 9i mod 32 (L2 bank rotation) | **634,507** |
| 5 | 1-row x 1024-col mvins | self-check FAIL (hardware rejects cols > 127) |
| 6 | mvout interleaved into the epilogue | 634,907 |
| 7 | printf probe, 16-row @ fake stride 512 -- OUTPUT LOST | 1,335,959 |
| 8 | printf probe, 3-row groups -- OUTPUT LOST | 1,299,397 |
| 9 | execute order matched to permuted load order | 637,154 |
| 10 | last K block's executes hidden under its own DMA | 634,934 |
| 11 | printf probe, host issue cost -- OUTPUT LOST | 822,414 |
| 12 | row groups {6,5,5} (10-12 requests in flight) | 682,974 |

The -1,401 came entirely from iteration 4 (chunk permutation). Nothing else
in the whole round was worth more than 400 cycles.

# WHAT THAT RETIRES -- DO NOT PROPOSE ANY OF THESE AGAIN

- **Rows per mvin.** Measured end to end at 3 / 4 / 5-6 / 8 / 16 rows. 4 is
  the optimum; see the table in the DMA hints below. More requests in flight
  is monotonically WORSE above 8. This whole axis is closed.
- **Wider mvins.** `cols > MAX_BLOCK_LEN*DIM` is rejected by the hardware
  (`mvin_cols_bits = 7`), so 64 B per row request is a hardware maximum and
  the stride-2048 request pattern cannot be made DRAM-contiguous without
  copying B, which costs a full extra pass.
- **Request address order.** Chunk permutation: -1,401 (kept). Matching the
  execute order to it: +2,647. Ascending K instead of descending: +17,104.
  Fake strides (512): no better than stride 2048.
- **Epilogue / overlap.** mvout interleaved: +400. Last block's executes
  hidden under its own DMA: +427. The serial epilogue costs nothing, because
  execute commands enqueue faster than the last loads return.
- **Host issue rate is NOT the bottleneck here.** The Round-3 note blamed it.
  Iteration 11 measured it at **~10.7 cycles per RoCC command**, not ~20:
  the kernel's ~49,000 commands cost ~526k of a 634.5k budget, so there is
  ~100k of host slack. Fewer, larger commands are free on the host side and
  still lose on the DMA side -- which is exactly what iteration 12 showed.
- **The M=512 sibling's 7.92 B/cycle is not a target.** It is a 1 MiB stream
  with a ~200-260 KiB warm-L2 tail baked into it. Scaled to a cold 4 MiB
  stream the same code shape measures 6.0-6.6 B/cycle. The "1.20x gap worth
  105,000 cycles" that Round 3's note advertised does not exist.

# THE ONLY THEORETICAL HEADROOM LEFT, AND WHY IT IS NOT WORTH AN ITERATION

The L2 is 512 KiB. The harness fills B immediately before the call, so some
tail of B is warm; descending-K already harvests it, and the arithmetic
(634,507 = w/16 + (4 MiB - w)/6.43 with w ~ 192 KiB) says ~192 KiB is warm
today. Even if a perfect schedule harvested the full 512 KiB it would be
worth ~30k cycles (5%), and no ordering has ever moved that number -- the
warm set is decided by the harness's fill and the L2's random replacement,
not by the kernel. C is 1 x 2048 int32 = 8 KiB, so the 64 KiB accumulator
holds the ENTIRE output for the whole K reduction: there are no passes, no
strips, no acc-bank pressure. Nothing else is on the table.

**Leave this kernel at 634,507. Cost the LM head at 6.6 B/cycle. If this
kernel is still in the queue, the iterations are better spent on
`llama-q8-gemv-gemmini-n16`, which has never had a single optimization
iteration and sits at 1.28x its floor.**"""


_GEMMINI_GEMM_DMA_HINTS = """\
# THE HARDWARE YOU ARE ACTUALLY DRIVING

Elaborated values for `GENV256D128GemminiShuttleConfig` (ground truth, from
`gemmini/src/main/scala/gemmini/Configs.scala`, `DMA.scala`,
`LoadController.scala`):

    DIM 16x16 int8, 256 MAC/cycle        scratchpad 256 KB, 4 banks
    accumulator 64 KB, ACC_ROWS = 1024   dma_buswidth 128 bit = 16 B/cycle
    **mbus (the path to DRAM) = 8 B/cycle** — rocket-chip BaseSubsystemConfig
    `MemoryBusKey => MemoryBusParams(beatBytes = 8)`, not overridden anywhere
    in this SoC's config chain. 16 B/cycle is the sbus/DMA-port width and is
    NOT what a cold weight byte gets; measured ceiling is 7.94 B/cycle.
    dma_maxbytes 64 B  (largest single DMA request, 4 beats)
    max_in_flight_mem_reqs 16            TLB 4 entries
    max bytes per row request = 64 B     max bytes per mvin command = 1024 B
    outstanding ld/st COMMANDS: nCmds = max_in_flight_mem_reqs / DIM + 1 = **2**

Three consequences, and they are what the few percent of headroom is made of:

- **Align tiles to `dma_maxbytes`.** A mvin row of a `DIM`-wide tile is only 16
  contiguous bytes when the source stride exceeds `DIM`, so it spends a whole
  DMA request on a quarter payload. Blocking B so each mvin'd row is 64
  contiguous bytes cuts the request count 4x. And because only **2** mvin
  commands are outstanding at once, each command should carry as many rows as
  it can (up to 1024 B) or the command tracker starves the 16-request pool.
- **Overlap DMA with the array.** This shape is COMPUTE bound (131,072-cycle
  floor vs an 81,920-cycle memory floor at the real 8 B/cycle mbus rate — the
  margin is 1.6x, not the 3.2x this note claimed before 2026-09-09), so every
  DMA cycle not hidden behind the array is a cycle lost outright, and the
  margin for hiding it is thinner than it looks. The ld / ex / st queues run concurrently;
  `gemmini_fence()` and over-conservative dependencies serialize them.
  Double-buffer the scratchpad so the next B tile arrives while the current one
  is being multiplied.
- **Reuse A, stream B exactly once.** `dim_I = 64` is only 4 row-tiles, so A
  (128 KiB) stays resident in the 256 KB scratchpad while B (512 KiB) streams
  past once. Keep K innermost per output tile and accumulate in the accumulator
  so no partial sum ever returns to DRAM.

Every iteration, before editing, write one line
`HYPOTHESIS: <what costs cycles> -> EXPECTED: <cycles>`, then check the
measurement against it and say whether the hypothesis survived."""


_GEMM_ROUND4 = """\


# STATUS OF THIS KERNEL: SOLVED (2026-09-09, after Round 4). READ THIS FIRST.

    compute floor (33.5M MAC / 256 per cycle)     131,072
    current best                                  137,246   (1.047x)
    pristine `tiled_matmul_auto`                  146,737

The remaining 6,174 cycles are accounted for exactly, and both items are
structural:

| item | cycles | why it cannot move |
|---|---|---|
| final 32 KiB C drain | ~4,100-5,000 | `LoopMatmul`'s accumulator base flips only between 0 and `ACC_ROWS/2` (= 32 tiles) and only after a C-producing command, so the LAST output command covers >= 32 tiles = 32 KiB. I=1 reaches only row-tiles 0 and 2; J=8 would need different B columns per i in one command; an explicit `mvout` is held by the loop unit until every loop retires. |
| DRAM cold start | <1,000 | measured: forcing the first command to `Kc = 1` (first compute needs ~2 KiB instead of 8 KiB) changed the total by **+34**. |
| steady state | ~0 | A 128 KiB + B 512 KiB + C 64 KiB = 704 KiB ~= 90k DMA cycles under a 131k-cycle compute time; each `LOOP_WS` computes 16,384 cycles while prefetching ~80 KiB (10-12k), so 4-6k of slack per command. |

# ROUND 4 (run 20260909-200556-5de3, 6 iterations)

Seed 138,892 -> best **137,246**, found on iteration 2 and then re-measured
identically on iterations 4, 5 and 6. **Iterations 4-6 were deliberate
no-change resubmissions** -- the agent declared the floor and had nothing
left to try. That is 3 of 6 iterations producing nothing; if this kernel is
scheduled again it will do the same.

| iter | change | cycles |
|---|---|---|
| 1 | single-pass J=16 (whole accumulator), half-height P/Q tail so the final drain is 32 KiB not 64 KiB | 137,391 |
| 2 | explicit scratchpad half-region ids + tail Q reuses P's B (`B=NULL, b_spad_id`), `tail_K` = full last K block | **137,246** |
| 3 | first `LOOP_WS` forced to `Kc = 1` (shorter cold start) | 137,280 |
| 4-6 | none (floor declared) | 137,246 x3 |

# RULED OUT (reasoned through or measured) -- do not propose again

- **Splitting J** ([8,4,4], [8,8]+tail, three passes): each extra pass
  re-reads A (+16k DMA cycles) and J <= 4 commands have no DMA slack, so the
  `st` at every block transition becomes exposed. Net loss 2-4k.
- **Larger `tile_K` (24/25) to cut command count**: puts A and B in the same
  scratchpad bank (read-port conflict) and shrinks the prefetch window of the
  first/penultimate command. Upside < 200 cycles, downside > 4k.
- **Explicit `mvout` to drain C early**: a non-loop command is blocked by
  `LoopMatmul` until all loops complete -> fully serial.
- **Dropping `gemmini_fence()`**: violates the contract (C not landed).
- **Shortening the cold start**: measured, +34.

**Leave this kernel at 137,246.** 95.5% of its cycles are the systolic array
doing real MACs; the only thing that would move it is a bigger array."""


LLAMA_Q8_GEMM = Kernel(
    name="llama-q8-gemm",
    target="gemmini",
    bench_dir="bareMetalC",
    # Unlike `gemmini-tiled-matmul-ws` (which hands the agent all of
    # `include/gemmini.h`), the agent here owns a small header that holds only
    # the compute call, so the optimization space is "how do I drive Gemmini
    # for THIS shape". It is a header, not a .c, because the Gemmini compile
    # line builds exactly one translation unit — `main_src` — and that one is
    # sealed. `include/*.h` is already in GEMMINI_SEALED_GLOBS, so the pristine
    # copy travels with the collateral and `gemmini_build_inputs` overwrites it
    # with the agent's bytes.
    kernel_rel="include/llama_q8_gemm_kernel.h",
    main_src="bareMetalC/llama_q8_gemm.c",
    elf_name="llama_q8_gemm.riscv",
    # Measured on GENV256D128GemminiShuttleConfig: 945 s of Verilator wall for
    # one M=256 run (the M=512 shape was correct but cost 1565 s, over the
    # 15-minute budget an inner-loop iteration gets). 3x that.
    sim_timeout_seconds=2850,
    sim_timeout_cycles=100_000_000,
    cycle_re=r"Cycles taken:\s*(\d+)",
    instret_re=None,
    metric_agg="first",
    reference_cycles=146_737,
    roofline_cycles=131_072,
    objective="Make `llama_q8_gemm` — a 64x2048x256 int8 GEMM, the prefill "
              "shape of Llama-3.2-1B — run in fewer **cycles** on Gemmini, "
              "without changing what it computes.",
    check_desc="`bareMetalC/llama_q8_gemm.c` (SEALED — you own only "
               "`include/llama_q8_gemm_kernel.h`) generates A and B from a "
               "fixed seed, times your `llama_q8_gemm` with `read_cycles()` "
               "and prints `Cycles taken: N` (that N is the objective), then "
               "recomputes **8 sampled output elements** with a scalar dot "
               "product and compares them exactly; any mismatch prints "
               "`MISMATCH` and `exit(1)`, which scores the attempt zero. It "
               "also prints a checksum over all 16384 outputs. The check is "
               "SAMPLED (a full scalar reference is 67M MACs, i.e. days of "
               "Verilator), so correctness of the rest is on you.",
    contract="""\
```c
static void llama_q8_gemm(size_t N, size_t K, size_t M,
                          const elem_t *A,   // N x K int8, row-major, stride K
                          const elem_t *B,   // K x M int8, row-major, stride M
                          acc_t *C);         // N x M int32, row-major, stride M
```

`C[i][j] = sum_k A[i][k] * B[k][j]`, exactly — no bias, no activation, no
output scaling, `int32` accumulation written out at full width (`full_C`).
Called once with `N = 64`, `K = 2048`, `M = 256`.

`A`, `B` and `C` are at fixed DRAM addresses (`0x81000000`, `0x81100000`,
`0x81300000`), 1 MiB-aligned, *outside* the ELF; the harness fills them before
calling you. `C` is zeroed by the caller.

You may replace the `tiled_matmul_auto` call with anything that produces the
same `C`: your own tiling loop over `sp_tiled_matmul_ws`, explicit
`gemmini_config_*` / `gemmini_extended_*` sequences, a different loop order.
You may NOT edit `include/gemmini.h`, `include/gemmini_params.h` (generated by
the hardware elaboration and overwritten anyway) or the benchmark. Read the
`DIM` / `BANK_NUM` / `BANK_ROWS` / `ACC_ROWS` / `MAX_BYTES` macros from
`gemmini_params.h`; never hard-code them.""",
    notes="""\
# Do the arithmetic before you touch the file — where the floor is

`N = 64`, `K = 2048`, `M = 256` -> **33,554,432 int8 MACs**. Derived with the
ROOFLINE METHOD at the top of this file.

    COMPUTE FLOOR
      The systolic array is `DIM x DIM = 16 x 16` and retires one 16-wide row
      per cycle once fed, i.e. `DIM*DIM = 256` MAC/cycle.
      compute floor = 33,554,432 / 256                    = 131,072 cycles

    MEMORY FLOOR   (8 B/cycle — the mbus, the only path to DRAM; corrected
                    2026-09-09 from 16 B/cycle, which is the sbus / Gemmini DMA
                    port and is NOT the bottleneck. Measured: ~7 B/cycle cold,
                    7.94 B/cycle best case — see the gemv-gemmini notes.)
      in  : B 2048*256 = 512 KiB  ->  65,536 c  (mvin, if streamed ONCE)
            A   64*2048 = 128 KiB ->  16,384 c
            total in    = 640 KiB ->  81,920 c
      out : C 64*256*4  =  64 KiB ->   8,192 c  (mvout, acc_t)
      memory floor = max(81,920, 8,192)                   =  81,920 cycles

    roofline = max(131,072, 81,920) = **131,072 cycles — COMPUTE binds**, by
    1.6x (it was quoted as 3.2x while the memory floor used 16 B/cycle; the
    roofline value itself is unchanged because compute still binds). This is the ONLY Gemmini entry in the registry where the array, not
    the weight stream, is the binding resource.

**So this shape is compute bound, unlike the 64x64x64
`gemmini-tiled-matmul-ws` and unlike every `llama-q8-gemv-gemmini-*` entry,
provided B is mvin'd only once.** That proviso is the whole game: the scratchpad is
`BANK_NUM * BANK_ROWS * DIM` = 256 KiB and B is 512 KiB, so B cannot be
resident. With only `N/DIM = 4` row-tiles of A, the natural schedule keeps A
resident and streams B once. A schedule that re-mvins B per I-tile turns 33k
cycles of DMA into 131k and puts you back in DMA-bound territory.

**131,072 cycles is the hard floor.** Anything claiming to beat it is an
arithmetic error.

# What the pristine kernel does

It calls stock `tiled_matmul_auto`, which runs a run-time search over
`tile_I/J/K` (bounded by `sqrt(mats_in_acc)` and the scratchpad partition
size) and then `tiled_matmul_outer`. The search is generic; it does not know
that `dim_I = 64` is tiny and `dim_K = 2048` is huge, which is exactly the
regime where a hand-picked loop order wins.

# Where the cycles above the floor plausibly live

1. **Loop order / B reuse.** `dim_K = 2048` is 128 K-tiles. Whether B is read
   once or `dim_I / (DIM * tile_I)` times is decided by the loop nest, and it
   is worth up to 200k cycles.
2. **Accumulator pressure.** `ACC_ROWS = 1024` rows = `ACC_ROWS / DIM = 64`
   16x16 acc tiles. The output is `4 x 16 = 64` tiles of `acc_t` — exactly the
   accumulator's capacity, with nothing to spare for double buffering, so the
   K-loop has to be innermost per output tile or partial sums spill to DRAM
   and are re-read.
3. **Tile-search overhead** in `tiled_matmul_auto` is inside the timed region.
   At a fixed known shape you can skip it entirely.
4. **Overlap.** Gemmini's mvin / compute / mvout queues run concurrently;
   `gemmini_fence()` and over-eager dependency tracking serialize them.
5. **Cold prologue / drained epilogue**: the first B mvin overlaps nothing and
   the last C mvout overlaps nothing. Small at this size, but real.

# READ THIS BEFORE PROPOSING ANYTHING: the budget is SMALL

The pristine kernel measures **146,737 cycles** against the 131,072-cycle
compute floor (memory floor 81,920 — still not binding) — **1.12x**, i.e.
89% of the array's peak throughput. Only ~15,700
cycles are up for grabs, and some of them are the unavoidable cold prologue.

That is a completely different situation from `gemmini-tiled-matmul-ws`
(64x64x64, 2324 cycles vs a 1024 floor = 2.3x), where the fixed per-call
overhead dominates a tiny problem. Here `dim_K = 2048` amortizes that overhead
over 128 K-tiles, so `tiled_matmul_auto`'s generic tiling is already close to
right, and B *is* being streamed roughly once.

Consequences for how you should think:

- A proposal that claims a large speedup is almost certainly wrong. Check it
  against the floor before you write any code.
- The realistic target is a few percent: the run-time tile search inside the
  timed region, redundant `gemmini_config_*` re-issues, and the cold first
  mvin / drained last mvout.
- A change that breaks correctness scores **zero**. At 1.12x the floor, the
  expected value of a risky restructure is negative. Prefer the small, safe,
  provable win.

# Nothing has been ruled out empirically yet

You are the first optimization session on this kernel. Read
`tiled_matmul_auto`, `tiled_matmul_outer` and `sp_tiled_matmul_ws` in
`include/gemmini.h` (you may read it; you may not edit it) before proposing
anything.""" + _GEMM_ROUND4 + _GEMMINI_GEMM_DMA_HINTS,
)


# ---------------------------------------------------------------------------
# Decode GEMV on Gemmini — is the accelerator even the right unit at N=1?
# ---------------------------------------------------------------------------
# `llama-q8-gemm` measures Gemmini at N = 64 (prefill). Extrapolating that
# number linearly in MACs down to N = 1 predicts a decode step far below the
# memory roofline, which cannot be right: at N = 1 every 16x16 weight tile is
# still mvin'd in full but only ONE of its 16 rows carries a valid activation,
# so the weight DMA is amortized over 1/16 of the work. These two entries
# measure the real thing at N = 1 (one token) and N = 16 (one full array row
# tile, i.e. the best case for a batched decode server).
#
# Same file split as `llama-q8-gemm`: a sealed `main_src` and an agent-owned
# header. There is one header PER N, not one header plus a -D, because
# `cache.baseline_key()` hashes the CONTENT of the agent-owned file — two
# variants with byte-identical headers would share one cached baseline.

_GEMV_GEMMINI_CONTRACT = """\
```c
static void llama_q8_gemv_gemmini(size_t N, size_t K, size_t M,
                                  const elem_t *A,   // N x K int8, stride K
                                  const elem_t *B,   // K x M int8, stride M
                                  acc_t *C);         // N x M int32, stride M
```

`C[i][j] = sum_k A[i][k] * B[k][j]`, exactly — no bias, no activation, no
output scaling, `int32` accumulation written out at full width (`full_C`).
Called once with `N = {n}`, `K = 2048`, `M = 512`.

`A`, `B` and `C` are at fixed DRAM addresses (`0x81000000`, `0x81100000`,
`0x81300000`), 1 MiB-aligned, *outside* the ELF; the harness fills them before
calling you. `C` is zeroed by the caller.

You may replace the `tiled_matmul_auto` call with anything that produces the
same `C`: your own tiling loop over `sp_tiled_matmul_ws`, explicit
`gemmini_config_*` / `gemmini_extended_*` sequences, a different loop order,
or a path that avoids feeding padded rows into the array at all. You may NOT
edit `include/gemmini.h`, `include/gemmini_params.h` (generated by the
hardware elaboration and overwritten anyway), the sealed harness body
`include/llama_q8_gemv_gemmini_body.h`, or the benchmark wrapper. Read the
`DIM` / `BANK_NUM` / `BANK_ROWS` / `ACC_ROWS` / `MAX_BYTES` macros from
`gemmini_params.h`; never hard-code them."""


def _gemmini_gemv(n: int, *, reference_cycles: int | None,
                  sim_timeout_seconds: int, notes: str) -> Kernel:
    """One N-variant of the Gemmini decode GEMV."""
    return Kernel(
        name=f"llama-q8-gemv-gemmini-n{n}",
        target="gemmini",
        bench_dir="bareMetalC",
        kernel_rel=f"include/llama_q8_gemv_gemmini_n{n}_kernel.h",
        main_src=f"bareMetalC/llama_q8_gemv_gemmini_n{n}.c",
        # The sealed harness body is a header, so `include/*.h` in
        # constants.GEMMINI_SEALED_GLOBS already carries it; nothing extra.
        elf_name=f"llama_q8_gemv_gemmini_n{n}.riscv",
        sim_timeout_seconds=sim_timeout_seconds,
        sim_timeout_cycles=100_000_000,
        cycle_re=r"Cycles taken:\s*(\d+)",
        instret_re=None,
        metric_agg="first",
        reference_cycles=reference_cycles,
        # Weight-DMA bound, NOT compute bound — see notes.
        # 2026-09-09: was 65_536 (= 1 MiB / 16 B per cycle, Gemmini's
        # `dma_buswidth = 128 bit`). That is the sbus/DMA-port width, not the
        # path to DRAM. rocket-chip's BaseSubsystemConfig sets
        # `MemoryBusKey => MemoryBusParams(beatBytes = 8)`; chipyard's
        # AbstractConfig does not override it and neither does this SoC's
        # config chain (docker/add_coexist_config.py touches only the sbus and
        # the tile beat width), so DRAM traffic crosses an 8 B/cycle mbus.
        # 1 MiB / 8 B per cycle = 131,072, and the measured mvin-only floor is
        # 132,109 — 0.8% above it, which is the confirmation.
        roofline_cycles=131_072,
        objective=f"Make `llama_q8_gemv_gemmini` — a {n}x2048x512 int8 "
                  f"GEMV/skinny-GEMM, the decode shape of Llama-3.2-1B — run "
                  f"in fewer **cycles** on Gemmini, without changing what it "
                  f"computes.",
        check_desc=f"`bareMetalC/llama_q8_gemv_gemmini_n{n}.c` and the sealed "
                   f"body `include/llama_q8_gemv_gemmini_body.h` (you own only "
                   f"`include/llama_q8_gemv_gemmini_n{n}_kernel.h`) generate A "
                   f"and B from a fixed seed, time your "
                   f"`llama_q8_gemv_gemmini` with `read_cycles()` and print "
                   f"`Cycles taken: N` (that N is the objective), then "
                   f"recompute **8 sampled output elements** with a scalar dot "
                   f"product in one fused pass over k and compare them "
                   f"exactly; any mismatch prints `MISMATCH` and `exit(1)`, "
                   f"which scores the attempt zero. It also prints a checksum "
                   f"over all {n * 512} outputs. The check is SAMPLED (a full "
                   f"scalar reference is {n * 2048 * 512 // 1000}k MACs), so "
                   f"correctness of the rest is on you.",
        contract=_GEMV_GEMMINI_CONTRACT.replace("{n}", str(n)),
        notes=notes,
    )


_GEMV_GEMMINI_ROOFLINE = """\
# Do the arithmetic before you touch the file — this one is MEMORY bound

`N = {n}`, `K = 2048`, `M = 512`.

The weights are the whole story. `B` is `2048 * 512 = 1 MiB` of int8 and every
byte must be moved into the scratchpad exactly once, no matter how small `N`
is. Derived with the ROOFLINE METHOD at the top of this file.

    MEMORY FLOOR
      in  : B 1 MiB (every byte once) + A {n}*2048 B + C zeros
            ~= 1,048,576 B / 8 B per cycle               = 131,072 cycles
      out : C {n}*512*4 B                                =    ~256 cycles
      memory floor = max(...)                            = 131,072 cycles
      (8 B/cycle is the **mbus**, the only path to DRAM: rocket-chip's
      `BaseSubsystemConfig` sets `MemoryBusKey => MemoryBusParams(beatBytes =
      8)` and nothing in this SoC's config chain overrides it. Gemmini's
      `dma_buswidth = 128 bit` and the 128-bit sbus are both upstream of it and
      are NOT the limit. This entry divided by 16 until 2026-09-09, which
      advertised 65,536 and sent three optimization rounds after a phantom 2x.)

    COMPUTE FLOOR
      MACs           = {n} * 2048 * 512 = {macs}
      array peak     = DIM * DIM = 16 * 16 = 256 MAC/cycle
      compute floor  = {macs} / 256 = {compute} cycles

    roofline = max({compute}, 131,072) = **131,072 cycles**

{bound_note}

# The hardware you are driving

- **Array**: `DIM = 16`, weight-stationary, 256 MAC/cycle peak once fed.
- **Scratchpad**: 256 KiB = `BANK_NUM = 4` banks x `BANK_ROWS` rows x `DIM`
  bytes. A and B tiles must COEXIST in it, so a single-buffered B tile can use
  at most about half of it. B is 1 MiB, i.e. 4x the whole scratchpad — it
  cannot be resident, it must be streamed.
- **Accumulator**: 64 KiB across 2 banks = `ACC_ROWS` rows of `DIM` int32.
  At N = 1 the output is one 512-wide int32 row = 2 KiB, so the accumulator is
  essentially EMPTY. There is room to hold the entire output resident across
  the whole K reduction and mvout exactly once.
- **DMA**: a 128-bit port onto a 128-bit sbus, but DRAM sits behind the
  **8 B/cycle mbus**, so **8 B/cycle is the real ceiling** and 7.94 B/cycle is
  what the best kernel measures. Read
  `DIM`/`BANK_NUM`/`BANK_ROWS`/`ACC_ROWS`/`MAX_BYTES` from
  `gemmini_params.h`; never hard-code them.

**131,072 cycles is the hard floor for this shape.** It is set by weight
traffic, not by the array, and no amount of scheduling can move it — the only
thing that would is not re-reading the weights (i.e. batching more tokens
against the same B, which is what N = 16 starts to do, or keeping B resident,
which is impossible: the scratchpad is `BANK_NUM * BANK_ROWS * DIM` = 256 KiB
and B is 1 MiB).

# Why this kernel exists

`llama-q8-gemm` measured Gemmini at `N = 64` and 146,737 cycles for a
64x2048x256 problem. Extrapolating that *linearly in MACs* down to a decode
step predicts a per-token cost far below the memory roofline for the full
1.24 GB of Llama-3.2-1B weights — which is impossible. The reason is exactly
the ratio above: at `N = 64` the weight mvin is amortized over 64 rows of
work, at `N = 1` over one. This entry replaces that extrapolation with a
measurement.

# Where the cycles above the floor plausibly live

1. **Padded rows.** `dim_I = {n}` is rounded up to a multiple of `DIM = 16`
   inside `tiled_matmul_auto`. At N = 1 that means 15 of every 16 rows fed
   through the array are zero padding. They cost array time, and they cost
   mvin bandwidth for the padded `A` tile.
2. **Tile-search overhead** in `tiled_matmul_auto` runs inside the timed
   region and does not know the shape is this lopsided.
3. **Overlap.** With so little compute per weight tile, mvin latency is much
   harder to hide than in the prefill GEMM: any `gemmini_fence()` or
   dependency stall in the K loop is directly on the critical path.
4. **Accumulator use.** The output is only `ceil({n}/16) x 32 = {acctiles}` acc
   tiles out of `ACC_ROWS / DIM = 64`, so the accumulator is nearly empty —
   there is room to keep more of the K reduction on-chip than the generic
   tiling assumes.

# Before you propose anything

Read `tiled_matmul_auto`, `tiled_matmul_outer` and `sp_tiled_matmul_ws` in
`include/gemmini.h` (you may read it; you may not edit it). Round 1 of the
optimization loop ran on the **N = 1** variant; its notes carry the full
ruled-out list and it mostly transfers to N = 16 as well.{measured}"""


_GEMV_GEMMINI_N1_MEASURED = """

# MEASURED — this is the number the whole "Saturn or Gemmini for decode?"
# question turns on

| what | cycles | vs floor |
|---|---|---|
| pristine `tiled_matmul_auto`, N=1 | **150,339** | 2.29x |
| Round 1 best (manual `sp_tiled_matmul_ws`) | **132,653** | 1.012x |
| Round 3 best (+ Zicbop hints) | **132,424** | 1.010x |
| mvin-only probe, same shape/order/cache state | 132,109 | 1.008x |
| weight-DMA floor (8 B/cycle mbus, the real link) | 131,072 | 1.00x |

Derived rates, against the *same* 1 MiB of weights:

    stock  weight bandwidth  = 1,048,576 B / 150,339 = 6.97 B/cycle  (87%)
    best   weight bandwidth  = 1,048,576 B / 132,424 = 7.92 B/cycle  (99%)
    (percentages are of the 8 B/cycle mbus, the actual link to DRAM)
    stock  array utilisation = 1,048,576 MAC / 150,339 = 6.97 MAC/c  (2.7%)

The array is 97% idle and that is CORRECT for this shape — it is waiting on
weights, exactly as the roofline says. Chasing MAC/cycle here is chasing the
wrong number; the only metric that matters is B/cycle of weight traffic.

**`tiled_matmul_auto` at N=1 sustains ~7 B/cycle, 87% of the 8 B/cycle the
mbus can carry; the tuned kernel reaches 99%.** And N=16 takes 168,367 cycles — 12% MORE time for 16x
the arithmetic. Same 1 MiB of weights, same ~7 B/cycle, array idle in both.
There is no clearer proof that this kernel is pure weight-DMA bound.

For scale: the same 512x2048 problem on Saturn (`llama-q8-gemv`, naive RVV)
takes 621,724 cycles = 1.69 B/cycle, so Gemmini is 4.1x faster at N=1 despite
using 2.7% of its array.

# THE COMPOSITE (Saturn + Gemmini) IDEA IS CLOSED — 2026-09-09

Earlier versions of this note computed a "composite floor" of 95,325 cycles
from `7 B/cycle (Gemmini) + 4 B/cycle (Saturn) = 11 B/cycle < 16 B/cycle
sbus`. **That arithmetic was wrong, and this is why:** the 16 B/cycle sbus is
not the shared resource. Both requesters go through the same 512 KiB L2 and
out the same **8 B/cycle mbus** to DRAM, and Gemmini alone already draws
7.92 of those 8 B/cycle. There is nothing to add. The rates cannot sum past 8,
so the composite floor is 1,048,576 / 8 = 131,072 — exactly the same floor
Gemmini reaches by itself.

The five measured attempts agree: 133,800 / 134,585 / 140,285 / two self-check
failures on this kernel, and 738,754 (vs 659,142) on the lmhead shape. Every
one of them was net-negative because host traffic displaces Gemmini's share of
a saturated link rather than adding to it.

**Do not propose Saturn/host co-compute for this kernel again, in any form,
with any mechanism, at any split size.** It is not a promising-but-unproven
idea; it is a refuted one with a mechanism that explains all five results.

# RULED OUT — Round 1, 8 iterations, out/loop/20260907-043120-a7d0/

| # | technique | cycles | why it did not help |
|---|---|---|---|
| 1 | **Manual weight-stationary tiling** via `sp_tiled_matmul_ws` directly: `tile_I=1`, `tile_J` = the whole M=512 (fits in half the accumulator), `tile_K` as large as half the scratchpad allows -> 9 `loop_ws` calls (8 of `tile_K=15` + a tail of 8). Whole 512-wide int32 output stays in the accumulator across the entire K reduction, ONE mvout at the end. K tiles visited in **descending** order. | **132,653** | **WINNER.** Streams B exactly once instead of stock's 4x walk in 128 B pieces. |
| 2 | Split the K reduction: Gemmini does rows 0-1791, host RVV does the last 1/8, vector-add the partial into C after a fence. Written with `<riscv_vector.h>` intrinsics. | BUILD FAIL | The Gemmini harness compiled `-march=rv64gc` — no `v` extension, so the intrinsics header did not resolve. **This is being fixed right now; assume `#include <riscv_vector.h>` WORKS for you.** |
| 3 | Same CPU/Gemmini K-split (host takes 1/8 = 256 rows), reimplemented as inline asm with `.option arch,+v` plus `csrs mstatus, MSTATUS_VS` at entry (crt.S leaves the vector unit off). Correct. | 134,585 | Slower than #1. The host slice became the critical path / stole bandwidth. |
| 4 | Same split with a much SMALLER host share (128 rows = 1/16), to test "the CPU slice was too big". | 133,800 | Still slower than #1. Refutes the slice-size theory; points at a shared memory port. |
| 5 | Re-ran #1 unchanged, to confirm determinism. | 132,653 | Reproducible. |
| 6 | **Ascending** K order instead of descending (one-line change, everything else identical) — a control experiment on #1's cache claim. | 149,757 | **+17,104 cycles.** Ascending forfeits the L2 residency; descending order is real and worth ~13k cycles. Do not "clean this up". |
| 7 | **Host-side software prefetch**: while Gemmini works on loop `kk`, the idle core issues batches of scalar `lb` at 64 B stride (8 lines in flight, 5,000-cycle `rdcycle` budget per window) to pull the next K tile's B into L2. | 142,221 | +9,568. The host's own traffic competes with Gemmini for the shared path instead of helping. |
| 8 | Restored #1, stopped. | 132,653 | Declared the empirical floor for this shape. |

Hardware facts Round 1 established empirically — these are worth more than the
attempts themselves:

- **Cold DRAM reads sustain only ~7.0-7.1 B/cycle**, not the 16 B/cycle the
  sbus can carry. (From iteration 6: 1 MiB / 149,757 cycles with no cache
  help at all.)
- **The harness fills B in DRAM immediately before calling you, so the TAIL of
  B is still warm in the (random-replacement) L2** — worth roughly 200-290 KiB
  of the 1 MiB, served at up to ~16 B/cycle. Consuming K tiles in descending
  order harvests that before read-misses on the head of B evict it. This is
  a property of the BENCHMARK, and it is why 132,653 already beats the
  ~148k cycles a pure-DRAM model predicts.
- Byte budget for the 132,653: ~835 KiB of true misses at ~7.1 B/cycle
  (~118k cycles) + ~215 KiB of L2 hits at ~16 B/cycle (~13k) + ~1.5k of
  fixed fill/drain/mvout/fence overhead.
- **9 `loop_ws` calls is the minimum** the half-scratchpad constraint allows
  for this tiling.
- Gemmini's DMA and the host core appear to share one tile-egress/L2/DRAM
  path — every attempt to run host memory traffic concurrently (#3, #4, #7)
  was net-negative. See the honesty warning above for why this may not be the
  last word.

# WHAT IS LEFT ON THIS KERNEL

Almost nothing, and that is the honest answer. 132,424 is 1.010x the
131,072-cycle link floor and 1.002x the mvin-only probe that used the same
request shape, order and cache state. Round 1 moved 6.97 -> 7.90 B/cycle by
streaming B exactly once; Rounds 2 and 3 spent ~25 iterations and found 229
more cycles, because there are only ~1,300 left and ~1,000 of those are DMA
start-up and drain.

Direction (a), keeping the DMA queue saturated, is DONE for this shape: the
9 `loop_ws` calls already emit 16-row x 64 B mvins with the hardware loop
generating the commands, which is the maximum in-flight request count the
2-slot command tracker permits. Direction (b), giving Saturn a slice, is
CLOSED (see above).

Read the Round-4 status block appended at the end of these notes before you
plan anything."""


_GEMV_N1_ROUND7 = """\


# ROUND 7 (2026-09-13): THIS IS A **PROBE** ITERATION. THE NUMBER MATTERS MORE
# THAN THE SCORE.

You have 2 iterations on this kernel and it is already at 132,424 = 1.010x its
link floor. **Do not try to beat 132,424.** Two rounds and ~25 iterations have
found 229 cycles. Spend these iterations producing measurements nobody has,
printed with `printf`, and say so explicitly in your write-up. An iteration
that scores 140,000 but answers the question below is a SUCCESS; an iteration
that scores 132,500 and answers nothing is a failure.

## The question: the in-flight curve was measured right and explained wrong

Round 4 measured, end to end on the lmhead shape, that rows-per-mvin 4
(= 8 requests in flight) is the optimum and that 8 and 16 rows (16 in flight)
are ~9% worse. The DMA-hints section below explains that with a story about
"a full DRAM round trip exposed as a bubble". **That story cannot be right on
this simulator, and here is the proof:**

  * The Verilator harness instantiates **`mm_magic_t`**, not DRAMSim2.
    `testchipip/src/main/resources/testchipip/csrc/SimDRAM.cc:90-97` picks
    `mm_dramsim2_t` only when the simulator is launched with `+dramsim`, and
    `loop/nodes.py:run_kernel` never passes `dramsim_ini_files`, so it is not.
  * `mm_magic_t::tick` (`testchipip/.../csrc/mm.cc:44-110`) has
    `ar_ready()` hard-wired to `true` and pushes **every beat of a read burst
    into the response queue in the same cycle the address fires**. There is
    no latency model, no bank, no row buffer, no refresh, no reordering. The
    only thing that throttles it is the AXI R channel draining one beat per
    cycle at the mbus's 8 B.

So there is **no DRAM-side latency to hide** in this experiment. Every cycle
above `bytes / 8` is an ON-CHIP cost, and the "cold DRAM sustains 7.0-7.1
B/cycle" line repeated throughout these notes is a FIT to on-chip behaviour
that was then given a DRAM name. Whatever makes 16 requests in flight worse
than 8 is inside the SoC.

## The leading suspect, with numbers

    Gemmini StreamReader  max_in_flight_mem_reqs = 16   (XactTracker nXacts)
    SiFive InclusiveCache L2   MSHRs = 12
        (elaborated DTS, `sifive,mshr-count = <12>`, in
         out/coexist/elaborate.GENV256D128GemminiShuttleConfig.log:299 and
         out/coexist/build.control.log:124 — it is in every elaborate log for
         this config)

**Gemmini is allowed to have more read transactions outstanding than the L2
has MSHRs to track them.** Past 12, the extra Gets cannot allocate and the L2
backpressures the sbus, which stalls Gemmini's StreamReader in
`waiting_for_dma_req_ready` — the same symptom Round 4 attributed to DRAM.
That predicts a knee somewhere in `[8, 12]` and monotonic degradation above
it, which is exactly the measured shape. It does NOT by itself explain why 10
and 12 already measured worse than 8, so treat it as a hypothesis with a
plausible mechanism, not as the answer. Other on-chip candidates worth naming
in your HYPOTHESIS line: the sbus xbar's arbitration between the Gemmini
master and the Shuttle D$ master, and InclusiveCache's own directory-access
serialisation on a miss burst.

## WHAT TO ACTUALLY DO

**Iteration 1 — the in-flight sweep, with the numbers printed.**
Keep the winning 9-`loop_ws` structure for the SCORED part if you like, but
add a probe section that runs the *same* 1 MiB weight stream as an mvin-only
loop five times, at five in-flight depths, timing each with `read_cycles()`
and printing one line per depth:

    PROBE inflight=6  cycles=NNNNNN
    PROBE inflight=8  cycles=NNNNNN
    PROBE inflight=10 cycles=NNNNNN
    PROBE inflight=12 cycles=NNNNNN
    PROBE inflight=16 cycles=NNNNNN

In-flight depth is `2 x (rows per mvin)` because `DMACommandTracker` keeps
`nCmds = max_in_flight_mem_reqs/DIM + 1 = 2` mvin commands open at a time and
`LoadController` issues one row per cycle per open command. So rows-per-mvin
of 3 / 4 / 5 / 6 / 8 gives 6 / 8 / 10 / 12 / 16. If you want depths the row
count cannot express, throttle explicitly: issue R rows, then spin on
`rdcycle` for a fixed number of cycles before the next mvin, and report the
effective depth you believe you achieved. **Print the depth you actually
programmed, not the one you intended.**

Flush the L2 between depths (stream >= 1 MiB of unrelated DRAM through it with
mvins you do not time) so every depth sees the same cache state. Say in your
write-up which depth was measured first, because the first one is the only one
that is genuinely cold for the whole SoC.

**Iteration 2 — the cold-state number the entire Llama cost model rests on.**
Every rate quoted in these notes (7.92 B/cycle here, 6.6 B/cycle on lmhead)
was measured with the harness's warm-L2 tail in it — the benchmark fills B
immediately before calling you, leaving ~200-260 KiB resident, and descending
K order harvests it for 17,104 cycles. That is a benchmark artifact; a real
decode step has just had its L2 destroyed by the previous layer's weights.
So: flush the L2 with >= 1 MiB of unrelated DMA traffic, THEN run the winning
kernel, and print

    PROBE flush_cycles=NNNNNN
    PROBE cold_gemv_cycles=NNNNNN
    PROBE cold_bytes_per_cycle_x100=NNNN

Round 4's iteration 1 tried exactly this and every number was lost, because
back then the harness only forwarded the simulator log on a FAILING run. **It
forwards it on passing runs now** (`out/loop/<run_id>/simlog_NN.txt` plus a
"=== benchmark stdout (tail) ===" section in the feedback, 4 KB cap). Prefix
every line with `PROBE `, one number per line, and it comes back.

Expected shape of the answer: the warm-tail credit is ~17k cycles, so a cold
run should land near 149,000-150,000 and the cold rate near 7.0 B/cycle. If it
comes back materially better than that, the warm tail is smaller than believed
and `llama-layer-fused-n1`'s cold baseline should be re-read in that light. If
it comes back much worse, every Gemmini number in the Llama projection is
optimistic and that is the single most important correction available this
round.

## Companion entry

`llama-layer-fused-n1` (new this round) runs this exact GEMV shape in a
deliberately COLD harness, in the same timed region as one attention head's
Saturn work. Its baseline is the direct measurement of the cold number you are
being asked to probe for here, from the other direction. If the two disagree,
say so — two independent measurements of the same quantity disagreeing is the
most informative result either run can produce."""



LLAMA_Q8_GEMV_GEMMINI_N1 = _gemmini_gemv(
    1,
    reference_cycles=150_339,
    # Measured: 1124 s of Verilator wall (this run shared a worker with the
    # N=16 one, so it is a slight over-estimate). 3x that.
    sim_timeout_seconds=3400,
    notes=_GEMV_GEMMINI_ROOFLINE.format(
        n=1, macs="1,048,576", compute="4,096", acctiles=32,
        bound_note="""\
So at N = 1 the array could in principle finish in 4,096 cycles and then sit
idle for 126,976 waiting on weights: the DMA bound is **32x** the compute
bound. This is a memory-bound kernel wearing a systolic array's clothes.""",
        measured=_GEMV_GEMMINI_N1_MEASURED)
    + _GEMV_N1_ROUND4 + _GEMV_N1_ROUND7 + _GEMMINI_DMA_HINTS,
)

_GEMV_N16_ROUND5 = """\


# ROUND 5 RESULT (2026-09-11, out/loop/20260911-065653-8115/, 10 iterations)

    weight-DMA floor (1 MiB / 8 B per cycle)      131,072
    pristine `tiled_matmul_auto`, N=16 (baseline) 168,367   (1.28x)
    ROUND 5 BEST (iteration 2)                    142,458   (1.087x)
    its N=1 sibling, SAME 1 MiB of weights        132,424   (1.010x)

Iteration 1 transplanted the N=1 winner's structure (hand `gemmini_loop_ws`
stream, tile_I=1, tile_J=32 so all of C stays in the accumulator, K
DESCENDING, B streamed once, one mvout) -> 144,521. Iteration 2 set
`tile_K = BANK_ROWS/(tile_J*DIM) = 8` so each loop's B slice fills exactly ONE
scratchpad bank (alternating banks 1 and 3) and A sits alone in bank 0/2 ->
**142,458**. Eight further iterations found nothing; the cycle sequence was
144521, **142458**, 768839(probe), 158708, 155636, 142609, 143005,
1036276(probe), 145220, 168980(probe).

# WHERE THE 142,458 GOES — MEASURED, NOT MODELLED

Every term below is a printf probe from this run (iterations 3, 8, 10), not
an estimate:

| term | cycles | how measured |
|---|---|---|
| B streamed cold, mvin only, no ex/mvout | 141,805 | iter-3 `PROBE T3_mvin_Bonly_16row` |
| full kernel, same (cold/post) state | 149,764 | iter-3 `PROBE T5_full_poststate` |
| full kernel, scored state (warm B tail) | 142,734 | iter-3 `PROBE T1_full_scoredstate` |
| A (32 KiB) standalone, cold | 5,593 | iter-8 `PROBE A2048_cold` |
| A in situ (marginal cost inside the stream) | ~900 | T1 - T3 |
| C (32 KiB) mvout standalone | 2,109 | iter-10 `PROBE mvout32` |
| host finishes issuing all loop commands at | 105,790 | iter-10 `PROBE issue_done_at` |
| drain after last command issued | 36,283 | iter-10 `PROBE tail_after_issue` |

So the account closes as: **cold B stream 141.8k + A in situ ~0.9k + exposed
tail <=1.4k - warm-L2 harvest ~2k ~= 142.4k.**

  * **compute is free.** 65,536 cycles of array time (16 B/cycle of B
    consumption, exactly half the DMA's supply rate) is entirely hidden — the
    full kernel is only ~900 cycles above a mvin-only stream of the same B.
  * **the 32 KiB mvout of C is free** (2,109 standalone, fully overlapped).
  * **A is nearly free** (~900 in situ) even though it costs 5,593 alone.
  * the whole 11.4k above the floor sits INSIDE the B stream: 1 MiB /
    141,805 = 7.39 B/cycle, and there is no ex, no A and no mvout in that
    number.

# HARDWARE FACTS ESTABLISHED (these kill whole families of ideas)

  * **DMA throughput is set by a fixed ~7-8.6 cycle/request issue cadence and
    a 16-deep in-flight pool, NOT by DRAM locality.** Iteration 8 read the
    same 32 KiB with row stride 2048 / 512 / 64: 5,593 / 5,787 / 5,647 cold
    (within 3%) and 3,630 warm. Stride, DRAM page and row-conflict arguments
    are DEAD for this machine. `MAX_BYTES = 64` fixes the request size, so
    request shape is not a tunable.
  * warm (L2-hit) DMA reads only reach ~9 B/cycle, cold ~5.8-7.4 B/cycle.
  * 1-row mvin commands are catastrophic (only 2 open commands -> 2 requests
    in flight): iteration 4 -> 158,708. 4-row commands are mildly better than
    16-row on a mixed A+B stream (iter-3 T4 148,560 vs T2 153,012) but not on
    the real kernel (iter 5 -> 155,636).
  * **any host memory traffic concurrent with the Gemmini stream is a net
    loss**, confirmed a second time: iteration 9's 512 L2-hit scalar loads
    (to flush dirty L1D lines) cost +2,762. Same conclusion as Round 1's
    `lb` prefetch (+9,568) and the Zicbop `prefetch.r` attempt (iteration 6,
    142,609 = noise). Host and Gemmini share one path to L2/DRAM.
  * splitting the mvout of C in halves to start it earlier is NOT expressible:
    `loop_ws` maps the accumulator as `c_addr = (i*J + j)*DIM`, so a J-split
    does not produce a contiguous mvout region.

# RULED OUT IN ROUND 5 — DO NOT RETRY

| # | idea | result |
|---|---|---|
| 1 | N=1 structure transplant, tile_K=15 | 144,521 |
| 2 | tile_K=8, one bank per B slice (bank-conflict fix) | **142,458 BEST** |
| 4 | hand A mvin, 1 row x 64 col commands, `a=NULL`+`a_spad_id` | 158,708 |
| 5 | same but 4 rows x 64 col commands | 155,636 |
| 6 | Zicbop `prefetch.r` of the next A slice from the idle host | 142,609 |
| 7 | tile_K=4 so ldA emits one command per loop | 143,005 |
| 9 | host L2-hit sweep to flush dirty L1D before mvout | 145,220 |

Dead hypotheses with their mechanism: A's stride-2048 scatter (iter 8 probe),
A's command shape (iters 4/5/7), dirty-L1D coherence probes (iter 9), exposed
ex/mvout tail (iter 10: mvout32 is only 2,109), host-side prefetch (iter 6).

# WHAT IS ACTUALLY LEFT: THE WARM-L2 HARVEST DEFICIT (~10k, UNEXPLAINED)

This is the ONE anomaly Round 5 did not close, and it is quantified:

    N=1 : cold (ascending) 149,757 -> warm (descending) 132,424   = -17,333
    N=16: cold (T5)        149,764 -> warm (T1)         142,734   =  -7,030

Both kernels stream the SAME 1 MiB of B, both descend K, both run against the
same harness fill order (A, then B, then C zeroed). Converting the saving to
bytes at (1/7.0 - 1/16) cycles/byte: **N=1 harvests ~215 KiB of warm L2 tail,
N=16 only ~87 KiB.** ~128 KiB of warm tail is being lost at N=16.

Partial explanation (worth maybe half of it): at N=16 the harness zeroes a
32 KiB C (vs 2 KiB) right after filling B, and the kernel itself allocates
32 KiB of A (vs 2 KiB) into a 512 KiB random-replacement L2 while the tail is
still being consumed — ~60 KiB of extra eviction pressure that N=1 does not
have. That does not obviously account for 128 KiB.

Second candidate mechanism, untested: at N=16 the array consumes B at exactly
16 B/cycle, the same rate an L2 hit can deliver it, and B is only DOUBLE
buffered (banks 1 and 3, 64 KiB each). During the warm phase the DMA is
therefore rate-matched to ex and can never run ahead to bank the cheap bytes;
at N=1 ex consumes 16x slower so the DMA sprints through the warm tail freely.

**TARGET IF THAT IS REAL: ~136,000-139,000. NOT 133,000** — the cold portion
of B (>=768 KiB at 7.0-7.4 B/cycle) is a hard ~110k on its own.

# FIRST ITERATION OF ANY ROUND 6 ON THIS KERNEL (probe first, do not guess)

    HYPOTHESIS: the warm B tail (~256 KiB) is either not being harvested at
                all or is evicted before use; at 16 B/cycle the first four
                loops (tile_K=8, 64 KiB of B each) should take ~4,100 cycles
                each, at cold rate ~8,700.
    CHANGE:     best kernel unchanged; rdcycle timestamp after each of the 16
                loop issues AND after a fence per loop, printed as
                PROBE loop_NN=<cycles>. Run it twice in one build: scored
                state and a cold replay, so the per-loop warm delta is direct.
    EXPECTED:   if loops 0-3 are ~4,100 in the scored state, the tail IS
                harvested and 142,458 is the floor -> declare SOLVED.
                If they are ~8,000+, the tail is being lost and the next
                change is to deepen B buffering from 2 banks to 3 (pack the
                2 KiB/loop A slice into the tail rows of bank 0 instead of
                giving it two whole banks) so the DMA can run ahead of ex
                through the warm region.

Anything that touches A's shape, A's stride, the mvout tail, or host-side
prefetching has already been measured and is a waste of an iteration."""


_GEMV_N16_STATUS = """\


# STATUS OF THIS KERNEL: SOLVED (2026-09-11, after Round 6). READ THIS FIRST.

    weight-DMA floor (1 MiB / 8 B per cycle, mbus)   131,072   1.00x
    pristine `tiled_matmul_auto` (baseline)          168,367   1.28x
    **FINAL: 142,458** (Round 5 iter 2, unchanged)   142,458   1.087x

Round 6 spent all four iterations proving, with per-loop `rdcycle` probes,
that every cycle of the 11,386 above the nominal floor is hardware state, not
kernel structure. Do NOT open another round on this kernel. The two branches
the Round-5 note left open (deepen B buffering to 3 banks; hand-issue B) are
both CLOSED with measurements below."""


_GEMV_N16_ROUND6 = """\


# ROUND 6 RESULT (2026-09-11, out/loop/20260911-115840-5f95/, 4 iterations)

    142,458 STANDS. Nothing in Round 6 beat it.

    | iter | what | cycles |
    |---|---|---|
    | 1 | per-loop rdcycle PROBE (scored + 2 replays), kernel unchanged | 688,048 (probe) |
    | 2 | hand-issued B: `b=NULL` + `b_spad_id=slot+1`, 4-row mvin2 | 214,211 |
    | 3 | 3-pass PROBE: scored / clean-cold / host-dirtied-L2 replay | 1,992,395 (probe) |
    | 4 | PROBE: swap first two K slices + time a 2nd mvout | 268,314 (probe) |

# THE PER-LOOP NUMBERS (iteration 1, 16 loops x 64 KiB of B, K DESCENDING)

`s_gap` = host issue gap in the SCORED state, `r_gap` = clean cold replay.
Loops 0-2 are absorbed by the command queue, so steady state starts at loop 3.

    loop   00    01    02     03     04     05     06     07
    s_gap 136    24    24   8351   5949   5967   6206   6651
    r_gap  80    17    19   9613   9481   9612   9356   9226

    loop   08    09    10     11     12     13     14     15   drain
    s_gap 7082  7405  7837   8885   9586  10287  10740  10833  36061
    r_gap 9467  9729  9437   9326   9260   9346   9613   9521  30903

    s_total = 142,024   r_total (clean cold) = 154,014

# VERDICT ON THE ROUND-5 DECISION CRITERION: the criterion was WRONG, and the
# answer it was looking for is nonetheless "warm is already saturated".

The Round-5 note said "if loops 0-3 are ~4,100 (16 B/cycle) the tail IS
harvested -> SOLVED". Measured warm steady state is **5,848-5,967 cycles per
64 KiB = 11.0-11.2 B/cycle**, not 4,100. But 16 B/cycle was never a DMA rate:
it is the ARRAY's consumption rate. Round 5 itself measured the warm (L2-hit)
DMA path at only ~9 B/cycle. The scored run reaches 11.2 B/cycle on the warm
loops, i.e. it is ALREADY ABOVE the previously measured warm ceiling. There is
no 4,100-cycle loop available on this machine at any buffering depth.

The decay 5,848 -> 10,833 is exactly L2 residency running out, not a buffering
stall: loop 3-5 run at 11.2 B/cycle, and the last loops at 6.07 B/cycle, below
even the clean-cold replay's flat 6.97 B/cycle.

# ITERATION 3 IS THE DECISIVE EXPERIMENT — THE CURVE IS STATE, NOT KERNEL

Three passes of the SAME best kernel in one call: (s) scored state, (r) clean
cold replay, (d) replay after the host re-dirties L2 exactly the way the
harness fill does (rewrite every byte of B with itself, then zero C).

    loop    03    04    05    06    07    08    09    10    11    12    13    14    15
    s_gap 8314  5848  5984  6334  6765  7098  7617  8084  9258  9963 10233 10397 10737
    d_gap 8349  5848  6011  6308  6769  7075  7463  7908  8774  9608 10080 10342 10745
    r_gap 9069  9105  9280  9275  9632  9582  9748  9414  9568  9551  9487  9878  9633

    s_total = 142,892   d_total = 141,213   r_total = 154,742

(d) reproduces (s) loop for loop (141,213 vs 142,892); (r) is flat. So the
whole shape of the curve — the warm head AND the sub-cold tail — is decided by
what the harness left in L2/L1D before the kernel starts. The kernel already
extracts the warm head optimally (K descending reads the most recently written
bytes first); the ~9k the tail loops spend BELOW the clean-cold rate is the
writeback cost of evicting the harness's ~450 KiB of dirty fill lines. No
software in the kernel can avoid that: an MMIO flush costs ~50 cycles/line for
~7k lines, and a host read inserted in the warm phase loses a 4.7-cycle hit to
save a 1.4-cycle writeback.

# BRANCH B (DEEPEN B BUFFERING TO 3 BANKS) IS STRUCTURALLY IMPOSSIBLE

`BANK_NUM=4, BANK_ROWS=4096` -> a `loop_ws` slot owns `spad_half = 8192` rows
= exactly TWO banks, and it must hold A as well. B can therefore never exceed
~1.94 banks inside a slot; tile_K=15 (7,680 rows, the maximum) was already
measured in Round 5 iteration 1 -> 144,521, worse, because A then shares a
bank with B. "3 banks of B" requires abandoning `loop_ws`'s ldB and hand-
issuing B, which is what iteration 2 did:

  * **hand-issued B costs ~52 cycles per command at N=16** (4,096 commands ->
    214,211). Self-check passed, so the `b_spad_id = slot+1` / bank-1/bank-3
    layout was correct; the cost is arbitration. `LoopMatmul`'s command output
    wins the arbiter over the host's direct path, and at N=16 the ex queue is
    often full, so host mvins only slip through in the gaps. (The same trick
    costs only 38.7 cycles/command in the N=1 lmhead kernel, whose ex queue is
    nearly empty — that is why it worked there and cannot work here.)
  * **Therefore: ANY hand-issued Gemmini command is dead at N=16.** B's command
    shape is whatever `loop_ws` emits (16 rows x 64 B) and is not a tunable.

# ITERATION 4: THE LAST TWO UNATTRIBUTED ITEMS

Swapping the first two K slices did NOT move the big first gap (loop 3 still
8,330 with a different k_start), so the ~2.5k that loop 3 carries above the
5.85k steady state is **pipeline/queue startup, not an L1D probe cost**. A
second mvout of the same accumulator region timed 2,097 cycles, matching the
2,109 standalone mvout of Round 5, so there is **no L1D-probe surcharge on the
mvout** either; the 35,883-cycle drain is simply the DMA backlog the host ran
ahead of, not an exposed tail.

# FINAL ACCOUNT OF 142,458 — EVERY TERM MEASURED

| term | cycles |
|---|---|
| warm segment, ~448 KiB at 11.2 B/cycle (DMA hit-path max) | ~40,000 |
| pipeline / queue startup | ~2,500 |
| cold segment, ~576 KiB at 6.1-7.0 B/cycle | ~90,000 |
|   ...of which dirty-fill writeback contention (state, not kernel) | ~9,000 |
| exposed tail (last tile ex + 32 KiB mvout + fence) | ~3,900 |

# DO NOT RETRY (Round 6 additions to the Round 5 list)

| idea | result |
|---|---|
| hand-issued B, 4-row mvin2 into the loop's own slot | 214,211 |
| deepen B to 3 banks inside `loop_ws` | impossible: slot = 2 banks |
| flushing / re-dirtying L2 from the host | iter 3: the cost is state-bound |
| reordering K slices to dodge an L1D probe | iter 4: there is no probe cost |

**CONCLUSION: 142,458 (1.087x the 8 B/cycle weight-DMA floor) is the floor of
this microarchitecture for this kernel under `loop_ws`, the fixed 16-row DMA
request shape, and the dirty L2 the harness leaves behind. The ideal "clean
cold + full warm harvest" number is ~133k; the ~9k difference is hardware
state. SOLVED — close the loop on this kernel.**"""


LLAMA_Q8_GEMV_GEMMINI_N16 = _gemmini_gemv(
    16,
    reference_cycles=168_367,
    sim_timeout_seconds=3400,
    notes=_GEMV_GEMMINI_ROOFLINE.format(
        n=16, macs="16,777,216", compute="65,536", acctiles=32,
        bound_note="""\
At N = 16 the compute floor is 65,536 cycles and the memory floor is 131,072,
so weights still bind by 2x. (Before 2026-09-09 this entry divided the weight
bytes by 16 B/cycle instead of the mbus's 8 and therefore claimed the two
floors were exactly equal here.) The break-even batch size — where the array
finally consumes weights as fast as the 8 B/cycle link delivers them — is
**N = 32**, not 16. Below it you are paying for weight bandwidth you cannot
use; above it you are back in the prefill regime.""",
        measured='\n\n# MEASURED\n\n| what | cycles | vs floor |\n|---|---|---|\n| pristine `tiled_matmul_auto`, N=16 | **168,367** | 1.28x |\n| N=1 variant, same weights (best) | 132,424 | 1.010x |\n| weight-DMA floor (8 B/cycle mbus) | 131,072 | 1.00x |\n| compute floor (256 MAC/cycle) | 65,536 | — |\n\n    weight bandwidth  = 1,048,576 B / 168,367 = 6.23 B/cycle  (78% of 8 B/c)\n    array utilisation = 16,777,216 MAC / 168,367 = 99.6 MAC/c (39% of 256)\n\n**16x the arithmetic for 12% more cycles.** Both N=1 and N=16 move the same\n1 MiB of weights and both land near 6-7 B/cycle, so the cost of this kernel is\nset almost entirely by weight traffic. Batching decode requests up to N=16 is\ntherefore close to free on this hardware; past N=16 you leave the GEMV regime\nand `llama-q8-gemm` is the relevant measurement.\n\nThe gap to the corrected floor is 1.28x, not the 2.57x this table showed\nbefore 2026-09-09, and Round 6 closed it: the kernel now sits at **142,458 = 1.087x** and the\nremaining 11.4k is measured hardware state (see STATUS below), not headroom.')
    + _GEMV_N16_STATUS + _GEMV_N16_ROUND5 + _GEMV_N16_ROUND6
    + _GEMMINI_DMA_HINTS,
)


# ---------------------------------------------------------------------------
# The non-GEMM operators of a Llama-3.2-1B decoder layer, on Saturn
# ---------------------------------------------------------------------------
# Together these are ~1% of decode cycles, so they exist to be MEASURED, not
# optimized: the point is to replace `roofline x fallback_factor` guesses in
# out/llama-profile/measured_cycles.json with real Verilator numbers.
#
# All seven follow the `llama-q8-gemv` pattern exactly: a sealed `_main.c` with
# main() + an LCG data generator + a scalar golden model, a sealed `.h` with
# the shape and a DRAM memory map starting at 0x81000000 (nothing big in .bss:
# the TSI loader zeroes it at ~12 s/KiB), an `empty.S` so the `*.S` glob has
# something to match, and ONE editable `.c`.
#
# dtype: fp32 activations throughout. loop/llama_model.py costs every one of
# these kernels at `acc_bytes = 4`, which is the same assumption. The two
# attention kernels also hold the KV cache in fp32, where ModelSpec assumes
# int8 (`kv_bytes = 1`); that makes them move 4x the bytes, i.e. the measured
# number is the CONSERVATIVE end of the range. Quantising the cache is a
# separate piece of work, not a kernel optimization.

_LLAMA_EW_CONTRACT_TAIL = """

The buffers are at fixed DRAM addresses *outside* the ELF (see the sealed
`.h`), filled at run time from a fixed LCG seed. There is no warm-up call, so
the first pass over the data is cold. Only the one `.c` file above reaches the
compiler from your workspace; the `.h` and the `_main.c` are sealed."""


LLAMA_RMSNORM = Kernel(
    name="llama-rmsnorm",
    target="saturn",
    bench_dir="llama-rmsnorm",
    kernel_rel="llama-rmsnorm/llama_rmsnorm.c",
    elf_name="llama-rmsnorm.riscv",
    bench_globs=_saturn_globs("llama-rmsnorm"),
    # Measured on GENV256D128GemminiShuttleConfig: 213 s of Verilator wall
    # for one run (ELF load + boot + data fill + kernel + scalar check),
    # with seven of these baselines running concurrently. 3x that.
    sim_timeout_seconds=900,
    sim_timeout_cycles=8_000_000,
    reference_cycles=5_454,
    roofline_cycles=1536,
    objective="Make `llama_rmsnorm` — the fp32 RMSNorm over a 2048-wide "
              "Llama-3.2-1B hidden state — run in fewer **cycles** (`mcycle`), "
              "without changing what it computes.",
    check_desc="`llama-rmsnorm_main.c` (SEALED) fills `x` and `w` from a fixed "
               "LCG seed, calls `setStats()` around your `llama_rmsnorm`, then "
               "recomputes **all 2048 outputs** with a scalar double-precision "
               "reference and compares to 1e-4 relative / 1e-5 absolute. Any "
               "mismatch prints `MISMATCH` and returns 1, which scores the "
               "attempt zero. A checksum over the whole output is printed too. "
               "You get back: pass/fail, `mcycle`, `minstret`.",
    contract="""\
```c
void llama_rmsnorm(size_t n, const float *x, const float *w, float *y,
                   float eps);
```

`ss = (1/n) * sum_i x[i]*x[i]`, then `y[i] = x[i] * rsqrt(ss + eps) * w[i]`.
RMSNorm, so there is NO mean subtraction. Called once with `n = 2048`,
`eps = 1e-5`. The reduction may be reassociated (the check has a 1e-4 relative
tolerance) but must be over all `n` terms.""" + _LLAMA_EW_CONTRACT_TAIL,
    notes="""\
# Where the floor is

`n = 2048` fp32. Two things have to happen: one reduction pass and one scaling
pass. Derived with the ROOFLINE METHOD at the top of this file (fp32 = 4
lanes/cycle, memory = 16 B/cycle per pipe, `max(read, write)` not the sum).

    COMPUTE FLOOR
      pass 1 (sum of squares)   2048 FMA / 4              =   512 c
      pass 2 (x * scale * w)    2 * 2048 flop / 4         = 1,024 c
      1 reduction (`vfredusum`) 4 c, + rsqrt via
        `vfrsqrt7` + 1 NR step on a SCALAR              ~=     8 c
      compute floor                                      = ~1,544 c

    MEMORY FLOOR
      reads  : x 8 KiB (pass 1) + x 8 KiB again (pass 2, unavoidable — 2048
               fp32 is 8 KiB and the whole register file at LMUL=8 is 1 KiB)
               + w 8 KiB                    = 24 KiB / 16 B/c = 1,536 c
      writes : y 8 KiB                       =  8 KiB / 16 B/c =   512 c
      memory floor = max(1,536, 512)                          = 1,536 c

    roofline = max(1,544, 1,536) ~= **1,536 cycles — the two are within 0.5%,
    so BOTH bind and there is no slack anywhere.**

The second read of `x` hits L1, so it is free of DRAM traffic — but it still
occupies the load pipe, which is why it is counted.

# What the pristine kernel does

Two separate loops at LMUL=1 (`e32m1` = 8 lanes, a 2-beat chime), one
accumulator, one `vfredusum`, a scalar `1/sqrt`. At LMUL=1 the chime is 2
cycles and the FMA pipe is 4 stages deep, so the single dependent accumulator
chain in pass 1 is very likely serializing — the classic LMUL-too-low
signature.

# Directions worth exploring

- Raise LMUL (`e32m8` = 64 lanes, 16-beat chime) so the dependent accumulate
  chain is fully hidden, and/or use several independent accumulators.
- Fuse the two passes? You cannot: `scale` is not known until the whole
  reduction is done. But you CAN keep `x` in registers across both passes when
  `n` is small enough — 2048 fp32 is 32 KiB, which does not fit, so the honest
  version of this idea is blocking.
- The multiply by `scale` and by `w[i]` is two ops; folding `scale` into a
  pre-scaled copy of `w` would be cheating (the caller owns `w`).

# MEASURED — pristine baseline on GENV256D128GemminiShuttleConfig
(Verilator, 2026-09-05): **5,454 cycles**, 3,896 instructions,
i.e. 3.55x the 1,536-cycle roofline above (compute floor ~1,544, memory floor
1,536 — balanced) and IPC 0.71.
This is the number in out/llama-profile/measured_cycles.json.
""",
)


LLAMA_ROPE = Kernel(
    name="llama-rope",
    target="saturn",
    bench_dir="llama-rope",
    kernel_rel="llama-rope/llama_rope.c",
    elf_name="llama-rope.riscv",
    bench_globs=_saturn_globs("llama-rope"),
    # Measured on GENV256D128GemminiShuttleConfig: 309 s of Verilator wall
    # for one run (ELF load + boot + data fill + kernel + scalar check),
    # with seven of these baselines running concurrently. 3x that.
    sim_timeout_seconds=950,
    sim_timeout_cycles=8_000_000,
    reference_cycles=4_695,
    roofline_cycles=1280,
    objective="Make `llama_rope` — rotary position embedding over the 32 query "
              "heads and 8 KV heads of one Llama-3.2-1B decode step — run in "
              "fewer **cycles** (`mcycle`), without changing what it computes.",
    check_desc="`llama-rope_main.c` (SEALED) fills `q` and `k` from a fixed LCG "
               "seed, keeps pristine copies, builds the cos/sin table for "
               "position 511 with libm, calls `setStats()` around BOTH calls "
               "(Q then K), then recomputes **all 2560 rotated elements** with "
               "a scalar reference to 1e-4 relative. Mismatch prints "
               "`MISMATCH` and returns 1. You get back: pass/fail, `mcycle`, "
               "`minstret`.",
    contract="""\
```c
void llama_rope(size_t nheads, size_t head_dim, float *x,
                const float *cs, const float *sn);   // IN PLACE
```

HuggingFace Llama `rotate_half` convention — the head vector is split into
HALVES, not interleaved pairs. With `H = head_dim/2`, for each head:

    out[i]     = x[i]   * cs[i] - x[i+H] * sn[i]
    out[i + H] = x[i+H] * cs[i] + x[i]   * sn[i]      i in [0, H)

`cs` and `sn` are `H` long and are SHARED by every head — they are an input,
because a real implementation precomputes the table once per position.
Called twice per decode step: `(32, 64, q, ...)` then `(8, 64, k, ...)`; the
objective brackets both.""" + _LLAMA_EW_CONTRACT_TAIL,
    notes="""\
# Where the floor is

2048 Q elements + 512 K elements = 2560 fp32, 4 flops per element pair
(2 multiplies + 2 FMAs produce 2 outputs), so 2 flops per output element.
Derived with the ROOFLINE METHOD at the top of this file.

    COMPUTE FLOOR
      2,560 elements * 2 flop / 4 fp32 lanes per cycle = 1,280 cycles
      no reductions anywhere (R = 0), no transcendentals — `cs`/`sn` are an
      INPUT, precomputed by the caller.

    MEMORY FLOOR
      reads  : 2,560 * 4 B = 10 KiB, plus the shared 64+64 fp32 cos/sin table
               (512 B, read ONCE into registers) = 10.5 KiB -> 672 cycles
      writes : 2,560 * 4 B = 10 KiB (in place)              -> 640 cycles
      memory floor = max(672, 640) = 672 cycles

    roofline = max(1,280, 672) = **1,280 cycles — COMPUTE binds**, by 1.9x.

Everything fits in L1 (10 KiB of data, a 512-byte table), so this is a pure
throughput problem with no reductions and no dependencies at all — the easiest
shape in the whole model, and the one with the most memory slack to hide loads
in.

# What the pristine kernel does

One head at a time at LMUL=1, with the 32-element cos/sin table re-loaded from
memory inside every head's inner loop (40 times over). `head_dim/2 = 32` is
exactly 4 strips at `e32m1`, so there is a loop-control instruction for every
4 vector instructions.

# Directions worth exploring

- The cos/sin table is 32 elements and is the SAME for all 40 heads: hoist it
  into registers once (`e32m4` holds 32 fp32 in one group) and never load it
  again.
- With `head_dim = 64` fixed, each head is one `e32m4` group of 32 for each
  half — the whole per-head computation can be 4 vector ops with no inner loop.
- Consecutive heads are contiguous in memory, so several heads can be processed
  per strip if you handle the half-split with a stride or a slide instead of
  two unit-stride loads. Watch out: **strided and indexed accesses on Saturn
  produce only one element address per cycle**, 4x slower than unit-stride, so
  a `vlse32` over the halves is very likely a loss.

# MEASURED — pristine baseline on GENV256D128GemminiShuttleConfig
(Verilator, 2026-09-05): **4,695 cycles**, 3,565 instructions,
i.e. 3.67x the 1,280-cycle roofline above (compute floor 1,280, memory floor
672 — compute binds) and IPC 0.76.
This is the number in out/llama-profile/measured_cycles.json.
""",
)


LLAMA_ATTN_SCORES = Kernel(
    name="llama-attn-scores",
    target="saturn",
    bench_dir="llama-attn-scores",
    kernel_rel="llama-attn-scores/llama_attn_scores.c",
    elf_name="llama-attn-scores.riscv",
    bench_globs=_saturn_globs("llama-attn-scores"),
    # Measured on GENV256D128GemminiShuttleConfig: 331 s of Verilator wall
    # for one run (ELF load + boot + data fill + kernel + scalar check),
    # with seven of these baselines running concurrently. 3x that.
    sim_timeout_seconds=1000,
    sim_timeout_cycles=8_000_000,
    reference_cycles=60_147,
    roofline_cycles=8192,
    objective="Make `llama_attn_scores` — QK^T/sqrt(d) for ONE query head "
              "against a 512-long fp32 KV cache — run in fewer **cycles** "
              "(`mcycle`), without changing what it computes.",
    check_desc="`llama-attn-scores_main.c` (SEALED) fills `K` and `q` from a "
               "fixed LCG seed, calls `setStats()` around your kernel, then "
               "recomputes **all 512 scores** with a scalar dot product to "
               "1e-4 relative and prints a checksum. Mismatch prints "
               "`MISMATCH` and returns 1. You get back: pass/fail, `mcycle`, "
               "`minstret`.",
    contract="""\
```c
void llama_attn_scores(size_t S, size_t d, const float *K, const float *q,
                       float *scores, float scale);
```

`scores[s] = (sum_j K[s*d + j] * q[j]) * scale` for `s` in `[0, S)`. `K` is
row-major with row stride exactly `d`. Called once with `S = 512`, `d = 64`,
`scale = 0.125` (= 1/sqrt(64), exact in fp32). One decode step runs 32 of
these per layer; the projection multiplies this per-head number by 32.
The reduction may be reassociated.""" + _LLAMA_EW_CONTRACT_TAIL,
    notes="""\
# Where the floor is

`S = 512`, `d = 64` -> 32,768 fp32 MACs, and the 128 KiB K slab is read once.
Derived with the ROOFLINE METHOD at the top of this file.

    COMPUTE FLOOR
      32,768 fp32 FMA / 4 fp32 lanes per cycle          = 8,192 c
      + 512 row reductions (`vfredusum`) at ~4 c each
        IF you keep them — the fast structure eliminates
        them (see below), so they are NOT in the floor   = (+2,048 c)
      compute floor                                      = 8,192 c

    MEMORY FLOOR
      reads  : K 512*64*4 = 128 KiB, each byte once, + q 256 B hoisted once
                                   = 131,328 B / 16 B/c = 8,208 c
      writes : scores 512*4 = 2 KiB ->  2,048 B / 16 B/c =   128 c
      memory floor = max(8,208, 128)                     = 8,208 c

    roofline = **8,192 cycles** — the two floors are within 0.2% of each other,
    so BOTH bind. (8,192 is quoted rather than 8,208 so the number stays a
    strict lower bound under any counting of the tiny `q` read.)

Exactly balanced means there is NO slack: to approach 8,192 the load pipe and
the FMA pipe have to be busy in the same cycle, every cycle. Saturn's load pipe
and arithmetic pipes are independent DLEN-wide sequencers with full chaining,
so this is achievable in principle, but only with the loads running ahead of
the arithmetic.

`q` is 256 bytes and is reused by all 512 rows: it belongs in registers, not
in the load stream. The pristine kernel re-loads it every row, which doubles
the load traffic to 16,384 cycles and makes memory the binding resource.

# What the pristine kernel does

One K row at a time at LMUL=1 (`d = 64` = 8 strips), one accumulator, and one
`vfredusum` per row on the critical path to `scores[s]` — 512 long-latency
serial reductions, none of which overlap each other.

# Directions worth exploring

- **Hoist `q`.** 64 fp32 is one `e32m8` group. Load it once.
- **Several rows per pass**, each with its own accumulator, so the reductions
  overlap and the dependent FMA chains interleave.
- **Amortize the reductions**: with R rows in flight the R `vfredusum`s can be
  issued back to back instead of one per row-length loop.
- At `d = 64` a whole row is one `e32m8` group (64 fp32 at VLEN=256), so the
  inner loop can disappear entirely — the FMA becomes one 16-beat instruction
  per row, which is long enough to hide the 4-stage pipe.

# MEASURED — pristine baseline on GENV256D128GemminiShuttleConfig
(Verilator, 2026-09-05): **60,147 cycles**, 35,884 instructions,
i.e. 7.34x the 8,192-cycle roofline above (compute floor 8,192, memory floor
8,208 — balanced) and IPC 0.60.
This is the number in out/llama-profile/measured_cycles.json.
""",
)


LLAMA_SOFTMAX = Kernel(
    name="llama-softmax",
    target="saturn",
    bench_dir="llama-softmax",
    kernel_rel="llama-softmax/llama_softmax.c",
    elf_name="llama-softmax.riscv",
    bench_globs=_saturn_globs("llama-softmax"),
    # Measured on GENV256D128GemminiShuttleConfig: 273 s of Verilator wall
    # for one run (ELF load + boot + data fill + kernel + scalar check),
    # with seven of these baselines running concurrently. 3x that.
    sim_timeout_seconds=850,
    sim_timeout_cycles=8_000_000,
    reference_cycles=21_906,
    roofline_cycles=1472,
    objective="Make `llama_softmax` — the attention softmax over one 512-long "
              "fp32 score row of a Llama-3.2-1B decode step — run in fewer "
              "**cycles** (`mcycle`), without changing what it computes.",
    check_desc="`llama-softmax_main.c` (SEALED) fills the score row from a "
               "fixed LCG seed (spread ~[-8,8], so the max subtraction is "
               "load-bearing), calls `setStats()` around your kernel, then "
               "recomputes **all 512 outputs** against a scalar libm `expf` "
               "reference to 1e-4 relative, and prints the output sum (which "
               "must be ~1.0). Mismatch prints `MISMATCH` and returns 1. You "
               "get back: pass/fail, `mcycle`, `minstret`.",
    contract="""\
```c
void llama_softmax(size_t n, const float *x, float *y);
```

`m = max_i x[i]`; `y[i] = exp(x[i]-m) / sum_j exp(x[j]-m)`. The max
subtraction is REQUIRED — it is what keeps the exponentials finite, and the
harness's data range is chosen so that dropping it is visible. Called once
with `n = 512`. The exponential may be any approximation that clears the 1e-4
relative tolerance; the reductions may be reassociated.""" + _LLAMA_EW_CONTRACT_TAIL,
    notes="""\
# STATUS OF THIS KERNEL: TERMINAL (2026-09-08, after Round 3). READ FIRST.

    pristine baseline                        21,906
    compute floor (see derivation below)      1,472   1.00x
    **FINAL: 1,555** (run 20260908-202543-fc79) 1,555  1.056x  = 14.1x speedup

    per round: R1 1,752 -> R2 1,568 -> R3 1,555. The last two rounds bought
    1.1% combined. 5.6% above a floor that is itself only a lower bound on the
    exp polynomial chosen. Do not schedule this kernel again.

NOTE: everything below this banner was written at the end of Round 1 and still
quotes 1,752 as "the winner". The structure it describes is right; the number
is stale — the final kernel is in the run directory above.

# Where the floor is

512 fp32. Memory is nothing (2 KiB in, 2 KiB out; it never leaves L1). This
kernel is **entirely** the cost of 512 exponentials plus three passes'
worth of streaming and two reductions. Derived with the ROOFLINE METHOD at the
top of this file; the assumed exp cost is **8 vector ops per element**.

    COMPUTE FLOOR
      pass 1, max          512 * 1 op / 4 lanes         =   128 c
      pass 2, exp + sum    512 * (8 + 1) ops / 4 lanes  = 1,152 c
      pass 3, * (1/S)      512 * 1 op / 4 lanes         =   128 c
      2 reductions (`vfredmax`, `vfredusum`) @ ~4 c     =     8 c
      scalar reciprocal of the sum (`vfrec7` + 1 NR)    =    ~4 c
      compute floor                                     = 1,420 c
      rounded up to a clean **1,472** to leave the tail
      handling and the `vsetvli`s inside the bound.

    MEMORY FLOOR
      reads  : x 2 KiB, read TWICE in pass 2 (once for the scaled argument,
               once for the range reduction) + y 2 KiB reloaded in pass 3
               = ~8 KiB                       -> 512 c
      writes : y 2 KiB written twice          -> 256 c
      memory floor = max(512, 256)            = 512 c

    roofline = max(1,472, 512) = **1,472 cycles — COMPUTE binds**, by ~3x.

**This was 2560 until 2026-09-07 and Round 1 BROKE it: the winning kernel
measured 1,752 cycles.** The old number assumed "~18 flops" for a 5th-order
Cephes exp with an explicit `2^n` multiply. That was too pessimistic in two
ways: (a) one FMA can fuse the `*log2e` scaling with the magic-number bias AND
absorb the max-subtraction (fold `-round(m*log2e)` into the magic constant, so
subtracting the max costs ONE SCALAR instruction, not a vector op per
element); (b) `2^n` is built by an integer shift + add on the bit pattern, not
by a second float multiply. Round 1's winner does both and lands at ~9 ops per
element for the whole exp+sum body. 8 ops is therefore a genuine floor, and
1,752 / 1,472 = 1.19x is the remaining headroom.

The floor MOVES with the exponential you choose — the 1e-4 relative tolerance
is what bounds how cheap it can get. Read the winner:
`out/loop/20260907-043200-f0d7/kernel_best.h`.

# What the pristine kernel does

Three separate passes at LMUL=1 (`vfredmax`, then exp+`vfadd` accumulate with
a store, then reload and `vfdiv` by the scalar sum). The exp is written out in
full in the kernel file, in the naive `vfmul` + `vfadd` Horner form rather than
`vfmadd`, and the last pass uses a full-precision divide where a single
reciprocal multiply would do.

# Directions worth exploring

- **`vfmadd` the Horner chain** — the pristine version spends 2 instructions
  per polynomial term where 1 will do. That is ~5 vector ops per strip saved.
- **Reciprocal instead of divide**: compute `1/s` once on the scalar side and
  `vfmul_vf` by it. `vfdiv` is a long, low-throughput operation.
- **Fuse passes 2 and 3**: you cannot, the sum is not known until pass 2 ends —
  but you CAN keep the exponentials in registers instead of storing and
  reloading them. 512 fp32 = 2 KiB = 8 `e32m8` groups; the register file has
  4 groups at LMUL=8, so a two-level blocking is the honest version.
- Raise LMUL so the long dependent polynomial chain is hidden.

# MEASURED — pristine baseline on GENV256D128GemminiShuttleConfig
(Verilator, 2026-09-05): **21,906 cycles**, 3,236 instructions,
i.e. **14.9x the 1,472-cycle roofline above** (compute floor 1,472, memory
floor 512 — compute binds) and IPC 0.15.

# MEASURED — Round 1 of the optimization loop (2026-09-07, Fable 5.1)

**1,752 cycles**, i.e. 12.5x the pristine baseline and **1.19x the roofline**.
This kernel is close to done; see `out/loop/20260907-043200-f0d7/kernel_best.h`
for what actually worked (LMUL split by phase — `e32m8` for the trivial
max/normalise passes, `e32m4` for the FMA-heavy exp pass; the folded
max-subtraction described above; a single `vfredusum` over a running
accumulator instead of one per strip; scalar reciprocal + `vfmul` instead of
`vfdiv`).
This is the number in out/llama-profile/measured_cycles.json.
""",
)


LLAMA_ATTN_PV = Kernel(
    name="llama-attn-pv",
    target="saturn",
    bench_dir="llama-attn-pv",
    kernel_rel="llama-attn-pv/llama_attn_pv.c",
    elf_name="llama-attn-pv.riscv",
    bench_globs=_saturn_globs("llama-attn-pv"),
    # Measured on GENV256D128GemminiShuttleConfig: 765 s of Verilator wall
    # for one run (ELF load + boot + data fill + kernel + scalar check),
    # with seven of these baselines running concurrently. 3x that.
    sim_timeout_seconds=2400,
    sim_timeout_cycles=8_000_000,
    reference_cycles=44_765,
    roofline_cycles=8192,
    objective="Make `llama_attn_pv` — probs@V for ONE query head against a "
              "512-long fp32 V cache — run in fewer **cycles** (`mcycle`), "
              "without changing what it computes.",
    check_desc="`llama-attn-pv_main.c` (SEALED) fills `V` from a fixed LCG "
               "seed and turns `P` into a real probability vector (exp then "
               "normalise), calls `setStats()` around your kernel, then "
               "recomputes **all 64 outputs** with a scalar reduction over the "
               "512 rows to 1e-4 relative and prints a checksum. Mismatch "
               "prints `MISMATCH` and returns 1. You get back: pass/fail, "
               "`mcycle`, `minstret`.",
    contract="""\
```c
void llama_attn_pv(size_t S, size_t d, const float *P, const float *V,
                   float *out);
```

`out[j] = sum_s P[s] * V[s*d + j]` for `j` in `[0, d)`. `V` is row-major with
row stride exactly `d`. `out` is zeroed by the caller. Called once with
`S = 512`, `d = 64`. One decode step runs 32 of these per layer; the
projection multiplies this per-head number by 32. The accumulation may be
reassociated.""" + _LLAMA_EW_CONTRACT_TAIL,
    notes="""\
# Where the floor is

Same arithmetic and the same traffic as `llama-attn-scores`, transposed.
Derived with the ROOFLINE METHOD at the top of this file.

    COMPUTE FLOOR
      512 * 64 = 32,768 fp32 FMA / 4 lanes per cycle    = 8,192 c
      reductions: **R = 0**. The accumulation runs over the OUTER loop, so
      there is no `vfredusum` anywhere — see below.
      compute floor                                     = 8,192 c

    MEMORY FLOOR
      reads  : V 512*64*4 = 128 KiB + P 512*4 = 2 KiB
                                    = 133,120 B / 16 B/c = 8,320 c
      writes : out 64*4 = 256 B                          =    16 c
      memory floor = max(8,320, 16)                      = 8,320 c

    roofline = **8,192 cycles** — both floors land within 1.6% of each other,
    so BOTH bind. (8,192 rather than 8,320 keeps it a strict lower bound.)

Balanced again, so again there is no slack and the load pipe must run
concurrently with the FMA pipe.

The structural difference from `attn-scores` is that there is **no reduction
instruction anywhere**: the output is 64 wide and the accumulation is over the
outer loop, so a plain `vfmacc.vf` per V row is the whole kernel. `d = 64` is
exactly one `e32m8` group at VLEN = 256, i.e. the entire accumulator fits in
ONE vector register group and never has to touch memory.

# What the pristine kernel does

Exactly the wrong thing: it keeps the accumulator in MEMORY and round-trips it
once per V row — LMUL=1, 8 strips, `vle32` + `vle32` + `vfmacc.vf` + `vse32`
per strip, 512 times. That is 4,096 loads and 4,096 stores of an accumulator
that could have lived in a register for the whole kernel, and it triples the
traffic on a kernel that is already memory-balanced.

# Directions worth exploring

- **Keep `out` in a register group.** `e32m8` holds 64 fp32; hoist the load
  before the `s` loop and store once at the end. This alone should be most of
  the available speedup.
- With one accumulator group the `vfmacc.vf` chain is loop-carried and 16 beats
  long at LMUL=8 — longer than the 4-stage FMA pipe, so it is already hidden.
  If you split into 2 or 4 accumulators at lower LMUL you get independence but
  a shorter chime; measure before assuming.
- Software-pipeline the V row loads one row ahead so the 128 KiB stream never
  stalls the FMA.

# MEASURED — pristine baseline on GENV256D128GemminiShuttleConfig
(Verilator, 2026-09-05): **44,765 cycles**, 37,413 instructions,
i.e. 5.46x the 8,192-cycle roofline above (compute floor 8,192, memory floor
8,320 — balanced) and IPC 0.84.
This is the number in out/llama-profile/measured_cycles.json.
""",
)


LLAMA_SILU_MUL = Kernel(
    name="llama-silu-mul",
    target="saturn",
    bench_dir="llama-silu-mul",
    kernel_rel="llama-silu-mul/llama_silu_mul.c",
    elf_name="llama-silu-mul.riscv",
    bench_globs=_saturn_globs("llama-silu-mul"),
    # Measured on GENV256D128GemminiShuttleConfig: 450 s of Verilator wall
    # for one run (ELF load + boot + data fill + kernel + scalar check),
    # with seven of these baselines running concurrently. 3x that.
    sim_timeout_seconds=1400,
    sim_timeout_cycles=8_000_000,
    reference_cycles=361_260,
    roofline_cycles=30_720,
    objective="Make `llama_silu_mul` — the SwiGLU activation over the "
              "8192-wide Llama-3.2-1B MLP intermediate — run in fewer "
              "**cycles** (`mcycle`), without changing what it computes.",
    check_desc="`llama-silu-mul_main.c` (SEALED) fills `gate` and `up` from a "
               "fixed LCG seed, calls `setStats()` around your kernel, then "
               "recomputes **512 sampled outputs** (stride 13, coprime with "
               "every usable vector length, wrapping over the whole array) "
               "against a scalar libm reference to 1e-4 relative, and prints a "
               "checksum over ALL 8192 outputs so a kernel that only computed "
               "the sampled lanes is visible in the log. Mismatch prints "
               "`MISMATCH` and returns 1. You get back: pass/fail, `mcycle`, "
               "`minstret`.",
    contract="""\
```c
void llama_silu_mul(size_t n, const float *gate, const float *up, float *out);
```

`out[i] = silu(gate[i]) * up[i]` with `silu(g) = g / (1 + exp(-g))`. Called
once with `n = 8192`. The sigmoid may be any approximation that clears the
1e-4 relative tolerance — including a reciprocal estimate plus Newton steps
instead of a full-precision divide.""" + _LLAMA_EW_CONTRACT_TAIL,
    notes="""\
# STATUS OF THIS KERNEL: TERMINAL (2026-09-08, after Round 3). READ FIRST.

    pristine baseline                         361,260
    compute floor (see derivation below)       30,720   1.00x
    **FINAL: 31,024** (run 20260908-202528-f504)         1.010x = 11.6x speedup

    per round: R1 66,811 -> R2 35,774 -> R3 31,024. 1.0% above the compute
    floor: there is nothing left. Do not schedule this kernel again.

NOTE: everything below this banner was written at the end of Round 1 and still
quotes 66,811 as the best. The techniques it lists are right; the number is
stale — the final kernel is in the run directory above.

# Where the floor is

Compute bound by a wide margin. Derived with the ROOFLINE METHOD at the top of
this file; the assumed transcendental costs are **exp = 8 vector ops/element**
and **1/x = 3 vector ops/element** (`vfrec7` + one Newton-Raphson step).

    COMPUTE FLOOR   (per element, `out = g / (1 + exp(-g)) * up`)
      negate g                                    1 op
      exp(-g)                                     8 ops
      + 1.0                                       1 op
      reciprocal (`vfrec7` + 1 NR step)           3 ops
      g * recip                                   1 op
      * up[i]                                     1 op
      total                                      15 ops/element
      8,192 * 15 / 4 fp32 lanes per cycle    = 30,720 cycles
      no reductions anywhere (R = 0).

    MEMORY FLOOR
      reads  : gate 32 KiB + up 32 KiB = 64 KiB -> 4,096 cycles
      writes : out 32 KiB                       -> 2,048 cycles
      memory floor = max(4,096, 2,048)          = 4,096 cycles

    roofline = max(30,720, 4,096) = **30,720 cycles — COMPUTE binds**, by 7.5x.

**`roofline_cycles` was 6,144 until 2026-09-07, and that was simply the WRONG
NUMBER to put in the field**: 6,144 was the memory floor, which does not bind
anywhere near here. Round 1 reached 66,811 cycles, still 2.2x above the
compute floor above, so nothing was falsified — the field was just reporting
the non-binding side of the roofline. It now reports `max()`, as it should.

The 15-op count is optimistic and deliberately so (a floor, not a target).
Round 1's winner needs ~25 ops/element, because 1e-4 over the full `gate`
range wants a wider polynomial than 4 terms: its exp alone is ~18 ops (clamp 2
+ `vfmul`/`vfcvt.x.f` under RNE 2 + two-constant `vfnmsac` range reduction 2 +
a degree-5 polynomial in POWER form 9 + exponent reassembly 3). At 25 ops the
floor would be 51,200 cycles and Round 1's 66,811 is 1.3x that. If you cannot
find a cheaper accurate sigmoid, 51,200 is the number to plan against;
30,720 is the number nothing may go below.

A cheaper sigmoid moves the floor itself — a direct minimax polynomial for
`1/(1+e^-g)` over a clamped range, or exploiting `silu(g) = g + silu(-g)`
symmetry to halve the argument range — but the 1e-4 check is the referee.

# What the pristine kernel does

LMUL=1, one strip at a time, `vsetvl` re-issued every iteration, the Cephes exp
written out in the naive `vfmul` + `vfadd` Horner form (2 instructions per
polynomial term instead of 1 `vfmadd`), and a full-precision `vfdiv_vv` for
`g / (1 + e)`.

# Directions worth exploring

- **`vfmadd` the Horner chain**: 5 vector ops per strip, for free.
- **Kill the `vfdiv`.** `vfrec7` gives 7 bits; one Newton-Raphson step gives
  ~14, two give full fp32 — and the tolerance is 1e-4, so one step is already
  more than enough. `vfdiv` is the single most expensive instruction in the
  loop.
- **Raise LMUL.** The polynomial is one long dependent chain; at LMUL=1 the
  2-beat chime is shorter than the 4-stage FMA pipe, so every term stalls. At
  `e32m8` the chain is fully hidden. This is likely the biggest single win.
- **Overlap the loads.** 64 KiB has to arrive while the polynomial runs;
  with the arithmetic 7x the memory floor there is plenty of room to hide it,
  but only if the loads are issued ahead.
- A cheaper sigmoid (a direct polynomial for `1/(1+e^-g)`, or exploiting
  `silu(g) = g - silu(-g)` symmetry) changes the floor itself — but the 1e-4
  check is the referee.

# MEASURED — pristine baseline on GENV256D128GemminiShuttleConfig
(Verilator, 2026-09-05): **361,260 cycles**, 48,180 instructions,
i.e. **11.8x the 30,720-cycle compute floor** and 88x the 4,096-cycle memory
floor — and IPC 0.13, which is the signature of a long dependent chain at a
2-beat chime plus a serialising `vfdiv`. Both of those are fixable without
changing the answer.
This is the number in out/llama-profile/measured_cycles.json.

# MEASURED — Round 1 of the optimization loop (2026-09-07, Fable 5.1)

**66,811 cycles** — 5.4x the pristine baseline, 2.2x the 30,720-cycle floor,
1.3x the 51,200-cycle floor implied by its own ~25-op-per-element sigmoid.
`out/loop/20260907-043150-7557/kernel_best.h`. What worked, and is therefore
NOT worth re-discovering:

- **LMUL = 4, not 8.** At `e32m8` the live set of the polynomial spills 11
  register groups per strip; at `e32m4` it spills 1. Spills, not chime length,
  are what bound LMUL here.
- **`vfcvt.x.f` under the default RNE rounding mode** to get `round(x*log2e)`
  in 2 ops, instead of the 10-op `floor(x*log2e + 0.5)` emulation.
- **Power-form polynomial (`x`, `x^2`, `x^3`, ...) instead of Horner.** RVV
  has `vfmacc.vf` (vector * scalar-coefficient, accumulate) but no
  vector-vector-multiply-plus-scalar-add, so Horner costs 16 ops where the
  power form costs 9 — and each old power dies immediately after producing the
  next, keeping the live set under 7 groups.
- **`vfrec7` + one Newton-Raphson step**, never `vfdiv`. The hardware iterative
  divider measured ~25 cycles/element and serialised the pipe.
- **One fused pass** — gate and up loaded once, silu and the multiply in the
  same loop body, no intermediate buffer.

Still open: a cheaper sigmoid (the op count IS the floor here), and better
overlap of the 64 KiB load stream, which at 4,096 cycles has enormous slack.
""",
)


LLAMA_ADD = Kernel(
    name="llama-add",
    target="saturn",
    bench_dir="llama-add",
    kernel_rel="llama-add/llama_add.c",
    elf_name="llama-add.riscv",
    bench_globs=_saturn_globs("llama-add"),
    # Measured on GENV256D128GemminiShuttleConfig: 182 s of Verilator wall
    # for one run (ELF load + boot + data fill + kernel + scalar check),
    # with seven of these baselines running concurrently. 3x that.
    sim_timeout_seconds=600,
    sim_timeout_cycles=8_000_000,
    reference_cycles=3_937,
    roofline_cycles=1024,
    objective="Make `llama_add` — the fp32 residual add over a 2048-wide "
              "Llama-3.2-1B hidden state — run in fewer **cycles** (`mcycle`), "
              "without changing what it computes.",
    check_desc="`llama-add_main.c` (SEALED) fills `x` and `y` from a fixed LCG "
               "seed, keeps a pristine copy of `x`, calls `setStats()` around "
               "your kernel, then checks **all 2048 outputs BIT-EXACTLY** "
               "against `x0[i] + y[i]` — a single fp32 add has exactly one "
               "correct answer, so there is no tolerance here. Mismatch prints "
               "`MISMATCH` and returns 1. You get back: pass/fail, `mcycle`, "
               "`minstret`.",
    contract="""\
```c
void llama_add(size_t n, float *x, const float *y);   // IN PLACE
```

`x[i] += y[i]`. Called once with `n = 2048`. Exactly one fp32 add per element,
no reassociation possible and none allowed — the check is bit-exact.""" + _LLAMA_EW_CONTRACT_TAIL,
    notes="""\
# Where the floor is

Pure memory. 2048 fp32, derived with the ROOFLINE METHOD at the top of this
file (load pipe and store pipe are INDEPENDENT 16 B/cycle sequencers, so the
memory floor is `max(read, write)`, not their sum).

    COMPUTE FLOOR
      2,048 fp32 adds / 4 lanes per cycle = 512 cycles. No reductions, no
      transcendentals, no reassociation (the check is bit-exact).

    MEMORY FLOOR
      reads  : x 8 KiB + y 8 KiB = 16 KiB -> 1,024 cycles
      writes : x 8 KiB                    ->   512 cycles
      memory floor = max(1,024, 512)      = 1,024 cycles

    roofline = max(512, 1,024) = **1,024 cycles — MEMORY (the load pipe)
    binds**, by 2x over compute.

**This was 1536 until 2026-09-07.** That number summed the read and write
streams; the two pipes are independent, so summing them is not a lower bound.
The old note already said the honest figure was 1,024 — now it IS 1,024.
This is the smallest kernel in the model and the one with the least room; do
not expect much.

# What the pristine kernel does

LMUL=1, `vsetvl` re-issued every iteration, 256 iterations of
`vle32` + `vle32` + `vfadd` + `vse32` — 4 vector instructions and ~3 scalar
loop-control instructions per 8 elements, i.e. the loop overhead is comparable
to the work.

# Directions worth exploring

- **Raise LMUL to 8** (`e32m8`, 64 fp32 per group): 32 iterations instead of
  256, and each memory op is a 16-beat streaming access instead of a 2-beat
  one. On a pure-bandwidth kernel this is almost the only lever.
- Hoist the `vsetvl` out of the loop (`n = 2048` is a multiple of every VLMAX
  here, so the tail can be handled once outside).
- Unroll by 2 groups so the two load streams and the store stream are all in
  flight at once.

# MEASURED — pristine baseline on GENV256D128GemminiShuttleConfig
(Verilator, 2026-09-05): **3,937 cycles**, 2,840 instructions,
i.e. **3.84x the 1,024-cycle roofline above** (compute floor 512, memory floor
1,024 — the load pipe binds) and IPC 0.72.
This is the number in out/llama-profile/measured_cycles.json.
""",
)


# --- int8-KV-cache variants of the two attention kernels -------------------
# The fp32 pair above holds the KV cache in fp32, which moves 4x the bytes
# ModelSpec assumes (`kv_bytes = 1`). These two are the same arithmetic with an
# int8 cache and a per-row fp32 dequant scale — the layout a real KV cache
# uses, because a row is one token's key/value vector, written once at append
# time. They are the entries measured_cycles.json now costs `attn_scores` and
# `attn_pv` from; the fp32 pair stays registered as the comparison point.

LLAMA_ATTN_SCORES_Q8KV = Kernel(
    name="llama-attn-scores-q8kv",
    target="saturn",
    bench_dir="llama-attn-scores-q8kv",
    kernel_rel="llama-attn-scores-q8kv/llama_attn_scores_q8kv.c",
    elf_name="llama-attn-scores-q8kv.riscv",
    bench_globs=_saturn_globs("llama-attn-scores-q8kv"),
    # Measured on GENV256D128GemminiShuttleConfig: 334 s of Verilator wall
    # for one run (ELF load + boot + data fill + kernel + full scalar check),
    # sharing the cluster with other baselines. 3x that.
    sim_timeout_seconds=1000,
    sim_timeout_cycles=8_000_000,
    reference_cycles=68_571,
    roofline_cycles=20_480,
    objective="Make `llama_attn_scores_q8kv` — QK^T/sqrt(d) for ONE query head "
              "against a 512-long **int8** KV cache with a per-row fp32 scale — "
              "run in fewer **cycles** (`mcycle`), without changing what it "
              "computes.",
    check_desc="`llama-attn-scores-q8kv_main.c` (SEALED) fills `K8` and the "
               "per-row scales from a fixed LCG seed, calls `setStats()` around "
               "your kernel, then recomputes **all 512 scores** with a scalar "
               "dequantise-and-dot reference to 1e-4 relative and prints a "
               "checksum. Mismatch prints `MISMATCH` and returns 1. You get "
               "back: pass/fail, `mcycle`, `minstret`.",
    contract="""\
```c
void llama_attn_scores_q8kv(size_t S, size_t d, const int8_t *K8,
                            const float *kscale, const float *q,
                            float *scores, float scale);
```

`scores[s] = kscale[s] * (sum_j (float)K8[s*d + j] * q[j]) * scale`.
`K8` is row-major int8 with row stride exactly `d`; `kscale` is one fp32
dequant scale per KV position; `q` is fp32. Called once with `S = 512`,
`d = 64`, `scale = 0.125`. One decode step runs 32 of these per layer.

The dequant scale is per ROW and the reduction is within a row, so it factors
straight out of the sum — apply it once to the finished dot product, never per
element. The reduction may be reassociated.""" + _LLAMA_EW_CONTRACT_TAIL,
    notes="""\
# Where the floor is — and why int8 KV does NOT make this cheaper

The cache shrank 4x, from 128 KiB to 32 KiB, and the memory floor shrank with
it — but `q` is fp32, so every one of the 32,768 int8 bytes has to be
*converted* before it can be multiplied, and that conversion is the kernel.
Derived with the ROOFLINE METHOD at the top of this file.

    COMPUTE FLOOR   (cheapest legal int8 -> fp32 -> FMA chain)
      `vsext.vf2`  int8 -> int16, 16-bit destination -> 8 elem/c -> 0.125 c/el
      `vfwcvt.f.x.v` int16 -> fp32, 32-bit dest      -> 4 elem/c -> 0.250 c/el
      `vfmacc.vv`  fp32 accumulate, 32-bit dest      -> 4 elem/c -> 0.250 c/el
      total                                                        0.625 c/el
      32,768 * 0.625                                            = 20,480 c
      + 512 row reductions (`vfredusum`) @ ~4 c   = (+2,048 c, avoidable —
        Round 1 showed the reductions can be replaced by a segmented-load
        transpose + tree-add, so they are NOT in the floor)

    MEMORY FLOOR
      reads  : K8 32 KiB + kscale 2 KiB + q 256 B (hoisted, read once)
                                        = 35,072 B / 16 B/c = 2,192 c
      writes : scores 2 KiB              =  2,048 B / 16 B/c =   128 c
      memory floor = max(2,192, 128)                         = 2,192 c

    roofline = max(20,480, 2,192) = **20,480 cycles — COMPUTE binds**, by 9.3x.

**This was 24,576 until 2026-09-07.** The old derivation assumed the widening
path is `vsext.vf4` -> `vfcvt.f.x` -> `vfmacc`, i.e. THREE 32-bit-destination
ops at 4 lanes/cycle. It is cheaper than that: go int8 -> int16 first
(`vsext.vf2` is a 16-bit destination, 8 lanes/cycle) and then use the WIDENING
convert `vfwcvt.f.x.v` (int16 -> fp32) instead of a same-width `vfcvt`. That
turns one of the three 4-lane ops into an 8-lane op and drops the floor by
17%. Nothing measured broke the old number; it was just pessimistic.

So this kernel is still **compute bound by ~9x**, where the fp32 version
(`llama-attn-scores`, floors 8,192 / 8,208) was exactly balanced. Quantising
the KV cache bought 96 KiB of bandwidth and paid for it with the conversion.
On this SoC, at this shape, that is a bad trade — and saying so is the point of
having both measurements.

**The way out is not to convert at all.** If `q` were quantised too, the whole
dot product is `vwmul` + `vwadd.wv` into int32 — one widening multiply and one
widening accumulate, no `vfcvt` anywhere — and a single fp32 multiply by
`kscale[s] * qscale` at the end. That is the `llama-attn-scores-int8` inner
loop, whose floor is `MACs / 8` = 32,768 / 8 = **4,096 cycles** — HALF the
fp32 kernel's floor and with a quarter of the traffic. Quantising `q` is a change to the CONTRACT, not to this file, so it
is out of scope for the optimizer — but it is the reason this benchmark
exists, and it belongs in the report rather than in the kernel.

# What the pristine kernel does

One K row at a time at LMUL=1: `vle8` into an `e8mf4` fractional group,
`vsext.vf4` to `i32m1`, `vfcvt.f.x` to `f32m1`, `vfmacc` — with `q` re-loaded
from memory for every one of the 512 rows, one accumulator, and one
`vfredusum` per row on the critical path.

# Directions worth exploring

- **Hoist `q`** — 64 fp32 is one `e32m8` group, loaded once, not 512 times.
- **Raise LMUL.** At `e8m2` -> `i32m8` the widening chain is 16 beats and the
  4-stage pipes disappear behind it. This is the single biggest lever, because
  the kernel is compute bound and the chain is dependent.
- **Several rows in flight** so the `vfredusum`s overlap.
- At `d = 64` a whole row is one `e32m8` group, so the inner loop can vanish.
- **Use `vsext.vf2` + `vfwcvt.f.x.v`, not `vsext.vf4` + `vfcvt.f.x`.** The
  first pair spends one 8-lane/cycle op and one 4-lane/cycle op where the
  second spends two 4-lane/cycle ops. That is the 17% the floor above already
  assumes; the pristine kernel does not do it.
# MEASURED — pristine baseline on GENV256D128GemminiShuttleConfig
(Verilator, 2026-09-05): **68,571 cycles**, 44,591 instructions,
i.e. **3.35x the 20,480-cycle compute floor** above (memory floor 2,192 — not
close to binding), IPC 0.65.

**The fp32-KV version of the same kernel measures 60,147 cycles** —
so the int8 cache is 1.14x SLOWER here despite moving a quarter of
the bytes, which is exactly what the two floors above predict. This is the
number out/llama-profile/measured_cycles.json now costs the model from; the
fp32 figure is kept there under `additional_measurements`.
""",
)


LLAMA_ATTN_PV_Q8KV = Kernel(
    name="llama-attn-pv-q8kv",
    target="saturn",
    bench_dir="llama-attn-pv-q8kv",
    kernel_rel="llama-attn-pv-q8kv/llama_attn_pv_q8kv.c",
    elf_name="llama-attn-pv-q8kv.riscv",
    bench_globs=_saturn_globs("llama-attn-pv-q8kv"),
    # Measured on GENV256D128GemminiShuttleConfig: 878 s of Verilator wall
    # for one run (ELF load + boot + data fill + kernel + full scalar check),
    # sharing the cluster with other baselines. 3x that.
    sim_timeout_seconds=2650,
    sim_timeout_cycles=8_000_000,
    reference_cycles=62_541,
    roofline_cycles=20_480,
    objective="Make `llama_attn_pv_q8kv` — probs@V for ONE query head against a "
              "512-long **int8** V cache with a per-row fp32 scale — run in "
              "fewer **cycles** (`mcycle`), without changing what it computes.",
    check_desc="`llama-attn-pv-q8kv_main.c` (SEALED) fills `V8` and the per-row "
               "scales from a fixed LCG seed and turns `P` into a real "
               "probability vector, calls `setStats()` around your kernel, then "
               "recomputes **all 64 outputs** with a scalar dequantise-and-"
               "reduce reference to 1e-4 relative and prints a checksum. "
               "Mismatch prints `MISMATCH` and returns 1. You get back: "
               "pass/fail, `mcycle`, `minstret`.",
    contract="""\
```c
void llama_attn_pv_q8kv(size_t S, size_t d, const float *P, const int8_t *V8,
                        const float *vscale, float *out);
```

`out[j] = sum_s P[s] * vscale[s] * (float)V8[s*d + j]`. `V8` is row-major int8
with row stride exactly `d`; `vscale` is one fp32 dequant scale per KV
position; `P` is fp32 and `out` is zeroed by the caller. Called once with
`S = 512`, `d = 64`. One decode step runs 32 of these per layer.

The scale is per ROW and the reduction is OVER rows, so it does not factor out
of the sum — but it folds into the row's scalar multiplier (`P[s]*vscale[s]`),
one scalar multiply per row and nothing per element. The accumulation may be
reassociated.""" + _LLAMA_EW_CONTRACT_TAIL,
    notes="""\
# Where the floor is — the same trade as attn-scores-q8kv

Derived with the ROOFLINE METHOD at the top of this file — same trade as
`llama-attn-scores-q8kv`, and the same cheapest-legal conversion chain.

    COMPUTE FLOOR   (per V8 element)
      `vsext.vf2`    int8 -> int16, 16-bit dest -> 8 elem/c -> 0.125 c/el
      `vfwcvt.f.x.v` int16 -> fp32, 32-bit dest -> 4 elem/c -> 0.250 c/el
      `vfmacc.vf`    fp32 accumulate           -> 4 elem/c -> 0.250 c/el
      32,768 * 0.625                                       = 20,480 c
      reductions: **R = 0** — the accumulation runs over the OUTER loop.

    MEMORY FLOOR
      reads  : V8 32 KiB + vscale 2 KiB + P 2 KiB
                                     = 36,864 B / 16 B/c = 2,304 c
      writes : out 64 * 4 B = 256 B  ->                       16 c
      memory floor = max(2,304, 16)                        = 2,304 c

    roofline = max(20,480, 2,304) = **20,480 cycles — COMPUTE binds**, by 8.9x.

**This was 24,576 until 2026-09-07**, on the assumption that int8 -> fp32
costs two 32-bit-destination operations (`vsext.vf4` + `vfcvt.f.x`). Going via
int16 (`vsext.vf2`, an 8-lane/cycle op) and then the WIDENING `vfwcvt.f.x.v`
is 17% cheaper. Nothing measured broke the old number; it was pessimistic.

Compute bound by ~9x, where the fp32 `llama-attn-pv` was balanced at ~8,192
both ways. The int8 cache saved 96 KiB of traffic and spent the conversion to
get it; at this shape that is a losing trade on this SoC. See
`llama-attn-scores-q8kv`'s notes for the way out (quantise `P` too and stay in
the integer pipe — a contract change, not a kernel change; it lands at a
**4,096**-cycle floor, see `llama-attn-pv-int8`).

The structural saving grace, as in the fp32 version, is that there is **no
reduction instruction anywhere**: the output is 64 wide, `d = 64` is exactly
one `e32m8` group at VLEN = 256, and the accumulation is over the outer loop.

# What the pristine kernel does

Two mistakes at once, on purpose. It keeps the 64-wide accumulator in MEMORY
and round-trips it once per V row (`vle32` + `vse32` per strip, 512 times, for
a value that fits in one register group and never had to leave it), and it
widens int8 -> int32 -> fp32 at LMUL=1 where the chain is 2 beats and the
pipes are 4 stages deep.

# Directions worth exploring

- **Keep `out` in an `e32m8` register group.** Load before the `s` loop, store
  once after it. This alone removes 4,096 loads and 4,096 stores.
- **Raise LMUL on the widening chain** so `vsext`/`vfcvt`/`vfmacc` stop
  stalling on each other.
- Software-pipeline the V row loads one row ahead — though with the arithmetic
  10x the memory floor there is a lot of slack to hide them in.
- The per-row `P[s]*vscale[s]` is scalar work on the critical path of every
  row; on an in-order Shuttle it is small but not free, and it can be
  precomputed for several rows at a time.
# MEASURED — pristine baseline on GENV256D128GemminiShuttleConfig
(Verilator, 2026-09-05): **62,541 cycles**, 46,632 instructions,
i.e. **3.05x the 20,480-cycle compute floor** above (memory floor 2,304 — not
close to binding), IPC 0.75.

**The fp32-KV version of the same kernel measures 44,765 cycles** —
so the int8 cache is 1.40x SLOWER here despite moving a quarter of
the bytes, which is exactly what the two floors above predict. This is the
number out/llama-profile/measured_cycles.json now costs the model from; the
fp32 figure is kept there under `additional_measurements`.
""",
)


# --- fully-quantised variants: int8 cache AND int8 activations -------------
# The third point of the comparison. `*-q8kv` quantised only the cache and got
# SLOWER, because an fp32 q/P forces an int8->int32->fp32 widening of every
# cached byte. These two quantise the activation side as well, which deletes
# that conversion entirely: the reduction stays in the integer pipe (`vwmul` +
# `vwadd.wv` into int32 — the `llama-q8-gemv` inner loop) and the only floating
# point left is one multiply per output.
#
# They carry a SECOND self-check the fp32/q8kv pair does not need: a
# quantisation-error gate on the relative L2 distance to a true fp32 reference.
# See each `check_desc` and the error budget written out in the `_main.c`.

LLAMA_ATTN_SCORES_INT8 = Kernel(
    name="llama-attn-scores-int8",
    target="saturn",
    bench_dir="llama-attn-scores-int8",
    kernel_rel="llama-attn-scores-int8/llama_attn_scores_int8.c",
    elf_name="llama-attn-scores-int8.riscv",
    bench_globs=_saturn_globs("llama-attn-scores-int8"),
    # Measured on GENV256D128GemminiShuttleConfig: 1073 s of Verilator wall
    # for one run. The harness is the heaviest of the three variants (it
    # generates fp32 data, quantises it, and runs BOTH an int32 and an fp32
    # golden model), which is why it is slower to simulate than the kernel
    # itself would suggest. 3x that.
    sim_timeout_seconds=3200,
    sim_timeout_cycles=8_000_000,
    reference_cycles=51_280,
    roofline_cycles=6144,
    objective="Make `llama_attn_scores_int8` — QK^T/sqrt(d) for ONE query head "
              "with an **int8 K cache and an int8 query**, int32 accumulation — "
              "run in fewer **cycles** (`mcycle`), without changing what it "
              "computes.",
    check_desc="`llama-attn-scores-int8_main.c` (SEALED) generates fp32 K and "
               "q, quantises K per row and q per vector, and runs TWO checks. "
               "**Gate A** recomputes all 512 scores with a scalar int32 dot "
               "product over exactly the bytes your kernel saw and compares to "
               "1e-4 relative — the integer sum has one correct answer, so "
               "this is as strict as the fp32 benchmark and it is what scores "
               "the attempt. **Gate B** compares against a true fp32 reference "
               "built from the ORIGINAL unquantised data and fails if the "
               "relative L2 error exceeds 3e-2; it validates the quantisation "
               "scheme, not your code, and the measured value is printed every "
               "run (the pristine kernel reports 1.7e-3, ~17x inside the gate). "
               "Mismatch prints `MISMATCH` and returns 1. You get back: "
               "pass/fail, `mcycle`, `minstret`.",
    contract="""\
```c
void llama_attn_scores_int8(size_t S, size_t d, const int8_t *K8,
                            const float *kscale, const int8_t *q8,
                            float *scores, float qscale, float scale);
```

`scores[s] = (float)(sum_j K8[s*d+j] * q8[j]) * kscale[s] * qscale * scale`.
`K8` is row-major int8 with row stride exactly `d`; `kscale` is one fp32
dequant scale per KV position; `q8` is int8 with the single fp32 `qscale`.
Called once with `S = 512`, `d = 64`, `scale = 0.125`.

The products are `int8 * int8` and **the accumulation and the dot product are
`int32` and must not saturate or round** — `int32` is wide enough here
(`|sum| <= 64 * 127 * 127 = 1,032,256`). Gate A is exact on the integer part,
so an int16 accumulator or a float accumulator will fail it.

The scales are per row and per vector respectively, so both factor straight
out of the reduction: apply them once to the finished dot product, never per
element.""" + _LLAMA_EW_CONTRACT_TAIL,
    notes="""\
# STATUS OF THIS KERNEL: TERMINAL BY EXHAUSTION (2026-09-08, after Round 3).

    pristine baseline                          51,280
    compute floor (see derivation below)        6,144   1.00x
    **FINAL: 8,233** (run 20260908-202458-85ab)          1.340x = 6.2x speedup

    per round: R1 8,574 -> R2 8,233 -> R3 8,233 (Round 3 reproduced Round 2
    EXACTLY and found nothing). This is the one kernel on the board still a
    third above its floor, but two full rounds have failed to move it, so it
    is closed by exhaustion, not by a proof. If anyone ever reopens it, the
    open question is the once-per-8-rows partial-sum step, not the load shape.

NOTE: everything below this banner was written at the end of Round 1 and still
quotes 8,574. The number is stale — the final kernel is in the run dir above.

# Where the floor is — and why this is the version that should win

Derived with the ROOFLINE METHOD at the top of this file. The key rule: an
int8 x int8 dot product runs at **8 MAC/cycle**, not 4.

    COMPUTE FLOOR
      `vwmacc` widens int8 x int8 into an int16 accumulator, a 16-bit
      destination -> DLEN/16 = 8 MAC/cycle.
      MAC floor           32,768 / 8                      = 4,096 c
      + reductions. `d = 64` means the sum is WITHIN a row, so something has
        to reduce 512 rows. At ~4 c of pipe occupancy per reduction that is
        +2,048 c; a naive `vredsum` per row costs 8-10 c and would be +4,608.
      compute floor       4,096 + 2,048                   = 6,144 c

    MEMORY FLOOR
      reads  : K8 32 KiB + kscale 2 KiB + q8 64 B (hoisted, read once)
                                       = 34,880 B / 16 B/c = 2,180 c
      writes : scores 2 KiB             =  2,048 B / 16 B/c =   128 c
      memory floor = max(2,180, 128)                        = 2,180 c

    roofline = max(6,144, 2,180) = **6,144 cycles — COMPUTE binds**, by 2.8x.

**This was 8,192 until 2026-09-07, and the DERIVATION was wrong even though
the number happened to survive** (Round 1 measured 8,574, just 5% above it).
The old note said the binding resource is the int32 accumulate at 4/cycle.
It is not: you do not accumulate every product into int32. You accumulate
PAIRS of products into int16 (`vwmul.vv` then `vwmacc.vv`; `2*127*127 =
32,258 < 32,767`, exact) and spend only ONE int32 `vwadd.wv` per pair, which
halves the 4-lane/cycle work and puts it on a pipe that chains. Round 1's
sibling kernel `llama-attn-pv-int8` used exactly this to reach 4,853 on the
same 32,768-MAC shape, i.e. **below the old 8,192 claim** — which is what
forced this recomputation.

The `+2,048` reduction term is the soft part of this floor and it is an
ESTIMATE, stated so you can disagree with it. If reductions were free the
floor would be 4,096; if they cost a naive 9 cycles each it would be ~8,700,
i.e. ABOVE Round 1's measured 8,574 — which would mean the model is wrong.
6,144 is the honest middle, and Round 1's 8,574 is 1.40x it.

Compare the three versions of this identical 512x64 problem:

| variant | cache | activation | memory floor | compute floor | binding |
|---|---|---|---|---|---|
| `llama-attn-scores` | fp32 128 KiB | fp32 | 8,208 | 8,192 | balanced |
| `llama-attn-scores-q8kv` | int8 32 KiB | fp32 | 2,192 | 20,480 | compute, 9.3x |
| `llama-attn-scores-int8` | int8 32 KiB | int8 | 2,180 | **6,144** | compute, 2.8x |

So quantising the query is what makes quantising the cache pay: it removes the
per-element conversion that `q8kv` had to spend, and it moves the MAC into an
8-lane/cycle int16 destination — a floor BELOW the fp32 version's while
keeping a quarter of the traffic. This is the configuration a real int8
inference stack uses, and it is the only one of the three with slack on BOTH
resources at once.

# Accuracy is not the constraint here

Measured relative L2 error against a true fp32 reference: **1.7e-3**. The gate
is 3e-2, so there is ~17x of margin — symmetric int8 with a per-row scale is
simply not the accuracy problem people assume it is at `d = 64`, because the
dot product's signal and its round-off both grow as `sqrt(d)`. Do not trade
cycles for accuracy you do not need; do not trade accuracy for cycles either,
because Gate A is exact on the integer part.

# What the pristine kernel does

One K row at a time at `e8m1` (32 int8 lanes, so `d = 64` is 2 strips), one
`i32m4` accumulator, one `vredsum` per row on the critical path, and `q8`
re-loaded from memory for all 512 rows. At LMUL=1 on the int8 side the chime is
`VLEN/DLEN = 2` beats, far too short to hide the 4-stage pipe, so the dependent
`vwadd.wv` chain almost certainly serialises — the same diagnosis as
`llama-q8-gemv`'s pristine baseline, which measured 2.37x its floor.

# Directions worth exploring

- **Hoist `q8`.** 64 int8 is a quarter of one `e8m1` register. Load it once,
  not 512 times.
- **Raise LMUL.** At `e8m2`/`e8m4` the accumulator becomes `i32m8`, the chime
  grows to 8/16 beats and the dependent accumulate chain is fully hidden.
- **Several rows per pass** so the `vredsum`s overlap each other instead of
  each blocking a row.
- `d = 64` is two `e8m1` strips or half an `e8m2` group; with a couple of rows
  packed per group the inner loop disappears.
- **Pair products in int16 before widening** (`vwmul.vv` + `vwmacc.vv`, then
  one `vwadd.wv` per pair). This is what makes the 8 MAC/cycle floor real.
# MEASURED — pristine baseline on GENV256D128GemminiShuttleConfig
(Verilator, 2026-09-05): **51,280 cycles**, 20,524 instructions,
i.e. **8.35x the 6,144-cycle compute floor** above (memory floor 2,180),
IPC 0.40.
Quantisation error vs a true fp32 reference: 0.0017 relative L2
(gate 3e-2, so 18x of margin).

All three dtype variants of this identical 512x64 shape, same naivety:

| variant | cache | activation | cycles | vs this |
|---|---|---|---|---|
| fp32 | fp32 128 KiB | fp32 | 60,147 | 1.17x slower |
| q8kv | int8 32 KiB | fp32 | 68,571 | 1.34x slower |
| **int8 (this)** | int8 32 KiB | int8 | **51,280** | 1.00x |

Quantising the cache alone LOSES (68,571 vs 60,147); quantising both
sides WINS. The whole result turns on the per-element int8 -> fp32
widening that an fp32 activation forces and an int8 one deletes.
measured_cycles.json currently costs the model from the q8kv entry and
keeps this one under `additional_measurements`; switching the model to
full int8 attention is a CONTRACT decision, not a kernel one.

# MEASURED — Round 1 of the optimization loop (2026-09-07, Fable 5.1)

**8,574 cycles** — 6.0x the pristine baseline, **1.40x the 6,144-cycle floor**.
`out/loop/20260907-043130-fa24/kernel_best.h`. What worked, and is therefore
NOT worth re-discovering:

- **Segmented loads (`vlseg8e8`, nf=8, `e8m1`, vl=32) to transpose 4 K rows
  into a lane layout where each lane already owns 8 columns of one row.** That
  converts a 64-wide horizontal per-row reduction into an 8-element partial-sum
  problem, deferred to a cheap once-per-8-rows step.
- **`q8` replicated 4x into a static buffer and segment-loaded ONCE** with the
  same lane mapping, so it lives in v8-v15 for the whole kernel.
- **Pair the int8 products in int16 before widening** (`vwmul.vv` +
  `vwmacc.vv`, `|2*127*127| = 32,258 < 32,767`), then join four such pair-sums
  into the `i32m4` accumulator with one `vwadd.vv` + two `vwadd.wv` — three
  widening adds where four separate widenings would have been needed.
- **NO `vredsum` on the critical path at all.** 32 int32 partials per 4-row
  block go to a small static scratch buffer; every 8 rows a `vlseg8e32`
  (vl=8) reloads them TRANSPOSED and 7 `vadd.vv` tree-reduce all 8 rows'
  dot products simultaneously.
- **Double-buffered K loads** (alternating v16-v23 / v0-v7): the segmented
  load of block n+1 is issued before block n's arithmetic runs, and the
  reduce+scale of block n-1 is issued after block n's compute, so both the
  cold-DRAM latency and the scratch store-to-load turnaround sit off the
  critical path.
- **Hand-written inline asm**, with instructions deliberately interleaved so
  no consumer sits directly behind its producer.
- One `vfcvt.f.x.v` + `vfmul.vv` (per-row `kscale`) + `vfmul.vf`
  (`qscale*scale`) per EIGHT rows — the floating-point tail is free.

Still open: the reduction term is the only thing between 8,574 and 4,096;
if the 8-row transpose can be widened or the tree-add folded into the
accumulate, this kernel has ~1.4x left.
""",
)


LLAMA_ATTN_PV_INT8 = Kernel(
    name="llama-attn-pv-int8",
    target="saturn",
    bench_dir="llama-attn-pv-int8",
    kernel_rel="llama-attn-pv-int8/llama_attn_pv_int8.c",
    elf_name="llama-attn-pv-int8.riscv",
    bench_globs=_saturn_globs("llama-attn-pv-int8"),
    # Measured on GENV256D128GemminiShuttleConfig: 1170 s of Verilator wall
    # for one run. The harness is the heaviest of the three variants (it
    # generates fp32 data, quantises it, and runs BOTH an int32 and an fp32
    # golden model), which is why it is slower to simulate than the kernel
    # itself would suggest. 3x that.
    sim_timeout_seconds=3500,
    sim_timeout_cycles=8_000_000,
    reference_cycles=22_856,
    roofline_cycles=4096,
    objective="Make `llama_attn_pv_int8` — probs@V for ONE query head with an "
              "**int8 V cache and int8 probabilities**, int32 accumulation — "
              "run in fewer **cycles** (`mcycle`), without changing what it "
              "computes.",
    check_desc="`llama-attn-pv-int8_main.c` (SEALED) generates fp32 V and P, "
               "quantises V per row, folds those row scales into P and "
               "quantises that with a single scale, then runs TWO checks. "
               "**Gate A** recomputes all 64 outputs with a scalar int32 "
               "accumulation over exactly the bytes your kernel saw and "
               "compares to 1e-4 relative — exact on the integer part, and it "
               "is what scores the attempt. **Gate B** compares against a true "
               "fp32 reference from the ORIGINAL data and fails above 3e-2 "
               "relative L2; the pristine kernel reports 2.9e-3, ~10x inside "
               "the gate, and the value is printed every run. Mismatch prints "
               "`MISMATCH` and returns 1. You get back: pass/fail, `mcycle`, "
               "`minstret`.",
    contract="""\
```c
void llama_attn_pv_int8(size_t S, size_t d, const int8_t *W8, const int8_t *V8,
                        int32_t *acc, float *out, float wscale);
```

`out[j] = (float)(sum_s W8[s] * V8[s*d+j]) * wscale`. `V8` is row-major int8
with row stride exactly `d`; `W8` is the int8 probability vector with the
single fp32 `wscale`; `acc` is a `d`-long int32 scratch buffer **zeroed by the
caller** (the same courtesy `out` gets in the other two variants). Called once
with `S = 512`, `d = 64`.

Products are `int8 * int8` and **the accumulation is `int32` and must not
saturate or round** (`|sum| <= 512 * 127 * 127 = 8,258,048`). Gate A is exact
on the integer part.

`V8`'s per-row dequant scales are NOT a parameter: because this reduction runs
ACROSS rows, a per-row scale cannot factor out of it, so the harness folds the
row scales into the probability vector before quantising it (512 scalar
multiplies, outside the kernel). See the sealed header for the derivation — it
is what buys the pure-integer inner loop.""" + _LLAMA_EW_CONTRACT_TAIL,
    notes="""\
# STATUS OF THIS KERNEL: TERMINAL (2026-09-08, after Round 3). READ FIRST.

    pristine baseline                          22,856
    compute floor (see derivation below)        4,096   1.00x
    **FINAL: 4,814** (run 20260908-202513-620b)          1.175x = 4.75x speedup

    per round: R1 4,853 -> R2 4,853 -> R3 4,814. Two rounds moved it by 0.8%
    total. Do not schedule this kernel again.

NOTE: everything below this banner was written at the end of Round 1 and still
quotes 4,853. The number is stale — the final kernel is in the run dir above.

# Where the floor is

Derived with the ROOFLINE METHOD at the top of this file. The key rule: an
int8 x int8 dot product runs at **8 MAC/cycle**, not 4.

    COMPUTE FLOOR
      `vwmacc` widens int8 x int8 into an int16 accumulator, a 16-bit
      destination -> DLEN/16 = 8 MAC/cycle.
      32,768 MAC / 8                                     = 4,096 c
      reductions: **R = 0.** The output is 64 wide and the accumulation runs
      over the OUTER loop, so no horizontal reduction exists anywhere in this
      kernel. That is why its floor is lower than `attn-scores-int8`'s.
      compute floor                                      = 4,096 c

    MEMORY FLOOR
      reads  : V8 32 KiB + W8 512 B + acc 256 B (the caller's zeros, read once)
                                       = 33,536 B / 16 B/c = 2,096 c
      writes : acc 256 B + out 256 B    =    512 B / 16 B/c =    32 c
      memory floor = max(2,096, 32)                         = 2,096 c

    roofline = max(4,096, 2,096) = **4,096 cycles — COMPUTE binds**, by 2.0x.

**This was 8,192 until 2026-09-07 and Round 1 BROKE it: the winning kernel
measured 4,853 cycles.** The old derivation said the binding resource is the
int32 `vwadd.wv` at 4 elements/cycle. It is not, because you do not widen every
product: pair two rows into one int16 accumulator (`vwmul.vx` then
`vwmacc.vx`; `|2*127*128| = 32,512 < 32,767`, exact) and spend one `vwadd.wv`
per PAIR. That halves the int32 work and puts it on a chaining pipe. 4,853 /
4,096 = 1.18x, so the new floor is tight but honest.

The pair-widen is real work — `32,768/2` int32-lane operations at 4/cycle is
another 4,096 cycles on the widening pipe — but Saturn's arithmetic pipes are
independent DLEN-wide sequencers with full chaining, and the measured 4,853
proves the two overlap almost completely. That overlap is exactly why the
widen is excluded from the floor.

Three-way comparison of the identical 512x64 problem:

| variant | cache | activation | memory floor | compute floor | binding |
|---|---|---|---|---|---|
| `llama-attn-pv` | fp32 128 KiB | fp32 | 8,320 | 8,192 | balanced |
| `llama-attn-pv-q8kv` | int8 32 KiB | fp32 | 2,304 | 20,480 | compute, 8.9x |
| `llama-attn-pv-int8` | int8 32 KiB | int8 | 2,096 | **4,096** | compute, 2.0x |

# The accumulator round-trip is deliberate, and it is the first thing to fix

The pristine kernel keeps the 64-wide int32 accumulator in MEMORY and reloads
and restores it once per V row: 512 rows x 2 strips x (`vle32` + `vse32`) =
2,048 extra memory operations moving 256 KiB that never had to leave the
register file. That is **more traffic than the V cache itself**, and it is the
same deliberate flaw the fp32 and q8kv baselines carry, so that all three
measurements of this shape differ only in dtype and not in how naive they are.

`d = 64` int32 is exactly two `i32m4` groups (or one `i32m8` at VLEN = 256),
so the whole accumulator fits in registers and the `acc` buffer can be touched
exactly twice — once to read the caller's zeros, once to write the result.

# Accuracy: the fold is where the error lives

Measured relative L2 error against a true fp32 reference: **2.9e-3**, gate
3e-2, so ~10x of margin. But note WHERE it comes from — not from V's per-row
int8 (that stays ~0.45% and mostly cancels over a 512-term sum) but from
re-quantising the folded `w = P * vscale` with a SINGLE scale across 512
values whose spread is set by the softmax. That is the half of this scheme to
watch if the shape or the temperature changes; the error budget is written out
in full in `llama-attn-pv-int8_main.c`.

# Directions worth exploring

- **Keep `acc` in registers.** One `i32m8` group holds all 64. This alone
  removes 2,048 memory operations and 256 KiB of traffic.
- **Raise LMUL** so the loop-carried `vwadd.wv` chain is longer than the
  4-stage pipe.
- **Unroll over rows**: `W8[s]` is a scalar operand, so several rows can be
  accumulated into independent groups and folded at the end.
- Software-pipeline the V row loads; with the arithmetic ~4x the memory floor
  there is room to hide the whole 32 KiB stream.
# MEASURED — pristine baseline on GENV256D128GemminiShuttleConfig
(Verilator, 2026-09-05): **22,856 cycles**, 16,877 instructions,
i.e. **5.58x the 4,096-cycle compute floor** above (memory floor 2,096),
IPC 0.74.
Quantisation error vs a true fp32 reference: 0.0029 relative L2
(gate 3e-2, so 10x of margin).

All three dtype variants of this identical 512x64 shape, same naivety:

| variant | cache | activation | cycles | vs this |
|---|---|---|---|---|
| fp32 | fp32 128 KiB | fp32 | 44,765 | 1.96x slower |
| q8kv | int8 32 KiB | fp32 | 62,541 | 2.74x slower |
| **int8 (this)** | int8 32 KiB | int8 | **22,856** | 1.00x |

Quantising the cache alone LOSES (62,541 vs 44,765); quantising both
sides WINS. The whole result turns on the per-element int8 -> fp32
widening that an fp32 activation forces and an int8 one deletes.
measured_cycles.json currently costs the model from the q8kv entry and
keeps this one under `additional_measurements`; switching the model to
full int8 attention is a CONTRACT decision, not a kernel one.

# MEASURED — Round 1 of the optimization loop (2026-09-07, Fable 5.1)

**4,853 cycles** — 4.7x the pristine baseline and **1.18x the 4,096-cycle
floor**. This is the kernel that falsified the old int8 rooflines across the
whole registry. `out/loop/20260907-043140-2246/kernel_best.h`. What worked,
and is therefore NOT worth re-discovering:

- **Pair two V rows into one int16 accumulator, widen once per pair.**
      vwmul.vx  v8, v0, w0     // i16m4 = row0 * W8[s]
      vwmacc.vx v8, w1, v2     // i16m4 += row1 * W8[s+1]
      vwadd.wv  v24, v24, v8   // ONE int32 widen for TWO rows
  Exact because `|2 * 127 * 128| = 32,512 < 32,767`. This is the single trick
  that makes the 8 MAC/cycle floor real.
- **Two alternating int32 accumulator groups** (v24 / v16), merged once at the
  very end with a single `vadd.vv`. A back-to-back RAW dependency on one
  accumulator cost ~3.5 cycles on top of the `vwadd.wv`'s 16 beats;
  interleaving hides it.
- **4 rows per loop iteration** (two pairs), amortising loop overhead and the
  scalar `lbu` loads that feed the `.vx` forms.
- **`e8m2`, vl = 64 = `d`** — one vector covers the whole output width, so
  there is no inner loop over columns at all.
- **Register-resident accumulator, touched exactly twice**: one `vle32.v` of
  the caller's zeros at entry, one store at exit. No spill across S = 512.
- **`vfcvt.f.x.v` + `vfmul.vf wscale` once per output element** (64 total).
- **Hand-written inline asm**, specifically to stop GCC inserting a 16-beat
  `vmv8r.v` per row because it would not let `vwadd.wv`'s destination alias
  its source group.

Still open: only 1.18x remains. Any further win has to come from overlapping
the 32 KiB V stream better, not from the arithmetic.
""",
)


# ---------------------------------------------------------------------------
# LM-head GEMV on Gemmini — the projection that dominates a decode step
# ---------------------------------------------------------------------------
# Same N = 1 decode shape as `llama-q8-gemv-gemmini-n1`, but against a tile of
# the LM head (vocab projection) instead of a tile of an attention/MLP
# projection: K = 2048, M = 2048, i.e. 4 MiB of int8 weights, 4x the n1 entry.
#
# Why it is separate rather than "n1 with a bigger M": the LM head of
# Llama-3.2-1B is 2048 x 128256 = ~250 MiB of int8, several times the rest of
# the model put together. A decode step streams it in full for ONE token, so
# whatever B/cycle Gemmini sustains on a big, cold weight tile is the single
# biggest term in the per-token cost. The M = 512 entries are small enough
# that fixed per-call and tile-search overheads are still visible in the
# number; at 4 MiB they are amortized ~4x further and what is left is much
# closer to pure weight-DMA behaviour.
#
# It has its OWN sealed body, `include/llama_q8_gemv_gemmini_lmhead_body.h`,
# because `include/llama_q8_gemv_gemmini_body.h` hard-codes M = 512 and the
# n1/n16 entries depend on it byte-for-byte.
#
# M = 2048, not 4096: Verilator wall time on this harness is dominated by the
# scalar DRAM fill of B, which is linear in M. The M = 512 entries take ~1100 s
# each; 4 MiB lands around an hour and 8 MiB around two, which is more machine
# time than the extra amortization is worth. If someone later wants M = 4096,
# the sealed body already takes M from the wrapper and its DRAM map leaves
# 9 MiB of room for B, so only `bareMetalC/llama_q8_gemv_gemmini_lmhead.c`
# changes (and `roofline_cycles` doubles to 1,048,576 at the corrected
# 8 B/cycle mbus rate; it read 524,288 here while the entry divided by 16).

LLAMA_Q8_GEMV_GEMMINI_LMHEAD = Kernel(
    name="llama-q8-gemv-gemmini-lmhead",
    target="gemmini",
    bench_dir="bareMetalC",
    kernel_rel="include/llama_q8_gemv_gemmini_lmhead_kernel.h",
    main_src="bareMetalC/llama_q8_gemv_gemmini_lmhead.c",
    # The sealed body is a header, so `include/*.h` in
    # constants.GEMMINI_SEALED_GLOBS already carries it; nothing extra.
    elf_name="llama_q8_gemv_gemmini_lmhead.riscv",
    # The B fill alone is ~4x the n1 harness's, which measured ~1100 s.
    sim_timeout_seconds=12000,
    sim_timeout_cycles=100_000_000,
    cycle_re=r"Cycles taken:\s*(\d+)",
    instret_re=None,
    metric_agg="first",
    reference_cycles=740_101,
    # Weight-DMA bound: 4 MiB / 8 B per cycle. Same reasoning as the n1
    # entry, four times the weights.
    # 2026-09-09: was 262_144 (4 MiB / 16 B per cycle). 16 B/cycle is the sbus
    # and Gemmini DMA-port width; DRAM sits behind an 8 B/cycle mbus
    # (rocket-chip BaseSubsystemConfig `MemoryBusKey => MemoryBusParams(
    # beatBytes = 8)`, not overridden by chipyard's AbstractConfig nor by
    # docker/add_coexist_config.py). 4 MiB / 8 = 524,288.
    roofline_cycles=524_288,
    objective="Make `llama_q8_gemv_gemmini` — a 1x2048x2048 int8 GEMV, one "
              "tile of the Llama-3.2-1B LM-head projection — run in fewer "
              "**cycles** on Gemmini, without changing what it computes.",
    check_desc="`bareMetalC/llama_q8_gemv_gemmini_lmhead.c` and the sealed "
               "body `include/llama_q8_gemv_gemmini_lmhead_body.h` (you own "
               "only `include/llama_q8_gemv_gemmini_lmhead_kernel.h`) "
               "generate A and B from a fixed seed, time your "
               "`llama_q8_gemv_gemmini` with `read_cycles()` and print "
               "`Cycles taken: N` (that N is the objective), then recompute "
               "**8 sampled output elements** with a scalar dot product in "
               "one fused pass over k and compare them exactly; any mismatch "
               "prints `MISMATCH` and `exit(1)`, which scores the attempt "
               "zero. It also prints a checksum over all 2048 outputs. The "
               "check is SAMPLED (a full scalar reference is 4,194k MACs), "
               "so correctness of the rest is on you.",
    contract="""\
```c
static void llama_q8_gemv_gemmini(size_t N, size_t K, size_t M,
                                  const elem_t *A,   // N x K int8, stride K
                                  const elem_t *B,   // K x M int8, stride M
                                  acc_t *C);         // N x M int32, stride M
```

`C[i][j] = sum_k A[i][k] * B[k][j]`, exactly — no bias, no activation, no
output scaling, `int32` accumulation written out at full width (`full_C`).
Called once with `N = 1`, `K = 2048`, `M = 2048`.

`A`, `B` and `C` are at fixed DRAM addresses (`0x81000000`, `0x81100000`,
`0x81A00000`), 1 MiB-aligned, *outside* the ELF; the harness fills them before
calling you. `C` is zeroed by the caller.

You may replace the `tiled_matmul_auto` call with anything that produces the
same `C`: your own tiling loop over `sp_tiled_matmul_ws`, explicit
`gemmini_config_*` / `gemmini_extended_*` sequences, a different loop order,
or a path that avoids feeding padded rows into the array at all. You may NOT
edit `include/gemmini.h`, `include/gemmini_params.h` (generated by the
hardware elaboration and overwritten anyway), the sealed harness body
`include/llama_q8_gemv_gemmini_lmhead_body.h`, or the benchmark wrapper. Read
the `DIM` / `BANK_NUM` / `BANK_ROWS` / `ACC_ROWS` / `MAX_BYTES` macros from
`gemmini_params.h`; never hard-code them.""",
    notes="""\
# Do the arithmetic before you touch the file — this one is MEMORY bound

`N = 1`, `K = 2048`, `M = 2048`.

The weights are the whole story. `B` is `2048 * 2048 = 4 MiB` of int8 and
every byte must be moved into the scratchpad exactly once, no matter how small
`N` is. At the 8 B/cycle the mbus delivers to a saturated Gemmini DMA
(`DIM * sizeof(elem_t)` = 16 B/cycle is the DMA port, not the DRAM path):

    B mvin floor = 4,194,304 B / 8 B per cycle = 524,288 cycles

(8 B/cycle, not the 16 B/cycle this entry assumed until 2026-09-09: Gemmini's
DMA port and the sbus are 128-bit, but DRAM is behind the **mbus**, which
rocket-chip's `BaseSubsystemConfig` fixes at `beatBytes = 8` and which nothing
in this SoC's config chain overrides.)

Compare that with the array:

    MACs           = 1 * 2048 * 2048 = 4,194,304
    array peak     = DIM * DIM = 256 MAC/cycle
    compute floor  = 4,194,304 / 256 = 16,384 cycles

So at N = 1 the array could in principle finish in 16,384 cycles and then sit
idle for 507,904 waiting on weights: the DMA bound is **32x** the compute
bound, exactly as in `llama-q8-gemv-gemmini-n1`. This is a memory-bound kernel
wearing a systolic array's clothes.

**524,288 cycles is the hard floor for this shape.** It is set by weight
traffic, not by the array, and no amount of scheduling can move it — the only
thing that would is not re-reading the weights, which is impossible here: the
scratchpad is `BANK_NUM * BANK_ROWS * DIM` = 256 KiB and B is 4 MiB.

# Why this kernel exists

The LM head of Llama-3.2-1B is `2048 x 128256` — about 250 MiB of int8, more
than the rest of the model combined. A decode step streams all of it to
produce logits for ONE token, so the B/cycle Gemmini sustains on a large,
cold weight tile is the single biggest term in per-token cost.
`llama-q8-gemv-gemmini-n1` measures the same N = 1 shape at only 1 MiB, where
fixed per-call cost and `tiled_matmul_auto`'s tile search are still a visible
fraction of the number. This entry is the same question with the overheads
amortized 4x further.

Compare the two directly: if this kernel lands at ~4x the n1 cycles, the
overhead was already negligible at 1 MiB and the ratio is pure bandwidth; if
it lands materially below 4x, the n1 number was overhead-inflated and the real
decode bandwidth is better than n1 suggests.

# Where the cycles above the floor plausibly live

1. **Padded rows.** `dim_I = 1` is rounded up to a multiple of `DIM = 16`
   inside `tiled_matmul_auto`. 15 of every 16 rows fed through the array are
   zero padding — array time, plus mvin bandwidth for the padded `A` tile.
2. **Tile-search overhead** in `tiled_matmul_auto` runs inside the timed
   region and does not know the shape is this lopsided. Its share of the
   number is 4x smaller here than at M = 512, which is part of the point.
3. **Overlap.** With so little compute per weight tile, mvin latency is much
   harder to hide than in the prefill GEMM: any `gemmini_fence()` or
   dependency stall in the K loop is directly on the critical path.
4. **Accumulator use.** The output is `ceil(1/16) x 128 = 128` acc tiles out of
   `ACC_ROWS / DIM`, so unlike the M = 512 entries the accumulator is NOT
   nearly empty — the J dimension is now big enough that the generic tiling
   has a real choice to make about how much of the K reduction stays on-chip.

# Nothing has been ruled out empirically yet

You are the first optimization session on this kernel. Read
`tiled_matmul_auto`, `tiled_matmul_outer` and `sp_tiled_matmul_ws` in
`include/gemmini.h` (you may read it; you may not edit it) before proposing
anything.

# MEASURED

| what | cycles | vs floor | B/cycle |
|---|---|---|---|
| pristine `tiled_matmul_auto`, M=2048 | **740,101** | 1.41x | 5.67 |
| best so far (v6, hand-issued) | **635,908** | 1.21x | 6.60 |
| `llama-q8-gemv-gemmini-n1`, M=512, best | 132,424 | 1.010x | 7.92 |
| weight-DMA floor (4 MiB / 8 B per cycle) | 524,288 | 1.00x | 8.00 |

4x the weights cost **4.92x** the cycles, not 4x. So the answer to the question
above is the *opposite* of the optimistic one: the M = 512 number was not
overhead-inflated — sustained weight bandwidth gets WORSE as the tile grows,
from 6.97 B/cycle down to 5.67 B/cycle (87% -> 71% of the 8 B/cycle mbus). Whatever
is costing the extra 19% per byte scales with the size of B, not with the
number of calls: DRAM/L2 behaviour on a 4 MiB cold stream, or a `stride_B` of
2048 making each mvin of a 16-wide tile touch 16 separate pages.

That matters for the model: costing the ~250 MiB LM head from the n1 number
under-predicts it by about a fifth, and the gap to the corrected roofline here (1.21x at the current best) is
the largest of any Gemmini entry — i.e. this is the most headroom on the board,
and all of it is in the weight path."""
    + _GEMV_LMHEAD_ROUND4 + _GEMMINI_DMA_HINTS,
)


# ---------------------------------------------------------------------------
# Layer fusion — does the Saturn attention work fit inside the Gemmini weight
# DMA's shadow?  (Round 7, 2026-09-13)
# ---------------------------------------------------------------------------

_LAYER_FUSED_CONTRACT = """\
```c
static void llama_layer_fused(size_t N, size_t Kdim, size_t M,
                              const elem_t *A, const elem_t *B, acc_t *C,
                              size_t S, size_t d,
                              const elem_t *K8, const float *kscale,
                              const elem_t *q8, float *scores, float *probs,
                              const elem_t *W8, const elem_t *V8,
                              int32_t *pvacc, float *pvout,
                              float qscale, float scale, float wscale);
```

Called ONCE with `N = 1`, `Kdim = 2048`, `M = 512`, `S = 512`, `d = 64`,
`qscale = 1.0f`, `scale = 0.125f`.  Four independent pieces of work, one
timed region.  You must produce all four results:

1. **Gemmini GEMV** — `C[j] = sum_k A[k] * B[k*M + j]` for `j` in `[0, 512)`,
   exact int32, no bias / activation / output scaling.  `B` is 1 MiB of int8.
2. **QK^T** — `scores[s] = (float)(sum_j K8[s*d+j] * q8[j]) * kscale[s] *
   qscale * scale`.  `K8` is row-major int8 with row stride exactly `d`; the
   dot product accumulates in **int32** and must not saturate or round
   (`|sum| <= 64*127*127 = 1,032,256`).
3. **softmax** — `m = max_s scores[s]`;
   `probs[s] = exp(scores[s]-m) / sum_t exp(scores[t]-m)`.  The max
   subtraction is REQUIRED; the data range is calibrated to about `[-8, 8]`
   so that dropping it is visible.  The exponential may be any approximation
   that clears the gate.
4. **probs@V** — `pvacc[j] = sum_s W8[s] * V8[s*d+j]` (exact int32,
   `|sum| <= 512*127*127`) and `pvout[j] = (float)pvacc[j] * wscale`.
   `V8` is row-major int8 with row stride exactly `d`.  **`W8` is a SEPARATE
   pre-quantised int8 probability vector supplied by the harness — it is NOT
   `probs`.**  That is deliberate: quantising `probs` inside the timed region
   would add work no existing kernel measures, and the point of this entry is
   the schedule, not a new quantiser.

**The ONLY data dependency among the four is `scores -> probs`.**  Everything
else may be reordered, interleaved, split, or issued in any order you like.
`C`, `scores`, `probs`, `pvacc` and `pvout` are all zeroed by the caller.

All arrays are at fixed DRAM addresses outside the ELF, 4 KiB- or 1 MiB-aligned
(see the sealed body for the map).  The harness streams 2 MiB of unrelated data
through the L2 with Gemmini's DMA immediately before starting the clock, so
**everything you read starts cold**.

You may NOT edit `include/gemmini.h`, `include/gemmini_params.h` (generated by
the hardware elaboration and overwritten anyway), the sealed harness body
`include/llama_layer_fused_body.h`, or the benchmark wrapper
`bareMetalC/llama_layer_fused_n1.c`.  Read the `DIM` / `BANK_NUM` /
`BANK_ROWS` / `ACC_ROWS` / `MAX_BYTES` macros from `gemmini_params.h`; never
hard-code them.

`mstatus.VS` is already enabled by the sealed body — `<riscv_vector.h>`
intrinsics and hand-written vector asm both work out of the box."""


_LAYER_FUSED_NOTES = """\
# WHAT THIS KERNEL IS FOR

This is the ONLY entry on the board whose objective is a **schedule**, not an
inner loop.  All four pieces of work are already at (or within 1-30% of) their
individual floors, and the file you are given contains the best known version
of each one:

| piece | source entry | best measured (its own harness) |
|---|---|---|
| Gemmini GEMV 1x2048x512 int8 | `llama-q8-gemv-gemmini-n1` | 132,424 (warm L2) |
| QK^T, S=512 d=64 int8        | `llama-attn-scores-int8`   |   8,233 |
| softmax, n=512 fp32          | `llama-softmax`            |   1,555 |
| probs@V, S=512 d=64 int8     | `llama-attn-pv-int8`       |   4,814 |

14,602 Saturn cycles and 132,424 Gemmini cycles, and until now the Llama cost
model (`loop/llama_project.py`) has been **adding** them, because nothing had
ever measured them in the same timed region.  Adding is the pessimistic
assumption, and this benchmark exists to find out whether it is the true one.

**Do not rewrite the four inner loops.  Change the boundary between them.**
An iteration that makes `lf_attn_pv` 200 cycles faster and leaves the fence
where it is has wasted itself: there are ~14,600 cycles on the table at the
boundary and ~0 inside the bodies.

# THE MECHANISM — why overlap is even possible here

Gemmini is a **RoCC accelerator**, not a library call.  `gemmini_extended_mvin`,
`gemmini_preload`, `gemmini_compute_*` and the `gemmini_loop_ws` hardware loop
are all *instructions the host issues and retires*; the DMA and the array then
run on their own.  What the host does next is its own business.  Concretely,
in the file you are given:

  * `sp_tiled_matmul_ws` expands into ~2 host commands per `loop_ws`
    invocation, and there are **9 of them** for this shape.  Between the 9
    issues the host core is doing *nothing* for ~14,700 cycles each.
    (The baseline already knows this: it fills that gap with a budgeted
    Zicbop `prefetch.r` burst, `LF_PREFETCH_BUDGET_CYCLES = 2500` per gap,
    worth -229 cycles when the L2 was warm.  That burst is the incumbent
    occupant of exactly the slack you want.)
  * The Saturn vector unit is a **separate functional unit on that same host
    core**.  Vector instructions issue from Shuttle's pipeline and execute in
    Saturn's own sequencers; they do not touch the RoCC port and they do not
    touch the Gemmini command queues.
  * `gemmini_fence()` is `asm volatile("fence")`, and on this core a `fence`
    stalls until the RoCC accelerator reports not-busy.  **That single
    instruction is what currently serialises the two halves.**  The baseline
    calls it at the end of `lf_gemv_gemmini`, before the first vector
    instruction.  The only fence that is *required* for correctness is one
    after the final `mvout`, before anything reads `C` — and nothing in the
    Saturn half reads `C`.

So the schedule you want looks like:

    issue loop_ws for K-tile 8          (Gemmini starts streaming)
    ... Saturn: QK^T rows 0..63 ...     (host + vector unit, concurrently)
    issue loop_ws for K-tile 7
    ... Saturn: QK^T rows 64..127 ...
    ...
    issue loop_ws for K-tile 0
    ... Saturn: softmax, probs@V ...
    gemmini_fence()                      (once, at the very end)

# THE SHARED RESOURCE — and why this is NOT the refuted idea

`llama-q8-gemv-gemmini-n1` carries a hard warning that host/Saturn co-compute
is CLOSED, with five measured failures.  Read it, then read this, because the
two are different experiments and the warning does not transfer:

**What was refuted** was giving Saturn a *slice of the same GEMV*: the host
computed rows 1792..2047 of the same `A*B` product while Gemmini did the rest.
That fails for a mechanical reason.  Both requesters go

    Gemmini StreamReader / Shuttle D$ --> tile xbar --> sbus (16 B/cycle)
      --> SiFive InclusiveCache L2, 512 KiB --> **mbus, 8 B/cycle** --> DRAM

and the weight stream alone already draws 7.92 of those 8 B/cycle.  A Saturn
slice of the GEMV reads *the same weight bytes* over *the same saturated
link*, so it cannot add bandwidth — it can only take Gemmini's share.  All
five attempts were net-negative and the mechanism explains every one.

**What is being tested here is different in exactly the way that matters**:
the Saturn work is **other data**, and there is 14x less of it.

    Gemmini weight traffic   1,048,576 B   =  93.5% of the bytes
    Saturn cold reads        ~72,768 B     =   6.5% of the bytes
    Saturn arithmetic        ~11,712 cycles of pipe occupancy

The Saturn half is 6.5% of the traffic and ~100% of it is *compute-bound on
its own unit*, not bandwidth-bound.  `llama-attn-scores-int8`'s own roofline
says so: compute floor 6,144 vs memory floor 2,180, compute binds by 2.8x.
So the question is not "can two requesters share a saturated link" (no) but
"can 11,712 cycles of vector arithmetic and 73 KiB of reads hide inside
131,072 cycles of DMA that leaves the host idle" (unknown — measure it).

The honest failure mode to watch for: those 73 KiB still cross the mbus and
will displace ~9,100 cycles worth of weight bytes if they are not already
absorbed.  A PERFECT overlap is therefore worth ~14,600 cycles, and a merely
good one might be worth only ~5,000.  Both are worth having; neither is free.

# ROOFLINE

Derived with the ROOFLINE METHOD at the top of this file.  Two resources, and
they are genuinely independent, so the fused floor is a `max`, not a sum.

    MEMORY FLOOR — everything crosses the one 8 B/cycle mbus
      B weights                        1,048,576 B   (every byte once)
      K8  512 x 64 int8                   32,768 B
      V8  512 x 64 int8                   32,768 B
      kscale 512 fp32                      2,048 B
      q8 64 + W8 512                         576 B
      write-allocate on scores/probs/
        pvacc/pvout/C                      ~4,608 B
      A 2048 + C out 2048                  ~4,096 B
      ------------------------------------------------
      total                            ~1,125,440 B
      memory floor = 1,125,440 / 8              = 140,680 cycles

    COMPUTE FLOOR — two different units, so take the max of the two
      Gemmini array: 1,048,576 MAC / 256 MAC per cycle  =  4,096 c
      Saturn pipes (sum, they share one arithmetic sequencer):
          QK^T      MAC 32,768 / 8 = 4,096  + ~2,048 reduction  = 6,144 c
          softmax   10 ops per 32-element group x 16 groups,
                    3 passes                                    = 1,472 c
          probs@V   MAC 32,768 / 8                              = 4,096 c
          Saturn total                                          = 11,712 c
      compute floor = max(4,096, 11,712)                   =  11,712 cycles

    roofline = max(140,680, 11,712) = **140,680 cycles — MEMORY binds, by 12x.**

Two things follow from that arithmetic and you should believe both:

  a) **The Saturn work is free in principle.**  11,712 cycles of arithmetic
     against 140,680 cycles of unavoidable DMA: there is 12x more shadow than
     there is work to hide in it.
  b) **The Saturn work's BYTES are not free.**  73 KiB at 8 B/cycle is 9,100
     cycles of mbus time that the weight stream does not get.  A schedule that
     perfectly hides the arithmetic but issues the K8/V8 reads at the worst
     moment can still lose.  Prefetching K8 and V8 *early*, while the first
     weight tile is still warm from nothing, is a legitimate idea.

140,680 is a HARD lower bound and it is not reachable: the measured COLD
Gemmini weight rate on this SoC is 6.0-7.1 B/cycle, not 8.0 (the 7.92 B/cycle
the n1 entry advertises includes a ~200-260 KiB warm-L2 tail that this harness
deliberately destroys).  At 7.0 B/cycle the weight stream alone is ~150,000
cycles, and that — not 140,680 — is the number the baseline will land near.
**Quote the baseline, not the roofline, when you predict.**

# THE COLD-STATE FLUSH — read this before you are surprised by the baseline

The sealed body streams 2 MiB of unrelated DRAM through the 512 KiB L2 with
Gemmini's own DMA immediately before `read_cycles()`.  This is deliberate and
it is the reason this entry's GEMV half will measure ~149k where
`llama-q8-gemv-gemmini-n1` measures 132,424:

  * that entry's harness fills B in DRAM immediately before calling the
    kernel, leaving ~200-260 KiB of B resident in the (random-replacement) L2;
  * consuming K tiles in DESCENDING order harvests that residue, and doing so
    is worth 17,104 measured cycles;
  * it is a property of the BENCHMARK, not of a decode step.  In a real Llama
    decode the previous layer's weights have just evicted everything.

So: **descending K order may no longer be worth anything here.**  It is one
line in `lf_gemv_gemmini` and flipping it is a clean, cheap control experiment
that produces a number nobody has.  Do not assume it still helps; do not
assume it still hurts.

# WHERE THE CYCLES ABOVE THE BASELINE PLAUSIBLY LIVE

1. **The fence.**  One `gemmini_fence()` at the end of `lf_gemv_gemmini`,
   worth the entire 14,602-cycle Saturn half.  This is the whole kernel.
2. **The Zicbop prefetch burst.**  It occupies the host slack between
   `loop_ws` issues — up to 2,500 cycles x 8 gaps = 20,000 host cycles — and
   it bought -229 cycles when the L2 was warm.  Cold, it may be worth more
   (nothing is resident) or nothing at all.  Either way it is *competing with
   the Saturn work for the same slack*, and you get to choose how to split it.
3. **Granularity.**  The Saturn half is 14,602 cycles and there are 8 usable
   gaps between the 9 `loop_ws` issues.  Splitting QK^T into 8 chunks of 64
   rows (~1,030 cycles each) fits; so does 2 chunks of 256.  The scores ->
   probs dependency means softmax cannot start until every chunk is done, so
   softmax + probs@V (~6,400 cycles) has to live in the last gap or after.
4. **The `sp_tiled_matmul_ws` call is not actually non-blocking.**  It issues
   into a finite ld/ex queue and `gemmini_loop_ws` waits for a free loop slot;
   measurements in the n1 entry say the call returns once loop `kk-2` has
   retired.  So the host regains control *during* tile `kk-1`, not after
   tile `kk`.  That is the real shape of the slack — verify it with a PROBE
   before you build a schedule that assumes otherwise.
5. **Vector/RoCC interaction is UNMEASURED.**  Nobody has ever run Saturn
   vector instructions concurrently with a Gemmini DMA on this SoC.  Shuttle
   is in-order: a long vector op occupies its sequencer but should not block
   the scalar RoCC issue behind it... *should not*.  If it does, the first
   PROBE will say so immediately and that is a valuable answer.
6. **The 73 KiB of Saturn reads on the mbus** (see the roofline).  Worth up to
   ~9,100 cycles if they land badly.

# MEASURED BASELINE — 206,304 cycles (2026-09-13, run 20260912-181641-b631)

| what | cycles |
|---|---|
| **BASELINE, sequential, cold** | **206,304** |
| roofline (memory floor, 8 B/cycle) | 140,680  (1.47x) |
| sum of the four kernels' own best numbers, WARM | 147,026 |
| ... Gemmini GEMV n1 (warm L2, its own harness) | 132,424 |
| ... QK^T + softmax + probs@V (warm, their own harnesses) | 14,602 |

**The baseline is 59,278 cycles ABOVE the warm sum, and that gap is the first
thing to explain.** It is the most informative number this entry has produced
and NOBODY KNOWS WHICH HALF IT BELONGS TO. Two candidate mechanisms, and they
predict DIFFERENT optimal schedules:

  a) **The GEMV half got slower because the L2 is cold.** The 132,424 figure
     includes a ~200-260 KiB warm-L2 tail worth 17,104 measured cycles, and
     descending-K order is what harvests it. Cold, the same code should land
     near 149,000-150,000. That accounts for ~17k of the 59k, not all of it.
  b) **The Saturn half got much slower because K8/V8/kscale are cold too.**
     In their own harnesses those arrays were written immediately before the
     kernel ran, so they were L1/L2 resident and the kernels are compute-bound
     by construction. Here they are 68 KiB of cold DRAM behind the same
     8 B/cycle mbus: ~8,500 cycles of pure link time the warm numbers never
     paid, PLUS miss latency the tight vector loops cannot hide
     (`lf_attn_scores` double-buffers K blocks in REGISTERS, one 256 B block
     ahead — nowhere near a DRAM round trip of slack).

If (b) dominates, the Saturn half is no longer 14,602 cycles of arithmetic to
hide; it is tens of thousands of cycles of mostly-stall, and the winning move
is to **get K8 and V8 into the L2 early** (64 KiB total, no scratchpad or
accumulator pressure at all) rather than to interleave more finely.

**ITERATION 1 SHOULD MEASURE THIS SPLIT AND NOTHING ELSE.** Time the four
phases separately with `read_cycles()` inside the timed region and print:

    PROBE gemv_cycles=NNNNNN
    PROBE scores_cycles=NNNNN
    PROBE softmax_cycles=NNNNN
    PROBE pv_cycles=NNNNN

The total stays ~206k either way, so the iteration costs nothing in score and
it turns one number into four. Do this before proposing any schedule.

# HOW TO SPEND AN ITERATION

Before editing anything, write exactly these three lines at the top of the
file, with specific numbers:

    HYPOTHESIS: <the mechanism costing cycles, with an arithmetic estimate of
                 how many, derived from the tables above — not "overhead",
                 not "stalls">
    CHANGE:     <the one structural thing you are changing>
    EXPECTED:   <a single predicted cycle count>

After the measurement, state in one line whether the hypothesis SURVIVED or
DIED and what that retires.  ONE mechanism per iteration.  A confident
prediction that misses by 10k is worth more than a 200-cycle tweak, because it
closes a branch.

**`printf` probes work and their output comes back.**  Since 2026-09-11 the
simulator log is saved every iteration (`out/loop/<run_id>/simlog_NN.txt`) and
`llm.format_feedback` appends a "=== benchmark stdout (tail) ===" section to
PASSING-run feedback, capped at 4 KB.  Prefix every probe line with `PROBE `
and print one number per line, e.g. `printf("PROBE gemv_done=%lu\\n", c);`.
`read_cycles()` / `rdcycle` inside the timed region costs a few cycles and is
absolutely worth it here — the single most useful number nobody has is **where
the Gemmini half actually finishes relative to the Saturn half**.

Probes worth more than a tweak, in order:
  1. Timestamp each of the 9 `loop_ws` issues and each Saturn phase boundary.
     That one run tells you the true shape of the slack, the true cold weight
     rate, and whether vector work delays RoCC issue — three unknowns at once.
  2. Run the Saturn half ALONE (skip the Gemmini call, fail the GEMV check on
     purpose is NOT acceptable — instead time it separately and print it) to
     get the cold-state Saturn cost, which is not 14,602: K8 and V8 are cold
     here and were warm in their own harnesses.
  3. GEMV-only, cold: the honest cold weight rate for this SoC, which the
     whole Llama cost model rests on and which is still a FIT, not a
     measurement.

# WHAT WOULD MAKE THIS A ROUND-7 WIN

The model-level claim this entry can support, if it works:
`llama_project.py` currently sums Gemmini and Saturn cycles.  The Saturn ops
are ~5% of decode time and ~1% of decode traffic.  If this benchmark shows
that ~N% of the Saturn time hides inside the Gemmini DMA, the projection can
legitimately be restated as a `max` over the overlapped region for that N, and
the per-token number moves by roughly the same few percent.  A *negative*
result is also publishable and closes the question for good — but it has to be
a measured negative with a mechanism, not a timeout."""



# ---------------------------------------------------------------- Round 8
_LAYER_FUSED_ROUND8 = """\
# ROUND 7 RESULTS (4 of 8 iterations; the round was cut short by a weekly
# usage cap, not by running out of ideas). Round 8 continues from the best.

    baseline (strictly sequential)   206,304
    iter1  naive fusion, no overlap  217,295   regression
    iter2  software pipelining       189,851   BEST, -8.0% vs baseline
    iter3  scheduling tweak          189,924   flat
    iter4  re-split + phase probe    223,640   regression, but see the probe

# PHASE DECOMPOSITION (iter4 `PROBE f 186245 s 196526 m 198168 p 205864`,
# cumulative timestamps -> per-phase costs):

    Gemmini GEMV + fence   186,245   90.5%
    attn-scores             10,281    5.0%
    attn-pv                  7,696    3.7%
    softmax                  1,642    0.8%

Gemmini dominates completely. That is WHY overlap paid and re-splitting did
not: there is only one big thing to hide the small things behind. The Saturn
tail actually measures 19,619 cycles, larger than the 14,602 this file
assumed, so there is more to hide than was budgeted -- and iter2 has not
necessarily hidden all of it.

# THE COLD RATE IS WORSE THAN THIS FILE SAYS ELSEWHERE

Round 7's companion probe (run 20260912-191830-e713) measured the SAME best
n1 kernel twice:

    warm (harness residue in L2)   132,435   7.91 B/cycle
    cold (L2 flushed first)        184,837   5.67 B/cycle

Every "cold DRAM sustains 7.0-7.1 B/cycle" line elsewhere in this file is a
fit to warm-tail-contaminated data. The true cold `loop_ws` rate is 5.67-5.68
B/cycle. This kernel runs cold by construction, so budget against 5.67.

# OPERATOR SIGN-OFF (2026-09-14): retiling the inner GEMV is APPROVED

Round 7 flagged "apply the 4-row-mvin finding to `lf_gemv_gemmini`" as needing
approval because it changes the fused kernel's inner loop structure. The
operator approved it. You may restructure the GEMV inside this kernel,
including its mvin row count and tiling, as long as the self-check still
passes and the timed region still covers all four phases. Expected ~15k.

# THE FOUR REMAINING EXPERIMENTS, IN PRIORITY ORDER

1. GEMV retiling (approved above): 4 rows per mvin, ~15k expected -> ~175k.
2. Prefetch K8/V8 earlier. The Saturn tail is 19,619 cycles and its operands
   are cold DRAM now (they were L1/L2-resident in their standalone harnesses,
   which is why those kernels looked compute-bound). Issue their loads before
   the GEMV's last K block rather than after the fence.
3. Descending vs ascending K order, re-tested cold. Descending was worth
   17,104 cycles WARM; with the L2 flushed there may be no warm tail left to
   harvest, in which case the ordering constraint is free to drop.
4. Finer QK^T chunking (8x64 rows) so attn-scores interleaves at a smaller
   grain into whatever DMA gaps remain.

One mechanism per iteration, HYPOTHESIS/CHANGE/EXPECTED at the top, and print
probe numbers with a `PROBE ` prefix -- passing runs now get the benchmark
stdout tail back in the feedback."""


# ---------------------------------------------------------------- Round 9
_LAYER_FUSED_ROUND9 = """\
# ROUND 8 RESULTS (iterations 5-8 of this kernel; seeded from 189,851)

    iter5  LF_B_ROWS 16 -> 4, Saturn work into host stalls   179,585  BEST
    iter6  diagnostic: same stream, Saturn serialized        198,688
    iter7  LF_B_ROWS 4 -> 8, chasing more in-flight reqs     199,347  regression
    iter8  rotate B slots across spad banks 1-3              181,860  close

    baseline 206,304 -> 179,585 = -13.0%.  Roofline 140,680 (unreachable).

# THE ONE NUMBER THAT DEFINES ROUND 9

iter6's probe reran iter5's EXACT weight stream with the Saturn phases forced
back into sequence:

    PROBE stream=161687 qk=16569 softmax=1613 pv=10190

    weight stream alone                 161,687
    Saturn phases, serialized            28,372
    iter5's actual total                179,585
    => Saturn cost still exposed         17,898   (63% of it)
    => Saturn cost already hidden        10,474   (37% of it)

So the overlap that won Round 7-8 is only ONE THIRD complete. If every Saturn
cycle could be issued inside a DMA stall, this kernel lands at ~161,700 --
another -10%. That is the whole of Round 9. Nothing else on this kernel is
worth an iteration.

# WHY 17,898 CYCLES ARE STILL EXPOSED -- READ BEFORE HYPOTHESISING

Facts established across rounds 7-8, do not re-derive them:
  * Gemmini dominates: 90.5% of the fused kernel is GEMV + fence (round 7
    iter4 phase probe). There is exactly one big thing to hide behind.
  * The Saturn operands (K8, V8, kscale) are COLD here -- 68 KiB of DRAM that
    was L1/L2-resident in the standalone attention harnesses. Those kernels
    only looked compute-bound because of that residency.
  * 4 rows per mvin beats 8 and 16 (iter5 vs iter7, and lmhead's own curve).
    8 rows regressed because of scratchpad BANK CONFLICTS, not latency --
    iter8 confirmed this by rotating slots across banks 1-3 and recovering
    most of the loss. Do not re-test 8 or 16 rows.
  * ExecuteController reads beat DMA writes on the single-ported scratchpad
    (Scratchpad.scala:522). Vector work that touches spad-adjacent state can
    stall the stream it is meant to hide inside.

Plausible mechanisms for the exposed 17,898, in the order worth testing:
  1. The Saturn loads are issued AFTER the weight stream has drained, so they
     pay full cold-DRAM latency with no DMA in flight to amortise them.
     Issuing K8/V8 loads BEFORE the last K block is the obvious fix and has
     never been tried in this kernel.
  2. The interleave grain is one whole phase (qk, then softmax, then pv). A
     DMA stall is not phase-sized. Chop QK^T into 8x64-row chunks and drop
     one chunk into each stall.
  3. softmax (1,613) has a serial dependency on all of qk and blocks all of
     pv. It may be the true critical path even though it is small.

# RULES FOR ROUND 9

One mechanism per iteration. HYPOTHESIS / CHANGE / EXPECTED at the top, with
EXPECTED as a single number derived from the 161,687 / 17,898 split above.
Print every measurement with a `PROBE ` prefix, one number per line -- passing
runs get the benchmark stdout tail back, and a probe whose numbers are lost is
a wasted hour of simulation. If you re-run a phase split, print it in exactly
iter6's format so the rounds stay comparable.

Results from this kernel are going into a paper. An iteration that produces a
clean measurement and a retired hypothesis is worth more than one that shaves
200 cycles by luck."""

# --------------------------------------------------------------- Round 10
_LAYER_FUSED_ROUND10 = """\
# ROUND 9 RESULTS (iterations 1-4, run 20260916-195958-36b4, seeded from 179,585)

    iter1  DMA-warm K8/V8/kscale into L2 ahead of use     185,374  regression
    iter2  halve the Saturn unit grain, SAT_EVERY 16->8   180,107  flat
    iter3  probe-only run on the unchanged iter5 best     183,430  probe
    iter4  incremental RoCC operands (19 -> 10 scalar)    179,033  BEST

    baseline 206,304 -> 179,033 = -13.2%.  Roofline 140,680 (unreachable).
    Round 8 best was 179,585, so Round 9 bought 552 cycles (-0.3%) and four
    retired hypotheses.  That is the honest summary: the MECHANISM is now
    understood and the cycles are not there.

# THE STRUCTURE OF 179,033 -- USE THESE NUMBERS, DO NOT RE-DERIVE THEM

Geometry: 1 MiB of B at 4 rows x 64 B per mvin2 = **4,096 mvins**, one
`lf_sat_unit` call every 16 mvins = **256 calls**, of which only the first
~130 do real Saturn work (they run out at mvin 2,080).

    pure weight stream, Saturn serialised out   161,687   39.47 c/mvin, 6.48 B/c
    exposed Saturn cost (179,033 - 161,687)      17,346    9.7% of the total
    ------------------------------------------------------------------
    total                                       179,033

iter3's probes split the stream in two, and the split is unambiguous:

    first 2,080 mvins (units active)   104,365   50.2 c/mvin
    last  2,016 mvins (units done)      78,863   39.1 c/mvin  == pure rate
    PROBE unit_host_cycles=48799 unit_calls=256   190.6 c/call
    PROBE mvin_stall_sampled=2558 mvin_samples=512   5.0 c per mvin2 issue

# DO NOT MISREAD `sat_done` -- THE OVERLAP IS 64% COMPLETE, NOT 100%

All three probe runs report `sat_done` far earlier than `gemv_end`
(106,708/185,332; 99,956/179,890; 104,365/183,228).  This does **not** mean
the Saturn work is fully hidden.  `sat_done` is early BY CONSTRUCTION: 130
working units x `LF_SAT_EVERY` 16 = 2,080 of the 4,096 mvins, so the Saturn
work is exhausted halfway down the stream whatever it costs.  What it costs is
visible in the RATE, not the timestamp: the first half runs 11 c/mvin slower
than the second half, which is exactly the residual exposure.

    host cycles held inside units        48,799   (190.6 c x 256 calls;
                                                   ~375 c per WORKING unit)
    of which absorbed into DMA slack     31,453   64%
    of which exposed as wall time        17,346   36%

Round 8's "17,898 still exposed" therefore became 17,346 -- essentially
unchanged.  Three rounds of scheduling work have moved the exposure by 3%.

Note also that the 48,799 cycles of host occupancy dwarf the 14,602-cycle warm
standalone Saturn cost: most of a unit is the in-order core waiting on COLD
K8/V8 misses that the Saturn VLSU does not decouple far enough to hide.

# FOUR HYPOTHESES RETIRED -- DO NOT RE-TEST ANY OF THEM

  * **Bytes are not the cost (iter1, DIED).** Warming the 68 KiB of Saturn
    operands with Gemmini's own DMA (mvin3, 512 B chunks, 4 units ahead) cost
    +5,789 cycles.  Those bytes joined the weight stream's request pool and
    paid full link time (~10.5k), whereas Saturn's own cold loads were riding
    the mbus bandwidth the 6.48 B/c stream leaves idle -- nearly free.
    Prefetching Saturn operands through Gemmini is a LOSS, permanently.
  * **Unit grain is not the cost (iter2, DIED).** Splitting every phase-1 unit
    into two halves and issuing every 8 mvins instead of 16 (same total vector
    instruction count) changed the total by +522 cycles.  The exposure scales
    with the TOTAL vector work, not with how it is chopped up, so "the unit
    overflows the ld queue's buffered DMA work" is wrong.
  * **The load queue never blocks the host (iter3, model D DIED).**
    `mvin_stall_sampled` = 5.0 c per mvin2 including two `rdcycle`s.  The ld
    reservation station is never full enough to stall issue; the host always
    has slack against the DMA's 39.5 c/mvin.
  * **The host scalar issue path is not the co-bottleneck (iter4, model E
    mostly DIED).** Rewriting every RoCC operand as an incrementally updated
    64-bit word cut the inner loop from 19 scalar + 3 RoCC to 10 + 3 (verified
    by disassembly).  Predicted -9.5k, delivered **-552**: ~0.27 c/mvin, not
    5 c/mvin.  Host scalar work is nearly free here.  It is still the best
    kernel, so keep the incremental operands -- but there is no second helping.

# WHAT IS LEFT, AND WHY IT IS SMALL

  * The stream floor itself. 161,687 = 6.48 B/c against an 8 B/c mbus.  The
    in-flight scan (6/8/10/12/16 -> 6.31/6.63/6.06/5.83/5.78 B/c) says the
    memory system tops out near 6.6 B/c, i.e. ~158k.  At most ~3.5k, and every
    knob on that curve has been turned.
  * The residual 17,346 of exposure.  The only surviving model is that the
    in-order Shuttle core is HELD for a whole vector unit (~375 c) -- mostly on
    cold K8/V8 miss latency -- and during the part of that window that exceeds
    the DMA work already queued, the stream idles.  The untested lever is not
    grain and not bytes but **distance**: issue unit n+1's `LF_LOADK` /
    `LF_PV4` vector loads during unit n, so the held window covers arithmetic
    (~112 c/unit warm) instead of miss latency.
  * `gemmini_loop_ws` is RULED OUT by arithmetic, do not spend an iteration on
    it.  It would free the host completely (2 commands instead of 4,096), which
    is exactly the right idea -- but its hardware FSM issues DIM-row (16-row)
    mvins, and the cold 16-row rate measured on this very machine is 5.67 B/c
    = 184,837 cycles for 1 MiB (round 7 companion probe).  Trading a 161,687
    stream for a 184,837 one to recover at most 17,346 lands at ~185k, worse
    than today's 179,033.  The lmhead lesson ("hand-issuing 33,000 commands is
    a disaster, loop_ws needs 2") does not transfer: here the hand-issued
    stream is 14% FASTER than loop_ws because 4-row mvins beat 16-row ones.

# VERDICT: effectively SOLVED (9.7% of theoretical room left, no live mechanism)

Round 10 is worth AT MOST ONE iteration, on the load-distance idea above
(expected ~173,000; anything at or above 179k means the exposure is
irreducible on this core and the kernel should be declared finished).  Do not
open a fifth scheduling iteration on the weight stream.

# WHAT THIS KERNEL IS FOR NOW: the overlap constant

The number this entry contributes to the paper and to the end-to-end
projection is the exposure ratio

    17,346 exposed / 28,372 serialised Saturn = 0.611

wired into `loop/llama_project.py --overlap` (off by default).  Applied to the
whole model it is worth only ~2% end to end, not 13.2%, because the fused
kernel's Saturn share (28,372 / 206,304 = 13.8%) is far larger than the Saturn
share of a full decode token in the projection.  The open question for the
paper is whether 0.611 extrapolates at all: the LM head is ~20.5% of decode
cycles, streams 4 MiB, and has NO attention work to hide underneath it.  A
lm_head-shaped fused kernel is the experiment that would settle that, and it
is a better use of simulator time than another 200 cycles here."""


LLAMA_LAYER_FUSED_N1 = Kernel(
    name="llama-layer-fused-n1",
    target="gemmini",
    bench_dir="bareMetalC",
    kernel_rel="include/llama_layer_fused_n1_kernel.h",
    main_src="bareMetalC/llama_layer_fused_n1.c",
    elf_name="llama_layer_fused_n1.riscv",
    # MEASURED 2026-09-13 (run 20260912-181641-b631): 3,434 s of wall for the
    # whole baseline-only run — ELF build + collateral + boot + a 1 MiB + 64 KiB
    # scalar fill + the 2 MiB DMA L2 flush + the 206k-cycle timed region + ~82k
    # scalar MACs of golden model.  This is the most expensive entry on the
    # board to simulate.  9000 s is ~2.6x that.
    sim_timeout_seconds=9000,
    sim_timeout_cycles=100_000_000,
    cycle_re=r"Cycles taken:\s*(\d+)",
    instret_re=None,
    metric_agg="first",
    # MEASURED 2026-09-13, run 20260912-181641-b631, self-check PASSED.
    # See out/loop/round7-prep.md.  1.47x the 140,680-cycle roofline.
    reference_cycles=206_304,
    # max(memory 140,680 ; compute 11,712) — full derivation in the notes.
    roofline_cycles=140_680,
    objective="Make one Llama-3.2-1B decoder-layer slice at decode N=1 — a "
              "1x2048x512 int8 GEMV on Gemmini (1 MiB of weights) TOGETHER "
              "with one attention head's QK^T, softmax and probs@V on Saturn "
              "— run in fewer **cycles**, by OVERLAPPING the Saturn work with "
              "the Gemmini weight DMA instead of running them back to back. "
              "Do not change what any of the four pieces computes, and do not "
              "rewrite their inner loops: they are already the best known.",
    check_desc="`bareMetalC/llama_layer_fused_n1.c` and the sealed body "
               "`include/llama_layer_fused_body.h` (you own only "
               "`include/llama_layer_fused_n1_kernel.h`) generate every input "
               "from a fixed seed, stream 2 MiB through the L2 with Gemmini's "
               "DMA so the machine is COLD, time your `llama_layer_fused` with "
               "`read_cycles()` and print `Cycles taken: N` (that N is the "
               "objective), then run FOUR checks. (1) GEMV: 8 sampled outputs "
               "recomputed with a scalar int32 dot product in one fused pass "
               "over k, compared EXACTLY. (2) scores: all 512, scalar int32 "
               "dot, 1e-4 relative — exact on the integer part. (3) softmax: "
               "all 512 against a local double-precision reference exp, 5e-3 "
               "relative, PLUS `sum(probs)` within 1e-3 of 1.0 (that sum is "
               "the tight gate). (4) probs@V: all 64, `pvacc` compared as "
               "EXACT int32 and `pvout` to 1e-4 relative. Any mismatch prints "
               "`MISMATCH` and `exit(1)`, which scores the attempt zero. "
               "Checksums over the full `C`, `scores` and `pvout` are printed "
               "every run, so writing only the sampled GEMV columns is caught.",
    contract=_LAYER_FUSED_CONTRACT,
    notes=(_LAYER_FUSED_NOTES + _LAYER_FUSED_ROUND8 + _LAYER_FUSED_ROUND9
           + _LAYER_FUSED_ROUND10 + _GEMMINI_DMA_HINTS),
)

# ===========================================================================
# llama-lmhead-fused-n1  (Round 10) — the LM-head-shaped overlap experiment
# ===========================================================================

_LMHEAD_FUSED_CONTRACT = """\
```c
static void llama_lmhead_fused(size_t N, size_t Kdim, size_t M,
                               const elem_t *A, const elem_t *B, acc_t *C,
                               size_t H, const float *X, const float *G,
                               float *xn, elem_t *xq, float eps, float qscale,
                               size_t L, const acc_t *LG, int32_t *amax);
```

Called ONCE with `N = 1`, `Kdim = 2048`, `M = 2048`, `H = 2048`, `L = 2048`,
`eps = 1e-5f`.  `qscale` is calibrated by the harness from the data.  FOUR
pieces of work, ONE timed region.  You must produce all four results:

1. **Gemmini LM-head GEMV** — `C[j] = sum_k A[k] * B[k*M + j]` for `j` in
   `[0, 2048)`, exact int32, no bias / activation / output scaling.  `B` is
   **4 MiB** of int8: the same shape as `llama-q8-gemv-gemmini-lmhead`.
2. **final RMSNorm** — `ss = (1/H) * sum_i X[i]*X[i]`;
   `xn[i] = X[i] * rsqrt(ss + eps) * G[i]`.  RMSNorm, so NO mean subtraction.
   The reduction may be reassociated but must cover all `H` terms.
3. **int8 quantisation** — `xq[i] = clamp(round(xn[i] * qscale), -127, +127)`.
   DEPENDS on `xn`.  Round-to-nearest; the gate tolerates one LSB per element
   (ties) but only 8 LSBs in aggregate.
4. **running argmax** — fold `LG[0..L)` into the running pair
   `(amax[0] = value, amax[1] = index)`, which the caller seeds with
   `(INT32_MIN, -1)`.  Ties break to the LOWEST index; the harness guarantees
   the maximum is unique, so there is exactly one right answer.

**The ONLY data dependency among the four is `xn -> xq`.**  Nothing in the
Saturn half reads `C`; nothing in the Gemmini half reads `xn`, `xq` or `amax`.
Everything else may be reordered, interleaved, split, or issued in any order.

`A` is the CURRENT token's already-quantised hidden state (an input).  `xq` is
the NEXT one (an output).  `LG` is a SEPARATE pre-supplied int32 logit vector
standing for the PREVIOUS tile's output.  Both separations are deliberate and
both are what a software-pipelined decode actually looks like: if the argmax
read `C`, or the GEMV read `xq`, a true dependency would serialise the halves
and this benchmark would measure nothing.  Do not "simplify" by wiring them
together — the self-check compares against references built from `X`, `G` and
`LG`, so you would simply fail.

`C`, `xn` and `xq` are zeroed by the caller; `amax` is seeded as above.

All arrays are at fixed DRAM addresses outside the ELF, 1 MiB- or 64 KiB-
aligned (see the sealed body for the map).  The harness streams 2 MiB of
unrelated data through the L2 with Gemmini's DMA immediately before starting
the clock, so **everything you read starts cold**.

You may NOT edit `include/gemmini.h`, `include/gemmini_params.h` (generated by
the hardware elaboration and overwritten anyway), the sealed harness body
`include/llama_lmhead_fused_body.h`, or the benchmark wrapper
`bareMetalC/llama_lmhead_fused_n1.c`.  Read the `DIM` / `BANK_NUM` /
`BANK_ROWS` / `ACC_ROWS` / `MAX_BYTES` / `MAX_BLOCK_LEN` macros from
`gemmini_params.h`; never hard-code them.

`mstatus.VS` is already enabled by the sealed body — `<riscv_vector.h>`
intrinsics and hand-written vector asm both work out of the box."""


_LMHEAD_FUSED_NOTES = """\
# WHAT THIS KERNEL IS FOR — the number matters more than the best

This entry exists to answer ONE question, and a measured ZERO is a publishable
answer:

    **Does `llama-layer-fused-n1`'s exposure ratio of 0.611 extrapolate to the
    LM head?**

`llama-layer-fused-n1` took a 1 MiB Gemmini GEMV plus one attention head's
worth of Saturn work from 206,304 sequential cycles to 179,033 overlapped
(-13.2%) across four rounds and twelve iterations.  Its final decomposition is

    pure weight stream, Saturn serialised out   161,687
    Saturn cost serialised                       28,372
    Saturn cost still EXPOSED after overlap       17,346
    exposure ratio   17,346 / 28,372           =  0.611

`loop/llama_project.py --overlap` currently extrapolates that 0.611 to the
whole decode step, and it is worth only 0.29-1.43% end to end because the
fused kernel's Saturn share (28,372 / 206,304 = 13.8%) is 6-20x the Saturn
share of a real decode token (0.7-2.2%).  That much is arithmetic.

The VALIDITY THREAT is different and this entry is the experiment that closes
it.  0.611 was measured on a kernel whose companion work is ATTENTION: 64 KiB
of KV cache, 28k cycles, arithmetic-shaped, sitting in the shadow of a 1 MiB
stream.  The single largest Gemmini phase of a decode step is the LM head:
~20.5% of decode cycles, 2048 x 128256 int8 = ~250 MiB streamed for ONE token,
and **no attention anywhere near it**.  The only concurrent work is the final
RMSNorm, the int8 quantisation of the next hidden state, and the running
argmax over the logits.  There is every reason to expect 0.611 to fail there,
and no measurement either way.

So: the deliverable is the EXPOSURE RATIO, not the cycle count.  A round that
lands at the baseline but reports a clean, mechanism-backed
"exposure = 1.0, and here is why" is a SUCCESS.  A round that shaves 3,000
cycles and cannot say where they came from is not.

# SHAPE AND SCALE — read before interpreting any number

One timed region, cold, containing:

| piece | shape | best known, WARM, in its own harness |
|---|---|---|
| Gemmini LM-head GEMV | `1 x 2048 x 2048` int8, **4 MiB** of B | 634,507 (`llama-q8-gemv-gemmini-lmhead`) |
| final RMSNorm | 2048 fp32 | 5,454 (`llama-rmsnorm`, never beaten) |
| int8 quantise | 2048 fp32 -> int8 | no sibling entry |
| running argmax | 2048 int32 | no sibling entry |

The real head is 2048 x 128256, so this is ONE 2048-column tile = 1/62.6 of
it.  The three Saturn pieces do NOT scale the same way, and the harness is
deliberately generous to the Saturn side:

  * the **running argmax is tile-matched and faithful**.  A decode step
    computes the head tile by tile; the natural software pipeline folds tile
    t-1's 2048 logits into the running max while tile t's weights stream.
    `L = 2048` per 4 MiB of B is exactly the true ratio.
  * the **RMSNorm and the quantiser happen once per TOKEN, not once per
    tile**.  Charging a whole token's worth of them to a single tile
    over-attributes the Saturn side by **62.6x**.

That over-attribution is on purpose: it makes whatever exposure you measure an
**UPPER BOUND**.  If the exposure ratio measured here is near 1.0 even with
62.6x too much Saturn work to hide, then at true scale it is 1.0 and the
question is closed.  Say that explicitly when you report.

A full-vocabulary argmax (128,256 int32 = 513 KiB) was considered and
REJECTED for this harness: it would add 12.5% to the byte count and turn the
experiment into a bandwidth-contention measurement instead of an overlap
measurement.  Here, as in `llama-layer-fused-n1`, the Saturn side is
compute-shaped, so the two exposure ratios are directly comparable.  Do not
propose adding vocabulary-scale work; it is a different experiment.

# ROOFLINE

Two resources, genuinely independent, so the fused floor is a `max`, not a sum.

    MEMORY FLOOR — everything crosses the one 8 B/cycle mbus
      B weights (every byte exactly once)      4,194,304 B
      A  1 x 2048 int8                             2,048 B
      C  out 2048 int32 (+ write-allocate)        16,384 B
      X  2048 fp32                                 8,192 B
      G  2048 fp32                                 8,192 B
      xn store 2048 fp32 (+ write-allocate)       16,384 B
      xq store 2048 int8 (+ write-allocate)        4,096 B
      LG 2048 int32                                8,192 B
      amax (one line, write-allocate)                 64 B
      -------------------------------------------------------
      total                                    4,257,856 B
      memory floor = 4,257,856 / 8                = 532,232 cycles

    COMPUTE FLOOR — two different units, so take the max of the two
      Gemmini array: 4,194,304 MAC / 256 MAC per cycle   = 16,384 c
      Saturn pipes (DLEN = 128 -> 4 fp32 or 4 int32 lanes/cycle):
          rmsnorm   reduce pass 512 + scale pass 1,024   =  1,536 c
                    (this is exactly `llama-rmsnorm`'s own roofline)
          quantise  6 element-ops (fmul, fcvt, max, min,
                    2 x nsrl) x 2048 / 4                 =  3,072 c
          argmax    2 passes x (load 512 + op 512)       =  2,048 c
          Saturn total                                   =  6,656 c
      compute floor = max(16,384, 6,656)                 = 16,384 cycles

    roofline = max(532,232, 16,384) = **532,232 cycles — MEMORY binds by 32x.**

Two consequences, and they are the whole story of this entry:

  a) **In principle the Saturn work is free.**  6,656 cycles of arithmetic
     against 532,232 cycles of unavoidable DMA: 80x more shadow than work.
  b) **The weight stream is 98.5% of the bytes.**  The Saturn side's 63,552 B
     is 1.5% of the traffic, i.e. ~7,900 cycles of mbus time.  Unlike
     `llama-layer-fused-n1` (6.5% of bytes), bandwidth displacement is a
     second-order term here.  If the Saturn cost stays exposed, it will NOT be
     because of bytes.

532,232 is a HARD lower bound and it is NOT reachable.  The measured rate for
this exact 4 MiB stream is 6.61 B/cycle WARM (634,507, that entry's Round-4
winner) and this harness runs COLD, so budget against the baseline, not the
roofline.  **Quote the baseline when you predict.**

# THE COLD-STATE FLUSH

The sealed body streams 2 MiB of unrelated DRAM through the 512 KiB L2 with
Gemmini's own DMA immediately before `read_cycles()`.  This matters less here
than it did for `llama-layer-fused-n1` — at 4 MiB the warm tail B's fill can
leave behind is at most 512 KiB, i.e. 12% of B rather than 25% — but it is not
nothing, and in particular:

  * the incumbent GEMV consumes K blocks in **DESCENDING** order specifically
    to harvest that tail.  Cold, that credit is gone.  Flipping to ascending
    is a one-line control experiment that produces a number nobody has.
    Do not assume it still helps; do not assume it still hurts.

# THE MECHANISM — why overlap is possible at all

Gemmini is a RoCC accelerator, not a library call.  `gemmini_extended_mvin2`,
`gemmini_extended_preload` and `gemmini_extended_compute_preloaded` are
*instructions the host issues and retires*; the DMA and the array then run on
their own.  Saturn is a **separate functional unit on the same host core**: its
instructions issue from Shuttle's pipeline, execute in Saturn's own sequencers,
and touch neither the RoCC port nor the Gemmini command queues.
`gemmini_fence()` is `asm volatile("fence")`, which stalls until the
accelerator reports not-busy — **that single instruction is what serialises
the baseline**, and the only fence required for correctness is one after the
final `mvout`, before anything reads `C`.

The incumbent GEMV hand-issues **16,384 mvin2 commands** (4 MiB at 4 rows x
64 B) plus 16,384 preload/compute pairs.  The host is therefore never idle for
long stretches the way it was under `loop_ws`; the slack is spread thin, in
~39-cycle gaps between command issues.  That is the shape of the opportunity
here, and it is DIFFERENT from `llama-layer-fused-n1`'s early rounds.

# EVERY VERIFIED HARDWARE FACT FROM ROUNDS 7-9 — DO NOT RE-DERIVE THESE

  * **mbus is 8 B/cycle.**  Gemmini's DMA port and the sbus are 128-bit
    (16 B/cycle), but DRAM sits behind the mbus, which rocket-chip's
    `BaseSubsystemConfig` fixes at `beatBytes = 8` and which nothing in this
    SoC's config chain overrides.  Every memory floor on the board uses 8.
  * **DRAM is `mm_magic_t`** — a zero-latency functional model.  There is no
    DRAM page or refresh behaviour to exploit; what you are measuring is the
    L2/mbus request path, not a memory controller.
  * **L2 is a 512 KiB SiFive InclusiveCache with MSHR = 12.**  That, not the
    link width, is what caps in-flight misses.
  * **Gemmini's DMA issues ONE 64 B TileLink request per mvin ROW and does
    not merge them.**  A 4-row mvin2 is 4 requests.
  * **`LoadController` has `nCmds = 2`.**  Only two mvin commands are in
    flight at the DMA level at any time.
  * **4 rows per mvin beats 8 and 16.**  Measured twice (lmhead's own curve:
    634k at 4 rows vs 690k/695k at 8/16; `llama-layer-fused-n1` iter5 vs
    iter7).  The reason is scratchpad **BANK CONFLICTS**, not latency —
    layer-fused iter8 confirmed it by rotating slots across banks 1-3 and
    recovering most of the loss.  **DO NOT re-test 8 or 16 rows.**
  * **`gemmini_loop_ws` is ruled out by arithmetic.**  Its hardware FSM issues
    DIM-row (16-row) mvins, and the cold 16-row rate on this machine is
    5.67 B/cycle.  It would free the host completely, which is the right idea,
    but at a 15% worse stream.  Hand-issuing wins here; do not spend an
    iteration on it.
  * **Warm vs cold, same 1 MiB kernel** (round-7 companion probe
    20260912-191830-e713): warm 132,435 = 7.91 B/cycle, cold 184,837 =
    5.67 B/cycle.  Any "cold DRAM sustains 7.0 B/cycle" line elsewhere on the
    board is a fit to warm-contaminated data.
  * **Issuing an mvin costs the host ~5 cycles** (layer-fused iter3:
    `mvin_stall_sampled` 2,558 over 512 samples, *including* two `rdcycle`s).
    The ld reservation station is never full enough to stall issue.
  * **Host scalar work is nearly free.**  Layer-fused iter4 cut the inner loop
    from 19 scalar + 3 RoCC ops to 10 + 3 (verified by disassembly), predicted
    -9.5k and delivered -552, i.e. 0.27 c/mvin.
  * **`ExecuteController` reads beat DMA writes on the single-ported
    scratchpad** (`Scratchpad.scala:522`).  Vector work is not affected, but
    execute scheduling is.
  * **In-flight scan on the 1 MiB stream** (6/8/10/12/16 commands ->
    6.31/6.63/6.06/5.83/5.78 B/cycle): the memory system tops out near
    6.6 B/cycle.  This 4 MiB stream already measures 6.61 B/cycle warm, i.e.
    it is AT that ceiling.  There is no bandwidth left to find.

# FIVE HYPOTHESES ALREADY RETIRED ON `llama-layer-fused-n1` — DO NOT RE-TEST

Different kernel, same core, same memory system, same mechanism.  Each of
these cost a full iteration there; none of them is worth one here.

  1. **Prefetching the Saturn operands through Gemmini's DMA (DIED, +5,789).**
     Warming the Saturn operands with `mvin3` ahead of use put those bytes in
     the weight stream's request pool, where they paid full link time.
     Saturn's own cold loads instead ride the bandwidth the stream leaves
     idle — nearly free.  **Bytes are not the cost.**
  2. **Finer interleave grain (DIED, +522).**  Halving the Saturn unit size
     and issuing twice as often, with the same total vector instruction count,
     changed nothing.  The exposure scales with TOTAL vector work, not with
     how it is chopped up.
  3. **Cutting host scalar operands (mostly DIED, -552 against -9,500
     predicted).**  See "host scalar work is nearly free" above.
  4. **8- and 16-row mvins (DIED twice).**  Bank conflicts.
  5. **Saturn taking a slice of the GEMV itself (DIED, five separate
     attempts).**  Both requesters cross the same saturated mbus and read the
     same weight bytes, so a Saturn slice cannot add bandwidth, only take
     Gemmini's share.  Do not revive host/Saturn co-compute on the GEMV.

The ONE surviving model from Round 9 is **distance, not grain or bytes**: the
in-order Shuttle core is HELD for the whole of a vector unit (~375 cycles
there, mostly cold-miss latency the Saturn VLSU does not decouple far enough
to hide), and during the part of that window that exceeds the DMA work already
queued, the stream idles.  The untested lever is issuing unit n+1's vector
LOADS during unit n, so the held window covers arithmetic instead of miss
latency.  That model has never been tested on ANY kernel, and the Saturn
operands here are only 24 KiB (X, G, LG) against layer-fused's 68 KiB.

# MEASURED BASELINE — 648,292 cycles (2026-09-17, run 20260917-145930-bf8a)

| what | cycles |
|---|---|
| **BASELINE, strictly sequential, cold** | **648,292** |
| roofline (memory floor, 8 B/cycle) | 532,232  (1.218x) |
| the SAME 4 MiB GEMV alone, WARM, its own harness | 634,507  (6.61 B/c) |
| `llama-rmsnorm` alone, warm, its own harness | 5,454 |
| quantise + argmax alone | never measured |

Effective rate: 4,194,304 B / 648,292 = **6.47 B/cycle**, against 6.61 B/cycle
for the same stream warm.  Read that table carefully, because it caps this
round before it starts:

    648,292 (fused, cold, sequential)
  - 634,507 (weight stream alone, WARM)
  = 13,785  <- EVERYTHING else: the cold-L2 penalty on the stream itself,
               PLUS all three Saturn phases, PLUS the fence.

**So perfect overlap is worth AT MOST 13,785 cycles (2.1%), and strictly less
than that**, because the cold-L2 penalty is part of the 13,785 and no schedule
can recover it — it is bytes that must cross the mbus.  Compare
`llama-layer-fused-n1`, where the Saturn half alone was 28,372 cycles on a
206,304-cycle baseline (13.8%).  Here the whole non-stream remainder is 2.1%.

That asymmetry is not a disappointment, it is **the finding**, and it is
visible before a single optimisation iteration: at LM-head scale there is
almost nothing to hide, because the companion work of a decode step does not
grow with the weight tile.  Round 10's job is to measure the split precisely
and state the mechanism, not to chase 13,785 cycles.

Note also how SMALL the cold penalty is here: 648,292 - 634,507 = 13,785 total,
whereas the 1 MiB GEMV paid 52,402 cycles (132,435 warm -> 184,837 cold) for
the same flush.  That is exactly what the arithmetic predicts — the warm tail
a 512 KiB L2 can hold is at most 12% of 4 MiB but was 25-50% of 1 MiB — and it
means DESCENDING vs ASCENDING K order is worth much less here than the 17,104
cycles it was worth in the warm 1 MiB harness.

# HOW TO SPEND AN ITERATION ON THIS KERNEL

Before editing anything, write exactly these three lines at the top of the
file, with specific numbers:

    HYPOTHESIS: <the mechanism costing cycles, with an arithmetic estimate of
                 how many, derived from the tables above — not "overhead",
                 not "stalls">
    CHANGE:     <the one structural thing you are changing>
    EXPECTED:   <a single predicted cycle count>

After the measurement, state in one line whether the hypothesis SURVIVED or
DIED and what that retires.  ONE mechanism per iteration.  A confident
prediction that misses by 20k is worth more than a 500-cycle tweak, because it
closes a branch.

**`printf` probes work and their output comes back.**  The simulator log is
saved every iteration (`out/loop/<run_id>/simlog_NN.txt`) and
`llm.format_feedback` appends a "=== benchmark stdout (tail) ===" section to
PASSING-run feedback, capped at 4 KB.  Prefix every probe line with `PROBE `
and print one number per line, e.g. `printf("PROBE gemv_done=%lu\\n", c);`.
`read_cycles()` / `rdcycle` inside the timed region costs a few cycles and is
absolutely worth it.

**ITERATION 1 MUST ESTABLISH THE DENOMINATOR AND NOTHING ELSE.**  The
exposure ratio needs two numbers this board does not have:

    PROBE stream=NNNNNN     the 4 MiB weight stream ALONE, cold, with the
                            Saturn phases serialised after it
    PROBE rms=NNNNN         cold cost of each Saturn phase, serialised
    PROBE quant=NNNNN
    PROBE argmax=NNNNN

Print them in exactly that format — `llama-layer-fused-n1` iter6 used
`PROBE stream=161687 qk=16569 softmax=1613 pv=10190` and the rounds must stay
comparable.  You get all four by timestamping the phase boundaries inside the
timed region of the UNCHANGED baseline, so iteration 1 costs nothing in score
and turns one number into five.  Then

    serialised Saturn = rms + quant + argmax
    exposure          = best_total - stream
    exposure ratio    = exposure / serialised Saturn

and THAT is what goes in the paper next to 0.611.

Iterations 2-4, in priority order:

  1. **Move the fence.**  Drop `gemmini_fence()` from the end of the GEMV,
     interleave the Saturn phases into the command stream (a `lhk_sat_unit`
     call every N mvin2 issues, exactly the structure
     `llama-layer-fused-n1` converged on), fence once at the very end before
     the checks.  This is the whole kernel.
  2. **Load distance.**  Issue the next Saturn unit's vector loads during the
     current one, so the window the in-order core is held for covers
     arithmetic rather than cold-miss latency.  This is the one surviving
     model from Round 9 and it has never been tested anywhere.
  3. **Ascending vs descending K**, cold.  One line, and it disentangles the
     warm-tail credit from the schedule.

# WHAT A RESULT LOOKS LIKE

    exposure ratio ~0.6  -> 0.611 extrapolates; `--overlap` is defensible
                            model-wide and the paper can say so.
    exposure ratio ~1.0  -> 0.611 is a property of attention-shaped companion
                            work and does NOT extrapolate.  `--overlap` must
                            be restricted to the attention phases, and the
                            LM head — 20.5% of decode — gets no credit.
                            **This is the outcome the arithmetic predicts**,
                            and proving it cleanly is worth more than 3,000
                            cycles.

Either way, report the mechanism.  "It did not overlap" is not a result;
"it did not overlap because the 4 MiB stream leaves only ~39-cycle host gaps
and one Saturn unit holds the in-order core for ~N cycles, of which M exceed
the queued DMA work" is."""


LLAMA_LMHEAD_FUSED_N1 = Kernel(
    name="llama-lmhead-fused-n1",
    target="gemmini",
    bench_dir="bareMetalC",
    kernel_rel="include/llama_lmhead_fused_n1_kernel.h",
    main_src="bareMetalC/llama_lmhead_fused_n1.c",
    elf_name="llama_lmhead_fused_n1.riscv",
    # MEASURED 2026-09-17 (run 20260917-145930-bf8a): 3,102 s of wall for the
    # whole baseline-only run — Ray connect + Gemmini collateral regeneration
    # + ELF build + boot + a 4 MiB + 2 MiB + 24 KiB data fill + the 2 MiB DMA
    # L2 flush + the 648k-cycle timed region + four golden models and four
    # checks. 9000 s is ~2.9x that, the same margin `llama-layer-fused-n1`
    # carries. Cheaper to simulate than that entry despite 4x the weights,
    # because its Saturn golden models are ~82k scalar MACs and these are ~6k.
    sim_timeout_seconds=9000,
    sim_timeout_cycles=100_000_000,
    cycle_re=r"Cycles taken:\s*(\d+)",
    instret_re=None,
    metric_agg="first",
    # MEASURED 2026-09-17, run 20260917-145930-bf8a, self-check PASSED.
    # See out/loop/round10-prep.md. 1.218x the 532,232-cycle roofline.
    reference_cycles=648_292,
    # max(memory 532,232 ; compute 16,384) — full derivation in the notes.
    roofline_cycles=532_232,
    objective="Make one Llama-3.2-1B LM-head TILE at decode N=1 — a "
              "1x2048x2048 int8 GEMV on Gemmini (4 MiB of weights) TOGETHER "
              "with the Saturn companion work of that decode step (the final "
              "fp32 RMSNorm over 2048, its int8 quantisation, and a running "
              "argmax over 2048 int32 logits) — run in fewer **cycles**, by "
              "OVERLAPPING the Saturn work with the Gemmini weight DMA "
              "instead of running them back to back. Do not change what any "
              "of the four pieces computes and do not rewrite their inner "
              "loops: the point of this entry is the exposure ratio, and "
              "changing the work makes it incomparable.",
    check_desc="`bareMetalC/llama_lmhead_fused_n1.c` and the sealed body "
               "`include/llama_lmhead_fused_body.h` (you own only "
               "`include/llama_lmhead_fused_n1_kernel.h`) generate every "
               "input from a fixed seed, stream 2 MiB through the L2 with "
               "Gemmini's DMA so the machine is COLD, time your "
               "`llama_lmhead_fused` with `read_cycles()` and print "
               "`Cycles taken: N` (that N is the objective), then run FOUR "
               "checks. (1) GEMV: 8 sampled outputs recomputed with a scalar "
               "int32 dot product in one fused pass over k, compared "
               "EXACTLY. (2) RMSNorm: all 2048 against a double-precision "
               "scalar reference, 1e-4 relative / 1e-5 absolute — the same "
               "gate `llama-rmsnorm` uses. (3) int8 quantisation: all 2048, "
               "at most 1 LSB per element (round-half ties) AND at most 8 "
               "LSBs summed over the whole vector, so a systematic off-by-one "
               "or a wrong scale fails even though every element passes. "
               "(4) running argmax: value AND index compared EXACTLY — the "
               "harness forces the maximum to be unique, so there is one "
               "right answer and no tolerance. Any mismatch prints "
               "`MISMATCH` and `exit(1)`, which scores the attempt zero. "
               "Checksums over the full `C`, `xn` and `xq` are printed every "
               "run, so writing only the sampled GEMV columns or only half "
               "of `xn` is caught.",
    contract=_LMHEAD_FUSED_CONTRACT,
    notes=(_LMHEAD_FUSED_NOTES + "\n\n" + _GEMMINI_DMA_HINTS),
)


KERNELS: dict[str, Kernel] = {
    k.name: k for k in (
        VEC_SGEMV, VEC_SOFTMAX, VEC_DOTPROD, LLAMA_Q8_GEMV,
        GEMMINI_TILED_MATMUL_WS, GEMMINI_TILED_MATMUL_WS_FAST,
        LLAMA_Q8_GEMM,
        LLAMA_Q8_GEMV_GEMMINI_N1, LLAMA_Q8_GEMV_GEMMINI_N16,
        LLAMA_Q8_GEMV_GEMMINI_LMHEAD,
        LLAMA_RMSNORM, LLAMA_ROPE, LLAMA_ATTN_SCORES, LLAMA_SOFTMAX,
        LLAMA_ATTN_PV, LLAMA_SILU_MUL, LLAMA_ADD,
        LLAMA_ATTN_SCORES_Q8KV, LLAMA_ATTN_PV_Q8KV,
        LLAMA_ATTN_SCORES_INT8, LLAMA_ATTN_PV_INT8,
        LLAMA_LAYER_FUSED_N1,
        LLAMA_LMHEAD_FUSED_N1,
    )
}


def get_kernel(name: str) -> Kernel:
    try:
        return KERNELS[name]
    except KeyError:
        raise SystemExit(
            f"unknown kernel {name!r}. Known: {', '.join(sorted(KERNELS))}"
        ) from None


def kernel_names() -> list[str]:
    return sorted(KERNELS)
