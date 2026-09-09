"""CBP-NG's score, with the pipeline depth made a parameter.

``cbp-ng/vfs.py`` and ``cbp-ng/predictor_metrics.py`` between them turn a
directory of per-trace ``.out`` lines into one Voltage-Frequency-Scaled Speedup
number.  They are reimplemented here for two reasons: the loop needs to score
in-process rather than by shelling out per variant, and -- the point of the
whole exercise -- ``predictor_metrics.py`` hard-codes

    p2_to_exec_stages = 9

at module scope.  That nine is the assumption under test.  Here it is an
argument, so the same measured counters can be re-scored at any depth without
re-running a single simulation.

Nothing else is changed: :func:`vfs` is a transcription of the published
formula, and :func:`aggregate` reproduces ``predictor_metrics.py``'s averaging
exactly (harmonic mean over IPC, arithmetic over CPI/EPI, and latencies taken
as the max over traces, rounded up to whole cycles).

The re-scoring this module exists for is checked against the organisers'
published results by ``python -m bp_evolve.vfs --selftest``, which reproduces
the depth-sweep table in the proposal from
``cbp-ng/docs/example_and_reference_predictor_results.csv``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Reference core, verbatim from cbp-ng/vfs.py.
IPC_REF = 8
CPI_REF = 0.0315
EPI_REF_CBP = 1000
ALPHA = 1.625
BETA = 4 * ALPHA / (ALPHA - 1) ** 2
GAMMA = 2 / (ALPHA - 1)
CBP_ENERGY_RATIO = 0.05


def vfs(ipc: float, cpi: float, epi: float) -> float:
    """Voltage-Frequency-Scaled Speedup for one (IPC, CPI, EPI) triple.

    ``ipc`` is prediction throughput, ``cpi`` the cycles lost per correct-path
    instruction to mispredictions, ``epi`` the predictor's dynamic energy per
    instruction.  Higher is better; the reference predictor scores ~0.97.
    """
    wpi_ref = IPC_REF * CPI_REF
    wpi = ipc * cpi
    speedup = (ipc / IPC_REF) * (1 + wpi_ref) / (1 + wpi)
    lam = 1 / (1 + wpi_ref / 2) - CBP_ENERGY_RATIO
    norm_epi = (
        (epi / EPI_REF_CBP) * CBP_ENERGY_RATIO + lam * speedup ** GAMMA
    ) * (1 + wpi / 2)
    return speedup * ALPHA * (
        1 - 2 / (1 + math.sqrt(1 + BETA / (speedup * norm_epi)))
    )


@dataclass
class TraceCounters:
    """One line of a CBP-NG ``.out`` file.

    The field order matches the CSV column order emitted by ``cbp``, so
    :meth:`from_out_line` is a positional unpack and stays correct if a field is
    renamed here.
    """
    name: str
    instructions: float
    branches: float
    conditional_branches: float
    predictions: float
    extra_cycles: float
    divergences: float
    divergences_at_end: float
    mispredictions: float
    p1_latency: float
    p2_latency: float
    energy_per_instruction: float

    @classmethod
    def from_out_line(cls, line: str) -> "TraceCounters":
        parts = line.strip().split(",")
        if len(parts) != 12:
            raise ValueError(
                f"expected 12 comma-separated fields in a CBP-NG .out line, "
                f"got {len(parts)}: {line[:120]!r}")
        name, *nums = parts
        return cls(name, *(float(x) for x in nums))


@dataclass
class Metrics:
    """What one predictor scored over a set of traces, at one depth."""
    ipc: float
    cpi: float
    epi: float
    mpi: float
    p1_latency: int
    p2_latency: int
    depth: int
    n_traces: int
    vfs: float = field(init=False)

    def __post_init__(self):
        self.vfs = vfs(self.ipc, self.cpi, self.epi)

    @property
    def mpki(self) -> float:
        """Mispredictions per kilo-instruction -- the cross-tier check value."""
        return self.mpi * 1000.0


def aggregate(counters: list[TraceCounters], depth: int) -> Metrics:
    """Fold per-trace counters into one Metrics at ``depth`` stages.

    Reimplements ``predictor_metrics.py``.  Two details there are easy to get
    wrong and are called out because the loop depends on them:

    * Latency is the **max over traces**, rounded up to a whole cycle, and is
      resolved before the per-trace loop -- a design's clock cannot vary by
      trace, so a per-trace latency would silently reward a design that is slow
      on one trace only.
    * When ``p2 <= p1`` the second level cannot override in time to matter, so
      P1 is ignored entirely and the block simply costs ``p2`` per prediction.
      This is the single-level case (``bimodalN``, ``gshareN``); getting it
      wrong makes every single-level design look slower than it is.
    """
    if not counters:
        raise ValueError("aggregate() needs at least one trace")

    p1 = max(math.ceil(c.p1_latency) for c in counters)
    p2 = max(math.ceil(c.p2_latency) for c in counters)

    n = 0
    inv_ipc = 0.0
    cpi_sum = 0.0
    epi_sum = 0.0
    mpi_sum = 0.0
    for c in counters:
        if c.instructions <= 0:
            raise ValueError(f"trace {c.name!r} reports {c.instructions} instructions")
        mpi = c.mispredictions / c.instructions
        if p2 <= p1:
            cycles = c.predictions * max(1, p2)
        else:
            cycles = (c.predictions * max(1, p1)
                      + c.divergences * p2
                      - c.divergences_at_end * max(1, p1))
        cycles += c.extra_cycles
        if cycles <= 0:
            raise ValueError(f"trace {c.name!r} yielded {cycles} cycles")

        n += 1
        inv_ipc += cycles / c.instructions           # harmonic mean over IPC
        cpi_sum += mpi * (depth + p2 - max(1, min(p1, p2)))
        epi_sum += c.energy_per_instruction
        mpi_sum += mpi

    return Metrics(
        ipc=n / inv_ipc,
        cpi=cpi_sum / n,
        epi=epi_sum / n,
        mpi=mpi_sum / n,
        p1_latency=p1,
        p2_latency=p2,
        depth=depth,
        n_traces=n,
    )


def sweep(counters: list[TraceCounters], depths) -> dict[int, Metrics]:
    """Score the same counters at every depth in ``depths``.

    This is the whole reason the loop is cheaper than it looks: a depth sweep
    costs no simulation at all, because the depth only enters through ``cpi``.
    """
    return {d: aggregate(counters, d) for d in depths}


def kendall_tau(a, b) -> float:
    """Rank correlation between two score vectors over the same designs.

    Plain tau-a: no tie correction, because VFS values are continuous and ties
    do not occur in practice.  Returns +1 for identical orderings, -1 for
    reversed.
    """
    a, b = list(a), list(b)
    if len(a) != len(b):
        raise ValueError("kendall_tau needs two equal-length vectors")
    n = len(a)
    if n < 2:
        raise ValueError("kendall_tau needs at least two designs")
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
    return (concordant - discordant) / (n * (n - 1) / 2)


# ---------------------------------------------------------------------------
# Self-test against the organisers' published numbers
# ---------------------------------------------------------------------------

def _selftest(csv_path: str) -> int:
    """Reproduce the proposal's depth table from the published CSV.

    Exists because every claim the loop makes about cost-model disagreement
    rests on this reimplementation matching ``predictor_metrics.py``.  The
    assertions below are the numbers quoted in the proposal.
    """
    import collections
    import csv as _csv

    rows = collections.defaultdict(list)
    with open(csv_path) as f:
        for r in _csv.DictReader(f):
            rows[r["predictor"]].append(TraceCounters(
                name=r["trace"],
                instructions=float(r["instructions"]),
                branches=float(r["branches"]),
                conditional_branches=float(r["conditional_branches"]),
                predictions=float(r["predictions"]),
                extra_cycles=float(r["extra_cycles"]),
                divergences=float(r["p1_p2_disagreements"]),
                divergences_at_end=float(r["p1_p2_disagreements_at_end"]),
                mispredictions=float(r["mispredictions"]),
                p1_latency=float(r["p1_latency"]),
                p2_latency=float(r["p2_latency"]),
                energy_per_instruction=float(r["energy_per_instruction"]),
            ))

    # never_taken and reference report no energy, so VFS cannot rank them as
    # designs; the five that do are the comparison set.
    designs = ["gshareN", "bimodalN", "tage", "gshare", "bimodal"]
    depths = [3, 9, 16, 30]

    print(f"{len(rows[designs[0]])} traces, "
          f"{sum(c.instructions for c in rows[designs[0]]) / 1e9:.2f}B instructions")
    print("\nVFS, distance to execute  " + "".join(f"{d:>9}" for d in depths))
    scores = {}
    for p in designs:
        scores[p] = {d: aggregate(rows[p], d).vfs for d in depths}
        print(f"  {p:<22}" + "".join(f"{scores[p][d]:9.3f}" for d in depths))
    base = [scores[p][9] for p in designs]
    print("  " + f"{'Kendall tau vs shipped':<22}" + "".join(
        f"{kendall_tau(base, [scores[p][d] for p in designs]):9.2f}" for d in depths))

    ok = True
    def check(what, got, want, tol=5e-4):
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {what}: {got:.4f} (expected {want})")

    print("\nproposal claims:")
    check("tage VFS @9", scores["tage"][9], 0.8732)
    check("gshareN VFS @9", scores["gshareN"][9], 0.9179)
    check("tage VFS @30", scores["tage"][30], 0.6400)
    check("tau @30", kendall_tau(base, [scores[p][30] for p in designs]), 0.40, 1e-9)

    rank9 = sorted(designs, key=lambda p: -scores[p][9]).index("tage") + 1
    rank16 = sorted(designs, key=lambda p: -scores[p][16]).index("tage") + 1
    ok &= (rank9 == 3 and rank16 == 1)
    print(f"  [{'ok' if rank9 == 3 else 'FAIL'}] TAGE rank @9: {rank9} (expected 3)")
    print(f"  [{'ok' if rank16 == 1 else 'FAIL'}] TAGE rank @16: {rank16} (expected 1)")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        path = args[1] if len(args) > 1 else \
            "/home/zxc12523/cbp-ng/docs/example_and_reference_predictor_results.csv"
        raise SystemExit(_selftest(path))
    print(__doc__)
