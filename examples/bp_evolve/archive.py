"""The MAP-Elites archive: the best design in each region, not the best design.

A leaderboard on VFS would work, and would be a mistake.  VFS rewards accuracy,
energy and latency together, and the cheapest way to buy accuracy in a TAGE is
to make the tables larger.  CBP-NG imposes no storage budget, so nothing stops
that; a scalar search converges on one enormous TAGE and reports it as a
result.  What the proposal actually needs is a *front* -- designs that trade
differently -- because the question is which parts of that front survive being
re-scored on a different core.

So the archive bins on what the simulator reports rather than on anything the
agent declares: energy per instruction, and the two prediction latencies.
Those three are the whole of CBP-NG's cost model.  A design that wins its bin
is kept even if a design in another bin scores higher, which is what keeps a
slow-and-accurate corner of the space alive long enough to be tested at depth
sixteen, where it turns out to win.

Occupancy of a bin is decided by VFS at the *shipped* depth.  Using a
depth-swept average instead would beg the question the loop exists to ask.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field, asdict


@dataclass
class Elite:
    """One archived design: what it is, what it scored, where it came from."""
    variant_id: str
    struct_name: str
    source: str
    parent_id: str | None
    generation: int
    template_args: str
    vfs: float                     # at the shipped depth -- the occupancy key
    epi: float
    p1_latency: int
    p2_latency: int
    mpki: float
    ipc: float
    cpi: float
    n_traces: int
    depth_scores: dict[int, float] = field(default_factory=dict)
    notes: str = ""

    def summary(self) -> dict:
        """The fields a prompt should see -- everything but the source."""
        d = asdict(self)
        d.pop("source")
        return d


class Archive:
    """Name-keyed MAP-Elites grid over (EPI bin, P1 latency, P2 latency).

    Not a general implementation: the descriptor is fixed, because the
    descriptor is an argument about what makes two branch predictors different,
    and burying it behind a callback would hide that argument.
    """

    def __init__(self, epi_edges, p1_bins, p2_bins):
        self.epi_edges = list(epi_edges)
        self.p1_bins = tuple(p1_bins)
        self.p2_bins = tuple(p2_bins)
        self.cells: dict[tuple[int, int, int], Elite] = {}
        self.rejected = 0          # variants that scored below their cell's elite

    # -- descriptor ---------------------------------------------------------

    def epi_bin(self, epi: float) -> int:
        """Index of the energy band ``epi`` falls in.

        ``bisect_right`` puts an EPI exactly on an edge in the *upper* band,
        so the edges read as inclusive lower bounds.
        """
        return bisect.bisect_right(self.epi_edges, epi)

    def _latency_bin(self, latency: int, bins) -> int:
        """Clamp to the last bin rather than growing the grid.

        A predictor with a 9-cycle second level is not usefully distinguished
        from one with 11; both are "too slow to override in time", and giving
        each its own cell would let a design survive by being uniquely bad.
        """
        return min(latency, bins[-1])

    def descriptor(self, elite: Elite) -> tuple[int, int, int]:
        return (self.epi_bin(elite.epi),
                self._latency_bin(elite.p1_latency, self.p1_bins),
                self._latency_bin(elite.p2_latency, self.p2_bins))

    # -- insertion ----------------------------------------------------------

    def add(self, elite: Elite) -> tuple[bool, tuple[int, int, int]]:
        """Insert if it beats its cell's occupant. Returns (accepted, cell).

        Re-adding a design that already holds its cell is a no-op that reports
        success, not a rejection.  The mapper runs again after Tier 1 fills in
        the ChampSim numbers, and without this the second pass would compare a
        design against itself, lose (``>`` is strict), increment ``rejected``,
        and relabel an archived elite as ``no_improvement``.
        """
        cell = self.descriptor(elite)
        current = self.cells.get(cell)
        if current is not None and current.variant_id == elite.variant_id:
            self.cells[cell] = elite      # refresh with the fuller record
            return True, cell
        if current is None or elite.vfs > current.vfs:
            self.cells[cell] = elite
            return True, cell
        self.rejected += 1
        return False, cell

    # -- reads --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.cells)

    @property
    def elites(self) -> list[Elite]:
        """Every occupant, best first."""
        return sorted(self.cells.values(), key=lambda e: -e.vfs)

    def best(self) -> Elite | None:
        e = self.elites
        return e[0] if e else None

    def pareto_front(self) -> list[Elite]:
        """Designs no other design beats on VFS, energy and P2 latency together.

        The front, not the leaderboard, is what gets promoted to Tier 1: a
        design that is second-best on VFS but half the energy is exactly the
        kind the fixed-depth score is suspected of misranking, and promoting
        only the VFS winner would never test that.
        """
        elites = self.elites
        front = []
        for cand in elites:
            dominated = any(
                other is not cand
                and other.vfs >= cand.vfs
                and other.epi <= cand.epi
                and other.p2_latency <= cand.p2_latency
                and (other.vfs > cand.vfs
                     or other.epi < cand.epi
                     or other.p2_latency < cand.p2_latency)
                for other in elites
            )
            if not dominated:
                front.append(cand)
        return front

    def select_parent(self, rng) -> Elite | None:
        """Draw a parent uniformly over *occupied cells*.

        Uniform over cells, not over VFS: weighting by score would pull every
        parent out of the same corner and undo the binning.  Curiosity about
        the sparse regions is the entire reason the grid exists.
        """
        if not self.cells:
            return None
        return self.cells[rng.choice(sorted(self.cells.keys()))]

    def coverage(self) -> float:
        """Occupied fraction of the grid -- the diversity number to watch."""
        total = (len(self.epi_edges) + 1) * len(self.p1_bins) * len(self.p2_bins)
        return len(self.cells) / total if total else 0.0

    # -- persistence --------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps({
            "epi_edges": self.epi_edges,
            "p1_bins": list(self.p1_bins),
            "p2_bins": list(self.p2_bins),
            "rejected": self.rejected,
            "cells": [{"cell": list(k), "elite": asdict(v)}
                      for k, v in sorted(self.cells.items())],
        }, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Archive":
        obj = json.loads(text)
        arch = cls(obj["epi_edges"], obj["p1_bins"], obj["p2_bins"])
        arch.rejected = obj.get("rejected", 0)
        for entry in obj["cells"]:
            e = dict(entry["elite"])
            # JSON object keys are strings; depth_scores is keyed by int depth.
            e["depth_scores"] = {int(k): v for k, v in (e.get("depth_scores") or {}).items()}
            arch.cells[tuple(entry["cell"])] = Elite(**e)
        return arch
