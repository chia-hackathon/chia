"""Stage 3 randomised programs, generated on the head.

riscv_extensions fills its stress pool with riscv-dv running under Xcelium on
a dedicated worker.  Titan cannot: **riscv-dv does not generate vector
instructions at all**, so that whole machine, and its Xcelium licence, would
produce nothing that exercises the unit under test.  The `gen` node type is
absent from cluster.yaml for this reason.

What replaces it is cheaper and more targeted.  The interesting random
variable here is not the instruction stream -- there are eight, after round
three --
it is the *tile geometry*, and that space is small enough to enumerate and
large enough to be worth sweeping: every legal (SEW, LAMBDA, LMUL, VL), each
with fresh random data, plus the register allocations and leading dimensions
they imply.  Generation is pure Python, so it runs on the head and needs no
worker at all.

The programs are the same self-checking paired form Stage 1 uses.  That means
Stage 3 keeps working before the Spike model lands: cosim adds an independent
third opinion, but the program still knows whether it passed.
"""
from __future__ import annotations

import argparse
import os
import random
import tempfile
from typing import Iterator, List, Optional, Sequence, Tuple

import ime_tests
import rvv_ref
from rvv_ref import TileGeometry


def geometry_mix(vlen: int, rng: random.Random,
                 count: int) -> Iterator[TileGeometry]:
    """Sample legal geometries, covering every one before repeating any.

    Shuffled round-robin rather than independent uniform draws: with a few
    hundred legal geometries and a budget of thousands of programs, uniform
    sampling would leave some geometries untouched for a long time purely by
    luck, and those are exactly the ones a bug hides in.
    """
    pool = [g for g in _every_geometry(vlen)
            if g.emul_c != 16 and _allocatable(g)]
    if not pool:
        raise ValueError(f"no IME geometries are legal at VLEN={vlen}")
    emitted = 0
    while emitted < count:
        rng.shuffle(pool)
        for geom in pool:
            if emitted >= count:
                return
            yield geom
            emitted += 1


def _every_geometry(vlen: int) -> Iterator[TileGeometry]:
    """Every geometry the stress pool draws from, tier by tier, in round order.

    Round two's vqmmacc.vv rides the same emitted program shape, so the mix
    needs nothing but a longer pool -- and the geometry space is what this
    file exists to sample, so leaving the widening half out would mean the
    only randomised coverage of the new instruction is none.  The same
    argument carries to round three: vwmmacc.vv and v8wmmacc.vv are the same
    shape at a different packing depth, and the transposing tier is the same
    shape with a column-major memory image, so all three are pool entries
    rather than new machinery.

    Tiers are concatenated, not interleaved, so the earlier sequences keep
    the positions -- and therefore the random draws -- they had before.
    """
    yield from rvv_ref.ime_legal_configs(vlen)
    yield from rvv_ref.ime_legal_configs(vlen, sews=rvv_ref.WIDENING_SEWS,
                                         ws=(4,))
    for w in (2, 8):
        yield from rvv_ref.ime_legal_configs(
            vlen, sews=rvv_ref.WIDENING_SEWS_BY_W[w], ws=(w,))
    yield from rvv_ref.ime_legal_configs(vlen, tloads=("t",))
    # Round four's floating-point tier, appended last for the same reason.
    # vfmmacc.vv rides the identical five-instruction program shape -- only
    # the arithmetic opcode and the reference path change -- so it is a pool
    # entry, not new machinery.  Its cases are drawn as rounding witnesses
    # (rvv_ref.random_fp_case), so a randomly sampled floating-point stress
    # program is as sharp as a directed one.
    yield from rvv_ref.ime_legal_configs(vlen, kinds=("fp",))


def _allocatable(geom: TileGeometry) -> bool:
    try:
        ime_tests.VectorAlloc.allocate(geom)
    except ValueError:
        return False
    return True


def generate(vlen: int, count: int, seed: int = 0
             ) -> List[Tuple[str, str, TileGeometry]]:
    """(name, assembly, geometry) for *count* randomised programs."""
    rng = random.Random(seed)
    out = []
    for index, geom in enumerate(geometry_mix(vlen, rng, count)):
        widen = "" if geom.w == 1 else f"_w{geom.w}"
        trans = "" if geom.tload == "op" else "_t"
        fp = "" if geom.kind == "int" else "_fp"
        name = (f"stress_{index:06d}_sew{geom.sew}{widen}{trans}{fp}"
                f"_lam{geom.lam}_lmul{geom.lmul}_n{geom.n}")
        case = rvv_ref.random_case(geom, rng)
        out.append((name, ime_tests.emit_test(geom, case, name), geom))
    return out


def fill_pool(pool_dir: str, vlen: int, count: int, seed: int = 0,
              work_dir: str = os.path.join(tempfile.gettempdir(), "titan-stress"),
              build=None, pool_add=None,
              extension: str = "ime") -> int:
    """Build *count* programs and stage them in the Stage 3 pool.

    ``build`` and ``pool_add`` are injected rather than imported so this
    module stays importable -- and its self-test runnable -- on a machine with
    no cluster and no RISC-V toolchain.  The loop passes
    ``nodes.build_ime_test`` and ``db_node.pool_add``.
    """
    if build is None or pool_add is None:  # pragma: no cover - loop wiring
        import db_node
        import nodes
        from chia.base.ChiaFunction import get
        build = lambda asm, name: get(
            nodes.build_ime_test.chia_remote(asm, name, work_dir,
                                             extension=extension))
        pool_add = lambda *a: get(db_node.pool_add.chia_remote(*a,
                                                              extension=extension))

    added = 0
    for name, asm, geom in generate(vlen, count, seed):
        elf = build(asm, name)
        if not elf:
            continue  # a program that will not assemble is a generator bug,
            # not a DUT bug; the loop's dump keeps the source for inspection
        pool_add(pool_dir, name, elf, asm, asm.count("\n"))
        added += 1
    return added


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def check_coverage() -> None:
    """One full cycle of the mix must touch every legal geometry exactly once."""
    rng = random.Random(1)
    for vlen in (128, 256, 512):
        pool = [g for g in _every_geometry(vlen)
                if g.emul_c != 16 and _allocatable(g)]
        seen = list(geometry_mix(vlen, rng, len(pool)))
        assert len(seen) == len(pool), vlen
        assert {g.describe() for g in seen} == {g.describe() for g in pool}, vlen


def check_reproducible_and_varied() -> None:
    """A seed must reproduce a run exactly; two seeds must not coincide.

    Reproducibility is what makes a Stage 3 divergence re-runnable at all --
    the pool keeps the .S, but being able to regenerate the whole batch from
    a seed is what lets you bisect one.  Names carry the geometry, so they
    move with the seed; what must be stable is name -> program.
    """
    first = generate(256, 40, seed=1)
    assert first == generate(256, 40, seed=1), "same seed must replay exactly"
    second = generate(256, 40, seed=2)
    assert [a for _, a, _ in first] != [a for _, a, _ in second], \
        "different seeds must produce different programs"
    for run in (first, second):
        names = [n for n, _, _ in run]
        assert len(set(names)) == len(names), "names must be unique in a run"


def check_emits() -> None:
    """Every generated program must survive ime_tests' own structural checks."""
    for name, asm, geom in generate(256, 60, seed=7):
        assert asm.count(".insn 4,") == 5, name
        assert "TITAN FAIL" in asm and "TITAN PASS" in asm, name
        assert ".globl main" in asm, name
        # The name must name the geometry it was generated from -- a stress
        # pool whose names lie is a pool you cannot bisect.
        assert ("_t_" in name) == (geom.tload == "t"), name
        assert (f"_w{geom.w}_" in name) == (geom.w != 1), name
        assert ("_fp_" in name) == (geom.kind == "fp"), name
        if geom.kind == "fp":
            # The floating-point programs carry the scalar rv64f / rv64d
            # reference, not the RVV one, and set both extension state
            # fields.  A stress program that silently fell back to the
            # integer path would be green for the wrong reason.
            assert "vmul.vv" not in asm and "vredsum.vs" not in asm, name
            assert "    csrwi frm, 0" in asm, name
            assert f"fmul.{ime_tests._FP_SUFFIX[geom.sew]} " in asm, name
            assert f"li    t0, {ime_tests.MSTATUS_FS_INITIAL}" in asm, name
        assert f"# {geom.mnemonic} " in asm, name
        assert f"# {geom.load_mnemonic} " in asm, name
        assert f"# {geom.store_mnemonic} " in asm, name


def check_pool_covers_every_instruction() -> None:
    """One full cycle of the mix must exercise all eight instructions.

    The pool is a geometry sampler, so an instruction that no geometry
    selects would be silently absent from Stage 3 -- which is how an
    instruction ends up with directed coverage only and no randomised
    coverage at all.
    """
    import constants
    pool = [g for g in _every_geometry(256)
            if g.emul_c != 16 and _allocatable(g)]
    seen = set()
    for geom in pool:
        seen.update((geom.mnemonic, geom.load_mnemonic, geom.store_mnemonic))
    assert seen == set(constants.ALL_INSNS), (
        f"stress pool covers {sorted(seen)}, "
        f"want {sorted(constants.ALL_INSNS)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vlen", type=int, default=256)
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()

    for check in (check_coverage, check_reproducible_and_varied, check_emits,
                  check_pool_covers_every_instruction):
        check()
        print(f"  ok  {check.__name__}")

    programs = generate(args.vlen, args.count)
    geometries = {g.describe() for _, _, g in programs}
    tally = {}
    for _, _, g in programs:
        key = (g.kind, g.w, g.tload)
        tally[key] = tally.get(key, 0) + 1
    breakdown = " + ".join(f"{n} {kind} W={w}/{tl}"
                           for (kind, w, tl), n in sorted(tally.items()))
    print(f"\nVLEN={args.vlen}: {len(programs)} stress programs "
          f"({breakdown}) covering "
          f"{len(geometries)} distinct tile geometries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
