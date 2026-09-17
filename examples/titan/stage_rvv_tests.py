#!/usr/bin/env python3
"""Build Saturn's riscv-vector-tests and stage them as the S2 gate.

Deliberately outside the loop.  A regression suite the loop builds for itself
is a regression suite the loop could get wrong, and S2's whole value is that
it passed before anyone touched Saturn.  Run this once, before the first loop
run, and again only when Saturn's submodule pin moves.

    chia job submit -- python examples/titan/stage_rvv_tests.py

What it does, on the chipyard worker:

  1. runs ``build-tests.sh`` in the saturn-vectors checkout, verbatim
  2. collects the stage2 ELFs for this VLEN
  3. writes them into ``DB_ROOT/tests/rvv/`` where ``db_node.fetch_tests``
     reads them

Two things about that script worth knowing before you read the code.

**The tests do not self-check.**  It builds with ``TEST_MODE=cosim``, which
compiles ``-DCOSIM_TEST_CASE`` and strips the self-verification: the suite is
designed to be judged by a co-simulator.  So ``titan_loop`` runs S2 through
``cosim_run`` and reads ``match``, not through a log grep.  Cosim needs no IME
model for this -- these are pure RVV 1.0 tests and the stock libriscv has
supported RVV for years, which is exactly why the S2 gate costs nothing.

**The pruning is part of the contract.**  build-tests.sh deletes the
vaes/vsha/vsm3/vsm4/vclmul/vghsh/vgmul and vfredusum/vfwredusum cases after
generating them -- Saturn does not implement those.  Running the script
verbatim keeps that pruning; hand-rolling the make invocations would quietly
stage tests the baseline already fails, and then S2 would report a
"regression" that predates the agent.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import List, Tuple

from chia.base.ChiaFunction import ChiaFunction, get
from chia.trace.profiler import get_profiler

import db_node
from constants import CHIPYARD_PATH, RUNTIME_ENV, VLEN

#: DB key S2 reads.  Must match titan_loop.RVV_REGRESSION_KEY.
RVV_REGRESSION_KEY = "rvv"

SATURN_REL = "generators/saturn"
BUILD_TIMEOUT_S = int(os.environ.get("TITAN_RVV_BUILD_TIMEOUT_S", "10800"))

#: Where build-tests.sh leaves the built ELFs, per VLEN and MODE.
_OUT_TEMPLATE = "riscv-vector-tests/out/v{vlen}x64{mode}/bin/stage2"
_MODES = ("machine", "virtual")


def _ensure_go(env: dict) -> str:
    """riscv-vector-tests' generator is written in Go, and the chipyard image
    does not ship it.  Install it from conda-forge into the container.

    The container is recreated by every `chia up`, so this reinstalls each
    time the cluster is rebuilt -- a few minutes, and self-healing.  The
    durable fix is a derived image (the way AETHER pinned its cosim fix into
    `chia-chisel-build-cosimfix:local`); this keeps the dependency visible in
    the script that needs it instead.
    """
    found = shutil.which("go", path=env.get("PATH"))
    if found:
        return found
    for mgr in ("mamba", "conda"):
        if not shutil.which(mgr):
            continue
        subprocess.run([mgr, "install", "-y", "-q", "-c", "conda-forge", "go"],
                       capture_output=True, text=True, timeout=1800)
        found = shutil.which("go", path=env.get("PATH"))
        if found:
            return found
    raise RuntimeError(
        "no Go toolchain and none installable: riscv-vector-tests' generator "
        "cannot be built, so the S2 suite cannot be produced here.")


#: Saturn pins riscv-vector-tests at 20200cc, whose `pspike` was written
#: against an older spike.  The installed spike (1.1.1-dev) changed three
#: `extension_t` signatures *and* the `sim_t` constructor, so pspike does not
#: compile -- which blocks the whole suite, cosim mode included: TEST_MODE
#: only switches a -D flag, it does not skip the patcher.
#:
#: It is not simply "the pin is too old".  The image's spike sits *between*
#: the pinned commit and upstream HEAD, so both ends fail, in opposite
#: directions.  Compiling pspike at each commit that touches it:
#:
#:     20200cc  (Saturn's pin)  old extension_t signatures      fail
#:     ef5e143  pspike API fix   still the old get_instructions  fail
#:     f21c047  latest spike     ---------------------------->  COMPILES
#:     b30515e  /  f76bff1       13-arg sim_t, image wants 12   fail
#:
#: So this pins f21c047 exactly rather than tracking a branch: newer is not
#: better here, it is a different kind of broken.  Re-derive this table if
#: the chipyard image's spike ever moves.
#:
#: Hand-porting pspike instead would be a poor trade: it is what produces the
#: tests' expected values, so a subtly wrong port yields a suite that is
#: wrong *and green*.
#:
#: This runs inside the container, so it does not touch the checkout in the
#: repo -- it is undone by the next `chia up`, like the Go install above.
#: Making it permanent means moving Saturn's own submodule pin.
VECTOR_TESTS_REF = os.environ.get("TITAN_VECTOR_TESTS_REF", "f21c047")


def _bump_vector_tests(saturn_dir: str) -> str:
    """Put riscv-vector-tests on the one commit whose pspike compiles here."""
    repo = os.path.join(saturn_dir, "riscv-vector-tests")
    # os.path.exists, not isdir: in a submodule `.git` is a *file* holding a
    # gitlink to the superproject's modules directory, so an isdir check
    # silently decides there is no repo here and skips the whole fix.
    if not os.path.exists(os.path.join(repo, ".git")):
        return f"riscv-vector-tests at {repo} is not a git checkout; skipping"

    def _git(*args, **kw):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True,
                              text=True, timeout=kw.pop("timeout", 300))

    want = _git("rev-parse", VECTOR_TESTS_REF).stdout.strip()
    if want and want == _git("rev-parse", "HEAD").stdout.strip():
        return f"riscv-vector-tests already at {VECTOR_TESTS_REF}"

    # The build itself dirties tracked files (Makefrag is generated), which
    # makes checkout refuse.  Discarding them is right in an ephemeral build
    # container and wrong nowhere else this runs.
    _git("checkout", "--", ".")
    if not want:
        _git("fetch", "--tags", "origin", timeout=900)
    checked = _git("checkout", "--detach", VECTOR_TESTS_REF)
    if checked.returncode != 0:
        return (f"could not check out {VECTOR_TESTS_REF}: "
                f"{(checked.stderr or '').strip()[-300:]}")
    # riscv-vector-tests has its own submodule, env/riscv-test-env, which
    # supplies the RVTEST_* and INIT assembler macros.  Moving HEAD moves the
    # gitlink but not the contents, and stage1 then fails to assemble with
    # "unrecognized opcode rvtest_rv64uvx" -- an error that looks like a
    # toolchain problem and is not.
    subs = _git("submodule", "update", "--init", "--recursive", timeout=900)
    head = _git("rev-parse", "--short", "HEAD").stdout.strip()
    note = f"riscv-vector-tests pinned to {head}"
    if subs.returncode != 0:
        note += f" (submodule update failed: {(subs.stderr or '').strip()[-200:]})"
    return note


@ChiaFunction(resources={"chipyard": 0.9})
def build_rvv_tests(chipyard_path: str = CHIPYARD_PATH, vlen: int = VLEN,
                    timeout_seconds: int = BUILD_TIMEOUT_S,
                    extension: str = "") -> Tuple[bool, str, List[Tuple[str, bytes]]]:
    """Run build-tests.sh and return (ok, log, [(name, elf_bytes)]).

    ELFs come back by value, like every other artifact here, so the database
    node needs no shared filesystem with the build node.
    """
    if extension:
        get_profiler().add_info({"extension": extension})
    saturn_dir = os.path.join(chipyard_path, SATURN_REL)
    script = os.path.join(saturn_dir, "build-tests.sh")
    if not os.path.isfile(script):
        return False, f"no build-tests.sh at {script}", []

    env = dict(os.environ)
    try:
        go = _ensure_go(env)
    except RuntimeError as exc:
        return False, str(exc), []
    pspike_note = _bump_vector_tests(saturn_dir)

    proc = subprocess.run(["bash", "build-tests.sh"], cwd=saturn_dir,
                          capture_output=True, text=True, env=env,
                          timeout=timeout_seconds)
    # Keep the setup notes on the front of every log.  Truncating to the tail
    # alone drops exactly the lines that say what was installed and which
    # commit was checked out -- which is what you need first when the build
    # fails, and what was missing the two times it did.
    notes = os.linesep.join([f"go: {go}", pspike_note])
    output = (proc.stdout or "") + (proc.stderr or "")
    log = notes + os.linesep + output[-20000:]

    # A partial suite is worth staging.  build-tests.sh does machine mode
    # first and virtual mode second, and with this spike the virtual half
    # fails: pspike trips a page-table assertion and its message gets
    # interleaved into the generated .S files, corrupting them.  The machine
    # half is complete and independent, and 841 real RVV tests is a far
    # better S2 gate than none at all.
    #
    # So on failure take machine mode only -- never the mode the build died
    # in, whose outputs may be the corrupted ones.
    modes = _MODES if proc.returncode == 0 else ("machine",)
    if proc.returncode != 0:
        log += os.linesep + (
            "NOTE: build-tests.sh failed; staging machine mode only. "
            "Virtual-mode (VM load/store) tests are NOT in the S2 gate.")

    collected: List[Tuple[str, bytes]] = []
    for mode in modes:
        out_dir = os.path.join(
            saturn_dir, _OUT_TEMPLATE.format(vlen=vlen, mode=mode))
        if not os.path.isdir(out_dir):
            continue
        for name in sorted(os.listdir(out_dir)):
            path = os.path.join(out_dir, name)
            # The generator emits .dump alongside each ELF; only the ELF runs.
            if not os.path.isfile(path) or name.endswith(".dump"):
                continue
            with open(path, "rb") as fh:
                collected.append((f"{mode}_{name}", fh.read()))
    return True, log, collected


@ChiaFunction(resources={"database": 0.9})
def stage(tests: List[Tuple[str, bytes]], key: str = RVV_REGRESSION_KEY,
          extension: str = "") -> int:
    """Write the suite into DB_ROOT/tests/<key>/, replacing what is there."""
    if extension:
        get_profiler().add_info({"extension": extension})
    dest = os.path.join(db_node.DB_ROOT, "tests", key)
    os.makedirs(dest, exist_ok=True)
    for stale in os.listdir(dest):
        path = os.path.join(dest, stale)
        if os.path.isfile(path):
            os.remove(path)
    for name, elf in tests:
        with open(os.path.join(dest, name), "wb") as fh:
            fh.write(elf)
    return len(tests)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vlen", type=int, default=VLEN)
    parser.add_argument("--dry-run", action="store_true",
                        help="build but do not stage")
    args = parser.parse_args()

    import ray
    ray.init(address="auto", runtime_env=RUNTIME_ENV)

    print(f"building riscv-vector-tests for VLEN={args.vlen} "
          f"(this is a long build; TEST_MODE=cosim)")
    ok, log, tests = get(build_rvv_tests.chia_remote(CHIPYARD_PATH, args.vlen,
                                                     extension="ime"))
    if not ok and not tests:
        print(log)
        print("\nbuild-tests.sh failed and produced nothing stageable; "
              "check the tail above.", file=sys.stderr)
        return 1
    if not ok:
        print(log[-3000:])
        print("\nbuild-tests.sh did not finish cleanly, but the machine-mode "
              "half is complete and is being staged. Virtual-mode tests are "
              "absent from the S2 gate -- see the NOTE above.",
              file=sys.stderr)
    if not tests:
        print(log[-4000:])
        print(f"\nbuild succeeded but no ELFs were found under "
              f"{_OUT_TEMPLATE.format(vlen=args.vlen, mode='machine')}. "
              f"The output layout may have moved -- check the script's rm -rf "
              f"paths against this repo's checkout.", file=sys.stderr)
        return 1

    print(f"built {len(tests)} test ELFs")
    if args.dry_run:
        for name, elf in tests[:20]:
            print(f"  {name}  {len(elf)} bytes")
        print("  ... (dry run, nothing staged)")
        return 0

    staged = get(stage.chia_remote(tests, RVV_REGRESSION_KEY, extension="ime"))
    print(f"staged {staged} tests under DB_ROOT/tests/{RVV_REGRESSION_KEY}/")
    print("titan_loop's S2 gate will now find them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
