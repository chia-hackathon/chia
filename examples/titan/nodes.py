"""@ChiaFunction workers for the Titan loop — one wrapper per cluster role.

Always dispatched with `.chia_remote(...)` / `.options(...).chia_remote(...)`
and awaited with `get(...)` (field guide §5). The chisel_build workers
(`reset_chipyard`, `collect_diff`, `build_saturn`) are dispatched into the
pipeline's `chipyard` placement group so the LLM's editor BashTool and the
build share one container exclusively.
"""

import hashlib
import os
import subprocess
import tempfile
import uuid

from chia.base.ChiaFunction import ChiaFunction
from chia.trace.profiler import get_profiler
from chia.chipyard.chisel_build_node import ChiselBuildNode
from chia.chipyard.cosim_node import CosimNode
from chia.chipyard.riscv_build_node import RiscvBuildNode
from chia.chipyard.spike_build_node import SpikeBuildNode
from chia.chipyard.verilator_run_node import VerilatorRunNode
from chia.chipyard.state_def import (
    BuildArtifact,
    BuildTarget,
    CosimResult,
    RunResult,
    SpikeBuildArtifact,
    SpikeResult,
)

from constants import (
    BUILD_MAKE_JOBS,
    BUILD_TIMEOUT_S,
    CHIPYARD_DIFF_SUBMODULES,
    CHIPYARD_PATH,
    CONFIG_PACKAGE,
    COSIM_CONFIG,
    COSIM_VRUN,
    IME_CFLAGS,
    SATURN_REPO_REL,
    SPIKE_ISA,
    SPIKE_SRC_REL,
    SIM_TIMEOUT_CYCLES,
    SIM_ZERO_INIT_DEFINES,
    STRESS_TEST_MAX_CYCLES,
    VERILATOR_THREADS,
    VLEN,
)


# --- chisel_build node (pin to the pipeline's placement group) -------------

@ChiaFunction(resources={"chipyard": 0.9})
def reset_chipyard(chipyard_path: str = CHIPYARD_PATH, extension: str = "") -> str:
    """Reset the chipyard repo AND every submodule in CHIPYARD_DIFF_SUBMODULES
    to a pristine baseline so every experiment starts clean: reset + clean
    chipyard, then each submodule (to the commit chipyard pins). Saturn ships
    no BOOM-style single target -- the agent edits Saturn, rocket-chip (the
    vtype/vconfig CSR machinery), and shuttle (the host tile), so all three
    have to come back to the pinned commit or a "clean" run silently carries
    over a previous iteration's edits."""
    if extension: get_profiler().add_info({"extension": extension})
    subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=chipyard_path, check=False)
    subprocess.run(["git", "clean", "-fd"], cwd=chipyard_path, check=False)
    parts = []
    for sm_rel in CHIPYARD_DIFF_SUBMODULES:
        sm_path = os.path.join(chipyard_path, sm_rel)
        r = subprocess.run(["git", "ls-tree", "HEAD", sm_rel],
                           cwd=chipyard_path, capture_output=True, text=True)
        target = r.stdout.split()[2] if (r.returncode == 0 and r.stdout.strip()) else "HEAD"
        subprocess.run(["git", "reset", "--hard", target], cwd=sm_path, check=False)
        subprocess.run(["git", "clean", "-fd"], cwd=sm_path, check=False)
        parts.append(f"{sm_rel}@{target[:10]}")
    return "Reset chipyard + " + ", ".join(parts)


@ChiaFunction(resources={"chipyard": 0.9})
def collect_diff(chipyard_path: str = CHIPYARD_PATH, extension: str = "") -> str:
    """The LLM's accumulated edits as ONE chipyard-rooted unified diff: the
    chipyard-level changes (the cosim config) plus every submodule in
    CHIPYARD_DIFF_SUBMODULES, each prefixed so a single `git apply` reseeds
    all of them. This MUST include generators/rocket-chip: the vtype/vl CSR
    state (VType, VConfig) that the IME instructions read for their tile
    geometry lives in rocket-chip, not in Saturn, so the agent's rocket-chip
    edits are as load-bearing as its Saturn edits -- a diff that omits
    rocket-chip cannot reproduce this run. Intent-to-add untracked files first
    so new sources appear. For probes."""
    if extension: get_profiler().add_info({"extension": extension})
    def _diff(cwd: str, prefix: str = "") -> str:
        subprocess.run(["git", "add", "-N", "."], cwd=cwd, check=False)
        args = ["git", "diff", "--ignore-submodules"]
        if prefix:
            args += [f"--src-prefix=a/{prefix}/", f"--dst-prefix=b/{prefix}/"]
        d = subprocess.run(args, cwd=cwd, capture_output=True, text=True).stdout
        subprocess.run(["git", "reset"], cwd=cwd, check=False)
        return d
    parts = [_diff(chipyard_path)]
    for sm_rel in CHIPYARD_DIFF_SUBMODULES:
        parts.append(_diff(os.path.join(chipyard_path, sm_rel), sm_rel))
    return "".join(parts)


@ChiaFunction(resources={"chipyard": 0.9})
def apply_diff(diff: str, chipyard_path: str = CHIPYARD_PATH,
               extension: str = "") -> str:
    """Seed the (freshly reset) tree with a prior run's probe diff, so a
    pipeline can resume from a known implementation instead of re-deriving
    it. Reseeds the tree from a `collect_diff` output (``git apply`` from
    the chipyard root; the diff's a/<submodule>/ prefixes are made for
    this).

    Used by ``--model-diff`` to reuse a converged Stage 0 Spike model from an
    earlier run instead of paying for the model agent again.  Returns "" on
    success, else git's stderr."""
    if extension: get_profiler().add_info({"extension": extension})
    r = subprocess.run(["git", "apply", "--whitespace=nowarn", "-"],
                       cwd=chipyard_path, input=diff, capture_output=True,
                       text=True)
    return "" if r.returncode == 0 else (r.stderr or r.stdout or "git apply failed")


@ChiaFunction(resources={"chipyard": 0.9})
def write_files(dir_path: str, files: dict, exclude_rel: str = "",
                chipyard_path: str = CHIPYARD_PATH, extension: str = "") -> str:
    """Drop a batch of text files into the chipyard container.

    The loop's artifacts live on the head (``TITAN_LOG_ROOT``), which the
    agent's container cannot see; until r5 that is why the loop pasted 300-line
    tails into the prompt instead of saying "read the log". This copies them
    to where the agent's bash tool actually is, once per iteration.

    ``exclude_rel`` names a path to add to ``.git/info/exclude`` (idempotently)
    before the first write. Not cosmetic: ``collect_diff`` runs ``git add -N .``
    to catch new sources, so an untracked ``titan_logs/`` would ride along in
    every probe diff, and the diff is what reseeds a run. Being ignored also
    survives ``reset_chipyard``'s ``git clean -fd`` -- which has no ``-x``, so
    it leaves ignored paths alone.

    Takes ``chipyard 0.9`` like the other chisel_build nodes and must be
    dispatched with the pipeline's ``pg_opts``: a chipyard task scheduled
    outside the placement group can never be placed, because the group has
    reserved the node's whole ``chipyard`` resource.
    """
    if extension: get_profiler().add_info({"extension": extension})
    if exclude_rel:
        exclude_file = os.path.join(chipyard_path, ".git", "info", "exclude")
        try:
            os.makedirs(os.path.dirname(exclude_file), exist_ok=True)
            existing = ""
            if os.path.exists(exclude_file):
                with open(exclude_file) as fh:
                    existing = fh.read()
            # A directory gets a trailing slash; a single file must NOT get
            # one, or gitignore matches directories only and the file stays
            # untracked-but-visible -- which is how the loop's generated S2
            # cosim config would have ridden along in every probe diff.
            entry = exclude_rel.rstrip("/")
            if not os.path.splitext(entry)[1]:
                entry += "/"
            if entry not in existing.split():
                with open(exclude_file, "a") as fh:
                    fh.write(f"\n# titan: loop artifacts, never part of a diff\n"
                             f"{entry}\n")
        except OSError as exc:                    # not fatal: the logs matter
            pass                                  # more than the ignore entry

    written = 0
    for rel, text in (files or {}).items():
        # Defensive: a relative path only, and no escaping the directory.
        rel = os.path.normpath(rel).lstrip("/")
        if rel.startswith(".."):
            continue
        target = os.path.join(dir_path, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8", errors="replace") as fh:
            fh.write(text if isinstance(text, str) else str(text))
        written += 1
    return f"wrote {written} file(s) to {dir_path}"


@ChiaFunction(resources={"chipyard": 0.9})
def build_saturn(config: str = COSIM_CONFIG, golden_model=None,
                 extension: str = "") -> BuildArtifact:
    """Elaborate + verilate the current Saturn source into a Verilator sim.

    `config` defaults to the cosim config; the baseline PPA synth passes a
    stock config that builds in a fresh tree (the cosim config is the LLM's
    to write).

    `golden_model` is a SpikeBuildArtifact from `build_spike`, and it is what
    makes cospike mean anything here. The stock image's libriscv has never
    heard of Zvvm, so linking against it would give a Stage 3 in which the
    golden model traps on the first matrix instruction. Passing the model the
    loop built links its bytes in instead. Stages 1 and 2 leave it None --
    they have no Spike and are designed not to need one."""
    if extension: get_profiler().add_info({"extension": extension})
    node = ChiselBuildNode(
        chipyard_path=CHIPYARD_PATH,
        config=config,
        config_package=CONFIG_PACKAGE,
        target=BuildTarget.VERILATOR,
        make_jobs=BUILD_MAKE_JOBS,
        timeout_seconds=BUILD_TIMEOUT_S,
        extra_make_args={"VERILATOR_THREADS": str(VERILATOR_THREADS),
                         "EXTRA_SIM_PREPROC_DEFINES": SIM_ZERO_INIT_DEFINES},
        clean_sim=True,                # required alongside a golden model:
                                       # make will not relink for a new lib
        collect_generated_src=True,    # RTL for the sky130 PPA synth
        golden_model=golden_model,
    )
    return node.build()


@ChiaFunction(resources={"chipyard": 0.9})
def build_spike(extension: str = "") -> SpikeBuildArtifact:
    """Build libriscv from the riscv-isa-sim checkout the model agent edits.

    Returns the shared library by value, so it needs no shared filesystem and
    no worker of its own -- it shares the chipyard image and machine with the
    Chisel build, which is why cluster.yaml has no spike node type.

    Note this node exists in chia but no example had ever called it: upstream
    Spike already implements Zb*/Zk*/Zicond, so riscv_extensions never needed
    to rebuild it. Zvvm is a draft Spike has never heard of, which is what
    makes the Stage 2 model a deliverable rather than a dependency."""
    if extension: get_profiler().add_info({"extension": extension})
    return SpikeBuildNode(
        chipyard_path=CHIPYARD_PATH,
        spike_rel=SPIKE_SRC_REL,
        make_jobs=BUILD_MAKE_JOBS,
        timeout_seconds=BUILD_TIMEOUT_S,
        install=True,
    ).build()


@ChiaFunction(resources={"chipyard": 0.4}, num_cpus=1)
def spike_run(elf_content: bytes, elf_name: str, work_dir: str,
              isa: str = SPIKE_ISA, vlen: int = VLEN,
              timeout_seconds: int = 900, extension: str = "") -> SpikeResult:
    """Run one ELF on Spike alone and capture what it printed.

    chia has a `SpikeResult` dataclass and no producer for it -- there is a
    cosimulation node but nothing that runs Spike on its own. Stage 2 needs
    exactly that: the model has to be judged before any DUT exists to
    cosimulate it against. The directed programs print their own verdict, so
    stdout is the result.

    The VLEN coupling is not optional: see the comment on the argv below."""
    if extension: get_profiler().add_info({"extension": extension})
    task_dir = os.path.join(work_dir, uuid.uuid4().hex[:8])
    os.makedirs(task_dir, exist_ok=True)
    elf_path = os.path.join(task_dir, elf_name)
    with open(elf_path, "wb") as fh:
        fh.write(elf_content)
    os.chmod(elf_path, 0o755)

    # VLEN goes in the ISA string, not in --varch.  Spike 1.1.1 has no
    # --varch option at all (it was removed in favour of the Zvl*b
    # extensions), and passing it makes spike exit before running anything.
    # Either way the coupling is the point: the programs are generated for a
    # specific VLEN and derive every tile geometry from it, so a spike at the
    # default width computes a different -- and perfectly legal -- answer.
    full_isa = isa if "zvl" in isa else f"{isa}_zvl{vlen}b"
    argv = ["spike", f"--isa={full_isa}", elf_path]
    try:
        proc = subprocess.run(argv, cwd=task_dir, capture_output=True,
                              text=True, timeout=timeout_seconds)
        returncode, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = -1
        out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = "timed out"
    return SpikeResult(
        test_binary_name=elf_name,
        isa=full_isa,
        log=out + err,
        commit_log="",
        returncode=returncode,
        success=returncode == 0,
    )


# --- riscv_build node: compile emitted differential directed tests ----------
# Stage 1 verifies against riscv-vector-tests / directed IME programs: ime_tests
# emits one program per instruction / geometry; we cross-compile each here
# (htif_nano harness) and cosim it. Built once per run — the test programs
# don't change across implement iterations.

ISA_CACHE_DIR = os.path.join(tempfile.gettempdir(), "titan_isacache")  # node-local ($TMPDIR)


@ChiaFunction(resources={"riscv_build": 1}, num_cpus=2)
def build_ime_test(asm: str, program: str, work_dir: str,
                   extension: str = "") -> bytes | None:
    """Cross-compile one emitted IME test (.S) into a baremetal ELF via the
    riscv-harness. `extra_cflags` MUST come from constants.IME_CFLAGS: the
    harness Makefile's baseline -march (rv64imafd_zicsr_zifencei) has no
    vector extension at all, so without it the assembler rejects the RVV
    scaffolding around the .insn-encoded IME instructions. Returns the ELF
    bytes, or None on build failure. Cached by content (the .S fully
    determines the ELF, since march is fixed), so re-runs and resumes reuse
    the same binaries instead of recompiling."""
    if extension: get_profiler().add_info({"extension": extension})
    cached = os.path.join(ISA_CACHE_DIR, hashlib.sha1(asm.encode()).hexdigest())
    if os.path.exists(cached):
        with open(cached, "rb") as f:
            return f.read()
    art = RiscvBuildNode().build(asm.encode(), program, work_dir, target="verilator",
                                 lang="asm", extra_cflags=" ".join(IME_CFLAGS))
    if not art.success:
        return None
    os.makedirs(ISA_CACHE_DIR, exist_ok=True)
    with open(cached, "wb") as f:
        f.write(art.binary_content)
    return art.binary_content


# --- verilator_run node (the DUT) ------------------------------------------
# Directed test ELFs are SELF-CHECKING: the DUT run alone is the verdict.
# Spike appears only inside the random-stress cosim (CosimNode), once the
# Stage 2 Spike model can decode Zvvm.

@ChiaFunction(resources={"verilator_run": COSIM_VRUN}, num_cpus=VERILATOR_THREADS)
def verilator_run_remote(
    artifact: BuildArtifact,
    elf_content: bytes,
    elf_name: str,
    work_dir: str,
    extension: str = "",
) -> RunResult:
    """Run one ELF on the Verilator simulation of the freshly-built core."""
    if extension: get_profiler().add_info({"extension": extension})
    return VerilatorRunNode().run(
        artifact=artifact,
        test_binary_content=elf_content,
        test_binary_name=elf_name,
        work_dir=work_dir,
        plusargs={"+loadmem": elf_name},
        timeout_cycles=SIM_TIMEOUT_CYCLES,
        verbose=False,
    )


# --- stress test (S3): head-generated, cosim on a cosim node ----------------
# riscv-dv cannot reach vector instructions at all (its ISA model has no RVV,
# let alone Zvvm), so there is no gen_to_pool node here and no `dv`/`xcelium`
# worker in this cluster. Titan's Stage 3 randomised programs are enumerated
# and emitted directly on the head by ime_stress.py (legal tile geometries x
# operand sweeps), then handed straight to cosim_run below.

@ChiaFunction(resources={"verilator_run": COSIM_VRUN}, num_cpus=VERILATOR_THREADS)
def cosim_run(artifact: BuildArtifact, elf_content: bytes, elf_name: str,
              instr: int, work_dir: str, extension: str = "") -> CosimResult:
    """Co-simulate one ELF in lockstep: spike rides inside the sim (cospike)
    and the run aborts at the first divergence. Budget is STRESS_TEST_MAX_CYCLES.

    Unusable in Stage 1: upstream Spike has no Zvvm decoder, so cospike
    aborts (illegal instruction) at the first IME instruction regardless of
    whether the DUT is correct. This node only becomes live once the Stage 2
    Spike model (spike/Zvvm functional model) exists to give cospike a
    reference that actually knows the extension.
    """
    if extension: get_profiler().add_info({"extension": extension})
    return CosimNode().run(artifact, elf_content, elf_name, work_dir,
                           timeout_cycles=STRESS_TEST_MAX_CYCLES)
