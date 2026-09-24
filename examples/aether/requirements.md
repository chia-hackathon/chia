# aether — requirements and preconditions

Everything needed to run `loop/loop.py` yourself. This file is the honest
inventory: some of it is *not* included in this snapshot, and those gaps are
called out explicitly under [Known gaps](#known-gaps).

> **Precondition, stated up front: this loop assumes a CHIA cluster is already
> up and running.** `loop/loop.py` calls `ray.init(address=...)` and submits
> work to named Ray resources (`llm`, `chipyard`, `riscv_build`,
> `verilator_run`, `database`) that only exist once `chia up` has brought the
> cluster's nodes and Docker containers online. **This directory does not
> start the cluster** — `cluster/up.sh` is a thin wrapper you run separately,
> before the loop. If the cluster is down, the loop hangs waiting for
> resources rather than failing fast.

## 1. Python environment

Python **3.10+** (the original run used 3.10.19 in a conda env named
`chia_env`).

The loop imports the `chia` package, which is **not** vendored here: it is the
repository this example lives in. Install it editable from the repository root
(two levels up from `examples/aether/`):

```bash
cd <repo root>          # the dir with pyproject.toml / name = "chialoops"
pip install -e .
```

That pulls the pinned runtime dependencies from `pyproject.toml`. The versions
the original run actually had installed:

| package | version used |
|---|---|
| `ray[default]` | 2.54.0 (pinned in `pyproject.toml`) |
| `mcp` | 1.27.1 |
| `pydantic` | 2.12.4 |
| `fastapi` | 0.121.0 |
| `pyyaml` | 6.0.3 |
| `graphviz` | 0.21 |
| `boto3` | 1.43.83 (`>=1.28.0`) |
| `google-genai` | 2.8.0 (`>=1.64.0`) |
| `requests` | 2.34.2 (`>=2.31.0`) |
| `chialoops` (this repo) | 1.0.1, installed editable |

`pytest` (`pip install -e '.[test]'`) is only needed for
`loop/tests/test_llm_feedback.py`.

The Ray version must match **exactly** between the host driver and every
container, or Ray refuses the connection. A system-packaged Ray (e.g. 2.9.1)
will not work — always invoke `python`/`ray`/`chia` from the env where
`chialoops` is installed.

The data/analysis tools (`ledger.py`, `journal.py`, `llama_project.py`,
`llama_batch_project.py`, `hwconfig.py`) need **no** cluster and no extra
packages beyond the standard library — they read the SQLite DB and JSON under
`results/`.

## 2. Required environment variables

| variable | used by | default | what it is |
|---|---|---|---|
| `AETHER_HEAD_IP` | `cluster/cluster.yaml` | *required* | IP of the host running the Ray nodes (`provider.head_ip`, `compatible_ips`). |
| `AETHER_RAY_TMPDIR` | `cluster/cluster.yaml` | *required* | Ray temp/spill dir, on a **large** filesystem. Keep the path short — AF_UNIX socket paths cap at 107 bytes. Each containerized node type also gets `<dir>_ct_{llm,build,cosim,riscv}` on the host, because the container users have different uids and sharing one dir gives `Permission denied`. |
| `AETHER_WORKDIR` | `cluster/cluster.yaml` | *required* | Host dir whose `work/{kernel,sim,build}` subdirs are bind-mounted into the containers. |
| `RAY_ADDRESS` | `loop/loop.py` | `auto` | Ray head to connect to, e.g. `10.0.0.5:6379`. Set it explicitly if the machine runs **more than one** Ray cluster — `auto` picks whichever it finds. |
| `AETHER_ROOT` | `loop/constants.py`, `loop/hwconfig.py` | this directory | Root for host-side inputs/outputs (`out/loop/`, `repos/`). |
| `AETHER_OUT_DIR` | `loop/constants.py` | `$AETHER_ROOT/out/loop` | Per-run artifact dir. |
| `AETHER_BENCHMARKS_DIR` | `loop/constants.py` | `$AETHER_ROOT/repos/saturn/benchmarks` | Saturn benchmark tree the loop compiles against. |
| `AETHER_DB` | `loop/db.py` | `$AETHER_ROOT/...` (see `constants.DB_PATH`) | SQLite results DB. Point it at `results/loop/aether.db` to re-derive the published reports. |
| `AETHER_DB_WAL`, `AETHER_DB_BUSY_TIMEOUT` | `loop/db.py` | off / 30s | WAL is unsafe on network filesystems, hence off. |
| `AETHER_GEMMINI_SRC` | `cluster/install-gemmini-kernels.sh` | `repos/gemmini/software/gemmini-rocc-tests` if present, else `kernels/` | Source tree the Gemmini kernel files are copied from. |
| `AETHER_BUILD_CONTAINER` | `cluster/install-gemmini-kernels.sh` | `aether-chisel-build-$USER-0` | chisel-build container name. |
| `CHIPYARD_PATH` | `loop/constants.py` | `/home/ray/chipyard` | chipyard path **inside** the chisel-build container. |
| `AETHER_{AGENT,BUILD,SIM}_WORK_DIR` | `loop/constants.py` | `/tmp/aether-{kernel,build,sim}` | Container-internal work dirs. Change only in lockstep with the bind-mounts in `cluster/cluster.yaml`. |
| `AETHER_DB_TOOL` | `loop/llm.py` | on | Set `0` to deny the agent the DB query tool. |
| `ANTHROPIC_API_KEY` *or* a `~/.claude` credential dir | the `llm` node | *required* | See §4. |
| `SSH_AUTH_SOCK` | `cluster/up.sh`, `cluster/cluster.yaml` | falls back to `/dev/null` | Forwarded ssh-agent, needed for the chisel-build container's `git@github.com:` remotes. |

`cluster/up.sh` refuses to launch if the three required `AETHER_*` variables
are unset, because chia passes an unset `${VAR}` through to the YAML
**literally** rather than erroring.

## 3. Docker images

`cluster/cluster.yaml` names four images, all run with `pull_before_run: False`
(they must already be present locally — the chisel-build image is ~30 GB and
layer-verifying it on every `chia up` times out):

| node type | image | source |
|---|---|---|
| `llm` | `ghcr.io/ucb-bar/chia-claude-code:latest` | public; `dockerfiles/ClaudeCodeDockerfile` at the repo root |
| `cosim` | `ghcr.io/ucb-bar/chia-verilator-run:latest` | public; `dockerfiles/VerilatorRunDockerfile` |
| `riscv_build` | `ghcr.io/ucb-bar/chia-riscv-cross:latest` | public; `dockerfiles/RiscvCrossDockerfile` |
| `build` | **`chia-chisel-build-aether-cosim:local`** | **locally built, not published** — see Known gaps |

The `build` image is `ghcr.io/ucb-bar/chia-chisel-build:latest` plus three
patch layers: (1) `debug_rob` DPI output initialization, (2) a vectorized
DebugROB DPI so a multi-issue Shuttle core does not interleave push/pop over
one shared deque, (3) cospike custom-extension support, and (4) the
Saturn+Gemmini co-existence configs added to `GemminiConfigs.scala` by
`docker/add_coexist_config.py` (the only one of those patch scripts included
here).

## 4. LLM access

The optimizing agent is the **Claude Code CLI** (`claude --print`) running
inside the `llm` container (`chia/models/claude.py`;
`LLM_MODEL = "claude-fable-5-1"`, `--effort high`, 1800 s timeout in
`loop/constants.py`). You need **your own** Anthropic credentials — either
`ANTHROPIC_API_KEY` in the container env, or a `~/.claude` credential
directory, which `cluster/cluster.yaml` bind-mounts as
`-v ${HOME}/.claude:/home/ray/.claude`. Budget accordingly: the published
10 rounds / 159 LLM-invoking iterations (248 DB rows over 63 runs) cost
**$180.16** total.

## 5. Hardware / RTL side

| item | value |
|---|---|
| Simulator config | `GENV256D128GemminiShuttleConfig` (cosim variant: `GENV256D128GemminiShuttleCosimConfig`) |
| Config package | `chipyard` |
| Config fragments | `saturn.shuttle.WithShuttleVectorUnit(256, 128, saturn.common.VectorParams.genParams)`, `gemmini.DefaultGemminiConfig`, `shuttle.common.WithShuttleDebugROB`, `shuttle.common.WithShuttleTileBeatBytes(16)`, `WithSystemBusWidth(128)`, `WithNShuttleCores` (the two accelerator fragments must sit **left** of `WithNShuttleCores` in the Config chain — both rewrite `TilesLocated`/`BuildRoCC`) |
| Chipyard | `ucb-bar/chipyard`, branch **`chia_artifact`** (what `dockerfiles/ChipyardDockerfile` clones) |
| Gemmini | `ucb-bar/gemmini` @ `8c3f9923a44a2fe2c7930587be297d6d4f8c09ca` |
| Saturn | `ucb-bar/saturn-vectors` @ `dfe75de2a8868d51d42c20612821431a9fcf645e` |
| Shuttle | `ucb-bar/shuttle` @ `622f08b9fd3697e3fd75e7799885ecdf9e2f13f2` |
| llama.cpp (reference only, not in the loop) | `ggml-org/llama.cpp` @ `0ef4d560e12c1a46470265c1abd31dd47c777d23` |
| Cross-compiler | `riscv64-unknown-elf-gcc`, `-march=rv64gcv_zfh_zvfh -mabi=lp64d` (Gemmini kernels use the gemmini-rocc-tests flags) |
| Simulation | Verilator RTL, `mm_magic_t` DRAM model (**not** DRAMSim2, **not** FireSim/FPGA) |

There is **no** submodule pointer for these in this repository — the commits
above were read from the original working tree's `repos/` checkouts.

## 6. Kernel sources

- **Gemmini kernels** (this project's own): `kernels/bareMetalC/*.c` +
  `kernels/include/*.h` — exactly the 16 files
  `cluster/install-gemmini-kernels.sh` installs. They must be copied into the
  chisel-build container's `generators/gemmini/software/gemmini-rocc-tests`
  tree after **every** `chia up`, because that tree lives inside the container
  and is not bind-mounted.
- **Saturn kernels**: `kernels/saturn/*` — a **flat** copy of what upstream
  keeps as one directory per benchmark under `saturn/benchmarks/`
  (`llama-softmax/llama_softmax.c`, etc.). Place them in your own
  saturn-vectors benchmark tree, one dir per kernel name, matching each
  `Kernel(bench_dir=...)` in `loop/kernels.py`.
- **Not included**: the upstream sanity benchmarks `vec-sgemv` (the *default*
  `--kernel`!), `vec-softmax`, `vec-dotprod`, and Gemmini's own
  `tiled_matmul_ws.c` / `include/gemmini.h`. Those come from the
  saturn-vectors and gemmini checkouts, not from here.

## 7. Known gaps

An outside reader should expect to supply these themselves; none of them can
be reconstructed from this directory alone.

1. **Anthropic credentials and budget** (§4).
2. **A running CHIA cluster** — `chia up` against a host you control, with
   Docker and a matching Ray. Nothing here starts it.
3. **The `chia-chisel-build-aether-cosim:local` image.** Only
   `docker/add_coexist_config.py` (the Saturn+Gemmini config fragment) is
   included; the surrounding Dockerfiles and the `debug_rob` / cospike patch
   scripts live in the original working tree's `docker/` directory and are not
   part of this snapshot.
4. **Chipyard / Saturn / Shuttle / Gemmini checkouts** at the commits in §5 —
   not vendored, no submodules.
5. **The upstream sanity + Gemmini library kernel sources** listed in §6.
6. **Per-iteration agent transcripts** (`out/loop/<run_id>/agent_NN.txt`,
   `kernel_NN.h`). `results/loop/` carries the DB, the ledger and the
   iteration journal, but not the transcripts, so re-running
   `loop/journal.py` reproduces every numeric column and leaves the
   "hypothesis" column blank.
7. **Elaboration logs** (`out/coexist/*.log`) cited by
   `results/evidence/hardware-config-excerpts.md`; only the excerpts are here.

## 8. Reproducing the published reports (no cluster needed)

```bash
cd examples/aether/loop && mkdir -p out
export AETHER_DB=../results/loop/aether.db
python ledger.py  --rounds-file ../results/loop/rounds.json \
                  --md-out out/ledger.md --json-out out/ledger.json
python journal.py --rounds-file ../results/loop/rounds.json \
                  --ledger-json ../results/loop/ledger.json \
                  --md-out out/iterations.md --json-out out/iterations.json
python llama_project.py --self-test
python llama_project.py --measured ../results/projections/measured_cycles_round9.json
python llama_batch_project.py --measured ../results/projections/measured_cycles_round9.json
python hwconfig.py --self-test
```

`ledger.md` reproduces byte-for-byte; `ledger.json` differs only in its
`generated_at` / `rounds_file` provenance fields. `iterations.md` reproduces
every numeric column (the hypothesis column needs the transcripts, gap 6).
