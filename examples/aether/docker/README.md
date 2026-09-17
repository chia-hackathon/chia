# Docker images

The loop runs its Chisel elaboration and Verilator build inside
`chia-chisel-build-aether-cosim:local`, which is **not published** — build it
locally from the Dockerfiles here. Each one is a thin layer (seconds to build);
the expensive work is the Verilator elaboration that happens later, inside the
container, when a config is first built.

## Layer chain

```
ghcr.io/ucb-bar/chia-chisel-build:latest      (upstream CHIA image)
  └── CosimFixDockerfile     -> chia-chisel-build-cosimfix:local
        (Bug 1: debug_rob DPI outputs)
        └── AetherCosimDockerfile -> chia-chisel-build-aether-cosim:local
              (Bug 2: vectorized DebugROB DPI for multi-issue Shuttle,
               cospike custom-extension support, Rocket+Gemmini cosim config,
               and the Saturn+Gemmini co-existence configs)
```

`GemminiCosimDockerfile` is the earlier sibling layer that
`AetherCosimDockerfile` merges in; it is kept for provenance and is not needed
if you build the chain above.

## Build

```bash
docker build -f CosimFixDockerfile     -t chia-chisel-build-cosimfix:local     .
docker build -f AetherCosimDockerfile  -t chia-chisel-build-aether-cosim:local .
```

The image name is what `cluster/cluster.yaml` refers to for the `build` node
type; change it in both places if you retag.

## What the layers patch, and why

Three RTL/co-simulation bugs block a Saturn + Gemmini Shuttle tile from running
under cospike lockstep. The patches are applied by the `fix_*.py` scripts and
`debug_rob-pop-init.patch` in this directory, each of which re-verifies its edit
(grep plus `-fsyntax-only`) so a silent upstream change fails the build instead
of the simulation:

| file | fixes |
|---|---|
| `fix_debug_rob.py` | `debug_rob` DPI outputs missing from the generated harness |
| `fix_debug_rob_vec.py` | one shared deque interleaved by a multi-issue core's N push/pop blackboxes |
| `fix_cospike_extension.py` | cospike rejecting a custom ISA extension string |
| `debug_rob-pop-init.patch` | uninitialised pop in the DebugROB blackbox |

## Config generation

`add_coexist_config.py` writes the `GENV256D128GemminiShuttleConfig` and
`...CosimConfig` fragments into the container's Chipyard checkout;
`add_gemmini_cosim_config.py` is the Rocket+Gemmini precursor. `loop/hwconfig.py`
renders the same text, so the two must stay in sync — the loop's digest covers
the rendered header, not these scripts.

## Building the simulators by hand

`build_all.sh` runs inside the build container and elaborates both configs,
writing per-config logs and a `build.summary` to the mounted `/work-out`. The
loop does this itself through `nodes.build_simulator()`; the script is here for
reproducing a build outside the loop.
