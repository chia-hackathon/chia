# Engineering notes

The measurements, the traps, and the things that only showed up by running the
thing. Split out of `README.md`, which describes the architecture; this file is
the record behind it. Nothing here is required reading to use the loop.

## Contents

* [Running it on GCP](#running-it-on-gcp) — the cluster that adds cloud Tier-0/1 capacity, and the global quota that shapes it
* [Making the three tiers comparable](#making-the-three-tiers-comparable) — the trace converter, the single-source port, and the port-fidelity gate that was wrong twice
* [Design decisions worth knowing](#design-decisions-worth-knowing) — why fitness is VFS alone, why the archive bins on cost
* [Status](#status) — what is verified, what is not yet run
* [There is no honest window, so there is no window](#there-is-no-honest-window-so-there-is-no-window) — why the championship's 40M is an upper bound, not a measurement window
* [What a full pass costs](#what-a-full-pass-costs) — throughput, scheduling, and two sampling mistakes
* [What the loop has already caught](#what-the-loop-has-already-caught) — bugs and agent failures invisible to any check short of running it

## Running it on GCP

**`cluster_gcp.yaml` is not included in this repository.** It carries
site-specific GCP settings -- project, zone, bucket, image names -- and is left
out deliberately; the section below describes what it declared so the config
can be reconstructed. Everything else here runs without it.

`cluster_gcp.yaml` adds four `t2d-standard-8` VMs in us-west1 to the Tier-0
and Tier-1 pools and leaves Tier 2 and the LLM pool on the local machine,
which keeps its own 16 Tier-0 slots. The experiment is unchanged -- same 168
traces, same 126/42 split, same run-to-the-end windows -- so results from the
two configs are comparable and the archive schema is the same.

```bash
export THIS_MACHINE=$(hostname -I | awk '{print $1}')
export TS_AUTHKEY=tskey-auth-...          # reusable + ephemeral
export BPE_GCS_BUCKET=<your-gcs-bucket>
gcloud auth application-default login     # once

./gcp/stage.sh                            # once, ~45 min, ~$0.30
gcloud compute instances list             # MUST be empty; see the quota note
python -m chia.cli.main up examples/bp_evolve/cluster_gcp.yaml -y
RAY_ADDRESS=$THIS_MACHINE:6379 ray status # cbp_ng must read 48, not 16
~/bp_evolve_data/launch_full_gcp.sh

python -m chia.cli.main down examples/bp_evolve/cluster_gcp.yaml -y
gcloud compute instances list             # always confirm; VMs bill by the second
```

**The quota that decides the shape of this is global, and `gcloud compute
regions describe` does not show it.** `CPUS_ALL_REGIONS` is a project-wide cap
on concurrently running vCPUs across every region, and here it is **32**:

```bash
gcloud compute project-info describe --project=<p> \
  --flatten="quotas[]" --format="table(quotas.metric,quotas.limit)"
```

The regional numbers are real (us-west1: `CPUS` 100, `T2D_CPUS` 100,
`INSTANCES` 24) but they are ceilings *underneath* that one, so they authorise
nothing on their own. `PREEMPTIBLE_CPUS` is 0, which means `spot: true` is not
a cheaper option that degrades under pressure -- the launch is rejected.

32 vCPU is 32 physical cores as either 2 × `t2d-standard-16` or 4 ×
`t2d-standard-8`; the config takes the four smaller ones, since the price is
linear in vCPU and four VMs are four independent memory subsystems, which is
what a memory-bound simulator wants. t2d rather than n2 because t2d has no SMT
-- `t2d-standard-8` is 8 real cores where `n2-standard-8` is 4 with SMT. **The
config uses the entire 32 vCPU budget with no headroom**, so any leftover
instance (`stage.sh`'s throwaway `e2-standard-8`, say) makes the fourth VM fail
to create.

`CPUS_ALL_REGIONS` is adjustable and this project is eligible for an increase
(`gcloud alpha quotas info describe CPUS-ALL-REGIONS-per-project
--service=compute.googleapis.com` → `isEligible=True`). Raising it to 128 is
the single change that would make this worth much more than it currently is.

**Only Tiers 0 and 1 move, and that is not a hedge.** Tier 0 runs 378
whole-trace simulations a generation and Tier 2 runs four gem5 points on one
promoted elite. Moving Tier 2 would buy a couple of minutes and cost a 15 GB
image and a forty-minute `scons` on every bring-up; `bpe-gem5:local` is already
built here. `--arm offline` never contacts the LLM pool at all, so it stays
where its credentials are. The local `cbp_ng` pool stays too -- at this cluster
size the desktop is roughly 40% of total throughput, not a rounding error.

**Nothing is pushed to a registry, and nothing large is uploaded.** Three
things have to reach the VMs and each takes the cheap path:

| | how | why not the obvious way |
|---|---|---|
| images (8.9 GB) | built on the VM | all but ~400 MB is a public Ray base the VM pulls at in-cloud bandwidth; `bpe-cbpng` is `rayproject/ray:2.54.0-cpu` plus gcc 12, and `champsim-bp-test` is this repo's `ChampSimDockerfile` |
| traces (19.5 GB) | staged to GCS by a throwaway VM that re-fetches the source tarball | `download.log` clocks that tarball at 2.05 MB/s **down**; a home link is slower up, and the workers' in-region reads out of the bucket are free |
| build context (6 MB) | through the same bucket | it cannot be cloned: `examples/bp_evolve/` is untracked, `ChampSimDockerfile` is locally modified, and cbp-ng is a private checkout |

The ChampSim traces are *rebuilt* on the staging VM rather than copied, using
this repo's `cbp2champsim` with no instruction cap. Tiers 0 and 1 must see the
same branches or the port self-check measures the workload instead of the
predictor, so the converter is pinned rather than assumed.

**Expect about 2.4x, and time generation 1 before believing it.** Adding 32
t2d cores to this desktop's ten is not a 4x speedup. t2d is slower per core --
a sibling project on this account measured its single-thread at 2.2x slower
than this i5-12600K, turning HARCOM's 278 kIPS solo here into roughly 126 kIPS
there -- and HARCOM does not scale linearly in slots, since sixteen of them
here yield 122 kIPS each against 278 solo. Four VMs × eight slots × ~126 kIPS ×
~0.7 for contention is ~2.8 MIPS of cloud on top of the 1.96 MIPS measured
locally:

| | local, measured | local + GCP, estimated |
|---|---|---|
| aggregate throughput | 1.96 MIPS | ~4.8 MIPS |
| Tier-0 wall per generation | 99 min | ~41 min |
| full generation | 112 min | ~54 min |
| ten generations | ~19 h | ~9 h |
| compute cost | — | ~$13 at ~$1.4/hr |

Promotion is local and does not shrink, which is why the generation figure
improves by less than the Tier-0 one. If the per-slot rate comes in well under
126 kIPS the slots are contending for memory bandwidth, and *fewer* of them
will finish the set sooner -- lower `cbp_ng` in the config rather than trusting
the estimate.

One thing the move does **not** fix: Tier 1 still runs ChampSim one trace at a
time. The local run showed `1.0/80.0 CPU, 1.0 used of 1.0 reserved in placement
groups` while `champsim: 8` sat free, so the pool was never the constraint --
the placement group was. `cluster_gcp.yaml` declares `champsim: 8` on all four
VMs, which is necessary for that fix and not sufficient for it.

`launch_full_gcp.sh` lowers `BPE_CBP_RUN_FLOOR_KIPS` from 40 to 20. The Tier-0
run timeout is sized per trace from its instruction count against that floor,
and the floor is a property of the machine rather than of the trace: t2d being
~2.2x slower per core means the same traces need a proportionally longer budget
to keep the same margin. Getting this wrong is not subtle -- a flat 1800s
timeout killed the six traces past 110M instructions on every variant, and since
one failed trace fails the variant, every variant failed while still costing a
full hour of compute per generation.

## Making the three tiers comparable

Two things had to exist before Tiers 1 and 2 could say anything, and neither is
obvious from the figure.

**The same branches.** CBP-NG and ChampSim read different trace formats, so one
of them has to be translated. `tools/cbp2champsim.cpp` translates the *trace*
rather than the predictor, because a trace converter can be checked against
ground truth: cbp-ng reports its own instruction and branch totals for every
trace it runs, and `convert_traces.py --verify` compares them. On
`gcc_test_trace.gz` the instruction and branch counts match exactly at every
warmup point tried; the conditional-branch count is one lower in cbp-ng, which
is the branch straddling its warmup boundary and not a desynchronised reader.

The converter keeps what cbp-ng's own reader throws away. `trace_reader.hpp`
reads the effective address of every load and store and discards it, because a
branch-prediction harness has no use for it — ChampSim does, or its caches see
no memory traffic at all. It also has to choose register names carefully:
ChampSim infers the *kind* of a branch from which registers an instruction
reads and writes, so a wrong mapping turns a return into a conditional branch
and changes which structure is even being asked.

**The same predictor.** `ports/tage_core.h.in` holds the algorithm — table
geometry, geometric history lengths, the folded-history hash, tag construction,
provider and alternate selection, the meta counter, allocation with u-bit
clearing and periodic reset. The ChampSim and gem5 files are adapters over it,
about forty lines each. Writing TAGE twice would have made every cross-tier
disagreement ambiguous between "the cost models differ" and "the two ports
differ".

One thing the port cannot reproduce: CBP-NG predicts a whole cache line per
cycle and advances its global history once per *prediction block*. Neither
ChampSim nor gem5 has a block hook — both ask per branch. Identical predictors
fed histories that advance at different rates compute different indices, so the
ports do not reproduce Tier-0 MPKI exactly.

![How the port-fidelity check integrates into the loop](docs/port_fidelity.png)

*Source: [`docs/port_fidelity.dot`](docs/port_fidelity.dot) — regenerate with
`dot -Tpng -Gdpi=140 docs/port_fidelity.dot -o docs/port_fidelity.png`.*

**How large is that?** The obvious way to find out — compare Tier-1 MPKI to
Tier-0 MPKI — gave 0.702 on the seed and was wrong, in a way worth recording
because the same trap is waiting in any two-simulator comparison. It compares
two *harnesses* along with the two predictors, and here the harnesses dominated:
ChampSim reopens a trace it runs off the end of, so a 40M-instruction Tier-1
window over a 3.0M-instruction trace ran the same code thirteen times, against
Tier 0's single pass. The ratio it produced was not even constant — 0.808,
0.823 and 0.480 across three designs in one sweep — because a design that
converges faster gains more from seeing the program again. A calibration
constant that depends on the design being calibrated is not one.

`tools/port_selfcheck.py` asks the narrow question instead: render
`ports/tage_core.h.in` at a parameter set, drive it over Tier 0's own traces
and window with `tools/tage_selfcheck.cpp`, and print conditional MPKI. No
simulator on either side, about a second per trace, and what is left between it
and Tier-0 MPKI is the interface and nothing else. Across four designs spanning
6.2 to 14.4 Tier-0 MPKI:

| design | params | Tier-0 MPKI | ported algorithm | ratio |
| --- | --- | --- | --- | --- |
| seed | `6,8,11,12,11,100,14,6` | 6.198 | 5.965 | 0.962 |
| gen001_2 | `6,9,11,12,9,100,14,6` | 13.930 | 13.170 | 0.945 |
| gen002_0 | `7,8,11,12,9,100,14,6` | 14.363 | 13.587 | 0.946 |
| gen003_0 | `6,10,12,13,11,100,15,6` | 6.111 | 5.984 | 0.979 |

**2–6%, one-sided, and stable across a 2.3× spread in MPKI** — that is the
real cost of the interface, and the port is faithful.

The clearest evidence is what happens when the two gates are run over the same
sweep. Every design that reached Tier 1 in `out_run1`, judged both ways:

| variant | old verdict | Tier-1/Tier-0 | self-check/Tier-0 | new verdict |
| --- | --- | --- | --- | --- |
| gen001_2 | no_improvement | 0.808 | 0.945 | pass |
| gen002_0 | no_improvement | 0.823 | 0.946 | pass |
| gen003_0 | **port_mismatch** | 0.480 | 0.979 | **pass** |
| gen004_1 | no_improvement | 0.785 | 0.968 | pass |
| gen005_0 | no_improvement | 0.812 | 0.951 | pass |
| gen006_0 | **port_mismatch** | 0.723 | 0.961 | **pass** |

The old column spans 1.7×; the new one spans 1.04×. And the second design the
old gate rejected, `gen006_0`, was the first variant in the whole sweep to beat
the seed (VFS 0.7683 against 0.7653). **A fidelity gate that measures the
harness does not merely add noise — it preferentially discards whatever the
search just found**, because a design that changes the predictor is also the
design most likely to move a ratio that was never about the predictor. The per-branch history is
the *more* accurate of the two, which is what you would expect: the block
version indexes with a history one block stale.

So the gate holds each port to `PORT_SELFCHECK_TOLERANCE` (10%) against its own
self-check rather than to a calibrated Tier-1 ratio. The ratio path survives
only for designs with no template arguments to re-render — the LLM arm writes
HARCOM directly — and its docstring says not to trust it. `TIER1_SIM_INSTRUCTIONS`
now sizes the Tier-1 window to the trace, and `run_fn_tier1` reports it loudly
when ChampSim wraps anyway.

**What the port deliberately drops.** The seed is a two-level design: a 1-cycle
gshare (P1, sized by `LOGP1`/`GHIST1`) that a 2-cycle TAGE (P2) overrides a
cycle later. `cbp.hpp` counts a misprediction only when P2 is wrong, and P1
shares no state with P2, so dropping P1 is exactly right for MPKI — and exactly
wrong for latency and energy, which Tier 0 charges for it. Neither
`predict_branch` nor gem5's `lookup` can express "answer now, correct yourself
one cycle later", so there is nowhere to put it. The consequence: two designs
differing only in `LOGP1` or `GHIST1` are distinct at Tier 0 and identical at
Tiers 1 and 2. `agents.offline_port` names the dropped parameters in its
rationale and returns them under `unported` rather than hiding it.

## Design decisions worth knowing

**Fitness is VFS at the shipped depth, and only that.** Folding the depth sweep
into fitness would answer the loop's own question inside the fitness function.
The search optimises the score CBP-NG actually ships; whether that score was
worth optimising is the *finding*, measured separately from the depth table the
same counters give for free.

**The archive bins on cost, not on score.** Regions are (energy band, P1
latency, P2 latency) — the whole of CBP-NG's cost model, since it imposes no
storage budget. A leaderboard on VFS would converge on one enormous TAGE,
because with no storage budget the cheapest accuracy is always a bigger table.
The archive keeps the slow-and-accurate corner alive long enough to be tested
at depth sixteen, where it turns out to win.

**Held-out traces never enter the inner loop.** The subset the agent's feedback
is computed from and the set a final claim is scored on are disjoint, chosen
once from a seeded shuffle. A design that wins on the traces its own feedback
came from has not been shown to generalise.

**A design that reaches around HARCOM is withdrawn, not repaired.**
`harcom_superuser`, a `#pragma` that silences the warnings the design is judged
under, `reinterpret_cast` on an opaque type — these still compile, still run,
and still produce a VFS score. It is just no longer a score about hardware.
Offering a repair round would say the rule is negotiable.

**The port is part of the work.** A HARCOM predictor is called for every
instruction with a block-based, two-level, timed interface; ChampSim and gem5
call a predictor once per branch. Every promoted design must reproduce its
Tier-0 MPKI on the Tier-1 branch stream before any cross-tier claim is made —
Tiers 0 and 1 share a trace, so a gap there is the translation, not the cost
model, and a mistranslation would look exactly like the disagreement the loop
exists to measure. gem5 executes binaries rather than traces, so Tier 2 changes
the workload as well and is reported as corroboration, not as the same
measurement.

## Status

**Working, on a real cluster**, against real `cbp-ng`, ChampSim and gem5
checkouts. `chia up examples/bp_evolve/cluster.yaml` brings up four pools on one
machine and the loop runs all three tiers through them.

* `vfs.py --selftest` reproduces the published depth table exactly — the six
  numbers quoted in the proposal all check out.
* The lint gate passes all ten shipped predictors with zero findings, and
  catches branch-on-value, `harcom_superuser`, warning-suppression pragmas,
  `stdout` writes, `fopen` and `reinterpret_cast` on synthetic cases. Rules run
  against source with comments and string literals blanked, so a rule name in a
  comment does not trip itself.
* **Tier 0** builds a variant in ~8 s and scores it. Variant selection is by
  `-include` + `-DPREDICTOR=`, so two variants build concurrently in one
  checkout without racing on a shared file, and the build leaves no residue.
* **The trace converter** reproduces cbp-ng's own instruction and branch counts
  exactly (see above), so Tiers 0 and 1 are driven from the same branch stream.
* **Tier 1** ports the design, builds a ChampSim binary with it and runs:
  conditional MPKI 4.336 against Tier-0's 6.175, and **140.6 instructions in
  the ROB at a mispredict where CBP-NG charges a flat nine-stage penalty**.
  That gap is the tier's whole reason to exist.
* **Tier 2** installs the design into gem5 as a SimObject, rebuilds
  incrementally and sweeps the pipeline depth CBP-NG holds fixed. On `bsearch`,
  same predictor, same program:

  | distance to execute | 6 | 9 | 16 | 25 |
  |---|---|---|---|---|
  | gem5 IPC | 0.553 | 0.411 | 0.258 | 0.186 |
  | conditional mispredicts | 117,247 | 117,121 | 117,542 | 117,345 |

  The predictor is exactly as accurate at every depth and costs 3× as much at
  25 stages as at 6. Nothing in a score computed at a fixed nine stages can say
  that.
* The offline arm completes a multi-generation sweep: distinct designs land in
  distinct archive cells, coverage grows, and the lineage DB records every
  variant with its parent, its template arguments, its full depth sweep and its
  cross-tier tau. `--score-held-out` rebuilds the archived front from its own
  recorded source and re-runs it.
* The params arm runs the same cascade end to end with a model choosing the
  parameters, and the model is demonstrably reading the archive rather than
  perturbing at random. Its recorded rationales name the cell they are aiming
  at and cite the previous generation's measured result to get there -- one
  proposal held the second-level configuration fixed to preserve `P2=2` while
  pushing `LOGP1` alone across the `P1` threshold; the next observed that a
  design with `NUMG=5, LOGG=10` had still come out at `P2=2` and dropped every
  parameter to its floor. It also leaves the box the RNG never left: the offline
  arm's 204 designs used nine `GHIST` values between 95 and 104, and the params
  arm reached 40, 60, 200, 250, 300 and 400 within two generations.

**Not yet run:**

* A sweep on all 168 official traces. **The traces are now installed and the
  code runs them; the sweep itself is what is outstanding.** All 168 are in
  place in both formats -- 168 CBP-NG `.gz` with 168 distinct md5s, and 168
  ChampSim conversions whose instruction counts match `trace_lengths.json`
  entry for entry -- and `launch_full.sh` is the ten-generation run over them.

  Every number in the sections above predates that and comes from a single
  2M-instruction trace (`gcc_test_trace.gz`) copied six times to fill a trace
  directory -- six byte-identical files, md5 `44a8275328...`, so the
  "unweighted mean over four traces" was one trace's number stated four times.
  Read those results as mechanism checks, not as claims about branch
  prediction.

### There is no honest window, so there is no window

The championship runs `./cbp <trace> test 1000000 40000000` -- warm up 1M
instructions, then measure 40M. On the shipped test trace that second number is
inert: the trace holds 3,005,000 instructions, `cbp.hpp` stops at end of file,
and the run measures 2,004,999. ChampSim, handed the same 40M, does *not* stop
at end of file -- it reopens the trace and keeps going, 13 times over. Tier 1
was scoring a predictor that had seen the program thirteen times against Tier
0's single pass, and that, not any property of a predictor, is where the
"calibrated port ratio" came from.

Measuring all 168 official traces settles what 40M actually means:

| | instructions |
| --- | --- |
| shortest trace | 13,020,331 |
| median | 24,282,226 |
| 75th percentile | 39,999,984 |
| longest | 130,000,052 |
| total across 168 | 4,989,128,151 |
| **traces reaching 41M (1M warmup + 40M)** | **15** |

The 75th percentile sitting one instruction short of 40M is the tell: **most of
the set is capped at ~40M by construction.** So the championship's `40000000`
is an upper bound meaning "warm up, then run to the end," not a measurement
window -- and a literal 40M measurement is available on only 15 of 168 traces.

An earlier pass took the other branch: keep the 40M window and install only
those 15. That is a defensible experiment and a badly biased sample -- those
traces qualify *because* they are the longest-running programs in the set. The
current setup keeps all 168 and drops the window instead, which is what the
championship number meant in the first place. There is no single window that is
honest across a 13.0M-to-130.0M spread: any value either wraps the short traces
or truncates the long ones, and neither failure shows up in the MPKI.

The two tiers reach "to the end" differently, because they have to:

* **Tier 0** is given `BPE_SIM_INSTRUCTIONS = 2e9`, past the longest trace, so
  `cbp.hpp`'s `while (!warmed_up || ninstr < measurement_instructions)` is left
  by `out_of_instructions` at EOF rather than by the counter. The window is
  inert on purpose.
* **Tier 1** cannot use the same trick: ChampSim reopens the file. Its window is
  therefore per-trace, `tier1_window_for()` returning
  `length - warmup - TIER1_TAIL_MARGIN` from `trace_lengths.json`, with 100k of
  slack because ChampSim fetches past the instruction it is retiring and a
  window set to exactly `length - warmup` can still wrap on the last few. A
  trace whose length is not in the manifest falls back to the global constant
  *and says so* in `run_diagnostics`, naming the trace -- a fidelity check that
  silently guessed would be worse than one that admits it does not know.

`trace_lengths.json` is not a probe of the CBP-NG files; it is the converter's
own output count for each ChampSim trace, which is the number Tier 1's window is
actually compared against. All 168 agree.

### What a full pass costs

HARCOM's rate is per *prediction*, so it varies by branch density rather than
by trace length -- a factor of 2.5 across this set:

| trace | solo | |
| --- | --- | --- |
| `int_32_trace` | 317 kIPS | |
| `502-gcc-all_16112_trace` | 228 kIPS | |
| `compress_44_trace` | 124 kIPS | under 67 contended sixteen ways |

Plan with the aggregate over a whole pass, not with a per-trace rate. Measured
here:

| | |
| --- | --- |
| one variant alone (3.88G instructions) | 31 min, **2.09 MIPS** |
| a generation of three, overlapped | 99 min Tier 0, **1.96 MIPS** |
| the same generation incl. promotion | 112 min |

Sixteen cores buy about seven, not sixteen: HARCOM is memory-bound and the slots
contend. Ten generations is therefore **~19 hours**.

An earlier version of this section quoted 187 kIPS per slot and 3.00 MIPS, from
timing the first sixteen traces alphabetically. That sample missed the heavy
`compress_*` traces and was 50% optimistic -- the same sampling mistake, in
miniature, that the old 15-trace subset made. Measure over a whole pass.

That is the price of the two fixes above, and it is why the loop had to learn to
overlap:

| | before | now |
| --- | --- | --- |
| traces per variant | 8 (of 15 installed) | 126 (of 168) |
| instructions per variant | 0.19G | 3.88G |
| variants per generation | 3, one after another | 3, concurrent |
| `cbp_ng` slots | 12 | 16 |
| Tier-0 wall per generation | ~4 min | ~99 min |

Two scheduling details matter more than the slot count once the traces are this
uneven. **Run timeouts are per trace.** A flat 1800s was fine when everything was
capped at 40M, but running to the end means a 120M trace at 67 kIPS needs over
half an hour; the flat number killed all six traces past 110M, on every variant,
and since one failed trace fails the variant the archive never grew past the
seed while each generation still cost a full hour. `cbp_run_timeout_for` sizes
each timeout from the trace's own length against a 40 kIPS floor -- well under
the worst measured rate, because a too-generous timeout idles one slot on a hung
run while a too-tight one produces nothing at all.

**Traces are dispatched longest first.** Ray hands out queued tasks in
submission order, so a 130M trace submitted late runs alone on one slot for its
last twenty minutes while fifteen sit idle. Longest-first lets the short traces
fill in behind it -- the standard longest-processing-time rule for minimising
makespan. Results are permuted back into the caller's order, since the per-trace
counters are the audit trail for the aggregate and a list that reorders itself by
length cannot be diffed against a previous run.

A generation now runs in three phases -- propose (sequential, so `rng` fixes the
run), build and score (concurrent), archive (sequential, in proposal order).
The middle phase is safe to overlap because `_evaluate_core` never reaches
`archive.add`: it touches the archive only through `result_mapper_fn` on a build
failure, which returns at the lint/build branch. Keeping the archiving on the
main thread in proposal order means which cells fill does not depend on which
thread finished first, so the seed still reproduces the run. The cost is that a
generation's variants are all proposed from the archive as it stood at the start
of it and cannot build on each other within the generation -- ordinary batched
MAP-Elites.

The port self-check was parallelised for the same reason, though it turned out to
be the cheap half: it sustains 6.7 MIPS per core -- roughly 36x HARCOM, since
there is no library between it and the rendered core -- so the whole 126-trace
inner set is about ten core-minutes, well under a minute at `--jobs 16`. It runs
`--jobs` traces at once and collects them in argument order, since the per-trace
lines are the audit trail for the aggregate.

It also had to be *fixed* before any of that mattered. The gate looked up each
Tier-0 counter line in a map of the driver's trace files, but the two sides used
different names: CBP-NG labels a run `web_114_trace` while the map was keyed
`web_114_trace.gz`, so no lookup ever hit. The gate reported "no CBP-NG traces on
the driver" and fell back to the weak Tier-1/Tier-0 ratio on every promotion --
the failure it exists to prevent, arriving quietly. Both sides now go through
`_trace_key`, which strips a compression suffix and is idempotent. It cannot use
`splitext`: trace names carry dots (`gmsh-5.4132_0_trace`), and the old
`split(".")[0]` collapsed the 168 traces into 155 labels, making 13 pairs of
distinct workloads indistinguishable in the counter lines.

* The LLM arm. `agents.py` follows [`../reveng`](../reveng)'s pattern, the four
  prompts are written, and the `llm` pool is up — but every result above comes
  from the offline arm, which mutates TAGE's template parameters and ports by
  rendering a template. It cannot find a new prediction algorithm; it only
  resizes the one it was given. **A sweep where the offline arm matches the LLM
  arm is not a sweep where the agent did well — it is one where the search
  space collapsed to table sizes.**
* Tier 2 on more than one design at a time. The gem5 grid is depths ×
  workloads, and with `TIER2_MAX_PROMOTED = 2` a generation is 24 gem5 runs
  against a pool of 4.

The held-out split is now a real one: `split_traces` holds out a seeded quarter
-- 42 of the 168 -- that never reaches feedback, leaving 126 to evolve on, and
`--score-held-out` re-runs the archived front against them afterwards. That is
what "use all 168" buys; every trace is used, and no trace is used for both.

The offline arm's depth sweep still reports tau = +1.000 everywhere: resizing
TAGE's tables does not produce the latency diversity that causes rank
inversions. That is a property of the arm, not of the traces, and it is
precisely why it is a control and not an experiment.

## What the loop has already caught

Every one of these was invisible to any check short of running the thing.

* **Every variant scored identically**, because `template_args` was computed and
  never reached the compiler. A smoke test that only asked "did it run" would
  have passed.
* **An archived elite was relabelled `no_improvement`**: the mapper runs again
  after Tier 1 fills in the ChampSim numbers, `Archive.add` compared a design
  against itself, and lost on a strict `>`.
* **`chia up` reported four healthy workers that were one container.** All four
  node types defaulted to the same container name on the same host, so ChampSim
  and gem5 tasks would have queued against a container with neither installed.
  Each node type now names its own.
* **The gem5 SimObject was declared twice**, because the installer's
  idempotency check matched on `class EvolvedBP(BranchPredictor):` and the base
  class had since changed to `ConditionalPredictor`. gem5 died at startup inside
  pybind with `error_already_set`, which says nothing about duplicate
  SimObjects. Idempotency checks have to be looser than the thing they guard.
* **Three separate host/worker path confusions**, all the same shape: a bind
  mount is `~/bp_evolve_data/cbp_traces` outside the container and
  `/home/ray/cbp-ng/traces` inside it, and the loop enumerates on the driver
  while simulating on the worker. Each failed as something that looked like a
  missing file rather than a path meant for the other machine.
* **A 5% absolute port-fidelity gate would have rejected a correct port**, and
  the fix for it was wrong twice over. The measured Tier-1/Tier-0 ratio was
  0.702; calibrating against it looked principled and was not, because the
  number was mostly ChampSim looping the trace thirteen times and it moved
  between 0.48 and 0.82 across designs. Measuring the port with no simulator on
  either side puts the real gap at 2–6%. Two lessons: a gate that compares two
  harnesses is not measuring what it names, and a "systematic" constant that
  has only ever been measured once is a guess.
* **The bounds agent read "never visited" as "explored and rejected."** Given
  only the archive — which holds elites, one per occupied cell — five
  cold-context calls on the same generation-39 front narrowed `GHIST` from
  `[40,400]` five times out of five, describing the rest as dead space. Across
  the 204 designs the sweep actually evaluated, `GHIST` had never left 95-104:
  100 appeared 43 times, and the nine values ever tried spanned nine of the
  range's 361. The agent was not wrong about the elites; it was answering a
  question the elites cannot answer. On the same evidence it also split 3/6
  between "every elite *uses* `LOGLB=7`, so lock it at `[7,7]`" and "every
  design is *pinned at the ceiling* 7, so raise it" — the same observation,
  two framings, in its own wording.

  Adding an occupancy table (all 204 designs, all eight parameters treated
  alike, no parameter named in prose) moved every one of those: `LOGLB`
  ceiling raised 3/6 → 5/5, `LOGLB` locked at 7 2/6 → 0/5, `GHIST` narrowed
  6/6 → 0/5. Three caveats belong with that result. The ceiling only ever moved
  to 8, never near the arithmetic cliff at 11 that `validate_bounds` guards.
  This was not a blind test — `LOGLB` was known to be the interesting parameter
  when the prompt was written — so both prompt versions and both sets of
  numbers have to be reported together. And the `GHIST` step size is a defect
  in the mutation operator that no prompt change fixes.
* **The held-out check passed, and passed too easily to mean much on its own.**
  The generation-39 front's seven elites re-scored on the 42 traces no feedback
  came from give Kendall tau `+1.000`: no pair reorders, at any depth from 3 to
  30. But the held-out penalty is nearly a constant — mean `-0.0287`, standard
  deviation `0.0019` — while the elites are spread over `0.2538` of VFS, so the
  designs are separated by 134x the jitter that would have to reorder them. With
  n=7 that ordering was never in much danger. The honest reading is **no
  evidence the search fitted its 126 traces**, not "generalises well" — and the
  fact that the two trace sets behave so alike suggests a random split does not
  make the held-out set an out-of-distribution test at all. Splitting by
  workload class would.
