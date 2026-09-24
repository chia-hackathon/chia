The build or the tests just failed. Find the root cause and fix it.

Start with `read_status`: it groups results by `EMUL_C`, `LAMBDA`, `SEW` and
`LMUL`. The pattern is the diagnosis. A fault confined to one `EMUL_C` points
at accumulator group addressing; one confined to a single `SEW` points at
element-width handling; one that appears only at `LMUL > 1` points at the K
dimension; one that appears everywhere points at something structural.

## Hard rules

- **Form a hypothesis about a specific signal, handshake or state element
  before you edit.** Say what you think is wrong and why the evidence fits.
  Changing things to see what happens costs an iteration each time.
- Do not disable, gate out, or detach any part of the matrix path to make a
  build succeed. A design that elaborates because the unit is unreachable has
  not made progress.
- Do not revert wholesale. If a previous change was wrong, say which and why.
- Do not weaken, skip or special-case a test. If you think a test is wrong,
  say so via `finish` and stop; do not edit it.
- Test your fix with `run_directed_start("failing")` before you finish, and
  call `run_directed_wait` until it answers — never end a turn with a job
  still running, or with instrumentation left in the tree: the loop grades
  the tree exactly as you leave it. Do not run sbt, make or verilator by
  hand; `run_directed_start` is that build.
- If the failure you were given is an RVV regression, reproduce it yourself
  with `run_rvv_start("failing")` → `run_rvv_wait` before you finish.
  One cosim run is not a verdict — the trace bridge flips ~12% of runs on an
  unchanged binary (`titan_runs/nondet/`). Use `reps=3` and judge by majority
  before an RVV failure drives an RTL change; one pass does not clear a test.
- Do not spawn sub-agents. Do the work in this session.
- **Never make a structural check pass by changing what it counts.** An
  earlier iteration kept `fus.size` constant by merging two FU factories so
  that the check guarding the issue-path shape stopped firing. That hides
  the problem. If a structural check fires, either the structure is wrong or
  the check is wrong -- say which, and fix that one. Narrowing a test's
  coverage, relaxing a tolerance or dropping a geometry to turn red green
  falls under the same rule.

## Four failures that are not what they look like

- **Floating-point RVV tests failing after a matrix change.** Check the FU
  lists first. `MatrixFPMultiplyFactory()` placed in `integerMatrix` rather
  than its own `fpMatrix` once broke 23 `.vf` tests in the Stage 2
  regression and took three iterations to localise, because nothing in the
  symptom pointed at the matrix unit. Round six accumulates in floating
  point and lands in the same place. Look there before you debug arithmetic.

- **Every version silent, all stopping at the same simulated time.** That is
  the `+max-cycles` budget, not your RTL and not the test. A genuine hang or
  functional failure does not line up to the same cycle count across
  unrelated builds. Compare stop times before you look for a bug.

- **A single lockstep or cosim divergence.** The trace bridge is
  run-to-run nondeterministic on an unchanged binary (~12%,
  `titan_runs/nondet/`). One run is never a verdict, in either direction:
  use `reps=3` and judge by majority before it drives an RTL change, and do
  not treat one pass as clearing a test either.

- **Numbers that are wrong in a plausible, consistent way.** Before blaming
  the datapath, check whether the *model* of a format is right. The adoc
  does not restate the OFP8/OFP4 bit encodings -- spec 1910-1913 is a
  normative reference to OCP Microscaling Formats (MX) v1.0 -- so E4M3
  having no infinities, E5M2 being IEEE-shaped and E2M1 having neither NaN
  nor infinity all have to come from OCP (OFP8 v1.0 text:
  `specs/ime/ocp-ofp8-v1.0.txt`; OCP MX / E2M1 is not available). A wrong
  special-value rule produces exactly this symptom. (E8M0 *is* restated in full at spec
  1990-1993, and note it has no zero, infinity or subnormal encoding: byte
  0x00 is the ordinary finite value 2^-127, and only 0xFF is NaN.)

## Reading a directed failure

The message gives you one line per failing test — geometry, the coordinate of
the first disagreeing C element, and the two values there — plus, for the
first few, the differing elements and a dump of the physical M x M C tile from
both paths (`TITAN CDUMP` is yours, `TITAN CREF` is the reference).

Read the dumps before you theorise. Zeros where the reference has data mean
the accumulator was never written; the right values in the wrong places mean
the tile index is wrong; one bad row out of eight means one register of the C
group is. A coordinate alone cannot tell those apart, and five iterations were
spent trying.

Anything not quoted is in `${AGENT_LOG_DIR}/<run>/iter<N>/`, in full, on the
machine your bash tool runs on: `sim_<test>.log` per program, plus the build
output, the status file and the summary json. Go and read them. Both paths computed the same thing from the
same data; the RVV 1.0 path is not under test. Work backwards from the named
element: which A row and which B row feed it, which registers those live in
under that geometry, and which stage of your datapath touches them.

A SKIP is not a failure, but the rule is narrower than "the DUT said a
different lambda". `vtype.lambda` is WARL: the legal answer is the largest
supported nonzero lambda **at or below** the request, and only if nothing that
small is supported may the implementation fall back to its smallest value.
Answering with a *larger* lambda while smaller supported values exist is a
`bad_geometry` failure, reported as one, and points at the vsetvl lambda
selection rather than at any datapath.

Every skip is also a geometry nobody tested. Widening the supported set is
part of the job, not a bonus.

Generator sources are under `${CHIPYARD_SRC_PATH}`.

An `ime_cl_` or `ime_clq_` failure means something specific: those programs load the
C tile with `vmtl.v`, run one multiply-accumulate, and then read the C register group
back with an ordinary `vse<SEW>.v` instead of `vmts.v`. They are therefore the only
tests that observe the *register-side* C index directly. A failure there is a
`mat_C_idx` bug -- the spec routes C(i,j) through `tile_reg_idx`, and a linear
`i*N_max + j` gives its exact transpose whenever EMUL_C > 1. It is not a datapath,
accumulator or tile-transfer fault, and the pair tests cannot see it because they
write and read C through the same permutation.
