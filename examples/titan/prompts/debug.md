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
- Do not spawn sub-agents. Do the work in this session.

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
