Your Spike model just failed to build, or a test program disagreed with
itself while running on it.

Start with `read_status`: results are grouped by `EMUL_C`, `LAMBDA`, `SEW`
and `LMUL`, and the pattern is usually the diagnosis. A fault confined to one
`EMUL_C` points at accumulator group addressing; one confined to a single
`SEW` points at element width; one that appears only at `LMUL > 1` points at
the K dimension; one that appears everywhere points at the geometry
derivation itself.

## Reading a failure

`TITAN FAIL row=<i> col=<j>` means the two paths in one program computed
different values for that element of C. Both ran on your Spike. The other
path is plain RVV 1.0, which Spike has supported for years and which is not
under test. **Your model is what is wrong.** Work backwards from the named
element: which A row and which B row feed it, where the geometry rules say
those live in the register groups, and what your code does with them.

A high skip rate is its own kind of failure. If your model is clamping
`lambda` for geometries the spec permits, you are declining to be tested
rather than passing.

## Hard rules

- Form a hypothesis about a specific rule in the spec before editing, and say
  which one. Re-read that section. This is a transcription task: a wrong
  result almost always means a misread line, not a subtle bug.
- Do not read the hardware implementation under `generators/saturn/`, and do
  not read the Titan reference model or test generators. Your independence
  from them is the only thing that makes your model worth having.
- Do not weaken, skip or special-case a test.
- Do not build or run anything yourself.

Spike's source is at `${SPIKE_SRC_PATH}`.
