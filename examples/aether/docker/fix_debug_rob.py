#!/usr/bin/env python3
"""
Bug 1 fix for chipyard cosim flakiness.

generators/rocket-chip/src/main/resources/csrc/debug_rob.cc :: debug_rob_pop_trace()
returns early (ROB missing / empty / still waiting on writeback) after writing only
*trace_valid = 0.  All other DPI output pointers are left untouched.  Verilator's
generated DPI wrapper allocates uninitialized stack temporaries for those outputs and
copies them back into the RTL unconditionally, so trace_0_cause / _exception /
_interrupt / _iaddr pick up stack garbage on every non-retiring cycle.
testchipip's cospike.v gates on

    if (trace_0_valid || trace_0_exception || trace_0_cause) begin

(no valid gate), so garbage in cause spuriously fires cospike -> "Unknown interrupt
<garbage>" / bogus PC mismatches, non-deterministically.

Fix: zero every output argument at the top of debug_rob_pop_trace.
Deterministic exact-string replacement (no line numbers involved).
"""
import sys

path = sys.argv[1]
src = open(path).read()

ANCHOR = """  *trace_valid = 0;
  if (debug_robs.find(hartid) == debug_robs.end()) return;"""

NEW = """  *trace_valid = 0;
  /* Bug 1 fix: initialize *all* DPI outputs. Verilator copies these back into
     the RTL unconditionally, and testchipip/cospike.v triggers on
     (trace_0_valid || trace_0_exception || trace_0_cause) without a valid gate,
     so leaving them as uninitialized stack garbage causes nondeterministic
     spurious cospike traps. */
  *trace_iaddr = 0;
  *trace_insn = 0;
  *trace_priv = 0;
  *trace_exception = 0;
  *trace_interrupt = 0;
  *trace_cause = 0;
  *trace_tval = 0;
  memset(trace_wdata, 0, WDATA_BYTES);
  if (debug_robs.find(hartid) == debug_robs.end()) return;"""

n = src.count(ANCHOR)
if n != 1:
    sys.exit("FATAL: expected exactly 1 anchor match in %s, found %d" % (path, n))

open(path, "w").write(src.replace(ANCHOR, NEW))
print("PATCHED %s" % path)
