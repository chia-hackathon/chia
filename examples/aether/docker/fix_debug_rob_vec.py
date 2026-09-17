#!/usr/bin/env python3
"""
Bug 2 fix for chipyard cosim on multi-issue (Shuttle) cores.

Root cause
----------
rocket-chip's DebugROB DPI is keyed only by hartid:

    std::map<int, debug_rob_t*> debug_robs;   // key = hartid

Shuttle instantiates one DebugROBPushTrace + one DebugROBPopTrace blackbox per
retire slot, all with the same hartid, so `retireWidth` independent
`always @(posedge clock)` / `always @(negedge clock)` blocks push into and pop
from *the same* std::deque.  Verilator does not guarantee any particular
evaluation order between those always blocks, so both the push order and the
pop order of the slots within a cycle are arbitrary.  In practice the order is
reversed, which makes cospike see instructions out of program order (the very
first bootrom instruction already mismatches: spike 10004 != DUT 10000).

Wrong fix (deliberately NOT done here)
--------------------------------------
Giving every slot its own deque (key = hartid*8+slot) breaks program order in a
subtler way: the DebugROB exists to hold instructions back until their
writeback data arrives, so a slot-1 instruction would overtake a slot-0
instruction that is still waiting on wb.

Fix
---
Vectorize the DPI: one blackbox per hart that takes/returns all retireWidth
slots at once.  Ordering is then decided entirely in C++ (slot 0 first), and
the "waiting for wb" stall applies to the whole group (if slot i's head of the
ROB is not ready, slots i+1.. emit nothing this cycle), which preserves program
order exactly.

The same is done for pushWb across the retire slots, because two instructions
retiring in the same cycle can write the same architectural register, and the
wb data is matched to traces by (tag, deque order).

The scalar DPI functions/blackboxes are left completely untouched, so Rocket
(retireWidth == 1) keeps its existing, working path.
"""
import sys, os

CHIPYARD = sys.argv[1] if len(sys.argv) > 1 else "/home/ray/chipyard"

MAXSLOTS = 8

def rd(p):
    with open(p) as f:
        return f.read()

def wr(p, s):
    with open(p, "w") as f:
        f.write(s)

def sub_once(path, old, new, what):
    src = rd(path)
    n = src.count(old)
    if n != 1:
        sys.exit("FATAL: %s: expected exactly 1 match for %s anchor, found %d" % (path, what, n))
    wr(path, src.replace(old, new))
    print("PATCHED %s (%s)" % (path, what))

# ---------------------------------------------------------------- debug_rob.cc
CC = os.path.join(CHIPYARD, "generators/rocket-chip/src/main/resources/csrc/debug_rob.cc")

CC_ADD = r'''

/* ==========================================================================
 * Bug 2 fix: vectorized DebugROB DPI for multi-retire cores (e.g. Shuttle).
 *
 * One call per hart per cycle handles all retire slots, so the intra-cycle
 * ordering of the slots is decided here (slot 0 first) instead of being left
 * to Verilator's arbitrary ordering of N identical always blocks.
 *
 * The scalar entry points above are untouched and still used by Rocket.
 * ========================================================================== */
#define DEBUG_ROB_MAX_SLOTS (8)
#define WDATA_WORDS (WDATA_BYTES / 8)

extern "C" void debug_rob_push_trace_vec(int hartid,
                                         int nslots,
                                         long long int should_wb,
                                         long long int has_wb,
                                         long long int* wb_tag,
                                         long long int trace_valid,
                                         long long int* trace_iaddr,
                                         long long int* trace_insn,
                                         long long int* trace_priv,
                                         long long int trace_exception,
                                         long long int trace_interrupt,
                                         long long int* trace_cause,
                                         long long int* trace_tval,
                                         long long int* trace_wdata) {
  if (debug_robs.find(hartid) == debug_robs.end())
    debug_robs[hartid] = new debug_rob_t;

  if (nslots > DEBUG_ROB_MAX_SLOTS) nslots = DEBUG_ROB_MAX_SLOTS;

  for (int i = 0; i < nslots; i++) {
    if (!((trace_valid >> i) & 1)) continue;
    tagged_traced_insn_t* insn = new tagged_traced_insn_t;
    insn->iaddr     = trace_iaddr[i];
    insn->insn      = trace_insn[i];
    insn->priv      = (uint8_t)trace_priv[i];
    insn->exception = (trace_exception >> i) & 1;
    insn->interrupt = (trace_interrupt >> i) & 1;
    insn->cause     = trace_cause[i];
    insn->tval      = trace_tval[i];
    insn->waiting   = ((should_wb >> i) & 1) && !((has_wb >> i) & 1);
    insn->wb_tag    = wb_tag[i];
    memcpy(insn->wdata, &trace_wdata[i * WDATA_WORDS], WDATA_BYTES);
    debug_robs[hartid]->rob.push_back(insn);
  }
}

extern "C" void debug_rob_push_wb_vec(int hartid,
                                      int nslots,
                                      long long int valid,
                                      long long int* wb_tag,
                                      long long int* wb_data) {
  if (debug_robs.find(hartid) == debug_robs.end())
    debug_robs[hartid] = new debug_rob_t;

  if (nslots > DEBUG_ROB_MAX_SLOTS) nslots = DEBUG_ROB_MAX_SLOTS;

  for (int i = 0; i < nslots; i++) {
    if (!((valid >> i) & 1)) continue;
    tagged_wb_data_t* data = new tagged_wb_data_t;
    data->tag = wb_tag[i];
    memcpy(data->data, &wb_data[i * WDATA_WORDS], WDATA_BYTES);
    debug_robs[hartid]->wb_datas.push_back(data);
  }
}

extern "C" void debug_rob_pop_trace_vec(int hartid,
                                        int nslots,
                                        long long int* trace_valid,
                                        long long int* trace_iaddr,
                                        long long int* trace_insn,
                                        long long int* trace_priv,
                                        long long int* trace_exception,
                                        long long int* trace_interrupt,
                                        long long int* trace_cause,
                                        long long int* trace_tval,
                                        long long int* trace_wdata) {
  /* Bug 1 fix applies here too: zero *every* DPI output, unconditionally. */
  *trace_valid     = 0;
  *trace_exception = 0;
  *trace_interrupt = 0;
  memset(trace_iaddr, 0, DEBUG_ROB_MAX_SLOTS * sizeof(long long int));
  memset(trace_insn,  0, DEBUG_ROB_MAX_SLOTS * sizeof(long long int));
  memset(trace_priv,  0, DEBUG_ROB_MAX_SLOTS * sizeof(long long int));
  memset(trace_cause, 0, DEBUG_ROB_MAX_SLOTS * sizeof(long long int));
  memset(trace_tval,  0, DEBUG_ROB_MAX_SLOTS * sizeof(long long int));
  memset(trace_wdata, 0, DEBUG_ROB_MAX_SLOTS * WDATA_BYTES);

  if (nslots > DEBUG_ROB_MAX_SLOTS) nslots = DEBUG_ROB_MAX_SLOTS;
  if (debug_robs.find(hartid) == debug_robs.end()) return;

  debug_rob_t* drob = debug_robs[hartid];
  std::deque<tagged_wb_data_t*> &wb_datas = drob->wb_datas;

  for (int i = 0; i < nslots; i++) {
    if (drob->rob.empty()) return;
    tagged_traced_insn_t* front = drob->rob.front();

    if (front->waiting) {
      for (auto it = wb_datas.begin(); it != wb_datas.end(); it++) {
        if ((*it)->tag == front->wb_tag) {
          memcpy(front->wdata, (*it)->data, WDATA_BYTES);
          front->waiting = false;
          delete (*it);
          wb_datas.erase(it);
          break;
        }
      }
    }
    /* Still waiting on writeback: stall the *whole* group.  Slots i..nslots-1
       stay invalid this cycle and are emitted later, in program order. */
    if (front->waiting) return;

    *trace_valid |= (1LL << i);
    trace_iaddr[i] = front->iaddr;
    trace_insn[i]  = front->insn;
    trace_priv[i]  = front->priv;
    if (front->exception) *trace_exception |= (1LL << i);
    if (front->interrupt) *trace_interrupt |= (1LL << i);
    trace_cause[i] = front->cause;
    trace_tval[i]  = front->tval;
    memcpy(&trace_wdata[i * WDATA_WORDS], front->wdata, WDATA_BYTES);
    drob->rob.pop_front();
    delete front;
  }
}
'''

src = rd(CC)
if "debug_rob_pop_trace_vec" in src:
    sys.exit("FATAL: %s already patched" % CC)
if "Bug 1 fix" not in src:
    sys.exit("FATAL: %s does not contain the Bug 1 fix -- wrong base image?" % CC)
wr(CC, src + CC_ADD)
print("PATCHED %s (vec DPI impl)" % CC)

# ----------------------------------------------------------------- debug_rob.v
V = os.path.join(CHIPYARD, "generators/rocket-chip/src/main/resources/vsrc/debug_rob.v")

V_ADD = r'''

// ===========================================================================
// Bug 2 fix: vectorized DebugROB blackboxes for multi-retire cores.
// All retire slots of one hart go through ONE blackbox instance, so the
// intra-cycle slot ordering is fixed by the C++ side rather than by the
// simulator's arbitrary scheduling of N identical always blocks.
// Ports are sized for DEBUG_ROB_MAX_SLOTS = 8; `nslots` says how many are live.
// ===========================================================================

import "DPI-C" function void debug_rob_push_trace_vec(input int     hartid,
                                                      input int     nslots,
                                                      input longint should_wb,
                                                      input longint has_wb,
                                                      input longint wb_tag[8],
                                                      input longint trace_valid,
                                                      input longint trace_iaddr[8],
                                                      input longint trace_insn[8],
                                                      input longint trace_priv[8],
                                                      input longint trace_exception,
                                                      input longint trace_interrupt,
                                                      input longint trace_cause[8],
                                                      input longint trace_tval[8],
                                                      input longint trace_wdata[64]);

import "DPI-C" function void debug_rob_push_wb_vec(input int     hartid,
                                                   input int     nslots,
                                                   input longint valid,
                                                   input longint wb_tag[8],
                                                   input longint wb_data[64]);

import "DPI-C" function void debug_rob_pop_trace_vec(input int      hartid,
                                                     input int      nslots,
                                                     output longint trace_valid,
                                                     output longint trace_iaddr[8],
                                                     output longint trace_insn[8],
                                                     output longint trace_priv[8],
                                                     output longint trace_exception,
                                                     output longint trace_interrupt,
                                                     output longint trace_cause[8],
                                                     output longint trace_tval[8],
                                                     output longint trace_wdata[64]);

module DebugROBPushTraceVec (
                             input           clock,
                             input           reset,
                             input [31:0]    hartid,
                             input [31:0]    nslots,
                             input [63:0]    should_wb,
                             input [63:0]    has_wb,
                             input [511:0]   wb_tag,
                             input [63:0]    trace_valid,
                             input [511:0]   trace_iaddr,
                             input [511:0]   trace_insn,
                             input [511:0]   trace_priv,
                             input [63:0]    trace_exception,
                             input [63:0]    trace_interrupt,
                             input [511:0]   trace_cause,
                             input [511:0]   trace_tval,
                             input [4095:0]  trace_wdata);

   longint __wb_tag[8];
   longint __trace_iaddr[8];
   longint __trace_insn[8];
   longint __trace_priv[8];
   longint __trace_cause[8];
   longint __trace_tval[8];
   longint __trace_wdata[64];
   genvar  i;

   for (i = 0; i < 8; i = i + 1) begin : gen_push_trace_vec_unpack
      assign __wb_tag[i]      = wb_tag[(i+1)*64-1:i*64];
      assign __trace_iaddr[i] = trace_iaddr[(i+1)*64-1:i*64];
      assign __trace_insn[i]  = trace_insn[(i+1)*64-1:i*64];
      assign __trace_priv[i]  = trace_priv[(i+1)*64-1:i*64];
      assign __trace_cause[i] = trace_cause[(i+1)*64-1:i*64];
      assign __trace_tval[i]  = trace_tval[(i+1)*64-1:i*64];
   end

   for (i = 0; i < 64; i = i + 1) begin : gen_push_trace_vec_wdata
      assign __trace_wdata[i] = trace_wdata[(i+1)*64-1:i*64];
   end

   always @(posedge clock) begin
      if (!reset) begin
         debug_rob_push_trace_vec(hartid, nslots,
                                  should_wb, has_wb, __wb_tag,
                                  trace_valid, __trace_iaddr, __trace_insn,
                                  __trace_priv, trace_exception, trace_interrupt,
                                  __trace_cause, __trace_tval, __trace_wdata);
      end
   end
endmodule; // DebugROBPushTraceVec

module DebugROBPushWbVec (
                          input          clock,
                          input          reset,
                          input [31:0]   hartid,
                          input [31:0]   nslots,
                          input [63:0]   valid,
                          input [511:0]  wb_tag,
                          input [4095:0] wb_data);

   longint __wb_tag[8];
   longint __wb_data[64];
   genvar  i;

   for (i = 0; i < 8; i = i + 1) begin : gen_push_wb_vec_tag
      assign __wb_tag[i] = wb_tag[(i+1)*64-1:i*64];
   end
   for (i = 0; i < 64; i = i + 1) begin : gen_push_wb_vec_data
      assign __wb_data[i] = wb_data[(i+1)*64-1:i*64];
   end

   always @(posedge clock) begin
      if (!reset) begin
         debug_rob_push_wb_vec(hartid, nslots, valid, __wb_tag, __wb_data);
      end
   end
endmodule; // DebugROBPushWbVec

module DebugROBPopTraceVec (
                            input           clock,
                            input           reset,
                            input [31:0]    hartid,
                            input [31:0]    nslots,
                            output [63:0]   trace_valid,
                            output [511:0]  trace_iaddr,
                            output [511:0]  trace_insn,
                            output [511:0]  trace_priv,
                            output [63:0]   trace_exception,
                            output [63:0]   trace_interrupt,
                            output [511:0]  trace_cause,
                            output [511:0]  trace_tval,
                            output [4095:0] trace_wdata);

   bit                                      r_reset;

   longint                                  __trace_valid;
   longint                                  __trace_iaddr[8];
   longint                                  __trace_insn[8];
   longint                                  __trace_priv[8];
   longint                                  __trace_exception;
   longint                                  __trace_interrupt;
   longint                                  __trace_cause[8];
   longint                                  __trace_tval[8];
   longint                                  __trace_wdata[64];

   reg [63:0]                               __trace_valid_reg;
   reg [511:0]                              __trace_iaddr_reg;
   reg [511:0]                              __trace_insn_reg;
   reg [511:0]                              __trace_priv_reg;
   reg [63:0]                               __trace_exception_reg;
   reg [63:0]                               __trace_interrupt_reg;
   reg [511:0]                              __trace_cause_reg;
   reg [511:0]                              __trace_tval_reg;
   reg [4095:0]                             __trace_wdata_reg;

   integer                                  k;

   always @(posedge clock) begin
      __trace_valid_reg     <= __trace_valid;
      __trace_exception_reg <= __trace_exception;
      __trace_interrupt_reg <= __trace_interrupt;
      for (k = 0; k < 8; k = k + 1) begin
         __trace_iaddr_reg[k*64 +: 64] <= __trace_iaddr[k];
         __trace_insn_reg[k*64 +: 64]  <= __trace_insn[k];
         __trace_priv_reg[k*64 +: 64]  <= __trace_priv[k];
         __trace_cause_reg[k*64 +: 64] <= __trace_cause[k];
         __trace_tval_reg[k*64 +: 64]  <= __trace_tval[k];
      end
      for (k = 0; k < 64; k = k + 1) begin
         __trace_wdata_reg[k*64 +: 64] <= __trace_wdata[k];
      end
   end

   assign trace_valid     = __trace_valid_reg;
   assign trace_iaddr     = __trace_iaddr_reg;
   assign trace_insn      = __trace_insn_reg;
   assign trace_priv      = __trace_priv_reg;
   assign trace_exception = __trace_exception_reg;
   assign trace_interrupt = __trace_interrupt_reg;
   assign trace_cause     = __trace_cause_reg;
   assign trace_tval      = __trace_tval_reg;
   assign trace_wdata     = __trace_wdata_reg;

   always @(negedge clock) begin
      r_reset <= reset;
      if (!reset && !r_reset) begin
         debug_rob_pop_trace_vec(hartid, nslots,
                                 __trace_valid, __trace_iaddr, __trace_insn,
                                 __trace_priv, __trace_exception, __trace_interrupt,
                                 __trace_cause, __trace_tval, __trace_wdata);
      end
   end
endmodule; // DebugROBPopTraceVec
'''

src = rd(V)
if "DebugROBPopTraceVec" in src:
    sys.exit("FATAL: %s already patched" % V)
wr(V, src + V_ADD)
print("PATCHED %s (vec blackbox verilog)" % V)

# -------------------------------------------------------------- DebugROB.scala
S = os.path.join(CHIPYARD, "generators/rocket-chip/src/main/scala/rocket/DebugROB.scala")

S_CLASSES = r'''
// Bug 2 fix: vectorized blackboxes.  One instance per hart handles all
// retireWidth slots at once, so the C++ side sees the slots in program order
// no matter how Verilator schedules the always blocks.
// Ports are sized for DebugROB.maxVecSlots (8); unused slots are tied off.
class DebugROBPushTraceVec(implicit val p: Parameters) extends BlackBox with HasBlackBoxResource with HasCoreParameters {
  val io = IO(new Bundle {
    val clock = Input(Clock())
    val reset = Input(Bool())
    val hartid = Input(UInt(32.W))
    val nslots = Input(UInt(32.W))
    val should_wb = Input(UInt(64.W))
    val has_wb = Input(UInt(64.W))
    val wb_tag = Input(UInt((DebugROB.maxVecSlots * 64).W))
    val trace_valid = Input(UInt(64.W))
    val trace_iaddr = Input(UInt((DebugROB.maxVecSlots * 64).W))
    val trace_insn = Input(UInt((DebugROB.maxVecSlots * 64).W))
    val trace_priv = Input(UInt((DebugROB.maxVecSlots * 64).W))
    val trace_exception = Input(UInt(64.W))
    val trace_interrupt = Input(UInt(64.W))
    val trace_cause = Input(UInt((DebugROB.maxVecSlots * 64).W))
    val trace_tval = Input(UInt((DebugROB.maxVecSlots * 64).W))
    val trace_wdata = Input(UInt((DebugROB.maxVecSlots * 512).W))
  })
  addResource("/csrc/debug_rob.cc")
  addResource("/vsrc/debug_rob.v")
}

class DebugROBPushWbVec(implicit val p: Parameters) extends BlackBox with HasBlackBoxResource with HasCoreParameters {
  val io = IO(new Bundle {
    val clock = Input(Clock())
    val reset = Input(Bool())
    val hartid = Input(UInt(32.W))
    val nslots = Input(UInt(32.W))
    val valid = Input(UInt(64.W))
    val wb_tag = Input(UInt((DebugROB.maxVecSlots * 64).W))
    val wb_data = Input(UInt((DebugROB.maxVecSlots * 512).W))
  })
  addResource("/csrc/debug_rob.cc")
  addResource("/vsrc/debug_rob.v")
}

class DebugROBPopTraceVec(implicit val p: Parameters) extends BlackBox with HasBlackBoxResource with HasCoreParameters {
  val io = IO(new Bundle {
    val clock = Input(Clock())
    val reset = Input(Bool())
    val hartid = Input(UInt(32.W))
    val nslots = Input(UInt(32.W))
    val trace_valid = Output(UInt(64.W))
    val trace_iaddr = Output(UInt((DebugROB.maxVecSlots * 64).W))
    val trace_insn = Output(UInt((DebugROB.maxVecSlots * 64).W))
    val trace_priv = Output(UInt((DebugROB.maxVecSlots * 64).W))
    val trace_exception = Output(UInt(64.W))
    val trace_interrupt = Output(UInt(64.W))
    val trace_cause = Output(UInt((DebugROB.maxVecSlots * 64).W))
    val trace_tval = Output(UInt((DebugROB.maxVecSlots * 64).W))
    val trace_wdata = Output(UInt((DebugROB.maxVecSlots * 512).W))
  })
  addResource("/csrc/debug_rob.cc")
  addResource("/vsrc/debug_rob.v")
}

object DebugROB {
'''

sub_once(S, "\nobject DebugROB {\n", S_CLASSES, "vec blackbox classes")

S_ANCHOR = """  def pushWb(clock: Clock, reset: Reset,
    hartid: UInt, valid: Bool, tag: UInt, data: UInt)(implicit p: Parameters): Unit = {
    val debug_rob_push_wb = Module(new DebugROBPushWb)
    debug_rob_push_wb.io.clock := clock
    debug_rob_push_wb.io.reset := reset
    debug_rob_push_wb.io.hartid := hartid
    debug_rob_push_wb.io.valid := valid
    debug_rob_push_wb.io.wb_tag := tag
    debug_rob_push_wb.io.wb_data := data
  }
}"""

S_NEW = S_ANCHOR[:-1] + r'''
  // ------------------------------------------------------------------------
  // Bug 2 fix: vectorized API for multi-retire cores.
  //
  // The scalar API above instantiates one blackbox per call site.  A core with
  // retireWidth > 1 therefore ends up with retireWidth producers and
  // retireWidth consumers on the *same* C++ deque (it is keyed only by
  // hartid), and Verilator is free to evaluate them in any order -- which
  // silently reorders the commit log and breaks cospike.
  //
  // These entry points funnel a whole retire group through a single blackbox,
  // so ordering is decided in C++.  Giving each slot its own deque would be
  // wrong: an instruction still waiting for its writeback data must hold back
  // the younger slots, which only works with one shared, ordered queue.
  // ------------------------------------------------------------------------
  val maxVecSlots = 8

  private def to64(x: UInt): UInt = x.pad(64)(63, 0)

  private def maskOf(bits: Seq[Bool]): UInt =
    VecInit(Seq.tabulate(maxVecSlots)(i => if (i < bits.size) bits(i) else false.B)).asUInt

  private def widen(traces: Seq[TracedInstruction]): Vec[WidenedTracedInstruction] = {
    val w = Wire(Vec(maxVecSlots, new WidenedTracedInstruction))
    for (i <- 0 until maxVecSlots) {
      if (i < traces.size) { w(i) := traces(i) }
      else { w(i) := 0.U.asTypeOf(new WidenedTracedInstruction) }
    }
    w
  }

  def pushTraceVec(clock: Clock, reset: Reset, hartid: UInt,
    traces: Seq[TracedInstruction],
    should_wb: Seq[Bool], has_wb: Seq[Bool], wb_tag: Seq[UInt])(implicit p: Parameters): Unit = {
    val n = traces.size
    require(n >= 1 && n <= maxVecSlots, s"DebugROB vectorized API supports 1..$maxVecSlots slots, got $n")
    require(should_wb.size == n && has_wb.size == n && wb_tag.size == n)
    val m = Module(new DebugROBPushTraceVec)
    val w = widen(traces)
    m.io.clock := clock
    m.io.reset := reset
    m.io.hartid := hartid
    m.io.nslots := n.U
    m.io.should_wb := maskOf(should_wb)
    m.io.has_wb := maskOf(has_wb)
    m.io.wb_tag := VecInit(Seq.tabulate(maxVecSlots)(i =>
      if (i < n) to64(wb_tag(i)) else 0.U(64.W))).asUInt
    m.io.trace_valid := maskOf((0 until n).map(i => w(i).valid))
    m.io.trace_exception := maskOf((0 until n).map(i => w(i).exception))
    m.io.trace_interrupt := maskOf((0 until n).map(i => w(i).interrupt))
    m.io.trace_iaddr := VecInit(w.map(t => to64(t.iaddr))).asUInt
    m.io.trace_insn := VecInit(w.map(t => to64(t.insn))).asUInt
    m.io.trace_priv := VecInit(w.map(t => to64(t.priv))).asUInt
    m.io.trace_cause := VecInit(w.map(t => to64(t.cause))).asUInt
    m.io.trace_tval := VecInit(w.map(t => to64(t.tval))).asUInt
    m.io.trace_wdata := VecInit(w.map(t => t.wdata.pad(512)(511, 0))).asUInt
  }

  def pushWbVec(clock: Clock, reset: Reset, hartid: UInt,
    valid: Seq[Bool], tag: Seq[UInt], data: Seq[UInt])(implicit p: Parameters): Unit = {
    val n = valid.size
    require(n >= 1 && n <= maxVecSlots, s"DebugROB vectorized API supports 1..$maxVecSlots slots, got $n")
    require(tag.size == n && data.size == n)
    val m = Module(new DebugROBPushWbVec)
    m.io.clock := clock
    m.io.reset := reset
    m.io.hartid := hartid
    m.io.nslots := n.U
    m.io.valid := maskOf(valid)
    m.io.wb_tag := VecInit(Seq.tabulate(maxVecSlots)(i =>
      if (i < n) to64(tag(i)) else 0.U(64.W))).asUInt
    m.io.wb_data := VecInit(Seq.tabulate(maxVecSlots)(i =>
      if (i < n) data(i).pad(512)(511, 0) else 0.U(512.W))).asUInt
  }

  def popTraceVec(clock: Clock, reset: Reset, hartid: UInt, n: Int)(implicit p: Parameters): Seq[TracedInstruction] = {
    require(n >= 1 && n <= maxVecSlots, s"DebugROB vectorized API supports 1..$maxVecSlots slots, got $n")
    val m = Module(new DebugROBPopTraceVec)
    m.io.clock := clock
    m.io.reset := reset
    m.io.hartid := hartid
    m.io.nslots := n.U
    (0 until n).map { i =>
      val w = Wire(new WidenedTracedInstruction)
      w.valid     := m.io.trace_valid(i)
      w.iaddr     := m.io.trace_iaddr(64 * i + 63, 64 * i)
      w.insn      := m.io.trace_insn(64 * i + 63, 64 * i)
      w.priv      := m.io.trace_priv(64 * i + 2, 64 * i)
      w.exception := m.io.trace_exception(i)
      w.interrupt := m.io.trace_interrupt(i)
      w.cause     := m.io.trace_cause(64 * i + 63, 64 * i)
      w.tval      := m.io.trace_tval(64 * i + 63, 64 * i)
      w.wdata     := m.io.trace_wdata(512 * i + 511, 512 * i)
      val t = Wire(new TracedInstruction)
      t := w
      t
    }
  }
}'''

sub_once(S, S_ANCHOR, S_NEW, "vec object API")

# ----------------------------------------------------- shuttle/exu/Core.scala
SH = os.path.join(CHIPYARD, "generators/shuttle/src/main/scala/exu/Core.scala")

SH_TRACE_OLD = """  val useDebugROB = shuttleParams.debugROB
  if (useDebugROB) {
    val trace = WireInit(csr.io.trace)
    for (i <- 0 until retireWidth) {
      val pc = if (usingVector) Mux(io.vector.get.com.retire_late, io.vector.get.com.pc, com_uops(i).bits.pc) else com_uops(i).bits.pc
      trace(i).valid := com_retire(i) || ((i == 0).B && (csr.io.exception || io.vector.map(_.com.retire_late).getOrElse(false.B)))
      trace(i).wdata.get := com_uops(i).bits.wdata.bits
      trace(i).iaddr := pc
      val ctrl = com_uops(i).bits.ctrl
      val rd = com_uops(i).bits.rd
      val should_wb = !io.vector.map(_.com.retire_late).getOrElse(false.B) && (ctrl.wfd || (ctrl.wxd && rd =/= 0.U)) && !csr.io.trace(i).exception
      DebugROB.pushTrace(clock, reset,
        io.hartid, trace(i),
        should_wb,
        false.B,
        rd + Mux(ctrl.wfd, 32.U, 0.U))
      io.trace.insns(i) := DebugROB.popTrace(clock, reset, io.hartid)
    }
  }"""

SH_TRACE_NEW = """  val useDebugROB = shuttleParams.debugROB
  if (useDebugROB) {
    val trace = WireInit(csr.io.trace)
    val debug_rob_should_wb = Wire(Vec(retireWidth, Bool()))
    val debug_rob_wb_tag = Wire(Vec(retireWidth, UInt(64.W)))
    for (i <- 0 until retireWidth) {
      val pc = if (usingVector) Mux(io.vector.get.com.retire_late, io.vector.get.com.pc, com_uops(i).bits.pc) else com_uops(i).bits.pc
      trace(i).valid := com_retire(i) || ((i == 0).B && (csr.io.exception || io.vector.map(_.com.retire_late).getOrElse(false.B)))
      trace(i).wdata.get := com_uops(i).bits.wdata.bits
      trace(i).iaddr := pc
      val ctrl = com_uops(i).bits.ctrl
      val rd = com_uops(i).bits.rd
      debug_rob_should_wb(i) := !io.vector.map(_.com.retire_late).getOrElse(false.B) && (ctrl.wfd || (ctrl.wxd && rd =/= 0.U)) && !csr.io.trace(i).exception
      debug_rob_wb_tag(i) := rd + Mux(ctrl.wfd, 32.U, 0.U)
    }
    // Bug 2 fix: one vectorized blackbox for the whole retire group instead of
    // retireWidth independent blackboxes sharing one hartid-keyed C++ deque.
    // With per-slot blackboxes Verilator picks the evaluation order of the N
    // always blocks, which reorders pushes and pops within a cycle and breaks
    // the commit log's program order (cospike then reports a PC mismatch on
    // the very first bootrom instruction).
    DebugROB.pushTraceVec(clock, reset, io.hartid,
      (0 until retireWidth).map(i => trace(i)),
      (0 until retireWidth).map(i => debug_rob_should_wb(i)),
      Seq.fill(retireWidth)(false.B),
      (0 until retireWidth).map(i => debug_rob_wb_tag(i)))
    val debug_rob_popped = DebugROB.popTraceVec(clock, reset, io.hartid, retireWidth)
    for (i <- 0 until retireWidth) {
      io.trace.insns(i) := debug_rob_popped(i)
    }
  }"""

sub_once(SH, SH_TRACE_OLD, SH_TRACE_NEW, "shuttle push/pop trace")

SH_WB_OLD = """  // wb
  for (i <- 0 until retireWidth) {
"""
SH_WB_NEW = """  // wb
  // Bug 2 fix: collect the per-slot DebugROB writebacks and push them through a
  // single vectorized blackbox after the loop.  Two instructions retiring in the
  // same cycle can target the same architectural register; the C++ side matches
  // wb data to traces by (tag, queue order), so the intra-cycle order of the
  // slots must be deterministic and equal to program order.
  val debug_rob_wb_valid = Wire(Vec(retireWidth, Bool()))
  val debug_rob_wb_tag_i = Wire(Vec(retireWidth, UInt(64.W)))
  val debug_rob_wb_data  = Wire(Vec(retireWidth, UInt(64.W)))
  debug_rob_wb_valid.foreach(_ := false.B)
  debug_rob_wb_tag_i.foreach(_ := 0.U)
  debug_rob_wb_data.foreach(_ := 0.U)
  for (i <- 0 until retireWidth) {
"""
sub_once(SH, SH_WB_OLD, SH_WB_NEW, "shuttle wb loop header")

SH_WB2_OLD = """    if (useDebugROB)
      DebugROB.pushWb(clock, reset, io.hartid,
        wen && uop.wdata.valid,
        uop.rd,
        uop.wdata.bits)


    wb_bypasses(i).valid := wb_uops(i).valid && wb_uops(i).bits.ctrl.wxd"""

SH_WB2_NEW = """    if (useDebugROB) {
      debug_rob_wb_valid(i) := wen && uop.wdata.valid
      debug_rob_wb_tag_i(i) := uop.rd
      debug_rob_wb_data(i)  := uop.wdata.bits
    }


    wb_bypasses(i).valid := wb_uops(i).valid && wb_uops(i).bits.ctrl.wxd"""

sub_once(SH, SH_WB2_OLD, SH_WB2_NEW, "shuttle wb per-slot capture")

SH_WB3_OLD = """    wb_bypasses(i).data := wb_uops(i).bits.wdata.bits
  }


  // ll wb"""
SH_WB3_NEW = """    wb_bypasses(i).data := wb_uops(i).bits.wdata.bits
  }
  if (useDebugROB) {
    DebugROB.pushWbVec(clock, reset, io.hartid,
      (0 until retireWidth).map(i => debug_rob_wb_valid(i)),
      (0 until retireWidth).map(i => debug_rob_wb_tag_i(i)),
      (0 until retireWidth).map(i => debug_rob_wb_data(i)))
  }


  // ll wb"""
sub_once(SH, SH_WB3_OLD, SH_WB3_NEW, "shuttle wb vectorized push")

print("ALL PATCHES APPLIED")
