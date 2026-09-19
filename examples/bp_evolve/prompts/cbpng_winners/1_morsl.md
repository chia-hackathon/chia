### 1st place: MORSL (Koizumi et al.), VFS 0.9935

Minimal-Overhead Rank-Based predictor with Summation-free correction and Lazy
access. On the 168 training traces: T 8.647 ins/cycle, 4.803 MPKI, EPI 524
fJ/inst, critical path 286 ps (one cycle), 37.5 KiB in 51 single-port SRAMs.

**Idea.** An ahead-pipelined, rank-based TAGE that is as accurate as a
conventional TAGE but pays for almost none of its latency or energy. Three
energy mechanisms sit on top: narrow-word tagged tables, a tagged corrector
instead of a statistical corrector, and an allocation-guided filter that skips
the long-history banks.

**Blocks and ranks.** A block runs to the first taken branch, the 4th
conditional branch, or 32 instructions, and may cross the 128-byte line.
Predictions are per *rank* (1st..4th conditional branch in the block), not per
instruction slot: rank prediction carried less information than offset
prediction and was cheaper in logic and energy under HARCOM. The upper 2 bits
of each 12-bit tag encode the lane (rank).

**Ahead-pipelining only the index.** The indices and bank mapping of the
tagged tables are computed one block ahead, from the previous block's PC and
history, and the RAM read is latched into a pipeline register. Tags are built
from the *current* block's PC and history and compared in the prediction
cycle. Because the tag carries the current context, no extra candidates are
read for the unknown exit path of the previous block; the only added hardware
is one stage of registers holding the ahead-read results. Ahead-pipelining
cost only about 1% MPKI (plain TAGE: 4.954 non-ahead, 5.006 one block ahead).
The bank + filter path is 580 ps but, one block ahead, its effective latency is
280 ps.

**What is *not* ahead.** The base bimodal (16K entries, 4 KiB, indexed by the
current block PC; 281 ps total including a 2:1 meta mux, 4:4 crossbar,
override mux) and TC-bias (258 ps) use the current PC directly because they
fit in the cycle and are more accurate that way. Choose per component whether
to go ahead.

**TAGE organisation.** 8 history lengths 5, 7, 11, 17, 33, 65, 126, 334
(blocks), dense at the short end (acts as pseudo-skewed associativity when
each entry holds a single tag), a relatively long shortest history (makes up
for the information lost by reading ahead). 8 physical banks of 2048 entries,
2:2 interleaving across bank pairs (bank interleaving: 4.954 -> 4.876 MPKI).
Entry = 12-bit tag, 3-bit counter, 1 useful bit. Path history in the lghist
style with a 6-bit value from the taken branch's own low address bits (not
target-only: blocks with the same target would become indistinguishable).
Final prediction: a meta counter chooses between the longest match and the
longest high-confidence match.

**Narrow words, split by write frequency.** Tag + prediction bit (which never
change on a correct prediction) go in one SRAM per bank; hysteresis + useful
bits (which change on every prediction) go in another, split into 4 banks to
emulate a second port. Merging SRAMs saved more energy than the accuracy it
cost.

**Tagged corrector instead of an SC.** An adder datapath does not fit and
costs energy, so auxiliary correlations are captured by override-style tagged
tables: TC-bias (128 entries, 7-bit tag, indexed by block PC, overrides when
its counter is strong and confidence high; -0.059 MPKI for +47 fJ) and
TC-history (128 sets x 2 ways, history = BrIMLI + path, index ahead / tag
late; -0.081 MPKI for +44 fJ). Each was worth only +0.0002 / +0.0003 VFS: at
one cycle, accuracy buys little.

**Allocation-guided access filter.** TAGE only allocates in long-history banks
after a shorter one failed, so "never allocated long here" means the long
banks cannot help. Two 1-bit, 2048-entry tables (one indexed by PC, one by PC
xor shortest history) record long allocations; unless both bits are 1 the
predictor runs in mini-TAGE mode reading only the 2 shortest banks. With a rare
random reset after mini-TAGE predicts a whole block correctly, it skips 35.9%
of tagged-bank reads, -87 fJ/inst for +0.021 MPKI. This was the biggest
energy win (EPI 520 -> 455 when added).

**HARCOM specifics.** Tables updated on correct predictions (TC-bias,
TC-history) read stale counters with banking alone; they needed entry
forwarding, guaranteed writes by stalling (`need_extra_cycle`), and 2 banks.
Cost about 0.1 IPC each, hardly any VFS. A 4-bit saturating meta-counter
update was the 286 ps critical path.
