### 3rd place: energy-efficient ahead-pipelined TAGE (Dang & Rotenberg), VFS 0.9742

On the CBP-NG suite: T 8.62 ins/cycle, CPI 0.052, EPI 1068 fJ/inst, one
cycle; 58.34 KB. The shipped `gshareN_ahead` scores 0.9674 and the shipped
`tage` 0.8728 on the same measurement.

**Idea.** A single one-cycle TAGE, no fast/slow P1-P2 pair: the pair's
disagreement bubbles and its extra misprediction cycle are exactly what holds
the stock `tage` back. Every table read is moved one block ahead so the
prediction cycle only compares and selects.

**What made the difference.** Latency went from 1.7 cycles to 0.9767 through
ahead-pipelining plus strategic logic placement, and T went from 4.47 to 8.58,
taking VFS from 0.7813 to 0.9718. Everything else together added 0.0024.
Once under one cycle, further latency cuts do not raise throughput; spend the
slack on components instead.

**Blocks.** Up to 4 branches per cycle; a block ends at a taken branch, a
misprediction, the end of a 256-instruction region, or the 4th branch. One
table per history length (10 tagged tables of 1024 entries, geometric
lengths up to 100); consecutive branches in a block are predicted by
different lengths. Longer maximum history would help MPKI but lengthens the
index path.

**Entries.** 11-bit tag = 2-bit branch position (rank) + 9-bit hash; 6-bit
*secondary tag*; 1 prediction bit, 2-bit hysteresis, 1 useful bit; each field
in its own SRAM (5 per table). The secondary tag disambiguates the part of
history not yet known when the ahead read starts; it is simply the predicted
target low bits of the previous block's final branch. That kept MPKI within
0.08 of the non-ahead design (with a 14-bit tag), and beat hashing all skipped
targets by 0.14 MPKI and 1% EPI.

**Prediction.** In the cycle before, compute index and hashed tag per table
from PC + folded history, read, latch everything. In the prediction cycle,
compare hashed tag, rank and secondary tag in parallel; per rank the longest
match provides, the second longest is kept as the alternate for useful-bit
updates; no match -> base. The base bimodal is ahead too: indexed by the
block PC, 8 banks x 4 predictions, one bank chosen late by the missing path.

**Update.** Flip a provider's prediction bit only when hysteresis is weak;
update useful bits only when provider and alternate disagree; on a
misprediction allocate up to 2 entries in longer tables (2 vs 1: -0.5%
mispredictions), clearing useful bits if no candidate is free. Correct
predictions only write hysteresis/useful, so they are cheap. Folded history is
updated with a compact hash of block PC, the final branch's position and
direction, and the next block PC.

**Timing tricks.** Put the main-tag comparison logic next to the main-tag
SRAM (the largest, so the critical path), keep registers near the arrays they
feed. The signal that suppresses the ahead table access right after a
recovery (when a branch in mid-block was predicted taken but fell through, as
in the "true block" rule) has huge fanout and gates the folded-history
update; they duplicated its logic.

**What did not pay.** Gating the upper 5 tables when there has been no
misprediction in 512 blocks and mispredictions x 1024 < branches: -25% energy
on some traces, but counter/gating overhead elsewhere and extra mispredictions
left VFS at 0.9741, so it was disabled in the submission.
