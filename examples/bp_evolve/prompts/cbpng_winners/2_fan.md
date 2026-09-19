### 2nd place: gshareN_tagtN_ahead (Jun Fan), VFS 0.9921

On the 168 training traces: T 8.893 ins/cycle, CPI 0.04705, EPI 312.6
fJ/inst, 5.228 MPKI, latency 0.96 cycles, 176 KB of RAM.

**Idea.** Start from the shipped `gshareN_ahead` (the reference source in
this prompt), which already has one-cycle throughput and very low energy, and
buy accuracy by replacing half of its RAM with two ahead-pipelined N-branch
tagged tables. Keep the table count at two to keep EPI low. Result: MPKI
slightly better than the stock `tage` (5.365 vs 5.489 before the last
hysteresis step), 1.5x its throughput (8.876 vs 5.769 IPC) and 4.5x lower
energy (279 vs 1265 fJ).

**Structure.** Up to N = 4 branch predictions per cycle (at most one taken),
256-instruction blocks (the example uses N = 7 and 1024).
T0: gshare, 8192 entries, each entry 8 paths x 4 counters, 18-bit history.
T1: tagged, 16384 entries, 9-bit tag, 4 counters, 20-bit history.
T2: tagged, 16384 entries, 11-bit tag, 4 counters (2-bit for timing), 80-bit
history. One tag covers all 4 branch predictions of the entry.

**Two stages.** BP0 (previous block B0): hash B0's address with each table's
history, read prediction bits and tags, flop them. BP1 (current block B1):
gshare's 8x4 bits are muxed by the *path* out of B0 (B0 address + number of
conditional branches + last direction; N+1 = 5 exits rounded up to 8), and the
tagged tables compare their tag against a hash of B1's address and current
history -- no path selection, no secondary tag. The longest matching table
provides all 4 predictions; gshare counts as always matching.

**Training, energy-first.** Prediction+tag bits in one RAM, hysteresis in
another split into 4 banks (one per branch), read only for the branch that
mispredicted. On a correct prediction the hysteresis is *written* to a fixed
strong value without being read, so the hysteresis write and the next
prediction read overlap and no extra cycle is needed. On a misprediction,
hysteresis is read and decremented, and the prediction/tag RAM of the provider
is rewritten only if its hysteresis is weak (an extra cycle). Allocation: T1
when gshare provided, T2 whenever gshare or T1 provided.

**Accuracy extras (small).** A 2-entry recent-PC training-direction bias
(-0.46% MPKI), hashing the B0->B1 path into the tagged index (-0.20%), 2-bit
hysteresis in T0/T1 (-2.55% MPKI, +12% EPI, +35% RAM, +0.0009 VFS). The two
tagged tables brought almost all of the gain; bias + path hashing together
moved VFS by only 0.0003.

**Timing is floorplan.** With the default placement the three prediction RAMs
were far apart and the latency was 1.17 cycles (-> 2 cycles). Stacking them
compactly brought it to 0.96 cycles. Declared-size and placement decide
whether you are one cycle or two.

**Energy details.** Skip the gshare read when its index equals the previous
block's and the entry has not been written since (reuse the pipeline
register; -1.6% EPI). Place T1's index register next to T1's RAM (-0.6%).
Avoid heavy buffering off the critical path (-0.35%).
