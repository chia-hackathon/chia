# Independent review: Titan round 12 (PPA/area), run r37

Reviewer: independent read-only review (no code edited, no sims run).
Inputs: `r12_seed_r35_frmfix.diff` (before) vs `r37_1006_2141/best_rtl.diff` (after). FU sources reconstructed in
`titan_runs/out/work/r37review/{r35,r37}_Matrix{FP,}MultiplyPipe.scala`, together with the check scripts (`chk_lzc.py`, `chk_rot.py`, `depth.py`).

**Verdict: ACCEPT-AFTER-TESTS.** I found no functional defect and no gaming of the functional checks. Cycle equality holds for every op, not just the 9 benchmarks.
Two things should still be done before accepting:
- (a) Close a real test gap. The vfmmacc SEW32/64 path was restructured the most, and nothing tests it with NaN, Inf, overflow or underflow.
- (b) Note openly that part of the delay gain (all of it for INT) comes from moving operand extraction into the issue cycle. That cycle is upstream of the FU, and neither PPA judge measures it.

---

## 0. Scope check

- Only `exu/fp/MatrixFPMultiplyPipe.scala` and `exu/int/MatrixMultiplyPipe.scala` differ semantically.
  - The `PipelinedFaultCheck.scala` difference is a diff-format artifact. The seed diff splits that file into two sections. r37 merges them, and the +/- lines are identical.
  - `shuttle/Core.scala` differs by one blank line.
- Port lists of both FUs are identical in the generated SV (`titan_scratch/r34_iter1/r37/{int_v0,int_a1,fp_v0,fp_d1}`).
- Nothing else is constant-tied. `set_fflags` is disabled, but it was disabled in r35 too.

## 1. What changed (architecture)

### INT FU (`MatrixMultiplyPipe`, 149 -> 129 lines)
1. **Operand extraction is retimed into the issue cycle.**
   - `a_ext/b_ext/c_ext = RegEnable(signExtend(extractElem(io.iss.op.*)), io.iss.valid)`.
   - Previously the FU extracted from the registered `op.rvs*_data`. Now nothing reads those dLen-wide `op` fields, so synthesis drops their flops. That is most of the area saving.
   - The enable matches the base-class `op := io.iss.op when io.iss.valid` exactly, so the stored values are the same.
2. **Two registers become one.** `c_ext_captured + product_sum` are replaced by one `acc_sum`, using `new_acc = (first_k ? c_ext : acc_sum) + product`.
   - This is a 2-input add instead of 3.
   - `acc_sum` is not updated on last_k, so write data stays stable under `io.write.ready` backpressure.
   - The result is the same modulo 2^64, and therefore modulo 2^SEW.
   - K=1 cells (first_k == last_k) still give c + product.
3. **Multiply style.** The multiply is `UInt*UInt` truncated to 64 bits instead of `SInt*SInt`. The low 64 bits are identical.

How 31.37 ns became 12.74 ns with no extra cycle:
- The agent's own analysis says the old critical path had about 11.7 ns of extraction mux plus sign-extend with fanout 72-82, followed by the multiply tree and the 3-input add.
- The extraction now sits before the a/b/c_ext flops, i.e. at the FU input ports. In standalone abc, input ports arrive at t=0.
- The 3-input add is now 2-input.
- The reg-to-reg path is now `b_ext -> mul -> add -> io_write_bits_data` (55 cells).

### FP FU (`MatrixFPMultiplyPipe`, -48.9 % area, 4997 -> 2616 flops)
- **One shared binary64 datapath.** It replaces the per-format H/BF/S/D multipliers, adders, rounders and their stage registers:
  - Recoded binary64 operand registers `mA`/`mB`.
  - One `MulFullRawFN(11,53)`.
  - One `AddRawFN(11,53)`, split into two halves.
  - One raw register `P` (`RawFloat(11,105)`), time-shared between the product, the exact group sum S, the block-scale product and the adder output.
  - One `RoundAnyRawFNToRecFN(11,55 -> fmt)` per format reads `mulRawReduce(P)`.
  - Narrow formats are widened exactly to D. Each result is then rounded once to fmt_C. A correctly rounded result of an exact or sticky-exact value is unique, so the results are bit-identical.
- **Final sums use the narrow rounders.** Final H/S sums are `resizeRawFloat(5,13 | 8,26)` of the D raw sum, then the format's own `RoundRawFNToRecFN`. This keeps those hardfloat module types under the FU for the P-SOC per-module check.
  - The resize is exact: the sum of two fmt_C values lies inside that raw exponent range. I checked the clamp bounds against hardfloat's `resizeRawFloat`.
- **fpw significand product uses the D multiplier.** At issue, `mA/mB := Cat(0(13), sig)`: a recoded "zero" whose fraction is the integer significand. `MulFullRawFN` multiplies `sig` without masking for zero (verified in hardfloat source), so `rawOut.sig(47,0) = sigA*sigB`.
  - This is correct but depends on hardfloat internals; see risk R3.
- **Exact 560-bit group sum: 2 adders instead of 14.** The S and -S registers rotate one 80-bit chunk per cycle through a single 80-bit adder each, over c1..c7.
  - The 560-bit alignment shifter is replaced by a 127-bit window `sig << (pos%80)` plus a one-hot chunk select.
  - I checked this against a 560-bit reference with 3,000 random multi-product groups including signs (`chk_rot.py`).
- **Cheaper normalisation.**
  - The 1024-bit LZC is replaced by a 64-bit-block LZC.
  - The 560x10 normalise shifter is replaced by a 128-bit window, with a prefix-OR sticky over the blocks below it.
  - I checked this bit-exactly against a reference over 200,000 random values, including trailing zeros and powers of two (`chk_lzc.py`).
- **E8M0 scale is spliced directly.** The scale is a recoded binary32 `Cat(0, byte+129, 0)`. I checked it against the hardfloat recode of r35's `e8m0ToFp32` for bytes 0..254. Byte 255 gives Inf, and its NaN status is tracked from the raw byte as before.
- **Inputs and v0 at issue.** As in the INT FU, A/B/C extraction (and the `fpwUnpack` used for the significand product) moves to the issue cycle, and the `v0_egs` prefetch write also happens at issue.
- **Accumulator operand read later.** The accumulator operand (`acc_input`) is no longer staged at c0. It is re-read at c12/c13. This is equivalent because `acc_reg` only changes at a uop's final stage, `c_raw` only changes at issue, and only one uop is in flight.
- **Control.** The 5-bit `cnt` becomes a 20-bit one-hot `ph`. Its bits are used as mux selects directly from flops.

### Cycle behaviour
The stage constants (`S_CH0..S_BFF`), the `retire_early` logic, `at_final` and the `wvalid` write logic are identical to r35. The INT `last` is unchanged. Per-uop FU occupancy is therefore the same for every family:

| uop | r35 | r37 |
|---|---|---|
| v0 prefetch | 1 (c0) | 1 |
| MXINT, not block-last | 1 | 1 |
| fpw / MXFP, not group-last | 8 (c0..c7) | 8 |
| all other non-last-k (H/S/D/OFP8 C) | 17 (c0..c16) | 17 |
| same, bf16 C | 19 (c0..c18) | 19 |
| last k | final stage +1, then wait for write.fire | same |
| INT, any uop | 1 (last_k: until write.fire) | same |

There is still one idle cycle between uops (`stall := valid`). Cycle equality is therefore structural, not an artifact of the benchmark set.

## 2. Gaming / functional-risk audit

| item | finding |
|---|---|
| One uop in flight | Holds. `ExecutionUnit`: `fu.io.iss.valid := io.iss.valid && fu_sel && !fu.io.stall`, and both FUs drive `stall := valid`. The base class asserts `!valid \|\| last`. All issue-time writes (`a/b/c_raw`, `mA/mB`, `v0_egs`) use the same enable as the `op` capture, so they cannot clobber an in-flight uop. |
| Accumulator widths | Unchanged: INT 64 b, int_acc 40 b, fpw S/-S 2x560 b, acc_reg 64 b. |
| Sticky bits | Block-LZC sticky verified (above). `mulRawReduce`, `resizeRawFloat` and `RoundAny` keep at least 56 bits plus sticky, which is enough for every target precision up to 53. |
| +-0 | The zero-sign trackers (`apz/anz`, RDN rule) and the OFP8 q_sign are unchanged. The D adder's zero sign uses `frm`, same as the S/H adder. |
| NaN/Inf/subnormal | Specials ride the hardfloat flags; NaN output is canonical either way. Subnormal fp32 x fp32 products (2^-298) are normal in D. The D -> S/H rounder gets negative `sAdjustedExp` there. hardfloat already handles that (totalUnderflow uses the untruncated exponent, as in `RecFNToRecFN` D->S). r35's `RoundAny(10,65 -> H)` had the same situation. |
| Block-scale NaN / maxFinite | `scaleToD`: H still rounds S->H under frm (to maxFinite under RTZ etc.), as in r35. BF16 and D are exact (r35 rounded them, but they were exact). The NaN test now reads `FType.D.isNaN(mA)`, the widened blk_scale held in mA from c3 to c11. OK. |
| Rounding modes incl. RMM | Every rounder takes `op.frm`. bf16 C still goes binary32 then bf16 (double rounding, p'=24 >= 2p+1). That is unchanged from r35 and safe for RNE, directed modes and RMM. |
| Tininess | Constant after-rounding, and fflags are disabled (pre-existing). No value impact. |
| SEW64 int, wrap, altfmt | `signExtend` now uses `iop.rvs*_eew` and `altfmt`, the same fields captured into `op`. The product's low 64 bits are sign-agnostic. OK. |
| Partial tiles / lambda / LMUL / vstart / mask | The FU is per-element. `wmask`, `wvd_eg` and `hazard` come from `op` unchanged. |
| Reset / X | New no-reset regs (`acc_sum`, `a/b/c_raw`, `mA/mB/P/P_inv`) are always written before they are read in any legal uop sequence. r35's `c_ext_captured` and `fpw_*_v` were also unreset. No issue under Verilator 2-state. |
| Dead-code tricks / constants | None found. Ports are identical. Every use of `op.rvs*_data` / `rvm_data` was replaced by issue-time capture of the same values. That removal is the legitimate area win, not a pruning artifact. The MXINT path assumes Int8/Int4 inputs only; that is pinned by the decoder and inherited from r11. |
| P-SOC per-module failure | The agent's analysis is credible: byte-identical modules moved, which is yosys IdString-order noise. The agent explicitly refused to re-roll FU text to steer the check. That is good behaviour; the judge needs fixing. |

### Risks found (none is a functional bug)
- **R1. Unmeasured timing moved upstream (moderate).**
  - Both FUs now compute combinationally from `io.iss.op`. That data comes from `Backend.scala`: VRF read response, then `Mux1H` (scalar/slide/zero), then the FU.
  - Measured on the judge's flat netlists (`depth.py`), the FU-input-to-flop depth went from about 0 to:
    - INT: 20 cells (from `rvs1_eew` / `a_pos`) and 17 cells (from data).
    - FP: 19 cells (from `rvs2_eew` / `b_pos`) and 16 cells (from data).
  - At the INT path's ~0.23 ns/cell, that is about 4-5 ns appended to the VRF-read issue cycle.
  - The standalone judge counts input ports from t=0. P-SOC is hierarchical and never sees cross-module paths. So this cost is invisible to every gate.
  - All of INT's 31.4 -> 12.7 ns comes from this relocation plus the 2-input add. Without it, INT would need either an extra cycle per uop or a failing delay.
  - Mitigation: Saturn already runs `extractElem(rvs*_data, eew, eidx)` in this same cycle (`Backend.scala` `rvs1_elem`), so the new logic is parallel to an existing path, plus sign-extend / `fpwUnpack`. Likely tolerable, but not demonstrated.
- **R2. Least-tested area changed the most.** vfmmacc SEW32/64 now runs on the D multiplier + `RoundAny(11,55 -> 8,24)`, and the S sum goes through `resize(8,26)` then `RoundRaw(S)`. The base and frm tiers for this family use an exponent-limited pool with no NaN, Inf, overflow, product underflow to subnormal, or total underflow. The theory says the result is equivalent, but no test exercises it.
- **R3. Fragile hardfloat dependence.** The fpw significand product relies on hardfloat passing `sig` through for zero-class recoded inputs. This is correct for the current hardfloat, but a hardfloat change that masks the significand of zeros would silently break every fpw/MXFP result. Only RTL tests would catch it. Add a comment or an elaboration-time assertion.
- **R4.** OFP8 C is RNE-only by disclosure, so the non-RNE paths of `ofp8Round` (unchanged code) are untested. MXFP bs=1 lacks RDN/RMM. Neither path changed this round, so these are low risk.

## 3. Spec spot checks (integrated-matrix-v0.9.0.adoc)
- 486-487, 1280: integer accumulation is modulo 2^SEW and C is always signed. The `acc_sum` mod 2^64 truncated to SEW conforms; `c_ext` is signed-extended.
- 1847-1870, Sail 5115-5122 / 5262 / 5348-5349 / 5399:
  - `decode_scale` goes to fmt_C under frm, then `fp_mul(sA,sB)` and `fp_mul(blk, S_r)` / `fp_add` in fmt_C.
  - The shared-D implementation computes the exact product/sum, then a single rounding to fmt_C under frm. That equals fmt_C arithmetic.
  - binary16 scale decode rounds under frm. bf16, S and D scale decodes are exact.
- vfmmacc (two roundings per k): the product is rounded to fmt_C at c3 (mB), then added and rounded at c13-c15. Conforms.
- fpw (S exact, then `fp_round_to_frm(S)`, then `fp_add`): unchanged structure. S is exact (560 b), rounded once at c10.

## 4. Test coverage and recommendations

Coverage today:
- Every compare is bit-exact against the Python/Sail-derived goldens. ime_f_ compares against the DUT's scalar FPU instead.
- Non-RNE tiers exist (RTZ/RDN/RUP/RMM) for vfmmacc SEW32/64, SEW16 (H/BF), fpw IEEE C, MXFP (bs=1 only RTZ/RUP) and MXINT. Each runs on one special geometry and is full-VL only.
- fpw/MXFP/MXINT frm tiers include huge/tiny/cancelling classes and scale range edges, so they do exercise overflow, maxFinite and underflow for those families.
- No NaN or Inf input under any non-RNE mode. No vstart or mask (not applicable to these FUs). K=1 coverage unconfirmed.

**S3 (`ime_stress.py`) does NOT cover the risks above.** It randomizes only geometry with the same data pools and forces `frm=0`. It is useful for geometry/sequence coverage, which this change barely touches, but adds nothing for R2-R4. No IME lockstep against Spike/Sail with random data exists in the .py code.

Prioritized additional directed tests (golden = rvv_ref / Sail, all five frm values):
1. **vfmmacc SEW32 and SEW64 special/extreme tier** (R2):
   - qNaN and sNaN in A, B, C; +-Inf; Inf x 0; Inf - Inf across k.
   - Products that overflow (Inf vs maxFinite per mode).
   - Products of subnormal x subnormal and normal x tiny (total underflow, pegMinNonzero under RUP/RDN).
   - Results that land exactly at the min subnormal / min normal boundary.
   - Exact cancellation acc + (-p) (sign under RDN vs others); -0 + -0.
2. **vfmmacc SEW16 fp16 and bf16, subnormal x subnormal products** (H: 2^-48, negative `sAdjustedExp` in `RoundAny(11,55 -> 5,11)`) under RUP/RDN/RMM. Also bf16 sums near 2^128 (S-intermediate overflow) at all modes.
3. **fpw W=4 bf16 / W=2 fp32 large cancellations**:
   - Groups like +2^127·x, -2^127·x, +tiny, which put |S|'s leading one in each 64-bit block 1..9 with nonzero bits in lower blocks. This exercises the new block LZC and the window/prefix-OR sticky in RTL.
   - Include sticky-only-below-window cases at RNE ties.
4. **MXFP/MXINT binary16 C with scale bytes 0, 1, 142/143 (H overflow edge), 254, 255** at RTZ/RUP/RMM. Also bs=1 at RDN/RMM.
5. **INT**:
   - A K=1 cell, if any geometry yields one.
   - SEW64 vmmacc with all 4 altfmt combinations and operands with bit 63 set.
   - Int4 hi/lo nibble with altfmt=1.
   - A last_k write under VRF write-port backpressure (any test where `io.write.ready` drops; check the waveform once).
6. **Mutants the suite should kill** (run once to calibrate the suite):
   - INT: drop `!op.matrix_last_k` from the `acc_sum` update (only killable with write backpressure, which tests the stability claim).
   - FP:
     - `e8m0RecS` +129 -> +128.
     - `fpw_stkC := false`.
     - `fpw_stkF := false`.
     - `resizeRawFloat(8,26)` -> `(8,24)`, i.e. a sticky loss.
     - `roundP` `invalidExc := false`.
     - `fpw_zero_sign` mode swap.
   - Any survivor marks a coverage hole on a path this round rewrote.
7. **Timing (R1)**:
   - Re-run the standalone judge with the FU wrapped by input registers on `io.iss.op` (or set an input arrival equal to the Backend VRF-read-to-iss delay).
   - Better: a flat Backend + FU STA on the issue path. This shows how much of the 31.4 -> 12.7 ns INT gain survives in context.
   - At minimum, state in the round report that the gain is a retiming into the issue cycle.

Accept once (1)-(3) pass and (7) shows the issue-cycle path is not the new SoC critical path, or it is accepted as a documented trade-off. Items (4)-(6) are recommended but lower risk.
