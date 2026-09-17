// See LICENSE for license details.
//
//**************************************************************************
// llama-lmhead-fused — one LM-head TILE of Llama-3.2-1B at decode N = 1, with
// the Gemmini weight DMA and the Saturn companion work in ONE timed region
//--------------------------------------------------------------------------
//
// SEALED HARNESS BODY. Included by the thin `bareMetalC/llama_lmhead_fused_n1.c`
// wrapper, which is the registry's `main_src`. It holds main(), the data
// generator, the L2 flush that puts the machine in a COLD state, the scalar
// golden models and all four self-checks. The optimizer never gets to change
// any of it: every build is reassembled from the pristine tree plus exactly
// one file, `include/llama_lmhead_fused_n1_kernel.h`.
//
// WHY THIS BENCHMARK EXISTS
// -------------------------
// `llama-layer-fused-n1` established that Saturn work CAN hide inside a
// Gemmini weight DMA: 206,304 sequential -> 179,033 overlapped (-13.2%),
// decomposing into a 161,687-cycle pure weight stream plus 17,346 cycles of
// still-exposed Saturn cost out of 28,372 serialised. That is an exposure
// ratio of 17,346 / 28,372 = 0.611, and `loop/llama_project.py --overlap`
// currently extrapolates it to the WHOLE decode step.
//
// That extrapolation has one obvious validity threat and this harness is the
// experiment that closes it. In `llama-layer-fused-n1` the Saturn half is
// 13.8% of the fused kernel and it is ATTENTION -- 64 KiB of KV cache, 28k
// cycles, genuinely compute-shaped work that wants to live in a DMA shadow.
// In a real decode step the Saturn share is 0.7-2.2%, and the single largest
// Gemmini phase is the LM head: ~20.5% of decode cycles, 2048 x 128256 int8
// = ~250 MiB of weights streamed for ONE token, with NO attention anywhere
// near it. The only work concurrent with the LM head is the final RMSNorm,
// the int8 quantisation of the next hidden state, and the running argmax over
// the logits. Whether 0.611 survives that change of shape is unknown and
// unmeasured, and a measured ZERO is as publishable as a measured win.
//
// WHAT IS IN THE TIMED REGION
// ---------------------------
//   GEMMINI : C[1 x 2048] = A[1 x 2048] * B[2048 x 2048]   (4 MiB int8)
//             exactly the `llama-q8-gemv-gemmini-lmhead` shape, so this
//             entry's stream is directly comparable with that entry's 634,507.
//   SATURN  : xn[0..2048)  = rmsnorm(X, G, eps)            (fp32, H = 2048)
//             xq[0..2048)  = clamp(rne(xn * qscale), -127, 127)   (int8)
//             amax         = running argmax over LG[0..2048) int32 logits
//
// SCALING -- READ THIS BEFORE INTERPRETING THE NUMBER
// ---------------------------------------------------
// The real head is 2048 x 128256. This benchmark is ONE 2048-column tile,
// i.e. 1/62.6 of it. The three Saturn pieces do not all scale the same way:
//
//   * the running argmax is TILE-MATCHED and faithful. A decode step computes
//     the head tile by tile; the natural software pipeline folds tile t-1's
//     2048 logits into the running max while tile t's weights stream. L = 2048
//     per 4 MiB of B is exactly the true ratio.
//   * the final RMSNorm and the int8 quantisation happen ONCE PER TOKEN, not
//     once per tile. Charging the whole token's worth of them to a single tile
//     OVER-ATTRIBUTES the Saturn side by 62.6x. That is deliberate: it makes
//     the measured exposure an UPPER BOUND. If the exposure ratio measured
//     here is small, it is smaller still at true scale, and the conclusion
//     ("0.611 does not extrapolate to the LM head") is safe.
//
// A full-vocabulary argmax (128,256 int32 = 513 KiB) was considered and
// REJECTED for this harness: it would add 12.5% to the byte count and turn the
// experiment into a bandwidth-contention measurement instead of an overlap
// measurement. The Saturn side here is compute-shaped, exactly as it was in
// `llama-layer-fused-n1`, so the two exposure ratios are comparable. Bandwidth
// contention is a separate question and deserves its own entry.
//
// WHY THE SATURN WORK IS INDEPENDENT OF THE GEMV (and must be)
// ------------------------------------------------------------
// If the argmax read `C`, or if the GEMV read `xq`, the two halves would be
// serialised by a true data dependency and the benchmark would measure nothing.
// Both are avoided the same way `llama-layer-fused-n1` avoids quantising
// `probs` inside its timed region -- by supplying the operands separately:
//   * `LG` is a SEPARATE pre-supplied int32 logit vector, standing for the
//     PREVIOUS tile's output. That is what a real streaming argmax consumes.
//   * `A` is the CURRENT token's already-quantised hidden state; `xq` is the
//     NEXT one. The rmsnorm/quantise pair produces `xq`, never `A`.
// This is the honest structure of a software-pipelined decode, not a
// convenience.
//
// COLD STATE
// ----------
// The harness streams 2 MiB of unrelated DRAM through the 512 KiB L2 with
// Gemmini's own DMA immediately before `read_cycles()`, on exactly the path
// the kernel will use. `llama-q8-gemv-gemmini-lmhead` measures 634,507 with a
// warm tail (its harness fills B right before the call, leaving up to 512 KiB
// of the 4 MiB resident, and its winning kernel consumes K blocks in
// DESCENDING order precisely to harvest it). Here that credit is GONE, so
// expect this entry's stream alone to land ABOVE 634,507, and expect
// descending-K order to be worth less than it was -- possibly nothing.
//
// Wall-clock discipline (Verilator runs this SoC at ~1.8k cycles/s):
//   * every array lives at a fixed DRAM address OUTSIDE the ELF. The loader
//     zeroes .bss at roughly 12 s per KiB over TSI, so a multi-MiB `static`
//     array would cost hours before main() runs. Only ~1 KiB of .bss exists.
//   * the byte fill is a 512-byte scalar-generated block XOR-ed with a
//     rotating 64-bit key, written 64 B (one cache line) per iteration with
//     eight independent stores so the in-order core keeps several line fills
//     in flight. That loop, not the kernel, dominates the Verilator wall time.
//   * the GEMV self-check is SAMPLED (8 outputs) and FUSED into one pass over
//     k. The three Saturn checks are FULL (2048 + 2048 + 1 outputs) because
//     together they are only ~6k scalar operations.
//   * no libm: the reference rsqrt is a double-precision Newton iteration on
//     top of `__builtin_sqrt`, which the toolchain lowers to `fsqrt.d`.

#ifndef LH_N
#error "include bareMetalC/llama_lmhead_fused_n1.c, not this file"
#endif
#ifndef LH_M
#error "the wrapper must define LH_M"
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

#include "include/gemmini_testutils.h"

// --------------------------------------------------------------------------
// Shapes
// --------------------------------------------------------------------------
#define LH_K     2048          // hidden size
#define LH_H     2048          // rmsnorm width == hidden size
#define LH_L     2048          // logits folded into the running argmax
#define LH_EPS   1e-5f

// DRAM (memory@80000000, 256 MiB) past the end of the ~64 KiB image.
//   0x81000000  A       1 x 2048 int8                       2 KiB
//   0x81100000  B       2048 x 2048 int8                    4 MiB -> 0x81500000
//   0x81600000  C       1 x 2048 int32                      8 KiB
//   0x81700000  X       2048 fp32                           8 KiB
//   0x81710000  G       2048 fp32                           8 KiB
//   0x81720000  xn      2048 fp32                           8 KiB
//   0x81730000  xq      2048 int8                           2 KiB
//   0x81740000  LG      2048 int32                          8 KiB
//   0x81750000  amax    2 int32                             8 B
//   0x81800000  reference arrays (never read by the kernel)
//   0x81A00000  flush buffer, 2 MiB   -> ends 0x81C00000
// All 1 MiB- or 64 KiB-aligned, so every row start is 16 B clean.
#define LH_A        ((elem_t  *)0x81000000UL)
#define LH_B        ((elem_t  *)0x81100000UL)
#define LH_C        ((acc_t   *)0x81600000UL)
#define LH_X        ((float   *)0x81700000UL)
#define LH_G        ((float   *)0x81710000UL)
#define LH_XN       ((float   *)0x81720000UL)
#define LH_XQ       ((elem_t  *)0x81730000UL)
#define LH_LG       ((acc_t   *)0x81740000UL)
#define LH_AMAX     ((int32_t *)0x81750000UL)
#define LH_REF_XN   ((float   *)0x81800000UL)
#define LH_REF_XQ   ((int8_t  *)0x81810000UL)
#define LH_FLUSH    ((elem_t  *)0x81A00000UL)
#define LH_FLUSH_BYTES (2UL * 1024 * 1024)

// --------------------------------------------------------------------------
// Deterministic pseudo-random fill (fixed seed -> reproducible every run)
// --------------------------------------------------------------------------

static uint64_t lh_rng = 0x243F6A8885A308D3ULL;

static inline uint64_t lh_rng_next(void) {
  lh_rng ^= lh_rng << 13;
  lh_rng ^= lh_rng >> 7;
  lh_rng ^= lh_rng << 17;
  return lh_rng;
}

#define LH_SEED_WORDS 64                 // 512 B, essentially the whole .bss
static uint64_t lh_seed_block[LH_SEED_WORDS];

// `bytes` is always a multiple of 64 here. Eight independent stores per
// iteration so the in-order core keeps several line fills in flight; this is
// the loop that dominates the Verilator wall time, not the kernel.
static void lh_fill_bytes(void *dst, size_t bytes) {
  uint64_t *d = (uint64_t *)dst;
  const size_t words = bytes / 8;
  for (size_t i = 0; i < words; i += 8) {
    const uint64_t key = lh_rng_next();
    const uint64_t *s = &lh_seed_block[i & (LH_SEED_WORDS - 1)];
    d[i + 0] = s[0] ^ key;
    d[i + 1] = s[1] ^ key;
    d[i + 2] = s[2] ^ key;
    d[i + 3] = s[3] ^ key;
    d[i + 4] = s[4] ^ key;
    d[i + 5] = s[5] ^ key;
    d[i + 6] = s[6] ^ key;
    d[i + 7] = s[7] ^ key;
  }
}

// --------------------------------------------------------------------------
// Reference helpers -- no libm
// --------------------------------------------------------------------------
static inline float lh_absf(float a) { return a < 0.0f ? -a : a; }
static inline double lh_absd(double a) { return a < 0.0 ? -a : a; }

static int lh_close(float got, float want, float rel, float abs_) {
  return lh_absf(got - want) <= abs_ + rel * lh_absf(want);
}

// Print a float as a scaled integer: this bare-metal printf has no %f.
static long lh_micro(float v) { return (long)(v * 1000000.0f); }

// round-half-away-from-zero, in double. The vector path will use RVV's
// default RNE (round-half-to-even), so the quantisation gate below allows a
// difference of 1 on exact ties -- see check 3.
static int32_t lh_rnd(double v) {
  return (int32_t)(v >= 0.0 ? v + 0.5 : v - 0.5);
}

static int8_t lh_q8(double v) {
  int32_t q = lh_rnd(v);
  if (q >  127) q =  127;
  if (q < -127) q = -127;
  return (int8_t)q;
}

// --------------------------------------------------------------------------
// GEMV golden model — 8 sampled outputs, ONE pass over k
// --------------------------------------------------------------------------
// The j's are clustered into two 64-byte lines of each B row on purpose: the
// obvious per-element reference walks B down a column with stride M, i.e. a
// fresh L2 miss on every one of 2048 steps per sample. Fused and clustered,
// the whole check is ~16k MACs and ~2k misses instead of ~16k misses.
// They straddle the DIM = 16 tile boundaries (0/15/16/17/31/32) and include
// the last column to catch a wrong stride or a dropped tail.
#define LH_NSAMP 8
static const size_t lh_sample_j[LH_NSAMP] = { 0, 1, 15, 16, 17, 31, 32,
                                              LH_M - 1 };

static void lh_ref_gemv(const elem_t *A, const elem_t *B, acc_t *out) {
  for (int s = 0; s < LH_NSAMP; s++) out[s] = 0;
  for (size_t k = 0; k < LH_K; k++) {
    const elem_t *brow = B + k * LH_M;
    const acc_t a = (acc_t)A[k];
    for (int s = 0; s < LH_NSAMP; s++)
      out[s] += a * (acc_t)brow[lh_sample_j[s]];
  }
}

// --------------------------------------------------------------------------
// L2 flush — stream 2 MiB of unrelated DRAM through the 512 KiB L2 using
// Gemmini's OWN DMA, so the eviction happens on exactly the path the kernel
// will use. 2048 mvins of 16 rows x 64 B = 1024 contiguous bytes each; the
// scratchpad destination rotates so consecutive commands never alias (an
// aliasing write-write pair would serialise in the ReservationStation and
// turn the flush into a latency test instead of a bandwidth stream).
// Cost: ~2 MiB / 8 B per cycle = ~262k cycles, OUTSIDE the timed region.
// --------------------------------------------------------------------------
static void lh_flush_l2(void) {
  const size_t sp_rows = (size_t)BANK_NUM * BANK_ROWS;
  gemmini_extended3_config_ld(64, MVIN_SCALE_IDENTITY, false, 0);
  size_t sp = 0;
  for (size_t off = 0; off < LH_FLUSH_BYTES; off += 1024) {
    gemmini_extended_mvin(LH_FLUSH + off, sp, 64, 16);
    sp += 16;
    if (sp + 16 > sp_rows) sp = 0;
  }
  gemmini_fence();
}

// --------------------------------------------------------------------------

int main(void)
{
  elem_t  *A    = LH_A;
  elem_t  *B    = LH_B;
  acc_t   *C    = LH_C;
  float   *X    = LH_X;
  float   *G    = LH_G;
  float   *xn   = LH_XN;
  elem_t  *xq   = LH_XQ;
  acc_t   *LG   = LH_LG;
  int32_t *amax = LH_AMAX;

  float  *ref_xn = LH_REF_XN;
  int8_t *ref_xq = LH_REF_XQ;

  printf("llama-lmhead-fused N,K,M = %d,%d,%d  H,L = %d,%d\n",
         LH_N, LH_K, LH_M, LH_H, LH_L);

  // crt.S leaves the vector unit OFF (mstatus.VS = 0), so the first Saturn
  // instruction would trap. Enable it here rather than in the agent-owned
  // file: it is a property of the environment, not of the schedule.
  // MSTATUS_VS = 0x600 (bits 10:9); `csrs` needs a register for a 12-bit imm.
  { uint64_t vs = 0x600UL; __asm__ volatile("csrs mstatus, %0" :: "r"(vs)); }
  __asm__ volatile("vsetivli x0, 1, e32, m1, ta, ma");   // smoke-test the unit

  gemmini_flush(0);

  // --- data: the int8 / raw-byte arrays ------------------------------------
  for (int i = 0; i < LH_SEED_WORDS; i++)
    lh_seed_block[i] = lh_rng_next();
  lh_fill_bytes(A, (size_t)LH_N * LH_K);
  lh_fill_bytes(B, (size_t)LH_K * LH_M);
  lh_fill_bytes(LH_FLUSH, LH_FLUSH_BYTES);

  // --- data: X and G must be REAL floats ----------------------------------
  // Raw random bytes reinterpreted as fp32 are NaNs, infinities and
  // denormals; the rmsnorm reduction would be meaningless. X is uniform in
  // about [-1, 1) and G (the per-channel gain) in [0.5, 1.5), so
  // ss = mean(X^2) ~ 1/3, the normaliser ~1.73 and xn lands in about
  // [-2.6, 2.6] -- a range where the fp32 rmsnorm has something to do and
  // the int8 quantiser uses its whole codebook.
  for (size_t i = 0; i < LH_H; i++) {
    uint64_t u = lh_rng_next();
    X[i] = (float)((int32_t)(u & 0xFFFFu) - 32768) * (1.0f / 32768.0f);
    G[i] = 0.5f + (float)((u >> 32) & 0xFFFFu) * (1.0f / 65536.0f);
  }

  // --- data: the logit vector standing for the PREVIOUS tile's output -----
  // int32 in about +-8.4M, i.e. the magnitude a 2048-term int8 dot product
  // actually reaches (2048 * 127 * 127 = 33.0M is the bound).
  for (size_t i = 0; i < LH_L; i++)
    LG[i] = (acc_t)((int32_t)((lh_rng_next() >> 40) & 0xFFFFFFu) - 8388608);

  // The argmax check compares BOTH the value and the index exactly, so the
  // maximum must be unique -- otherwise a kernel with a different tie-break
  // would fail spuriously. Force uniqueness by pushing every duplicate of the
  // maximum one below it. One pass, and it cannot create a new tie at the top.
  {
    acc_t m = LG[0];
    for (size_t i = 1; i < LH_L; i++) if (LG[i] > m) m = LG[i];
    int seen = 0;
    for (size_t i = 0; i < LH_L; i++)
      if (LG[i] == m) { if (seen) LG[i] = m - 1; else seen = 1; }
  }

  // --- golden model: rmsnorm, in double ------------------------------------
  double ss = 0.0;
  for (size_t i = 0; i < LH_H; i++) ss += (double)X[i] * (double)X[i];
  const double rms = 1.0 / __builtin_sqrt(ss / (double)LH_H + (double)LH_EPS);
  double amaxabs = 0.0;
  for (size_t i = 0; i < LH_H; i++) {
    const double v = (double)X[i] * rms * (double)G[i];
    ref_xn[i] = (float)v;
    if (lh_absd(v) > amaxabs) amaxabs = lh_absd(v);
  }

  // qscale is calibrated from the data so the int8 codebook is fully used and
  // the clamp is never the thing being tested.
  const float qscale = (float)(127.0 / (amaxabs > 0.0 ? amaxabs : 1.0));
  for (size_t i = 0; i < LH_H; i++)
    ref_xq[i] = lh_q8((double)ref_xn[i] * (double)qscale);

  // --- golden model: running argmax ---------------------------------------
  int32_t ref_amax_val = -2147483647 - 1;
  int32_t ref_amax_idx = -1;
  for (size_t i = 0; i < LH_L; i++)
    if ((int32_t)LG[i] > ref_amax_val) {
      ref_amax_val = (int32_t)LG[i];
      ref_amax_idx = (int32_t)i;
    }

  printf("PROBE ref_rms_micro=%ld\n", lh_micro((float)rms));
  printf("PROBE ref_xn_absmax_micro=%ld\n", lh_micro((float)amaxabs));
  printf("PROBE ref_qscale_micro=%ld\n", lh_micro(qscale));
  printf("PROBE ref_amax_val=%ld ref_amax_idx=%ld\n",
         (long)ref_amax_val, (long)ref_amax_idx);

  // --- caller-zeroed / caller-seeded outputs -------------------------------
  for (size_t i = 0; i < (size_t)LH_N * LH_M; i++) C[i] = 0;
  for (size_t i = 0; i < LH_H; i++) { xn[i] = 0.0f; xq[i] = 0; }
  amax[0] = -2147483647 - 1;      // running max value
  amax[1] = -1;                   // running max index

  // --- COLD STATE: evict everything the kernel will read -------------------
  printf("Flushing L2 (%lu B through Gemmini's DMA)\n",
         (unsigned long)LH_FLUSH_BYTES);
  lh_flush_l2();

  printf("Starting llama lmhead fused\n");
  uint64_t start = read_cycles();

  llama_lmhead_fused(LH_N, LH_K, LH_M, A, B, C,
                     LH_H, X, G, xn, xq, LH_EPS, qscale,
                     LH_L, LG, amax);

  uint64_t end = read_cycles();
  printf("Cycles taken: %lu\n", (unsigned long)(end - start));

  int bad = 0;

  // --- check 1: GEMV, 8 sampled outputs, EXACT int32 -----------------------
  {
    acc_t want[LH_NSAMP];
    lh_ref_gemv(A, B, want);
    for (int s = 0; s < LH_NSAMP; s++) {
      size_t j = lh_sample_j[s];
      acc_t got = C[j];
      if (got != want[s]) {
        printf("MISMATCH gemv C[0][%d]: got %d want %d\n",
               (int)j, (int)got, (int)want[s]);
        bad = 1;
      }
    }
  }

  // --- check 2: rmsnorm, ALL 2048, 1e-4 relative / 1e-5 absolute ----------
  // Same gate as `llama-rmsnorm`'s own harness. The reduction may be
  // reassociated but must cover all H terms; a dropped tail moves the
  // normaliser and every element fails at once.
  {
    int n = 0;
    for (size_t i = 0; i < LH_H; i++) {
      if (!lh_close(xn[i], ref_xn[i], 1e-4f, 1e-5f)) {
        if (n < 4)
          printf("MISMATCH xn[%d]: got_micro %ld want_micro %ld\n",
                 (int)i, lh_micro(xn[i]), lh_micro(ref_xn[i]));
        n++; bad = 1;
      }
    }
    if (n) printf("MISMATCH xn: %d of %d wrong\n", n, LH_H);
  }

  // --- check 3: int8 quantisation, ALL 2048 -------------------------------
  // Two gates. Per element, at most 1 LSB of difference: the reference rounds
  // half AWAY from zero in double, RVV's `vfcvt.x.f` rounds half to EVEN in
  // fp32, so an exact tie or a value one ULP from a tie may legitimately
  // differ by one. In AGGREGATE, at most 8 LSBs over all 2048 elements: a
  // kernel that is systematically off by one, or that uses a wrong scale,
  // blows this even though every element passes the per-element gate.
  {
    int n = 0;
    long total = 0;
    for (size_t i = 0; i < LH_H; i++) {
      int d = (int)xq[i] - (int)ref_xq[i];
      if (d < 0) d = -d;
      total += d;
      if (d > 1) {
        if (n < 4)
          printf("MISMATCH xq[%d]: got %d want %d\n",
                 (int)i, (int)xq[i], (int)ref_xq[i]);
        n++; bad = 1;
      }
    }
    if (n) printf("MISMATCH xq: %d of %d off by more than 1\n", n, LH_H);
    printf("xq total abs error = %ld  (limit 8)\n", total);
    if (total > 8) { printf("MISMATCH xq aggregate\n"); bad = 1; }
  }

  // --- check 4: running argmax, EXACT value AND index ---------------------
  // The maximum was made unique above, so there is exactly one right answer
  // and no tolerance is warranted. A kernel that folds only part of LG almost
  // certainly misses the maximum; one that folds it all but mis-tracks the
  // index gets the value right and the index wrong.
  {
    if (amax[0] != ref_amax_val || amax[1] != ref_amax_idx) {
      printf("MISMATCH argmax: got val %ld idx %ld want val %ld idx %ld\n",
             (long)amax[0], (long)amax[1],
             (long)ref_amax_val, (long)ref_amax_idx);
      bad = 1;
    }
  }

  // --- checksums over the FULL outputs ------------------------------------
  // A kernel that only writes the sampled GEMV columns, or only the first
  // half of `xn`, is caught here even though the GEMV gate is sampled.
  {
    int64_t csum = 0;
    for (size_t i = 0; i < (size_t)LH_N * LH_M; i++) csum += C[i];
    printf("C checksum = %ld\n", (long)csum);
    double xs = 0.0;
    int64_t qs = 0;
    for (size_t i = 0; i < LH_H; i++) { xs += (double)xn[i]; qs += xq[i]; }
    printf("xn sum_micro = %ld\n", lh_micro((float)xs));
    printf("xq checksum = %ld\n", (long)qs);
    printf("argmax val = %ld idx = %ld\n", (long)amax[0], (long)amax[1]);
  }

  if (bad) {
    printf("FAILED\n");
    exit(1);
  }
  printf("PASSED (gemv %d sampled, xn %d, xq %d, argmax 1)\n",
         LH_NSAMP, LH_H, LH_H);
  exit(0);
}
