// See LICENSE for license details.
//
//**************************************************************************
// llama-layer-fused — one Llama-3.2-1B decoder-layer SLICE at decode N = 1,
// with the Gemmini weight DMA and the Saturn vector work in ONE timed region
//--------------------------------------------------------------------------
//
// SEALED HARNESS BODY. Included by the thin `bareMetalC/llama_layer_fused_n1.c`
// wrapper, which is the registry's `main_src`. It holds main(), the data
// generator, the L2 flush that puts the machine in a COLD state, the scalar
// golden models and all four self-checks. The optimizer never gets to change
// any of it: every build is reassembled from the pristine tree plus exactly
// one file, `include/llama_layer_fused_n1_kernel.h`.
//
// WHY THIS BENCHMARK EXISTS
// -------------------------
// `llama-q8-gemv-gemmini-n1` measures the Gemmini decode GEMV alone: 1 MiB of
// int8 weights streamed at ~7.9 B/cycle over an 8 B/cycle mbus, 132,424
// cycles, and the 16x16 array 97% idle the whole time.
// `llama-attn-scores-int8` / `llama-softmax` / `llama-attn-pv-int8` measure
// the Saturn side of the same decoder layer: 8,233 + 1,555 + 4,814 = 14,602
// cycles that move ~70 KiB, i.e. 1% of the traffic and ~10% of the time.
//
// Up to Round 6 the Llama cost model (`loop/llama_project.py`) added those two
// numbers. It had to, because nothing had ever measured them TOGETHER. But
// Gemmini's `mvin`/`preload`/`compute` are asynchronous RoCC commands: once
// they are in the ld/ex queues the host core is free, and the Saturn vector
// unit is a *different* functional unit on that same host core. There is no
// architectural reason the attention math cannot execute inside the shadow of
// the weight DMA. This harness is the experiment that decides it.
//
// The baseline runs the two phases back to back, Gemmini fully fenced before
// the first vector instruction, and times the whole thing. The optimizer's job
// is to software-pipeline them. Nothing else about the work may change.
//
// COLD STATE
// ----------
// Every earlier Gemmini entry inherited a warm L2 tail: the harness fills B in
// DRAM immediately before calling the kernel, so ~200-260 KiB of the 1 MiB is
// still resident and descending-K order harvests it. That is a benchmark
// artifact worth ~17,000 cycles, and it makes any overlap measurement
// ambiguous (did the schedule win, or did the cache?). So this harness streams
// 2 MiB of UNRELATED data through the 512 KiB L2 with Gemmini's own DMA right
// before the timed region. Everything the kernel touches starts cold, on the
// same path the kernel will use. Expect the GEMV part to cost ~149k rather
// than ~132k for exactly that reason; that is the honest number.
//
// Wall-clock discipline (Verilator runs this SoC at ~1.8k cycles/s):
//   * every array lives at a fixed DRAM address OUTSIDE the ELF. The loader
//     zeroes .bss at roughly 12 s per KiB over TSI, so a 1 MiB `static` array
//     would cost hours before main() runs. Only ~1.3 KiB of .bss exists here.
//   * the fill is a 512-byte scalar-generated block XOR-ed with a rotating
//     64-bit key, copied 8 bytes at a time.
//   * the GEMV self-check is SAMPLED (8 outputs) and FUSED into one pass over
//     k. The three Saturn checks are FULL (512 + 512 + 64 outputs) because
//     they are only ~65k scalar MACs in total.
//   * the reference exp is a local double-precision minimax, NOT libm: it
//     avoids any dependence on newlib reentrancy in a -nostdlib build.

#ifndef LF_N
#error "include bareMetalC/llama_layer_fused_n1.c, not this file"
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

#include "include/gemmini_testutils.h"

// --------------------------------------------------------------------------
// Shapes — one decoder-layer slice of Llama-3.2-1B at decode N = 1, S = 512
// --------------------------------------------------------------------------
//   GEMMINI : C[1 x 512]  = A[1 x 2048] * B[2048 x 512]   (1 MiB int8 weights)
//             exactly the `llama-q8-gemv-gemmini-n1` shape, so the two cycle
//             counts are directly comparable.
//   SATURN  : scores[512] = QK^T/sqrt(d) for ONE query head, int8 K + int8 q
//             probs[512]  = softmax(scores)
//             pvout[64]   = probs@V for ONE head, int8 V + int8 probs
//             exactly the `llama-attn-scores-int8` / `llama-softmax` /
//             `llama-attn-pv-int8` shapes.
#define LF_K     2048          // hidden size
#define LF_M      512          // one output tile of a projection matrix
#define LF_S      512          // KV-cache length
#define LF_D       64          // head dim
#define LF_SCALE  0.125f       // 1/sqrt(64)
#define LF_QSCALE 1.0f

// DRAM (memory@80000000, 256 MiB) past the end of the ~64 KiB image.
//   0x81000000  A          1 x 2048 int8                      2 KiB
//   0x81100000  B          2048 x 512 int8                    1 MiB
//   0x81300000  C          1 x 512 int32                      2 KiB
//   0x81400000  K8         512 x 64 int8                     32 KiB
//   0x81500000  V8         512 x 64 int8                     32 KiB
//   0x81600000  kscale     512 fp32                           2 KiB
//   0x81700000  q8         64 int8    / +0x1000  W8  512 int8
//   0x81800000  scores 512 fp32 / +0x1000 probs / +0x2000 pvacc / +0x3000 pvout
//   0x81A00000  reference arrays (never read by the kernel)
//   0x81C00000  flush buffer, 2 MiB   -> ends 0x81E00000
// All 1 MiB- or 4 KiB-aligned, so every row start is 16 B clean.
#define LF_A          ((elem_t  *)0x81000000UL)
#define LF_B          ((elem_t  *)0x81100000UL)
#define LF_C          ((acc_t   *)0x81300000UL)
#define LF_K8         ((elem_t  *)0x81400000UL)
#define LF_V8         ((elem_t  *)0x81500000UL)
#define LF_KSCALE     ((float   *)0x81600000UL)
#define LF_Q8         ((elem_t  *)0x81700000UL)
#define LF_W8         ((elem_t  *)0x81701000UL)
#define LF_SCORES     ((float   *)0x81800000UL)
#define LF_PROBS      ((float   *)0x81801000UL)
#define LF_PVACC      ((int32_t *)0x81802000UL)
#define LF_PVOUT      ((float   *)0x81803000UL)
#define LF_REF_DOT    ((int32_t *)0x81A00000UL)
#define LF_REF_SCORES ((float   *)0x81A01000UL)
#define LF_REF_PROBS  ((float   *)0x81A02000UL)
#define LF_REF_PVACC  ((int32_t *)0x81A03000UL)
#define LF_REF_PVOUT  ((float   *)0x81A04000UL)
#define LF_FLUSH      ((elem_t  *)0x81C00000UL)
#define LF_FLUSH_BYTES (2UL * 1024 * 1024)

// --------------------------------------------------------------------------
// Deterministic pseudo-random fill (fixed seed -> reproducible every run)
// --------------------------------------------------------------------------

static uint64_t lf_rng = 0x243F6A8885A308D3ULL;

static inline uint64_t lf_rng_next(void) {
  lf_rng ^= lf_rng << 13;
  lf_rng ^= lf_rng >> 7;
  lf_rng ^= lf_rng << 17;
  return lf_rng;
}

#define LF_SEED_WORDS 64                 // 512 B, essentially the whole .bss
static uint64_t lf_seed_block[LF_SEED_WORDS];

static void lf_fill_bytes(void *dst, size_t bytes) {
  uint64_t *d = (uint64_t *)dst;
  const size_t words = bytes / 8;
  uint64_t key = 0;
  for (size_t i = 0; i < words; i++) {
    if ((i & (LF_SEED_WORDS - 1)) == 0) key = lf_rng_next();
    d[i] = lf_seed_block[i & (LF_SEED_WORDS - 1)] ^ key;
  }
}

// --------------------------------------------------------------------------
// Reference exp — local, double precision, no libm
// --------------------------------------------------------------------------
// exp(x) = 2^n * exp(r), n = round(x*log2e), |r| <= ln2/2 = 0.3466.
// A 10-term Taylor series in r has |error| < r^11/11! < 1.5e-13 relative, far
// inside the 5e-3 gate below, and 2^n is built by repeated doubling so no
// ldexp/scalbn (and therefore no libm) is needed. `-ffast-math` is on, but
// every step here is a single multiply or add at double precision, so
// reassociation cannot move the result by more than a few ULP.
static double lf_exp(double x) {
  int n = (int)(x * 1.4426950408889634 + (x >= 0.0 ? 0.5 : -0.5));
  double r = x - (double)n * 0.6931471805599453;
  double term = 1.0, sum = 1.0;
  for (int i = 1; i <= 10; i++) { term *= r / (double)i; sum += term; }
  double s = 1.0;
  if (n > 0)      for (int i = 0; i < n;  i++) s *= 2.0;
  else if (n < 0) for (int i = 0; i < -n; i++) s *= 0.5;
  return sum * s;
}

static inline float lf_absf(float a) { return a < 0.0f ? -a : a; }

// got within `rel` relative (plus `abs_` absolute) of want?
static int lf_close(float got, float want, float rel, float abs_) {
  return lf_absf(got - want) <= abs_ + rel * lf_absf(want);
}

// Print a float as a scaled integer: this bare-metal printf has no %f.
static long lf_micro(float v) { return (long)(v * 1000000.0f); }

// --------------------------------------------------------------------------
// GEMV golden model — 8 sampled outputs, ONE pass over k
// --------------------------------------------------------------------------
// The j's are clustered into two 64-byte lines of each B row on purpose: the
// obvious per-element reference walks B down a column with stride M, i.e. a
// fresh L2 miss on every one of 2048 steps per sample. Fused and clustered,
// the whole check is ~16k MACs and ~2k misses instead of ~16k misses.
// They still straddle the DIM = 16 tile boundaries (0/15/16/17/31/32) and
// include the last column (511) to catch a wrong stride or a dropped tail.
#define LF_NSAMP 8
static const size_t lf_sample_j[LF_NSAMP] = { 0, 1, 15, 16, 17, 31, 32, 511 };

static void lf_ref_gemv(const elem_t *A, const elem_t *B, acc_t *out) {
  for (int s = 0; s < LF_NSAMP; s++) out[s] = 0;
  for (size_t k = 0; k < LF_K; k++) {
    const elem_t *brow = B + k * LF_M;
    const acc_t a = (acc_t)A[k];
    for (int s = 0; s < LF_NSAMP; s++)
      out[s] += a * (acc_t)brow[lf_sample_j[s]];
  }
}

// --------------------------------------------------------------------------
// L2 flush — stream 2 MiB of unrelated DRAM through the 512 KiB L2 using
// Gemmini's OWN DMA, so the eviction happens on exactly the path the kernel
// will use. 2048 mvins of 16 rows x 64 B = 1024 contiguous bytes each; the
// scratchpad destination rotates so consecutive commands never alias (an
// aliasing write-write pair would serialise in the ReservationStation and
// turn the flush into a latency test instead of a bandwidth stream).
// Cost: ~2 MiB / 8 B per cycle = ~262k cycles, outside the timed region.
// --------------------------------------------------------------------------
static void lf_flush_l2(void) {
  const size_t sp_rows = (size_t)BANK_NUM * BANK_ROWS;
  gemmini_extended3_config_ld(64, MVIN_SCALE_IDENTITY, false, 0);
  size_t sp = 0;
  for (size_t off = 0; off < LF_FLUSH_BYTES; off += 1024) {
    gemmini_extended_mvin(LF_FLUSH + off, sp, 64, 16);
    sp += 16;
    if (sp + 16 > sp_rows) sp = 0;
  }
  gemmini_fence();
}

// --------------------------------------------------------------------------

int main(void)
{
  elem_t  *A      = LF_A;
  elem_t  *B      = LF_B;
  acc_t   *C      = LF_C;
  elem_t  *K8     = LF_K8;
  elem_t  *V8     = LF_V8;
  float   *kscale = LF_KSCALE;
  elem_t  *q8     = LF_Q8;
  elem_t  *W8     = LF_W8;
  float   *scores = LF_SCORES;
  float   *probs  = LF_PROBS;
  int32_t *pvacc  = LF_PVACC;
  float   *pvout  = LF_PVOUT;

  int32_t *ref_dot    = LF_REF_DOT;
  float   *ref_scores = LF_REF_SCORES;
  float   *ref_probs  = LF_REF_PROBS;
  int32_t *ref_pvacc  = LF_REF_PVACC;
  float   *ref_pvout  = LF_REF_PVOUT;

  printf("llama-layer-fused N,K,M = %d,%d,%d  S,d = %d,%d\n",
         LF_N, LF_K, LF_M, LF_S, LF_D);

  // crt.S leaves the vector unit OFF (mstatus.VS = 0), so the first Saturn
  // instruction would trap. Enable it here rather than in the agent-owned
  // file: it is a property of the environment, not of the schedule.
  // MSTATUS_VS = 0x600 (bits 10:9); `csrs` needs a register for a 12-bit imm.
  { uint64_t vs = 0x600UL; __asm__ volatile("csrs mstatus, %0" :: "r"(vs)); }
  __asm__ volatile("vsetivli x0, 1, e32, m1, ta, ma");   // smoke-test the unit

  gemmini_flush(0);

  // --- data -----------------------------------------------------------------
  for (int i = 0; i < LF_SEED_WORDS; i++)
    lf_seed_block[i] = lf_rng_next();
  lf_fill_bytes(A,  (size_t)LF_N * LF_K);
  lf_fill_bytes(B,  (size_t)LF_K * LF_M);
  lf_fill_bytes(K8, (size_t)LF_S * LF_D);
  lf_fill_bytes(V8, (size_t)LF_S * LF_D);
  lf_fill_bytes(q8, LF_D);
  lf_fill_bytes(W8, LF_S);
  lf_fill_bytes(LF_FLUSH, LF_FLUSH_BYTES);

  // --- golden model: QK^T integer dots (also needed to calibrate kscale) ---
  int32_t maxabs = 1;
  for (size_t s = 0; s < LF_S; s++) {
    const elem_t *krow = K8 + s * LF_D;
    int32_t dot = 0;
    for (size_t j = 0; j < LF_D; j++)
      dot += (int32_t)krow[j] * (int32_t)q8[j];
    ref_dot[s] = dot;
    int32_t a = dot < 0 ? -dot : dot;
    if (a > maxabs) maxabs = a;
  }

  // kscale is calibrated from the data so the fp32 scores land in roughly
  // [-8, 8]: wide enough that softmax's max-subtraction is load-bearing (the
  // `llama-softmax` harness uses the same range), narrow enough that exp never
  // approaches the normal-float limit. The 0.6..1.4 per-row jitter keeps the
  // per-KV-position scale genuinely per-row, as GQA int8 KV caches are.
  const float base = 8.0f / ((float)maxabs * LF_SCALE);
  for (size_t s = 0; s < LF_S; s++) {
    uint64_t u = lf_rng_next();
    kscale[s] = base * (0.6f + 0.8f * (float)(u & 0xFFFF) / 65536.0f);
  }

  // --- golden model: scores, softmax, pv -----------------------------------
  const float fs = LF_QSCALE * LF_SCALE;
  float smax = -3.4028235e38f;
  for (size_t s = 0; s < LF_S; s++) {
    ref_scores[s] = (float)ref_dot[s] * kscale[s] * fs;
    if (ref_scores[s] > smax) smax = ref_scores[s];
  }
  double esum = 0.0;
  for (size_t s = 0; s < LF_S; s++) {
    double e = lf_exp((double)ref_scores[s] - (double)smax);
    ref_probs[s] = (float)e;             // unnormalised for now
    esum += e;
  }
  for (size_t s = 0; s < LF_S; s++)
    ref_probs[s] = (float)((double)ref_probs[s] / esum);

  // wscale: the probability vector is quantised with ONE fp32 scale after the
  // per-row V scales have been folded in (see `llama-attn-pv-int8`'s contract).
  // Any positive constant exercises the same arithmetic; this one keeps pvout
  // O(1).
  const float wscale = 1.0f / (512.0f * 127.0f);
  for (size_t j = 0; j < LF_D; j++) ref_pvacc[j] = 0;
  for (size_t s = 0; s < LF_S; s++) {
    const elem_t *vrow = V8 + s * LF_D;
    const int32_t w = (int32_t)W8[s];
    for (size_t j = 0; j < LF_D; j++)
      ref_pvacc[j] += w * (int32_t)vrow[j];
  }
  for (size_t j = 0; j < LF_D; j++)
    ref_pvout[j] = (float)ref_pvacc[j] * wscale;

  {
    float smin = 3.4028235e38f;
    for (size_t s = 0; s < LF_S; s++)
      if (ref_scores[s] < smin) smin = ref_scores[s];
    printf("PROBE ref_score_min_micro=%ld\n", lf_micro(smin));
    printf("PROBE ref_score_max_micro=%ld\n", lf_micro(smax));
  }

  // --- caller-zeroed outputs (the same courtesy every sibling kernel gets) --
  for (size_t i = 0; i < (size_t)LF_N * LF_M; i++) C[i] = 0;
  for (size_t s = 0; s < LF_S; s++) { scores[s] = 0.0f; probs[s] = 0.0f; }
  for (size_t j = 0; j < LF_D; j++) { pvacc[j] = 0; pvout[j] = 0.0f; }

  // --- COLD STATE: evict everything the kernel will read -------------------
  printf("Flushing L2 (%lu B through Gemmini's DMA)\n",
         (unsigned long)LF_FLUSH_BYTES);
  lf_flush_l2();

  printf("Starting llama layer fused\n");
  uint64_t start = read_cycles();

  llama_layer_fused(LF_N, LF_K, LF_M, A, B, C,
                    LF_S, LF_D, K8, kscale, q8, scores, probs,
                    W8, V8, pvacc, pvout,
                    LF_QSCALE, LF_SCALE, wscale);

  uint64_t end = read_cycles();
  printf("Cycles taken: %lu\n", (unsigned long)(end - start));

  int bad = 0;

  // --- check 1: GEMV, 8 sampled outputs, exact int32 -----------------------
  {
    acc_t want[LF_NSAMP];
    lf_ref_gemv(A, B, want);
    for (int s = 0; s < LF_NSAMP; s++) {
      size_t j = lf_sample_j[s];
      acc_t got = C[j];
      if (got != want[s]) {
        printf("MISMATCH gemv C[0][%d]: got %d want %d\n",
               (int)j, (int)got, (int)want[s]);
        bad = 1;
      }
    }
  }

  // --- check 2: scores, all 512, 1e-4 relative ----------------------------
  // The integer dot has exactly one right answer, so this gate is as strict
  // as the `llama-attn-scores-int8` Gate A: only the final fp32 multiply by
  // kscale*qscale*scale can differ, and that is one rounding.
  {
    int n = 0;
    for (size_t s = 0; s < LF_S; s++) {
      if (!lf_close(scores[s], ref_scores[s], 1e-4f, 1e-6f)) {
        if (n < 4)
          printf("MISMATCH scores[%d]: got_micro %ld want_micro %ld\n",
                 (int)s, lf_micro(scores[s]), lf_micro(ref_scores[s]));
        n++; bad = 1;
      }
    }
    if (n) printf("MISMATCH scores: %d of %d wrong\n", n, LF_S);
  }

  // --- check 3: softmax, all 512, 5e-3 relative + sum == 1 ----------------
  // Tolerance rationale: the reference probabilities are built from the
  // reference SCORES, not from the kernel's, so this gate absorbs (a) the
  // 1e-4 the scores gate already allows, amplified by exp to at most
  // |x|*1e-4 <= 8e-4 relative, and (b) the exp approximation the softmax
  // kernel is allowed (1e-4 in its own harness). 5e-3 is ~5x that sum, and
  // the sum-to-one check below is the tight one: it catches a missing
  // normalisation or a dropped tail that a loose per-element gate would not.
  {
    int n = 0;
    float psum = 0.0f;
    for (size_t s = 0; s < LF_S; s++) {
      psum += probs[s];
      if (!lf_close(probs[s], ref_probs[s], 5e-3f, 1e-9f)) {
        if (n < 4)
          printf("MISMATCH probs[%d]: got_micro %ld want_micro %ld\n",
                 (int)s, lf_micro(probs[s]), lf_micro(ref_probs[s]));
        n++; bad = 1;
      }
    }
    if (n) printf("MISMATCH probs: %d of %d wrong\n", n, LF_S);
    printf("probs sum_micro = %ld  (want 1000000)\n", lf_micro(psum));
    if (!lf_close(psum, 1.0f, 1e-3f, 0.0f)) {
      printf("MISMATCH probs sum\n");
      bad = 1;
    }
  }

  // --- check 4: pv, all 64, exact int32 accumulator + 1e-4 on the fp out --
  {
    for (size_t j = 0; j < LF_D; j++) {
      if (pvacc[j] != ref_pvacc[j]) {
        printf("MISMATCH pvacc[%d]: got %d want %d\n",
               (int)j, (int)pvacc[j], (int)ref_pvacc[j]);
        bad = 1;
      }
      if (!lf_close(pvout[j], ref_pvout[j], 1e-4f, 1e-9f)) {
        printf("MISMATCH pvout[%d]: got_micro %ld want_micro %ld\n",
               (int)j, lf_micro(pvout[j]), lf_micro(ref_pvout[j]));
        bad = 1;
      }
    }
  }

  // --- checksums over the FULL outputs ------------------------------------
  // A kernel that only writes the sampled GEMV columns, or only the first
  // half of `scores`, is caught here even though the gates above are sampled
  // (GEMV) or tolerant (softmax).
  {
    int64_t csum = 0;
    for (size_t i = 0; i < (size_t)LF_N * LF_M; i++) csum += C[i];
    printf("C checksum = %ld\n", (long)csum);
    double ssum = 0.0, osum = 0.0;
    for (size_t s = 0; s < LF_S; s++) ssum += (double)scores[s];
    for (size_t j = 0; j < LF_D; j++) osum += (double)pvout[j];
    printf("scores sum_micro = %ld\n", lf_micro((float)ssum));
    printf("pvout sum_micro = %ld\n", lf_micro((float)osum));
  }

  if (bad) {
    printf("FAILED\n");
    exit(1);
  }
  printf("PASSED (gemv %d sampled, scores %d, probs %d, pv %d)\n",
         LF_NSAMP, LF_S, LF_S, LF_D);
  exit(0);
}
