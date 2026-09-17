// See LICENSE for license details.
//
//**************************************************************************
// llama-q8-gemv-gemmini — int8 GEMV/skinny-GEMM on Gemmini, decode shape
//--------------------------------------------------------------------------
//
// SEALED HARNESS BODY. Included by the two thin `bareMetalC/*.c` wrappers,
// which are the registry's `main_src`. It holds main(), the data generator,
// the scalar golden model and the self-check. The optimizer never gets to
// change any of it: every build is reassembled from the pristine tree plus
// exactly one file, `include/llama_q8_gemv_gemmini_n{1,16}_kernel.h`.
//
// Why this file lives in include/ and not next to the .c: the Gemmini compile
// line builds exactly ONE translation unit (the entry's `main_src`), and
// `include/*.h` is already in constants.GEMMINI_SEALED_GLOBS, so the whole
// harness travels with the collateral for free.
//
//   C[N x M] = A[N x K] * B[K x M],  int8 inputs, int32 output
//   N = LQ8_GV_N  decode batch (1 = one token, 16 = one full Gemmini tile row)
//   K = 2048      Llama-3.2-1B hidden size
//   M = 512       one output tile of a projection matrix
//
// M and K deliberately match the Saturn `llama-q8-gemv` kernel so the two
// numbers are directly comparable.
//
// THE QUESTION THIS KERNEL EXISTS TO ANSWER: at N = 1 the 16x16 array gets one
// valid row out of 16, so the weight matrix B still has to be moved in in full
// but only 1/16 of the array's MACs do useful work. Linearly extrapolating a
// large-N GEMM measurement down to N = 1 therefore *overstates* Gemmini's
// decode throughput; this harness measures it instead.
//
// Wall-clock discipline (Verilator runs this SoC at ~1.8k cycles/s):
//   * A, B and C are at fixed DRAM addresses OUTSIDE the ELF. The harness
//     loads the image over TSI and zeroes .bss at roughly 12 s per KiB, so a
//     1 MiB `static` array would cost hours before main() runs.
//   * the fill is a 512-byte scalar-generated block XOR-ed with a rotating
//     64-bit key, copied 8 bytes at a time.
//   * the self-check is SAMPLED and FUSED: 8 outputs, all in ONE pass over k.

#ifndef LQ8_GV_N
#error "include one of bareMetalC/llama_q8_gemv_gemmini_n{1,16}.c, not this file"
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

#include "include/gemmini_testutils.h"

#define LQ8_GV_K 2048
#define LQ8_GV_M 512

// DRAM (memory@80000000, 256 MiB) past the end of the ~64 KiB image:
//   0x81000000  A : N x 2048 int8            <= 32 KiB
//   0x81100000  B : 2048 x 512 int8          =   1 MiB
//   0x81300000  C : N x 512 int32            <= 32 KiB
// All 1 MiB-aligned, so every row start is `row_align(1)`-clean (16 B).
#define LQ8_GV_A ((elem_t *)0x81000000UL)
#define LQ8_GV_B ((elem_t *)0x81100000UL)
#define LQ8_GV_C ((acc_t  *)0x81300000UL)

// --------------------------------------------------------------------------
// Deterministic pseudo-random fill (fixed seed -> reproducible every run)
// --------------------------------------------------------------------------

static uint64_t rng = 0x243F6A8885A308D3ULL;

static inline uint64_t rng_next(void) {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return rng;
}

#define SEED_WORDS 64                 // 512 B, essentially the whole .bss
static uint64_t seed_block[SEED_WORDS];

static void fill_bytes(void *dst, size_t bytes) {
  uint64_t *d = (uint64_t *)dst;
  const size_t words = bytes / 8;
  uint64_t key = 0;
  for (size_t i = 0; i < words; i++) {
    if ((i & (SEED_WORDS - 1)) == 0) key = rng_next();
    d[i] = seed_block[i & (SEED_WORDS - 1)] ^ key;
  }
}

// --------------------------------------------------------------------------
// Scalar golden model — 8 sampled outputs, ONE pass over k
// --------------------------------------------------------------------------
// The j's are clustered into two 64-byte lines of each B row on purpose: the
// obvious per-element reference walks B down a column with stride M, i.e. a
// fresh L2 miss on every one of 2048 steps per sample. Fused and clustered,
// the whole check is ~16k MACs and ~4k misses instead of ~16k misses.
// They still straddle the DIM = 16 tile boundaries (0/15/16/17/31/32) and
// include the last column (511) to catch a wrong stride or a dropped tail.
#define N_SAMPLES 8
static const size_t sample_i_raw[N_SAMPLES] = { 0, 0,  1, 15,  8,  3, 15,  7 };
static const size_t sample_j[N_SAMPLES]     = { 0, 1, 15, 16, 17, 31, 32, 511 };

static size_t sample_i[N_SAMPLES];

static void ref_samples(const elem_t *A, const elem_t *B, acc_t *out) {
  for (int s = 0; s < N_SAMPLES; s++)
    out[s] = 0;
  for (size_t k = 0; k < LQ8_GV_K; k++) {
    const elem_t *brow = B + k * LQ8_GV_M;
    for (int s = 0; s < N_SAMPLES; s++)
      out[s] += (acc_t)A[sample_i[s] * LQ8_GV_K + k] * (acc_t)brow[sample_j[s]];
  }
}

// --------------------------------------------------------------------------

int main(void)
{
  elem_t *A = LQ8_GV_A;
  elem_t *B = LQ8_GV_B;
  acc_t  *C = LQ8_GV_C;

  printf("llama-q8-gemv-gemmini N,K,M = %d,%d,%d\n",
         LQ8_GV_N, LQ8_GV_K, LQ8_GV_M);

  gemmini_flush(0);

  for (int s = 0; s < N_SAMPLES; s++)
    sample_i[s] = sample_i_raw[s] % LQ8_GV_N;   // collapses to 0 when N == 1

  for (int i = 0; i < SEED_WORDS; i++)
    seed_block[i] = rng_next();
  fill_bytes(A, (size_t)LQ8_GV_N * LQ8_GV_K);
  fill_bytes(B, (size_t)LQ8_GV_K * LQ8_GV_M);
  for (size_t i = 0; i < (size_t)LQ8_GV_N * LQ8_GV_M; i++)
    C[i] = 0;

  printf("Starting gemmini gemv\n");
  uint64_t start = read_cycles();

  llama_q8_gemv_gemmini(LQ8_GV_N, LQ8_GV_K, LQ8_GV_M, A, B, C);

  uint64_t end = read_cycles();
  printf("Cycles taken: %lu\n", (unsigned long)(end - start));

  // --- sampled self-check -------------------------------------------------
  acc_t want_all[N_SAMPLES];
  ref_samples(A, B, want_all);

  int bad = 0;
  for (int s = 0; s < N_SAMPLES; s++) {
    size_t i = sample_i[s], j = sample_j[s];
    acc_t got = C[i * LQ8_GV_M + j], want = want_all[s];
    if (got != want) {
      printf("MISMATCH C[%d][%d]: got %d want %d\n",
             (int)i, (int)j, (int)got, (int)want);
      bad = 1;
    }
  }

  int64_t sum = 0;
  for (size_t i = 0; i < (size_t)LQ8_GV_N * LQ8_GV_M; i++)
    sum += C[i];
  printf("C checksum = %ld\n", (long)sum);

  if (bad) {
    printf("FAILED\n");
    exit(1);
  }
  printf("PASSED (%d sampled outputs)\n", N_SAMPLES);
  exit(0);
}
