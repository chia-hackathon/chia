// See LICENSE for license details.
//
//**************************************************************************
// llama-q8-gemv-gemmini-lmhead — int8 GEMV on Gemmini, LM-head shape
//--------------------------------------------------------------------------
//
// SEALED HARNESS BODY. Included by the thin `bareMetalC/*.c` wrapper, which
// is the registry's `main_src`. It holds main(), the data generator, the
// scalar golden model and the self-check. The optimizer never gets to change
// any of it: every build is reassembled from the pristine tree plus exactly
// one file, `include/llama_q8_gemv_gemmini_lmhead_kernel.h`.
//
// This is a SEPARATE body from `llama_q8_gemv_gemmini_body.h` on purpose:
// that one hard-codes M = 512 and the two decode entries (n1/n16) depend on
// it byte-for-byte. This one takes both N and M from the wrapper.
//
//   C[N x M] = A[N x K] * B[K x M],  int8 inputs, int32 output
//   N = LQ8_GV_N  = 1     one decoded token
//   K = LQ8_GV_K  = 2048  Llama-3.2-1B hidden size
//   M = LQ8_GV_M          one output tile of the LM head / vocab projection
//
// Why the LM head is its own benchmark: it is the ONE projection in
// Llama-3.2-1B that is not 2048-wide on the output side. The head is
// 2048 x 128256, i.e. ~250 MiB of int8 weights streamed for a single token —
// several times the whole rest of the model. Its per-tile behaviour is
// therefore the dominant term in a decode step, and the decode entries
// (M = 512) are too small to show what happens once B no longer fits
// anywhere near the on-chip hierarchy.
//
// Wall-clock discipline (Verilator runs this SoC at ~2k cycles/s):
//   * A, B and C are at fixed DRAM addresses OUTSIDE the ELF. The harness
//     loads the image over TSI and zeroes .bss at roughly 12 s per KiB, so a
//     multi-MiB `static` array would cost hours before main() runs.
//   * the fill is a 512-byte scalar-generated block XOR-ed with a rotating
//     64-bit key, written 64 B (one cache line) per iteration with eight
//     independent stores so the in-order core keeps several line fills in
//     flight — this loop, not the kernel, dominates the Verilator wall time.
//   * the self-check is SAMPLED and FUSED: 8 outputs, all in ONE pass over k.

#ifndef LQ8_GV_N
#error "include bareMetalC/llama_q8_gemv_gemmini_lmhead.c, not this file"
#endif
#ifndef LQ8_GV_M
#error "the wrapper must define LQ8_GV_M"
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

#include "include/gemmini_testutils.h"

#define LQ8_GV_K 2048

// DRAM (memory@80000000, 256 MiB) past the end of the ~64 KiB image:
//   0x81000000  A : N x 2048 int8              <=  32 KiB
//   0x81100000  B : 2048 x M int8              =  M/256 MiB  (<= 9 MiB of room)
//   0x81A00000  C : N x M int32                <= 256 KiB
// All 1 MiB-aligned, so every row start is `row_align(1)`-clean (16 B).
// The B window is sized for M up to 4608; a larger M must move C.
#define LQ8_GV_A ((elem_t *)0x81000000UL)
#define LQ8_GV_B ((elem_t *)0x81100000UL)
#define LQ8_GV_C ((acc_t  *)0x81A00000UL)

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

// `bytes` is always a multiple of 64 here (K*M and N*K both are).
static void fill_bytes(void *dst, size_t bytes) {
  uint64_t *d = (uint64_t *)dst;
  const size_t words = bytes / 8;
  for (size_t i = 0; i < words; i += 8) {
    const uint64_t key = rng_next();
    const uint64_t *s = &seed_block[i & (SEED_WORDS - 1)];
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
// Scalar golden model — 8 sampled outputs, ONE pass over k
// --------------------------------------------------------------------------
// The j's are clustered into two 64-byte lines of each B row on purpose: the
// obvious per-element reference walks B down a column with stride M, i.e. a
// fresh L2 miss on every one of 2048 steps per sample. Fused and clustered,
// the whole check is ~16k MACs and ~4k misses instead of ~16k misses.
// They still straddle the DIM = 16 tile boundaries (0/15/16/17/31/32) and
// include the last column (M-1) to catch a wrong stride or a dropped tail.
#define N_SAMPLES 8
static const size_t sample_i_raw[N_SAMPLES] = { 0, 0,  1, 15,  8,  3, 15,  7 };
static const size_t sample_j_raw[N_SAMPLES] = { 0, 1, 15, 16, 17, 31, 32,
                                                LQ8_GV_M - 1 };

static size_t sample_i[N_SAMPLES];
#define sample_j sample_j_raw

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

  printf("llama-q8-gemv-gemmini-lmhead N,K,M = %d,%d,%d\n",
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
