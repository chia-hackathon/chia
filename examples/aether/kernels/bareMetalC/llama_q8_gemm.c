// See LICENSE for license details.
//
//**************************************************************************
// llama-q8-gemm — int8 GEMM in the shape of a Llama-3.2-1B prefill step
//--------------------------------------------------------------------------
//
// SEALED HARNESS. main(), the pseudo-random data generator, the scalar golden
// model and the self-check live here. The optimizer only ever owns
// `include/llama_q8_gemm_kernel.h`; every build is reassembled from the
// pristine copy of this file plus the optimizer's header.
//
//   C[N x M] = A[N x K] * B[K x M],  int8 inputs, int32 output
//   N = 64    prefill tokens in the batch
//   K = 2048  Llama-3.2-1B hidden size
//   M = 256   one output tile of a projection matrix
//
// M was 512 in the first bring-up: correct, 315,825 Gemmini cycles, but 26 min
// of Verilator wall per run — over the 15 min budget an inner-loop iteration
// gets. Halving M halves the data fill and the matmul; the sampled check was
// also restructured (one fused pass over k for all 8 samples instead of eight
// separate strided walks down B), which is where most of the rest went.
//
// Wall-clock discipline (Verilator runs this SoC at ~1.8k cycles/s):
//   * A, B and C are at fixed DRAM addresses OUTSIDE the ELF. The Verilator
//     harness loads the image over TSI and zeroes .bss at roughly 12 s per
//     KiB, so 1.2 MiB of `static` arrays would cost hours before main() runs.
//   * data fill is a 512-byte scalar-generated block XOR-ed with a rotating
//     64-bit key, copied 8 bytes at a time (~2 cycles / 8 B), not a per-byte
//     PRNG (~5 cycles / byte).
//   * the self-check is SAMPLED: 8 output elements are recomputed with a
//     scalar dot product (~150k cycles). A full scalar reference would be
//     64*512*2048 = 67M MACs, i.e. days of Verilator.

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#include "include/gemmini_testutils.h"
#include "include/llama_q8_gemm_kernel.h"

#define LQ8_N 64
#define LQ8_K 2048
#define LQ8_M 256

// DRAM (memory@80000000, 256 MiB) past the end of the ~64 KiB image:
//   0x81000000  A : 64 x 2048 int8            = 128 KiB
//   0x81100000  B : 2048 x 256 int8           = 512 KiB
//   0x81300000  C : 64 x 256 int32            =  64 KiB
// All 1 MiB-aligned, so every row start is `row_align(1)`-clean (16 B).
#define LQ8_A ((elem_t *)0x81000000UL)
#define LQ8_B ((elem_t *)0x81100000UL)
#define LQ8_C ((acc_t  *)0x81300000UL)

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

#define SEED_WORDS 64                 // 512 B, the whole .bss of this program
static uint64_t seed_block[SEED_WORDS];

static void fill_bytes(void *dst, size_t bytes) {
  uint64_t *d = (uint64_t *)dst;
  const size_t words = bytes / 8;
  uint64_t key = 0;
  for (size_t i = 0; i < words; i++) {
    // A fresh key every 512 B keeps consecutive copies of the seed block from
    // being byte-identical, so no two rows of B are equal.
    if ((i & (SEED_WORDS - 1)) == 0) key = rng_next();
    d[i] = seed_block[i & (SEED_WORDS - 1)] ^ key;
  }
}

// --------------------------------------------------------------------------
// Scalar golden model — ONE output element. Deliberately trivial.
// --------------------------------------------------------------------------

// Sampled outputs: corners, tile boundaries (multiples of DIM = 16) and
// off-by-one neighbours, so a wrong stride, a wrong tile edge or a dropped
// tail tile all show up. The j's are clustered inside a few cache lines on
// purpose — see ref_samples().
#define N_SAMPLES 8
static const size_t sample_i[N_SAMPLES] = { 0, 0,  1, 15, 16,  31,  63,  63 };
static const size_t sample_j[N_SAMPLES] = { 0, 1, 15, 16, 17, 128, 255, 240 };

// All 8 references in ONE pass over k. The obvious per-element loop walks B
// down a column with stride M, i.e. a fresh cache line (and an L2 miss) on
// every one of 2048 steps, eight times over — that alone cost ~1.3M simulated
// cycles. Fused, each k touches one row of B and the eight j's land in a
// handful of lines, so the whole check is ~16k MACs with ~2k misses.
static void ref_samples(const elem_t *A, const elem_t *B, acc_t *out) {
  for (int s = 0; s < N_SAMPLES; s++)
    out[s] = 0;
  for (size_t k = 0; k < LQ8_K; k++) {
    const elem_t *brow = B + k * LQ8_M;
    for (int s = 0; s < N_SAMPLES; s++)
      out[s] += (acc_t)A[sample_i[s] * LQ8_K + k] * (acc_t)brow[sample_j[s]];
  }
}

// --------------------------------------------------------------------------

int main(void)
{
  elem_t *A = LQ8_A;
  elem_t *B = LQ8_B;
  acc_t  *C = LQ8_C;

  printf("llama-q8-gemm N,K,M = %d,%d,%d\n", LQ8_N, LQ8_K, LQ8_M);

  gemmini_flush(0);

  for (int i = 0; i < SEED_WORDS; i++)
    seed_block[i] = rng_next();
  fill_bytes(A, (size_t)LQ8_N * LQ8_K);
  fill_bytes(B, (size_t)LQ8_K * LQ8_M);
  for (size_t i = 0; i < (size_t)LQ8_N * LQ8_M; i++)
    C[i] = 0;

  printf("Starting gemmini matmul\n");
  uint64_t start = read_cycles();

  llama_q8_gemm(LQ8_N, LQ8_K, LQ8_M, A, B, C);

  uint64_t end = read_cycles();
  printf("Cycles taken: %lu\n", (unsigned long)(end - start));

  // --- sampled self-check -------------------------------------------------
  acc_t want_all[N_SAMPLES];
  ref_samples(A, B, want_all);

  int bad = 0;
  for (int s = 0; s < N_SAMPLES; s++) {
    size_t i = sample_i[s], j = sample_j[s];
    acc_t got = C[i * LQ8_M + j], want = want_all[s];
    if (got != want) {
      printf("MISMATCH C[%d][%d]: got %d want %d\n",
             (int)i, (int)j, (int)got, (int)want);
      bad = 1;
    }
  }

  // Checksum over EVERY output: cheap, and it makes a kernel that only filled
  // the sampled elements visible in the log.
  int64_t sum = 0;
  for (size_t i = 0; i < (size_t)LQ8_N * LQ8_M; i++)
    sum += C[i];
  printf("C checksum = %ld\n", (long)sum);

  if (bad) {
    printf("FAILED\n");
    exit(1);
  }
  printf("PASSED (%d sampled outputs)\n", N_SAMPLES);
  exit(0);
}
