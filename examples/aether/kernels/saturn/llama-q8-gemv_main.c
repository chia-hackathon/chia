// See LICENSE for license details.
//
//**************************************************************************
// llama-q8-gemv — int8 GEMV in the shape of a Llama-3.2-1B decode step
//--------------------------------------------------------------------------
//
// SEALED HARNESS. This file holds main(), the pseudo-random data generator,
// the scalar golden model and the self-check. The optimizer never gets to
// change it: every build is reassembled from the pristine copy of this file
// plus the optimizer's `llama_q8_gemv.c`.
//
// Wall-clock discipline (Verilator runs this SoC at ~1.8k cycles/s):
//   * no big static arrays  -> the TSI ELF load stays ~1 s, not ~1 h
//   * vector data fill      -> ~150k cycles instead of ~1M for 1 MiB
//   * SAMPLED self-check    -> 8 rows recomputed scalar (~150k cycles)
//     instead of all 512 (~9.4M cycles, i.e. ~90 min of Verilator).

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include "riscv_vector.h"
#include "util.h"

#include "llama_q8_gemv.h"

// --------------------------------------------------------------------------
// Deterministic pseudo-random data (fixed seed -> reproducible every run)
// --------------------------------------------------------------------------

static uint32_t lcg_state = 0x13579BDFu;

static inline uint32_t lcg_next(void) {
  lcg_state = lcg_state * 1103515245u + 12345u;
  return lcg_state >> 16;
}

// 256 bytes of scalar-generated noise: the seed pattern that the vector fill
// below broadcasts across W. Small enough that it costs nothing in .bss.
static int8_t seed_block[256];

static void fill_weights(int8_t *W, size_t bytes) {
  for (int i = 0; i < 256; i++)
    seed_block[i] = (int8_t)lcg_next();

  const size_t vl = __riscv_vsetvl_e8m8(256);   // VLEN=256, LMUL=8 -> 256 lanes
  vint8m8_t v = __riscv_vle8_v_i8m8(seed_block, vl);

  // Store the block, then perturb it by a fresh pseudo-random offset so that
  // no two 256-byte chunks (and therefore no two rows of W) are identical.
  for (size_t off = 0; off < bytes; off += vl) {
    __riscv_vse8_v_i8m8(W + off, v, vl);
    v = __riscv_vadd_vx_i8m8(v, (int8_t)lcg_next(), vl);
  }
}

static void fill_vector(int8_t *x, size_t n) {
  for (size_t i = 0; i < n; i++)
    x[i] = (int8_t)lcg_next();
}

// --------------------------------------------------------------------------
// Scalar golden model — ONE row. This is the only reference the check has, so
// it is deliberately trivial and obviously correct.
// --------------------------------------------------------------------------

static int32_t ref_row(const int8_t *W, const int8_t *x, size_t K, size_t m) {
  const int8_t *w = W + m * K;
  int32_t s = 0;
  for (size_t k = 0; k < K; k++)
    s += (int32_t)w[k] * (int32_t)x[k];
  return s;
}

// Rows the check recomputes: spread over the whole output so a wrong tile
// boundary, a wrong row stride or a dropped tail row all show up.
#define N_SAMPLES 8
static const size_t sample_rows[N_SAMPLES] = {
  0, 1, 37, LQ8_M / 4, LQ8_M / 2, LQ8_M / 2 + 1, LQ8_M - 33, LQ8_M - 1
};

// --------------------------------------------------------------------------

int main(int argc, char *argv[])
{
  int8_t  *W = LQ8_W;
  int8_t  *x = LQ8_X;
  int32_t *y = LQ8_Y;

  printf("llama-q8-gemv M,K = %d,%d\n", LQ8_M, LQ8_K);

  fill_weights(W, (size_t)LQ8_M * LQ8_K);
  fill_vector(x, LQ8_K);
  for (size_t m = 0; m < LQ8_M; m++)
    y[m] = 0;

  // Do the GEMV. mcycle/minstret around exactly this call is the objective.
  setStats(1);
  llama_q8_gemv(LQ8_M, LQ8_K, W, x, y);
  setStats(0);

  // --- sampled self-check ------------------------------------------------
  int bad = 0;
  for (int i = 0; i < N_SAMPLES; i++) {
    size_t m = sample_rows[i];
    int32_t got = y[m], want = ref_row(W, x, LQ8_K, m);
    if (got != want) {
      printf("MISMATCH row %d: got %d want %d\n", (int)m, got, want);
      bad = 1;
    }
  }

  // A checksum over EVERY output, so a kernel that only computed the sampled
  // rows is at least visible in the log even though the check cannot price a
  // full scalar reference.
  int64_t sum = 0;
  for (size_t m = 0; m < LQ8_M; m++)
    sum += y[m];
  printf("y checksum = %ld\n", (long)sum);

  if (bad) {
    printf("FAILED\n");
    return 1;
  }
  printf("PASSED (%d sampled rows)\n", N_SAMPLES);
  return 0;
}
