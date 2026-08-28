// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#define CHECK_RESULT 1

#define NO_BIAS 1
#define FULL_BIAS_WIDTH 1

#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#error variable-bitwidth bias not currently supported
#endif

#define MAT_DIM_I 14
#define MAT_DIM_K 1536
#define MAT_DIM_J 2048
/***
X:      [14, 1536]
WQᵀ:    [1536, 2048]
Q:      [14, 2048]
***/


void full_matmul(elem_t A[MAT_DIM_I][MAT_DIM_K], elem_t B[MAT_DIM_K][MAT_DIM_J], ACC_T D[MAT_DIM_I][MAT_DIM_J], full_t C_full[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t r = 0; r < MAT_DIM_I; r++)
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < MAT_DIM_K; k++)
        C_full[r][c] += A[r][k]*B[k][c];
    }
}



int full_is_equal(elem_t x[MAT_DIM_I][MAT_DIM_J], elem_t y[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i)
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      if (x[i][j] != y[i][j])
        return 0;
  return 1;
}

void full_matscale(full_t full[MAT_DIM_I][MAT_DIM_J], elem_t out[MAT_DIM_I][MAT_DIM_J], acc_scale_t scale) {
  for (size_t r = 0; r < MAT_DIM_I; r++)                             
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      // Scale element
      full_t scaled = ACC_SCALE(full[r][c], scale);

      // Saturate and cast element
#ifndef ELEM_T_IS_FLOAT
      full_t elem = scaled > elem_t_max ? elem_t_max : (scaled < elem_t_min ? elem_t_min : scaled);
      out[r][c] = elem;
#else
      out[r][c] = scaled; // TODO should we also saturate when using floats?
#endif
    }
} 

void print_output_stats(
    const char *name,
    elem_t matrix[MAT_DIM_I][MAT_DIM_J]
) {
  int min_value = (int)matrix[0][0];
  int max_value = (int)matrix[0][0];

  size_t zeros = 0;
  size_t saturated_max = 0;
  size_t saturated_min = 0;

  long checksum = 0;

  for (size_t i = 0; i < MAT_DIM_I; i++) {
    for (size_t j = 0; j < MAT_DIM_J; j++) {
      int value = (int)matrix[i][j];

      if (value < min_value) min_value = value;
      if (value > max_value) max_value = value;

      if (value == 0) zeros++;
      if (value == elem_t_max) saturated_max++;
      if (value == elem_t_min) saturated_min++;

      checksum += value;
    }
  }

  printf(
      "%s: min=%d max=%d zeros=%lu "
      "sat_min=%lu sat_max=%lu checksum=%ld\n",
      name,
      min_value,
      max_value,
      (unsigned long)zeros,
      (unsigned long)saturated_min,
      (unsigned long)saturated_max,
      checksum
  );
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(1);
    static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(1);
    static elem_t full_C[MAT_DIM_I][MAT_DIM_J] row_align(1);
    static ACC_T full_D[MAT_DIM_I][MAT_DIM_J] row_align_acc(1); // TODO don't use row_align_acc when ACC_T is elem_t

    static full_t gold_full[MAT_DIM_I][MAT_DIM_J];
    static elem_t gold[MAT_DIM_I][MAT_DIM_J];

#if CHECK_RESULT == 1
    // Initialize X: [14, 1536]
    for (size_t i = 0; i < MAT_DIM_I; i++) {
      for (size_t k = 0; k < MAT_DIM_K; k++) {
        full_A[i][k] = (elem_t)(((int)(i + k) % 5) - 2);
      }
    }

    // // Initialize WQ^T: [1536, 2048]
    for (size_t k = 0; k < MAT_DIM_K; k++) {
      for (size_t j = 0; j < MAT_DIM_J; j++) {
        full_B[k][j] = (elem_t)(((int)(2 * k + j) % 5) - 2);
      }
    }

    //No bias
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        full_D[i][j] = 0;
      }
    }


    printf("Inputs initialized:\n");
    printf("X shape: [%d, %d]\n", MAT_DIM_I, MAT_DIM_K);
    printf("WQ^T shape: [%d, %d]\n", MAT_DIM_K, MAT_DIM_J);
    printf("Q output shape: [%d, %d]\n", MAT_DIM_I, MAT_DIM_J);


    printf("Starting slow CPU matmul\n");

    unsigned long cpu_start = read_cycles();
    full_matmul(full_A, full_B, full_D, gold_full);
    unsigned long cpu_end = read_cycles();
    printf("Cycles taken: %lu\n", cpu_end - cpu_start);
    full_matscale(gold_full, gold, ACC_SCALE_IDENTITY);
#endif

    printf("Starting Gemmini WS matmul\n");
    unsigned long start = read_cycles();

    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)full_A, (elem_t*)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false,
            false, false,
            0,
            WS);

    unsigned long end = read_cycles();
    printf("Cycles taken: %lu\n", end - start);

#if CHECK_RESULT == 1
    print_output_stats("CPU gold", gold);
    print_output_stats("Gemmini", full_C);

    if (!full_is_equal(full_C, gold)) {
      printf("ERROR: CPU and Gemmini results do not match.\n");
      exit(1);
    }

    printf("PASS: CPU and Gemmini results match.\n");
#endif

  exit(0);
}

