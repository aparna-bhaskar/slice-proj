#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "include/gemmini_testutils.h"
#include "gemma4_real_q_data.h"


static elem_t cpu_output[REAL_Q_I][REAL_Q_J] row_align(1);
static elem_t gemmini_output[REAL_Q_I][REAL_Q_J] row_align(1);


static void run_cpu_q_projection(void) {
    for (size_t i = 0; i < REAL_Q_I; i++) {
        for (size_t j = 0; j < REAL_Q_J; j++) {
            acc_t sum = 0;

            for (size_t k = 0; k < REAL_Q_K; k++) {
                sum +=
                    (acc_t)REAL_Q_X[i][k]
                    * (acc_t)REAL_Q_W_TRANSPOSED[k][j];
            }

            cpu_output[i][j] =
                (elem_t)ACC_SCALE(sum, REAL_Q_C_SCALE);
        }
    }
}


static void print_stats(
    const char *name,
    const elem_t matrix[REAL_Q_I][REAL_Q_J]
) {
    int minimum = (int)matrix[0][0];
    int maximum = (int)matrix[0][0];

    unsigned long zeros = 0;
    unsigned long sat_min = 0;
    unsigned long sat_max = 0;

    long checksum = 0;

    for (size_t i = 0; i < REAL_Q_I; i++) {
        for (size_t j = 0; j < REAL_Q_J; j++) {
            int value = (int)matrix[i][j];

            if (value < minimum) {
                minimum = value;
            }

            if (value > maximum) {
                maximum = value;
            }

            if (value == 0) {
                zeros++;
            }

            if (value == -128) {
                sat_min++;
            }

            if (value == 127) {
                sat_max++;
            }

            checksum += value;
        }
    }

    printf(
        "%s: min=%d max=%d zeros=%lu "
        "sat_min=%lu sat_max=%lu checksum=%ld\n",
        name,
        minimum,
        maximum,
        zeros,
        sat_min,
        sat_max,
        checksum
    );
}


static int compare_matrices(
    const char *name,
    const elem_t left[REAL_Q_I][REAL_Q_J],
    const elem_t right[REAL_Q_I][REAL_Q_J]
) {
    unsigned long mismatches = 0;

    for (size_t i = 0; i < REAL_Q_I; i++) {
        for (size_t j = 0; j < REAL_Q_J; j++) {
            if (left[i][j] != right[i][j]) {
                if (mismatches < 10) {
                    printf(
                        "%s mismatch [%lu][%lu]: "
                        "%d versus %d\n",
                        name,
                        (unsigned long)i,
                        (unsigned long)j,
                        (int)left[i][j],
                        (int)right[i][j]
                    );
                }

                mismatches++;
            }
        }
    }

    printf(
        "%s mismatch count: %lu\n",
        name,
        mismatches
    );

    return mismatches == 0;
}


int main(void) {
    gemmini_flush(0);

    printf("Real Gemma 4 layer-0 Q projection\n");
    printf("X shape: [%d, %d]\n", REAL_Q_I, REAL_Q_K);
    printf("WQ^T shape: [%d, %d]\n", REAL_Q_K, REAL_Q_J);
    printf("Q shape: [%d, %d]\n", REAL_Q_I, REAL_Q_J);
    printf("Gemmini C scale: %f\n", (double)REAL_Q_C_SCALE);

    printf("\nStarting scalar CPU projection\n");

    unsigned long cpu_start = read_cycles();

    run_cpu_q_projection();

    unsigned long cpu_end = read_cycles();

    printf(
        "Scalar CPU cycles: %lu\n",
        cpu_end - cpu_start
    );

    printf("\nStarting Gemmini WS projection\n");

    unsigned long gemmini_start = read_cycles();

    tiled_matmul_auto(
        REAL_Q_I,
        REAL_Q_J,
        REAL_Q_K,

        (elem_t *)REAL_Q_X,
        (elem_t *)REAL_Q_W_TRANSPOSED,
        NULL,
        (elem_t *)gemmini_output,

        REAL_Q_K,
        REAL_Q_J,
        REAL_Q_J,
        REAL_Q_J,

        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,

        NO_ACTIVATION,
        REAL_Q_C_SCALE,
        0,
        false,

        false,
        false,
        false,
        false,

        0,
        WS
    );

    unsigned long gemmini_end = read_cycles();

    printf(
        "Gemmini WS Spike cycles: %lu\n",
        gemmini_end - gemmini_start
    );

    printf("\nOutput statistics\n");

    print_stats("CPU", cpu_output);
    print_stats("Gemmini", gemmini_output);
    print_stats("Python expected", REAL_Q_EXPECTED);

    printf("\nComparisons\n");

    int cpu_vs_gemmini = compare_matrices(
        "CPU vs Gemmini",
        cpu_output,
        gemmini_output
    );

    int cpu_vs_python = compare_matrices(
        "CPU vs Python expected",
        cpu_output,
        REAL_Q_EXPECTED
    );

    int gemmini_vs_python = compare_matrices(
        "Gemmini vs Python expected",
        gemmini_output,
        REAL_Q_EXPECTED
    );

    if (
        !cpu_vs_gemmini
        || !cpu_vs_python
        || !gemmini_vs_python
    ) {
        printf("\nFAIL: real Q projection did not match.\n");
        return 1;
    }

    printf(
        "\nPASS: real layer-0 Q projection matched "
        "across Python, CPU and Gemmini.\n"
    );

    return 0;
}