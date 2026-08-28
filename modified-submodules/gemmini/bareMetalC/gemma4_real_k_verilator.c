#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "include/gemmini_testutils.h"
#include "gemma4_real_k_data.h"


static elem_t gemmini_output[REAL_K_I][REAL_K_J] row_align(1);


static void print_stats(
    const char *name,
    const elem_t matrix[REAL_K_I][REAL_K_J]
) {
    int minimum = (int)matrix[0][0];
    int maximum = (int)matrix[0][0];

    unsigned long zeros = 0;
    unsigned long sat_min = 0;
    unsigned long sat_max = 0;
    long checksum = 0;

    for (size_t i = 0; i < REAL_K_I; i++) {
        for (size_t j = 0; j < REAL_K_J; j++) {
            int value = (int)matrix[i][j];

            if (value < minimum)
                minimum = value;

            if (value > maximum)
                maximum = value;

            if (value == 0)
                zeros++;

            if (value == -128)
                sat_min++;

            if (value == 127)
                sat_max++;

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


static int compare_to_expected(void) {
    unsigned long mismatches = 0;

    for (size_t i = 0; i < REAL_K_I; i++) {
        for (size_t j = 0; j < REAL_K_J; j++) {

            if (gemmini_output[i][j] != REAL_K_EXPECTED[i][j]) {

                if (mismatches < 10) {
                    printf(
                        "Mismatch [%lu][%lu]: "
                        "Gemmini=%d expected=%d\n",
                        (unsigned long)i,
                        (unsigned long)j,
                        (int)gemmini_output[i][j],
                        (int)REAL_K_EXPECTED[i][j]
                    );
                }

                mismatches++;
            }
        }
    }

    printf(
        "Gemmini vs Python expected mismatch count: %lu\n",
        mismatches
    );

    return mismatches == 0;
}


int main(void) {
    gemmini_flush(0);

    printf("FAST VERILATOR TEST\n");
    printf("Real Gemma 4 layer-0 K projection\n");

    printf(
        "X shape: [%d, %d]\n",
        REAL_K_I,
        REAL_K_K
    );

    printf(
        "WK^T shape: [%d, %d]\n",
        REAL_K_K,
        REAL_K_J
    );

    printf(
        "K shape: [%d, %d]\n",
        REAL_K_I,
        REAL_K_J
    );

    printf("\nStarting Gemmini WS K projection\n");

    unsigned long start = read_cycles();

    tiled_matmul_auto(
        REAL_K_I,
        REAL_K_J,
        REAL_K_K,

        (elem_t *)REAL_K_X,
        (elem_t *)REAL_K_W_TRANSPOSED,
        NULL,
        (elem_t *)gemmini_output,

        REAL_K_K,
        REAL_K_J,
        REAL_K_J,
        REAL_K_J,

        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,

        NO_ACTIVATION,
        REAL_K_C_SCALE,
        0,
        false,

        false,
        false,
        false,
        false,

        0,
        WS
    );

    unsigned long end = read_cycles();

    printf(
        "Gemmini WS cycles: %lu\n",
        end - start
    );

    printf("\nOutput statistics\n");

    print_stats("Gemmini", gemmini_output);
    print_stats("Python expected", REAL_K_EXPECTED);

    printf("\nComparison\n");

    if (!compare_to_expected()) {
        printf(
            "\nFAIL: Gemmini RTL K output does not "
            "match Python expected output.\n"
        );

        return 1;
    }

    printf(
        "\nPASS: Gemmini RTL K projection matches "
        "Python expected output.\n"
    );

    return 0;
}