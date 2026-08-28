#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "include/gemmini_testutils.h"

#include "gemma4_real_q_data.h"
#include "gemma4_real_k_data.h"
#include "gemma4_real_v_data.h"


#define SEQ_LEN   14
#define Q_HEADS   8
#define KV_HEADS  1
#define HEAD_DIM  256


static elem_t q_heads[Q_HEADS][SEQ_LEN][HEAD_DIM];
static elem_t k_heads[KV_HEADS][SEQ_LEN][HEAD_DIM];
static elem_t v_heads[KV_HEADS][SEQ_LEN][HEAD_DIM];


static void reshape_q(void) {
    for (size_t t = 0; t < SEQ_LEN; t++) {
        for (size_t h = 0; h < Q_HEADS; h++) {
            for (size_t d = 0; d < HEAD_DIM; d++) {

                size_t flat_index = h * HEAD_DIM + d;

                q_heads[h][t][d] =
                    REAL_Q_EXPECTED[t][flat_index];
            }
        }
    }
}


static void reshape_k(void) {
    for (size_t t = 0; t < SEQ_LEN; t++) {
        for (size_t d = 0; d < HEAD_DIM; d++) {
            k_heads[0][t][d] =
                REAL_K_EXPECTED[t][d];
        }
    }
}


static void reshape_v(void) {
    for (size_t t = 0; t < SEQ_LEN; t++) {
        for (size_t d = 0; d < HEAD_DIM; d++) {
            v_heads[0][t][d] =
                REAL_V_EXPECTED[t][d];
        }
    }
}


static unsigned long verify_q(void) {
    unsigned long mismatches = 0;

    for (size_t t = 0; t < SEQ_LEN; t++) {
        for (size_t h = 0; h < Q_HEADS; h++) {
            for (size_t d = 0; d < HEAD_DIM; d++) {

                size_t flat_index = h * HEAD_DIM + d;

                elem_t expected =
                    REAL_Q_EXPECTED[t][flat_index];

                elem_t actual =
                    q_heads[h][t][d];

                if (actual != expected) {

                    if (mismatches < 10) {
                        printf(
                            "Q mismatch h=%lu t=%lu d=%lu: "
                            "actual=%d expected=%d\n",
                            (unsigned long)h,
                            (unsigned long)t,
                            (unsigned long)d,
                            (int)actual,
                            (int)expected
                        );
                    }

                    mismatches++;
                }
            }
        }
    }

    return mismatches;
}


static unsigned long verify_k(void) {
    unsigned long mismatches = 0;

    for (size_t t = 0; t < SEQ_LEN; t++) {
        for (size_t d = 0; d < HEAD_DIM; d++) {

            elem_t expected =
                REAL_K_EXPECTED[t][d];

            elem_t actual =
                k_heads[0][t][d];

            if (actual != expected)
                mismatches++;
        }
    }

    return mismatches;
}


static unsigned long verify_v(void) {
    unsigned long mismatches = 0;

    for (size_t t = 0; t < SEQ_LEN; t++) {
        for (size_t d = 0; d < HEAD_DIM; d++) {

            elem_t expected =
                REAL_V_EXPECTED[t][d];

            elem_t actual =
                v_heads[0][t][d];

            if (actual != expected)
                mismatches++;
        }
    }

    return mismatches;
}


int main(void) {

    printf("Gemma 4 layer-0 Q/K/V reshape test\n\n");

    printf("Input shapes\n");
    printf("Q: [14, 2048]\n");
    printf("K: [14, 256]\n");
    printf("V: [14, 256]\n\n");

    unsigned long start = read_cycles();

    reshape_q();
    reshape_k();
    reshape_v();

    unsigned long end = read_cycles();

    printf("Reshape cycles: %lu\n\n", end - start);

    printf("Output shapes\n");
    printf("Q: [8, 14, 256]\n");
    printf("K: [1, 14, 256]\n");
    printf("V: [1, 14, 256]\n\n");

    unsigned long q_mismatches = verify_q();
    unsigned long k_mismatches = verify_k();
    unsigned long v_mismatches = verify_v();

    printf("Q reshape mismatch count: %lu\n", q_mismatches);
    printf("K reshape mismatch count: %lu\n", k_mismatches);
    printf("V reshape mismatch count: %lu\n", v_mismatches);

    printf("\nSample mapping\n");

    printf(
        "Q[token=0, flat=0]     -> "
        "Q[head=0, token=0, dim=0]: %d -> %d\n",
        (int)REAL_Q_EXPECTED[0][0],
        (int)q_heads[0][0][0]
    );

    printf(
        "Q[token=0, flat=256]   -> "
        "Q[head=1, token=0, dim=0]: %d -> %d\n",
        (int)REAL_Q_EXPECTED[0][256],
        (int)q_heads[1][0][0]
    );

    printf(
        "Q[token=0, flat=2047]  -> "
        "Q[head=7, token=0, dim=255]: %d -> %d\n",
        (int)REAL_Q_EXPECTED[0][2047],
        (int)q_heads[7][0][255]
    );

    if (
        q_mismatches == 0 &&
        k_mismatches == 0 &&
        v_mismatches == 0
    ) {
        printf(
            "\nPASS: Q/K/V reshapes are correct.\n"
        );

        return 0;
    }

    printf(
        "\nFAIL: Q/K/V reshape mismatch detected.\n"
    );

    return 1;
}