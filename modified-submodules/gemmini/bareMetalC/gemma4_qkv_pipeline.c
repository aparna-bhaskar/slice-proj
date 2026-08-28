#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>

#include "include/gemmini_testutils.h"

#include "gemma4_real_q_data.h"
#include "gemma4_real_k_data.h"
#include "gemma4_real_v_data.h"


#define SEQ_LEN   14
#define Q_HEADS   8
#define KV_HEADS  1
#define HEAD_DIM  256


// -----------------------------------------------------------------------------
// Actual Gemmini projection outputs
// -----------------------------------------------------------------------------

static elem_t q_proj[REAL_Q_I][REAL_Q_J] row_align(1);
static elem_t k_proj[REAL_K_I][REAL_K_J] row_align(1);
static elem_t v_proj[REAL_V_I][REAL_V_J] row_align(1);


// -----------------------------------------------------------------------------
// CPU-reshaped outputs
// -----------------------------------------------------------------------------

static elem_t q_heads[Q_HEADS][SEQ_LEN][HEAD_DIM];
static elem_t k_heads[KV_HEADS][SEQ_LEN][HEAD_DIM];
static elem_t v_heads[KV_HEADS][SEQ_LEN][HEAD_DIM];


// -----------------------------------------------------------------------------
// Verify that Q/K/V exporters produced the same quantized layer input.
//
// For a true integrated pipeline, Q, K and V must all begin with the SAME X.
// -----------------------------------------------------------------------------

static unsigned long check_shared_input(void) {
    unsigned long qk_mismatches = 0;
    unsigned long qv_mismatches = 0;

    const elem_t *q_x = (const elem_t *)REAL_Q_X;
    const elem_t *k_x = (const elem_t *)REAL_K_X;
    const elem_t *v_x = (const elem_t *)REAL_V_X;

    const size_t total = REAL_Q_I * REAL_Q_K;

    for (size_t i = 0; i < total; i++) {
        if (q_x[i] != k_x[i])
            qk_mismatches++;

        if (q_x[i] != v_x[i])
            qv_mismatches++;
    }

    printf(
        "Q input vs K input mismatch count: %lu\n",
        qk_mismatches
    );

    printf(
        "Q input vs V input mismatch count: %lu\n",
        qv_mismatches
    );

    return qk_mismatches + qv_mismatches;
}


// -----------------------------------------------------------------------------
// Gemmini projections
//
// IMPORTANT:
// We intentionally use REAL_Q_X as the input to ALL THREE projections.
// This makes this a real shared-X Q/K/V pipeline rather than three isolated
// projection tests.
// -----------------------------------------------------------------------------

static unsigned long run_q_projection(void) {
    unsigned long start = read_cycles();

    tiled_matmul_auto(
        REAL_Q_I,
        REAL_Q_J,
        REAL_Q_K,

        (elem_t *)REAL_Q_X,
        (elem_t *)REAL_Q_W_TRANSPOSED,
        NULL,
        (elem_t *)q_proj,

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

    unsigned long end = read_cycles();

    return end - start;
}


static unsigned long run_k_projection(void) {
    unsigned long start = read_cycles();

    tiled_matmul_auto(
        REAL_K_I,
        REAL_K_J,
        REAL_K_K,

        // Same shared X used by Q
        (elem_t *)REAL_Q_X,
        (elem_t *)REAL_K_W_TRANSPOSED,
        NULL,
        (elem_t *)k_proj,

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

    return end - start;
}


static unsigned long run_v_projection(void) {
    unsigned long start = read_cycles();

    tiled_matmul_auto(
        REAL_V_I,
        REAL_V_J,
        REAL_V_K,

        // Same shared X used by Q and K
        (elem_t *)REAL_Q_X,
        (elem_t *)REAL_V_W_TRANSPOSED,
        NULL,
        (elem_t *)v_proj,

        REAL_V_K,
        REAL_V_J,
        REAL_V_J,
        REAL_V_J,

        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,

        NO_ACTIVATION,
        REAL_V_C_SCALE,
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

    return end - start;
}


// -----------------------------------------------------------------------------
// CPU reshape
// -----------------------------------------------------------------------------

static void reshape_q(void) {
    for (size_t t = 0; t < SEQ_LEN; t++) {
        for (size_t h = 0; h < Q_HEADS; h++) {
            for (size_t d = 0; d < HEAD_DIM; d++) {
                const size_t flat_index =
                    h * HEAD_DIM + d;

                q_heads[h][t][d] =
                    q_proj[t][flat_index];
            }
        }
    }
}


static void reshape_k(void) {
    for (size_t t = 0; t < SEQ_LEN; t++) {
        for (size_t d = 0; d < HEAD_DIM; d++) {
            k_heads[0][t][d] =
                k_proj[t][d];
        }
    }
}


static void reshape_v(void) {
    for (size_t t = 0; t < SEQ_LEN; t++) {
        for (size_t d = 0; d < HEAD_DIM; d++) {
            v_heads[0][t][d] =
                v_proj[t][d];
        }
    }
}


// -----------------------------------------------------------------------------
// Verification
// -----------------------------------------------------------------------------

static unsigned long compare_flat(
    const elem_t *actual,
    const elem_t *expected,
    size_t total,
    const char *name
) {
    unsigned long mismatches = 0;

    for (size_t i = 0; i < total; i++) {
        if (actual[i] != expected[i]) {

            if (mismatches < 5) {
                printf(
                    "%s mismatch index=%lu: actual=%d expected=%d\n",
                    name,
                    (unsigned long)i,
                    (int)actual[i],
                    (int)expected[i]
                );
            }

            mismatches++;
        }
    }

    return mismatches;
}


static unsigned long verify_q_reshape(void) {
    unsigned long mismatches = 0;

    for (size_t t = 0; t < SEQ_LEN; t++) {
        for (size_t h = 0; h < Q_HEADS; h++) {
            for (size_t d = 0; d < HEAD_DIM; d++) {

                const size_t flat_index =
                    h * HEAD_DIM + d;

                const elem_t expected =
                    REAL_Q_EXPECTED[t][flat_index];

                if (q_heads[h][t][d] != expected)
                    mismatches++;
            }
        }
    }

    return mismatches;
}


static unsigned long verify_k_reshape(void) {
    unsigned long mismatches = 0;

    for (size_t t = 0; t < SEQ_LEN; t++) {
        for (size_t d = 0; d < HEAD_DIM; d++) {

            if (
                k_heads[0][t][d] !=
                REAL_K_EXPECTED[t][d]
            ) {
                mismatches++;
            }
        }
    }

    return mismatches;
}


static unsigned long verify_v_reshape(void) {
    unsigned long mismatches = 0;

    for (size_t t = 0; t < SEQ_LEN; t++) {
        for (size_t d = 0; d < HEAD_DIM; d++) {

            if (
                v_heads[0][t][d] !=
                REAL_V_EXPECTED[t][d]
            ) {
                mismatches++;
            }
        }
    }

    return mismatches;
}


// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(void) {

    gemmini_flush(0);

    printf(
        "Gemma 4 layer-0 integrated Q/K/V + reshape pipeline\n\n"
    );

    printf("Shared input X: [14, 1536]\n");
    printf("Q projection:   [14, 2048]\n");
    printf("K projection:   [14, 256]\n");
    printf("V projection:   [14, 256]\n\n");


    // -------------------------------------------------------------------------
    // Confirm Q/K/V really share one quantized input X
    // -------------------------------------------------------------------------

    printf("Checking shared input X\n");

    unsigned long input_mismatches =
        check_shared_input();

    if (input_mismatches != 0) {
        printf(
            "\nFAIL: Q/K/V exports do not contain "
            "the same quantized input X.\n"
        );

        return 1;
    }

    printf("PASS: Q/K/V share identical input X.\n\n");


    // -------------------------------------------------------------------------
    // Run all three projections on Gemmini
    // -------------------------------------------------------------------------

    printf("Running Gemmini Q projection\n");
    unsigned long q_cycles = run_q_projection();
    printf("Q cycles: %lu\n\n", q_cycles);

    printf("Running Gemmini K projection\n");
    unsigned long k_cycles = run_k_projection();
    printf("K cycles: %lu\n\n", k_cycles);

    printf("Running Gemmini V projection\n");
    unsigned long v_cycles = run_v_projection();
    printf("V cycles: %lu\n\n", v_cycles);


    // -------------------------------------------------------------------------
    // Check raw projection outputs
    // -------------------------------------------------------------------------

    unsigned long q_proj_mismatches =
        compare_flat(
            (const elem_t *)q_proj,
            (const elem_t *)REAL_Q_EXPECTED,
            REAL_Q_I * REAL_Q_J,
            "Q projection"
        );

    unsigned long k_proj_mismatches =
        compare_flat(
            (const elem_t *)k_proj,
            (const elem_t *)REAL_K_EXPECTED,
            REAL_K_I * REAL_K_J,
            "K projection"
        );

    unsigned long v_proj_mismatches =
        compare_flat(
            (const elem_t *)v_proj,
            (const elem_t *)REAL_V_EXPECTED,
            REAL_V_I * REAL_V_J,
            "V projection"
        );

    printf("Projection comparisons\n");

    printf(
        "Q projection mismatch count: %lu\n",
        q_proj_mismatches
    );

    printf(
        "K projection mismatch count: %lu\n",
        k_proj_mismatches
    );

    printf(
        "V projection mismatch count: %lu\n\n",
        v_proj_mismatches
    );


    // -------------------------------------------------------------------------
    // CPU reshape of ACTUAL Gemmini output buffers
    // -------------------------------------------------------------------------

    printf("Starting CPU Q/K/V reshape\n");

    unsigned long reshape_start = read_cycles();

    reshape_q();
    reshape_k();
    reshape_v();

    unsigned long reshape_end = read_cycles();

    printf(
        "CPU reshape cycles: %lu\n\n",
        reshape_end - reshape_start
    );

    printf("Reshaped output shapes\n");
    printf("Q: [8, 14, 256]\n");
    printf("K: [1, 14, 256]\n");
    printf("V: [1, 14, 256]\n\n");


    // -------------------------------------------------------------------------
    // Validate reshaped Gemmini results against Python references
    // -------------------------------------------------------------------------

    unsigned long q_reshape_mismatches =
        verify_q_reshape();

    unsigned long k_reshape_mismatches =
        verify_k_reshape();

    unsigned long v_reshape_mismatches =
        verify_v_reshape();

    printf("Reshape comparisons\n");

    printf(
        "Q reshape mismatch count: %lu\n",
        q_reshape_mismatches
    );

    printf(
        "K reshape mismatch count: %lu\n",
        k_reshape_mismatches
    );

    printf(
        "V reshape mismatch count: %lu\n",
        v_reshape_mismatches
    );


    // -------------------------------------------------------------------------
    // Final result
    // -------------------------------------------------------------------------

    if (
        q_proj_mismatches == 0 &&
        k_proj_mismatches == 0 &&
        v_proj_mismatches == 0 &&
        q_reshape_mismatches == 0 &&
        k_reshape_mismatches == 0 &&
        v_reshape_mismatches == 0
    ) {
        printf(
            "\nPASS: integrated Gemmini Q/K/V "
            "projections -> CPU reshape pipeline "
            "matches Python references.\n"
        );

        return 0;
    }

    printf(
        "\nFAIL: integrated Q/K/V pipeline "
        "contains mismatches.\n"
    );

    return 1;
}