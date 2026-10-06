#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "include/gemmini_testutils.h"

#include "gemma4_real_q_data.h"
#include "gemma4_real_k_data.h"
#include "gemma4_real_qk_data.h"
#include "gemma4_decoder_qk_rope_data.h"
#include "gemma4_decoder_attn_v_data.h"
#include "gemma4_decoder_runtime_attn_data.h"
#include "gemma4_decoder_norm_rope_data.h"
#include "gemma4_decoder_runtime_q_data.h"
#include "gemma4_decoder_runtime_k_data.h"
#include "gemma4_decoder_runtime_qk_data.h"
#include "gemma4_decoder_runtime_attention_full_data.h"
#include "gemma4_real_o_data.h"
#include "gemma4_real_gate_data.h"
#include "gemma4_real_up_data.h"
#include "gemma4_real_down_data.h"
#include "gemma4_decoder_runtime_o_data.h"
#include "gemma4_decoder_runtime_input_data.h"
#include "gemma4_decoder_runtime_ffn_norm_data.h"
#include "gemma4_decoder_runtime_ffn_input_data.h"
#include "gemma4_decoder_runtime_gate_up_data.h"
#include "gemma4_decoder_runtime_down_input_data.h"
#include "gemma4_decoder_runtime_down_data.h"
#include "gemma4_decoder_runtime_post_ffn_data.h"
#include "gemma4_decoder_post_ffn_norm_data.h"
#include "gemma4_decoder_runtime_per_layer_gate_data.h"
#include "gemma4_decoder_runtime_per_layer_product_data.h"
#include "gemma4_decoder_per_layer_input_data.h"
#include "gemma4_decoder_runtime_per_layer_proj_data.h"
#include "gemma4_decoder_final_norm_data.h"
#include "gemma4_decoder_runtime_final_data.h"
#include "gemma4_decoder_runtime_v_data.h"
#include "gemma4_real_v_data.h"


// -----------------------------------------------------------------------------
// Minimal errno support for newlib libm in this -nostdlib bare-metal build.
//
// expf() references __errno(), but the Gemmini bare-metal runtime does not
// provide the normal libc errno implementation.
// -----------------------------------------------------------------------------

static int baremetal_errno;

int *__errno(void)
{
    return &baremetal_errno;
}


#define SEQ_LEN   14
#define Q_HEADS   8
#define KV_HEADS  1
#define HEAD_DIM  256


// -----------------------------------------------------------------------------
// BF16 round-to-nearest-even.
//
// Same conversion used by the CPU decoder baseline after FP32 softmax.
// -----------------------------------------------------------------------------

static float round_to_bf16(float x)
{
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));

    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    bits &= 0xFFFF0000u;

    memcpy(&x, &bits, sizeof(x));
    return x;
}


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

/*
 * Q after:
 *   Gemmini projection -> dequantize -> RMSNorm -> RoPE -> requantize
 *
 * Layout matches the QK Gemmini input:
 *   [head][seq][head_dim]
 */
static elem_t runtime_q_rope[Q_HEADS][SEQ_LEN][HEAD_DIM] row_align(1);
static elem_t runtime_k_rope[KV_HEADS][SEQ_LEN][HEAD_DIM] row_align(1);
static elem_t runtime_v_norm[SEQ_LEN][HEAD_DIM] row_align(1);


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


// ============================================================
// Runtime layer input -> input RMSNorm -> shared Q/K/V INT8 input
//
// Original layer input: [14][1536] BF16-valued float
// Input RMSNorm: exact CPU baseline semantics
// Quantization scale: REAL_Q_X_SCALE
// ============================================================

static elem_t runtime_qkv_input[REAL_Q_I][REAL_Q_K]
    row_align(1);


static unsigned long prepare_runtime_qkv_input(void)
{
    unsigned long start = read_cycles();

    for (size_t t = 0; t < REAL_Q_I; t++) {

        double sum_sq = 0.0;

        for (size_t d = 0; d < REAL_Q_K; d++) {
            float x = DECODER_LAYER_INPUT[t][d];
            sum_sq += (double)x * (double)x;
        }

        float mean_squared =
            (float)(sum_sq / (double)REAL_Q_K) + 1.0e-6f;

        float inv_rms =
            powf(mean_squared, -0.5f);

        for (size_t d = 0; d < REAL_Q_K; d++) {

            float y =
                DECODER_LAYER_INPUT[t][d] *
                inv_rms *
                DECODER_INPUT_NORM_WEIGHT[d];

            // Match the CPU baseline's explicit BF16 boundary.
            y = round_to_bf16(y);

            int32_t q =
                (int32_t)roundf(
                    y / (float)REAL_Q_X_SCALE
                );

            if (q > 127)
                q = 127;
            else if (q < -127)
                q = -127;

            runtime_qkv_input[t][d] =
                (elem_t)q;
        }
    }

    unsigned long end = read_cycles();
    return end - start;
}


static size_t verify_runtime_qkv_input(void)
{
    size_t mismatches = 0;
    long checksum = 0;

    for (size_t t = 0; t < REAL_Q_I; t++) {
        for (size_t d = 0; d < REAL_Q_K; d++) {

            elem_t actual =
                runtime_qkv_input[t][d];

            elem_t expected =
                REAL_Q_X[t][d];

            checksum += (long)actual;

            if (actual != expected) {
                if (mismatches < 10) {
                    printf(
                        "Runtime QKV input mismatch "
                        "t=%lu d=%lu: "
                        "actual=%d expected=%d\n",
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

    printf(
        "Runtime QKV input checksum: %ld "
        "(expected -1274)\n",
        checksum
    );

    return mismatches;
}


static unsigned long run_q_projection(void) {
    unsigned long start = read_cycles();

    tiled_matmul_auto(
        REAL_Q_I,
        REAL_Q_J,
        REAL_Q_K,

        (elem_t *)runtime_qkv_input,
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
        (elem_t *)runtime_qkv_input,
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
        (elem_t *)runtime_qkv_input,
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
// Runtime Q RMSNorm + RoPE
//
// Input:
//   q_heads [head][seq][dim] -- INT8 Gemmini Q projection output
//
// Steps:
//   1. dequantize with REAL_Q_OUTPUT_SCALE
//   2. RMSNorm over HEAD_DIM
//   3. BF16 round
//   4. RoPE
//   5. BF16 round
//   6. requantize with DECODER_Q_ROPE_SCALE
//
// Output:
//   runtime_q_rope [head][seq][dim]
// -----------------------------------------------------------------------------

static unsigned long run_runtime_q_norm_rope(void)
{
    unsigned long start = read_cycles();

    for (size_t h = 0; h < Q_HEADS; h++) {
        for (size_t t = 0; t < SEQ_LEN; t++) {

            float q_norm[HEAD_DIM];

            // Match CPU baseline RMSNorm:
            // accumulate squares in double precision.
            double sum_sq = 0.0;

            for (size_t d = 0; d < HEAD_DIM; d++) {
                float x =
                    (float)q_heads[h][t][d] *
                    (float)REAL_Q_OUTPUT_SCALE;

                sum_sq += (double)x * (double)x;
            }

            float mean_squared =
                (float)(sum_sq / (double)HEAD_DIM) + 1.0e-6f;

            float inv_rms = powf(mean_squared, -0.5f);

            for (size_t d = 0; d < HEAD_DIM; d++) {
                float x =
                    (float)q_heads[h][t][d] *
                    (float)REAL_Q_OUTPUT_SCALE;

                float y =
                    x *
                    inv_rms *
                    DECODER_Q_NORM_WEIGHT[d];

                q_norm[d] = round_to_bf16(y);
            }

            // Gemma rotate_half:
            // [x1, x2] -> [-x2, x1]
            for (size_t d = 0; d < HEAD_DIM; d++) {
                const size_t half = HEAD_DIM / 2;
                const size_t rotated_d =
                    (d < half) ? d + half : d - half;

                float rotated =
                    (d < half)
                    ? -q_norm[rotated_d]
                    :  q_norm[rotated_d];

                float y =
                    q_norm[d] * DECODER_ROPE_COS[t][d] +
                    rotated   * DECODER_ROPE_SIN[t][d];

                y = round_to_bf16(y);

                int32_t q =
                    (int32_t)roundf(
                        y / (float)DECODER_Q_ROPE_SCALE
                    );

                if (q > 127)
                    q = 127;
                else if (q < -127)
                    q = -127;

                runtime_q_rope[h][t][d] = (elem_t)q;
            }
        }
    }

    unsigned long end = read_cycles();
    return end - start;
}


static unsigned long verify_runtime_q_rope(void)
{
    unsigned long mismatches = 0;
    long checksum = 0;

    for (size_t h = 0; h < Q_HEADS; h++) {
        for (size_t t = 0; t < SEQ_LEN; t++) {
            for (size_t d = 0; d < HEAD_DIM; d++) {

                elem_t actual =
                    runtime_q_rope[h][t][d];

                elem_t expected =
                    DECODER_RUNTIME_Q_ROPE[h][t][d];

                checksum += (long)actual;

                if (actual != expected) {
                    if (mismatches < 10) {
                        printf(
                            "runtime Q mismatch h=%lu t=%lu d=%lu: "
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

    printf(
        "Runtime Q RoPE checksum: %ld "
        "(expected 4760)\n",
        checksum
    );

    return mismatches;
}



// -----------------------------------------------------------------------------
// Runtime K RMSNorm + RoPE
//
// Gemmini K projection INT8
//     -> dequantize
//     -> RMSNorm
//     -> BF16
//     -> RoPE
//     -> BF16
//     -> requantize for Gemmini QK^T
// -----------------------------------------------------------------------------

static unsigned long run_runtime_k_norm_rope(void)
{
    unsigned long start = read_cycles();

    for (size_t t = 0; t < SEQ_LEN; t++) {

        float k_norm[HEAD_DIM];
        double sum_sq = 0.0;

        for (size_t d = 0; d < HEAD_DIM; d++) {
            float x =
                (float)k_heads[0][t][d] *
                (float)REAL_K_OUTPUT_SCALE;

            sum_sq += (double)x * (double)x;
        }

        float mean_squared =
            (float)(sum_sq / (double)HEAD_DIM) + 1.0e-6f;

        float inv_rms = powf(mean_squared, -0.5f);

        for (size_t d = 0; d < HEAD_DIM; d++) {
            float x =
                (float)k_heads[0][t][d] *
                (float)REAL_K_OUTPUT_SCALE;

            float y =
                x *
                inv_rms *
                DECODER_K_NORM_WEIGHT[d];

            k_norm[d] = round_to_bf16(y);
        }

        const size_t half = HEAD_DIM / 2;

        for (size_t d = 0; d < HEAD_DIM; d++) {

            const size_t rotated_d =
                (d < half) ? d + half : d - half;

            float rotated =
                (d < half)
                ? -k_norm[rotated_d]
                :  k_norm[rotated_d];

            float y =
                k_norm[d] * DECODER_ROPE_COS[t][d] +
                rotated   * DECODER_ROPE_SIN[t][d];

            y = round_to_bf16(y);

            int32_t q =
                (int32_t)roundf(
                    y / (float)DECODER_K_ROPE_SCALE
                );

            if (q > 127)
                q = 127;
            else if (q < -127)
                q = -127;

            runtime_k_rope[0][t][d] = (elem_t)q;
        }
    }

    unsigned long end = read_cycles();
    return end - start;
}


static unsigned long verify_runtime_k_rope(void)
{
    unsigned long mismatches = 0;
    long checksum = 0;

    for (size_t t = 0; t < SEQ_LEN; t++) {
        for (size_t d = 0; d < HEAD_DIM; d++) {

            elem_t actual =
                runtime_k_rope[0][t][d];

            elem_t expected =
                DECODER_RUNTIME_K_ROPE[0][t][d];

            checksum += (long)actual;

            if (actual != expected) {
                if (mismatches < 10) {
                    printf(
                        "runtime K mismatch t=%lu d=%lu: "
                        "actual=%d expected=%d\n",
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

    printf(
        "Runtime K RoPE checksum: %ld "
        "(expected -567)\n",
        checksum
    );

    return mismatches;
}



// -----------------------------------------------------------------------------
// Runtime V RMSNorm
//
// Gemmini V projection INT8
//     -> dequantize
//     -> RMSNorm
//     -> BF16
//     -> requantize for Gemmini Attention x V
//
// Gemma layer-0 V RMSNorm has no learned scale weight.
// -----------------------------------------------------------------------------

static unsigned long run_runtime_v_norm(void)
{
    unsigned long start = read_cycles();

    for (size_t t = 0; t < SEQ_LEN; t++) {

        double sum_sq = 0.0;

        for (size_t d = 0; d < HEAD_DIM; d++) {
            float x =
                (float)v_heads[0][t][d] *
                (float)REAL_V_OUTPUT_SCALE;

            sum_sq += (double)x * (double)x;
        }

        float mean_squared =
            (float)(sum_sq / (double)HEAD_DIM) + 1.0e-6f;

        float inv_rms =
            powf(mean_squared, -0.5f);

        for (size_t d = 0; d < HEAD_DIM; d++) {

            float x =
                (float)v_heads[0][t][d] *
                (float)REAL_V_OUTPUT_SCALE;

            float y =
                round_to_bf16(x * inv_rms);

            int32_t q =
                (int32_t)roundf(
                    y / (float)DECODER_V_NORM_SCALE
                );

            if (q > 127)
                q = 127;
            else if (q < -127)
                q = -127;

            runtime_v_norm[t][d] = (elem_t)q;
        }
    }

    unsigned long end = read_cycles();
    return end - start;
}


static unsigned long verify_runtime_v_norm(void)
{
    unsigned long mismatches = 0;
    long checksum = 0;

    for (size_t t = 0; t < SEQ_LEN; t++) {
        for (size_t d = 0; d < HEAD_DIM; d++) {

            elem_t actual =
                runtime_v_norm[t][d];

            elem_t expected =
                DECODER_RUNTIME_V_NORM[t][d];

            checksum += (long)actual;

            if (actual != expected) {
                if (mismatches < 10) {
                    printf(
                        "runtime V mismatch t=%lu d=%lu: "
                        "actual=%d expected=%d\n",
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

    printf(
        "Runtime V norm checksum: %ld "
        "(expected 2159)\n",
        checksum
    );

    return mismatches;
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


// ============================================================
// QK^T attention score kernel
//
// Q[h] : [SEQ_LEN][HEAD_DIM] = [14][256]
// K[0] : [SEQ_LEN][HEAD_DIM] = [14][256]
//
// For each query head h:
//     Q[h] @ K[0]^T -> [14][14]
//
// Gemma 4 uses 8 query heads and 1 KV head, so the same K head
// is shared across all 8 query heads.
// ============================================================

static acc_t qk_gemmini[Q_HEADS][SEQ_LEN][SEQ_LEN] row_align_acc(1);

static unsigned long run_qk_gemmini(void) {
    unsigned long start = read_cycles();

    for (size_t h = 0; h < Q_HEADS; h++) {
        tiled_matmul_auto(
            SEQ_LEN,                 // I = 14
            SEQ_LEN,                 // J = 14
            HEAD_DIM,                // K = 256

            (elem_t *)runtime_q_rope[h],     // A = runtime Q[h], [14][256]
            (elem_t *)runtime_k_rope[0],     // B = runtime K[0], [14][256]
            NULL,                     // no bias
            (acc_t *)qk_gemmini[h],   // C = raw accumulator output

            HEAD_DIM,                 // stride_A
            HEAD_DIM,                 // stride_B (physical K rows)
            SEQ_LEN,                  // stride_D (unused)
            SEQ_LEN,                  // stride_C

            MVIN_SCALE_IDENTITY,
            MVIN_SCALE_IDENTITY,
            MVIN_SCALE_IDENTITY,

            NO_ACTIVATION,
            ACC_SCALE_IDENTITY,
            0,

            false,                    // repeating_bias
            false,                    // transpose_A
            true,                     // transpose_B: Q @ K^T

            true,                     // full_C -> write acc_t
            false,                    // low_D

            0,
            WS
        );
    }

    unsigned long end = read_cycles();
    return end - start;
}

static size_t verify_qk(void) {
    size_t mismatches = 0;

    for (size_t h = 0; h < Q_HEADS; h++) {
        for (size_t i = 0; i < SEQ_LEN; i++) {
            for (size_t j = 0; j < SEQ_LEN; j++) {
                acc_t actual = qk_gemmini[h][i][j];
                acc_t expected = DECODER_RUNTIME_QK_EXPECTED[h][i][j];

                if (actual != expected) {
                    if (mismatches < 10) {
                        printf(
                            "QK mismatch h=%u i=%u j=%u: "
                            "gemmini=%d expected=%d\n",
                            (unsigned)h,
                            (unsigned)i,
                            (unsigned)j,
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


// ============================================================
// Runtime attention softmax
//
// Input:
//   qk_gemmini[h][i][j] = INT32 Gemmini Q @ K^T result
//
// Steps:
//   1. Dequantize QK score
//   2. Apply causal mask (SEQ_LEN=14 < sliding window 512)
//   3. Stable FP32 softmax
//   4. Round probability to BF16
//   5. Quantize probability to INT8 for Gemmini Attention x V
//
// Attention scale is 1/127, so probability 1.0 maps to 127.
// ============================================================

static elem_t runtime_attn[Q_HEADS][SEQ_LEN][SEQ_LEN]
    row_align(1);

static unsigned long run_runtime_softmax(void)
{
    unsigned long start = read_cycles();

    const float qk_scale =
        (float)DECODER_Q_ROPE_SCALE *
        (float)DECODER_K_ROPE_SCALE;

    const float attn_scale =
        (float)DECODER_ATTN_SCALE;

    for (size_t h = 0; h < Q_HEADS; h++) {
        for (size_t i = 0; i < SEQ_LEN; i++) {

            float row[SEQ_LEN];
            float max_value = -INFINITY;

            // Dequantize only causal entries j <= i.
            for (size_t j = 0; j <= i; j++) {
                float score =
                    (float)qk_gemmini[h][i][j] *
                    qk_scale;

                row[j] = score;

                if (score > max_value)
                    max_value = score;
            }

            // Stable exponential and row sum.
            float sum = 0.0f;

            for (size_t j = 0; j <= i; j++) {
                float e =
                    expf(row[j] - max_value);

                row[j] = e;
                sum += e;
            }

            // Normalize -> BF16 -> INT8.
            for (size_t j = 0; j <= i; j++) {
                float p_softmax =
                    round_to_bf16(row[j] / sum);

                int32_t q =
                    (int32_t)roundf(
                        p_softmax / attn_scale
                    );

                if (q > 127)
                    q = 127;

                if (q < -127)
                    q = -127;

                runtime_attn[h][i][j] =
                    (elem_t)q;
            }

            // Causal masked entries are exactly zero probability.
            for (size_t j = i + 1; j < SEQ_LEN; j++) {
                runtime_attn[h][i][j] = 0;
            }
        }
    }

    unsigned long end = read_cycles();
    return end - start;
}


static size_t verify_runtime_attn(void)
{
    size_t mismatches = 0;

    for (size_t h = 0; h < Q_HEADS; h++) {
        for (size_t i = 0; i < SEQ_LEN; i++) {
            for (size_t j = 0; j < SEQ_LEN; j++) {

                elem_t actual =
                    runtime_attn[h][i][j];

                elem_t expected =
                    DECODER_FULL_RUNTIME_ATTN[h][i][j];

                if (actual != expected) {
                    if (mismatches < 10) {
                        printf(
                            "Runtime attention mismatch "
                            "h=%u i=%u j=%u: "
                            "runtime=%d expected=%d\n",
                            (unsigned)h,
                            (unsigned)i,
                            (unsigned)j,
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


// ============================================================
// Attention x V
//
// Uses runtime attention produced from Gemmini QK^T followed by
// CPU causal FP32 softmax and BF16/INT8 quantization.
//
// V is still supplied from the golden post-V-RMSNorm INT8 boundary.
//
// For each query head:
//   [14][14] @ [14][256] -> [14][256]
// ============================================================

static acc_t attn_v_gemmini[Q_HEADS][SEQ_LEN][HEAD_DIM]
    row_align_acc(1);

static unsigned long run_attn_v_gemmini(void) {
    unsigned long start = read_cycles();

    for (size_t h = 0; h < Q_HEADS; h++) {
        tiled_matmul_auto(
            SEQ_LEN,
            HEAD_DIM,
            SEQ_LEN,

            (elem_t *)runtime_attn[h],
            (elem_t *)runtime_v_norm,
            NULL,
            (acc_t *)attn_v_gemmini[h],

            SEQ_LEN,
            HEAD_DIM,
            HEAD_DIM,
            HEAD_DIM,

            MVIN_SCALE_IDENTITY,
            MVIN_SCALE_IDENTITY,
            MVIN_SCALE_IDENTITY,

            NO_ACTIVATION,
            ACC_SCALE_IDENTITY,
            0,

            false,
            false,
            false,

            true,
            false,

            0,
            WS
        );
    }

    unsigned long end = read_cycles();
    return end - start;
}

static size_t verify_attn_v(void) {
    size_t mismatches = 0;

    for (size_t h = 0; h < Q_HEADS; h++) {
        for (size_t i = 0; i < SEQ_LEN; i++) {
            for (size_t d = 0; d < HEAD_DIM; d++) {

                acc_t actual =
                    attn_v_gemmini[h][i][d];

                acc_t expected =
                    DECODER_FULL_RUNTIME_ATTN_V_EXPECTED[h][i][d];

                if (actual != expected) {
                    if (mismatches < 10) {
                        printf(
                            "Attention x V mismatch "
                            "h=%u i=%u d=%u: "
                            "gemmini=%d expected=%d\n",
                            (unsigned)h,
                            (unsigned)i,
                            (unsigned)d,
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



// ============================================================
// Runtime Attention x V -> O projection input
//
// attn_v_gemmini is an INT32 accumulator representing:
//
//   real = accumulator * ATTN_SCALE * V_NORM_SCALE
//
// Attention output layout:
//
//   [head][token][dim] -> [token][head * HEAD_DIM + dim]
//
// The flattened result is requantized using REAL_O_X_SCALE.
// ============================================================

static elem_t runtime_o_input[REAL_O_I][REAL_O_K]
    row_align(1);

static elem_t runtime_o_proj[REAL_O_I][REAL_O_J]
    row_align(1);


static unsigned long prepare_runtime_o_input(void)
{
    unsigned long start = read_cycles();

    const float attn_v_scale =
        (float)DECODER_ATTN_SCALE *
        (float)DECODER_V_NORM_SCALE;

    const float o_input_scale =
        (float)REAL_O_X_SCALE;

    for (size_t t = 0; t < SEQ_LEN; t++) {
        for (size_t h = 0; h < Q_HEADS; h++) {
            for (size_t d = 0; d < HEAD_DIM; d++) {

                float real_value =
                    (float)attn_v_gemmini[h][t][d] *
                    attn_v_scale;

                int32_t q =
                    (int32_t)roundf(
                        real_value / o_input_scale
                    );

                if (q > 127)
                    q = 127;
                else if (q < -127)
                    q = -127;

                runtime_o_input[t][h * HEAD_DIM + d] =
                    (elem_t)q;
            }
        }
    }

    unsigned long end = read_cycles();
    return end - start;
}


static size_t verify_runtime_o_input(void)
{
    size_t mismatches = 0;
    long checksum = 0;

    for (size_t t = 0; t < REAL_O_I; t++) {
        for (size_t j = 0; j < REAL_O_K; j++) {

            elem_t actual =
                runtime_o_input[t][j];

            elem_t expected =
                DECODER_RUNTIME_O_INPUT_EXPECTED[t][j];

            checksum += (long)actual;

            if (actual != expected) {
                if (mismatches < 10) {
                    printf(
                        "Runtime O input mismatch "
                        "t=%lu j=%lu: "
                        "actual=%d expected=%d\n",
                        (unsigned long)t,
                        (unsigned long)j,
                        (int)actual,
                        (int)expected
                    );
                }

                mismatches++;
            }
        }
    }

    printf(
        "Runtime O input checksum: %ld "
        "(expected 19523)\n",
        checksum
    );

    return mismatches;
}


// ============================================================
// Gemmini O projection
//
// [14][2048] @ [2048][1536] -> [14][1536]
// ============================================================

static unsigned long run_runtime_o_projection(void)
{
    unsigned long start = read_cycles();

    tiled_matmul_auto(
        REAL_O_I,
        REAL_O_J,
        REAL_O_K,

        (elem_t *)runtime_o_input,
        (elem_t *)REAL_O_W_TRANSPOSED,
        NULL,
        (elem_t *)runtime_o_proj,

        REAL_O_K,
        REAL_O_J,
        REAL_O_J,
        REAL_O_J,

        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,

        NO_ACTIVATION,
        REAL_O_C_SCALE,
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


static size_t verify_runtime_o_projection(void)
{
    size_t mismatches = 0;
    long checksum = 0;

    for (size_t t = 0; t < REAL_O_I; t++) {
        for (size_t j = 0; j < REAL_O_J; j++) {

            elem_t actual =
                runtime_o_proj[t][j];

            elem_t expected =
                DECODER_RUNTIME_O_EXPECTED[t][j];

            checksum += (long)actual;

            if (actual != expected) {
                if (mismatches < 10) {
                    printf(
                        "Runtime O projection mismatch "
                        "t=%lu j=%lu: "
                        "actual=%d expected=%d\n",
                        (unsigned long)t,
                        (unsigned long)j,
                        (int)actual,
                        (int)expected
                    );
                }

                mismatches++;
            }
        }
    }

    printf(
        "Runtime O output checksum: %ld "
        "(expected 820)\n",
        checksum
    );

    return mismatches;
}



// ============================================================
// Runtime O projection -> post-attention RMSNorm
// -> first residual -> pre-FFN RMSNorm -> FFN INT8 input
// ============================================================

static float runtime_post_attn_norm[REAL_O_I][REAL_O_J];
static float runtime_first_residual[REAL_O_I][REAL_O_J];
static float runtime_pre_ffn_norm[REAL_O_I][REAL_O_J];

static elem_t runtime_ffn_input[REAL_O_I][REAL_O_J]
    row_align(1);


static unsigned long prepare_runtime_ffn_input(void)
{
    unsigned long start = read_cycles();

    // --------------------------------------------------------
    // 1. Dequantize O output and apply post-attention RMSNorm
    // --------------------------------------------------------

    for (size_t t = 0; t < REAL_O_I; t++) {

        double sum_sq = 0.0;

        for (size_t d = 0; d < REAL_O_J; d++) {
            float x =
                (float)runtime_o_proj[t][d] *
                (float)REAL_O_OUTPUT_SCALE;

            sum_sq += (double)x * (double)x;
        }

        float mean_squared =
            (float)(sum_sq / (double)REAL_O_J) +
            1.0e-6f;

        float inv_rms =
            powf(mean_squared, -0.5f);

        for (size_t d = 0; d < REAL_O_J; d++) {

            float x =
                (float)runtime_o_proj[t][d] *
                (float)REAL_O_OUTPUT_SCALE;

            float y =
                x *
                inv_rms *
                DECODER_POST_ATTN_NORM_WEIGHT[d];

            runtime_post_attn_norm[t][d] =
                round_to_bf16(y);
        }
    }


    // --------------------------------------------------------
    // 2. First residual:
    //    original layer input + normalized attention output
    // --------------------------------------------------------

    for (size_t t = 0; t < REAL_O_I; t++) {
        for (size_t d = 0; d < REAL_O_J; d++) {

            runtime_first_residual[t][d] =
                round_to_bf16(
                    DECODER_LAYER_INPUT[t][d] +
                    runtime_post_attn_norm[t][d]
                );
        }
    }


    // --------------------------------------------------------
    // 3. Pre-FFN RMSNorm
    // --------------------------------------------------------

    for (size_t t = 0; t < REAL_O_I; t++) {

        double sum_sq = 0.0;

        for (size_t d = 0; d < REAL_O_J; d++) {
            float x =
                runtime_first_residual[t][d];

            sum_sq += (double)x * (double)x;
        }

        float mean_squared =
            (float)(sum_sq / (double)REAL_O_J) +
            1.0e-6f;

        float inv_rms =
            powf(mean_squared, -0.5f);

        for (size_t d = 0; d < REAL_O_J; d++) {

            float y =
                runtime_first_residual[t][d] *
                inv_rms *
                DECODER_PRE_FFN_NORM_WEIGHT[d];

            runtime_pre_ffn_norm[t][d] =
                round_to_bf16(y);
        }
    }


    // --------------------------------------------------------
    // 4. Quantize shared Gate/Up input
    // --------------------------------------------------------

    for (size_t t = 0; t < REAL_O_I; t++) {
        for (size_t d = 0; d < REAL_O_J; d++) {

            int32_t q =
                (int32_t)roundf(
                    runtime_pre_ffn_norm[t][d] /
                    (float)REAL_GATE_X_SCALE
                );

            if (q > 127)
                q = 127;
            else if (q < -127)
                q = -127;

            runtime_ffn_input[t][d] =
                (elem_t)q;
        }
    }

    unsigned long end = read_cycles();
    return end - start;
}


static size_t verify_runtime_ffn_input(void)
{
    size_t mismatches = 0;
    long checksum = 0;

    for (size_t t = 0; t < REAL_O_I; t++) {
        for (size_t d = 0; d < REAL_O_J; d++) {

            // Verify post-attention RMSNorm exactly.
            if (
                runtime_post_attn_norm[t][d] !=
                DECODER_RUNTIME_POST_ATTN_NORM_EXPECTED[t][d]
            ) {
                if (mismatches < 10) {
                    printf(
                        "Post-attn norm mismatch "
                        "t=%lu d=%lu\n",
                        (unsigned long)t,
                        (unsigned long)d
                    );
                }
                mismatches++;
            }

            // Verify first residual exactly.
            if (
                runtime_first_residual[t][d] !=
                DECODER_RUNTIME_FIRST_RESIDUAL_EXPECTED[t][d]
            ) {
                if (mismatches < 10) {
                    printf(
                        "First residual mismatch "
                        "t=%lu d=%lu\n",
                        (unsigned long)t,
                        (unsigned long)d
                    );
                }
                mismatches++;
            }

            // Verify pre-FFN RMSNorm exactly.
            if (
                runtime_pre_ffn_norm[t][d] !=
                DECODER_RUNTIME_PRE_FFN_NORM_EXPECTED[t][d]
            ) {
                if (mismatches < 10) {
                    printf(
                        "Pre-FFN norm mismatch "
                        "t=%lu d=%lu\n",
                        (unsigned long)t,
                        (unsigned long)d
                    );
                }
                mismatches++;
            }

            elem_t actual =
                runtime_ffn_input[t][d];

            elem_t expected =
                DECODER_RUNTIME_FFN_INPUT_EXPECTED[t][d];

            checksum += (long)actual;

            if (actual != expected) {
                if (mismatches < 10) {
                    printf(
                        "Runtime FFN input mismatch "
                        "t=%lu d=%lu: "
                        "actual=%d expected=%d\n",
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

    printf(
        "Runtime FFN input checksum: %ld "
        "(expected 2151)\n",
        checksum
    );

    return mismatches;
}



// ============================================================
// Runtime Gemmini Gate + Up projections
//
// [14,1536] @ [1536,6144] -> [14,6144]
// ============================================================

static elem_t runtime_gate_proj[REAL_GATE_I][REAL_GATE_J]
    row_align(1);

static elem_t runtime_up_proj[REAL_UP_I][REAL_UP_J]
    row_align(1);


static unsigned long run_runtime_gate_projection(void)
{
    unsigned long start = read_cycles();

    tiled_matmul_auto(
        REAL_GATE_I,
        REAL_GATE_J,
        REAL_GATE_K,

        (elem_t *)runtime_ffn_input,
        (elem_t *)REAL_GATE_W_TRANSPOSED,
        NULL,
        (elem_t *)runtime_gate_proj,

        REAL_GATE_K,
        REAL_GATE_J,
        REAL_GATE_J,
        REAL_GATE_J,

        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,

        NO_ACTIVATION,
        REAL_GATE_C_SCALE,
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


static unsigned long run_runtime_up_projection(void)
{
    unsigned long start = read_cycles();

    tiled_matmul_auto(
        REAL_UP_I,
        REAL_UP_J,
        REAL_UP_K,

        (elem_t *)runtime_ffn_input,
        (elem_t *)REAL_UP_W_TRANSPOSED,
        NULL,
        (elem_t *)runtime_up_proj,

        REAL_UP_K,
        REAL_UP_J,
        REAL_UP_J,
        REAL_UP_J,

        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,

        NO_ACTIVATION,
        REAL_UP_C_SCALE,
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


static size_t verify_runtime_gate_up(void)
{
    size_t gate_mismatches = 0;
    size_t up_mismatches = 0;

    long gate_checksum = 0;
    long up_checksum = 0;

    for (size_t t = 0; t < REAL_GATE_I; t++) {
        for (size_t j = 0; j < REAL_GATE_J; j++) {

            elem_t gate_actual =
                runtime_gate_proj[t][j];

            elem_t gate_expected =
                DECODER_RUNTIME_GATE_EXPECTED[t][j];

            gate_checksum += (long)gate_actual;

            if (gate_actual != gate_expected) {
                if (gate_mismatches < 10) {
                    printf(
                        "Runtime Gate mismatch "
                        "t=%lu j=%lu: actual=%d expected=%d\n",
                        (unsigned long)t,
                        (unsigned long)j,
                        (int)gate_actual,
                        (int)gate_expected
                    );
                }

                gate_mismatches++;
            }


            elem_t up_actual =
                runtime_up_proj[t][j];

            elem_t up_expected =
                DECODER_RUNTIME_UP_EXPECTED[t][j];

            up_checksum += (long)up_actual;

            if (up_actual != up_expected) {
                if (up_mismatches < 10) {
                    printf(
                        "Runtime Up mismatch "
                        "t=%lu j=%lu: actual=%d expected=%d\n",
                        (unsigned long)t,
                        (unsigned long)j,
                        (int)up_actual,
                        (int)up_expected
                    );
                }

                up_mismatches++;
            }
        }
    }

    printf(
        "Runtime Gate checksum: %ld "
        "(expected -729534)\n",
        gate_checksum
    );

    printf(
        "Runtime Up checksum: %ld "
        "(expected -3445)\n",
        up_checksum
    );

    printf(
        "Runtime Gate mismatches: %lu\n",
        (unsigned long)gate_mismatches
    );

    printf(
        "Runtime Up mismatches: %lu\n",
        (unsigned long)up_mismatches
    );

    return gate_mismatches + up_mismatches;
}



// ============================================================
// Runtime GELU(gate) * up -> quantized Down input
// ============================================================

static float runtime_gate_up_product[REAL_GATE_I][REAL_GATE_J];

static elem_t runtime_down_input[REAL_DOWN_I][REAL_DOWN_K]
    row_align(1);


static unsigned long prepare_runtime_down_input(void)
{
    unsigned long start = read_cycles();

    const float gelu_coeff =
        0.7978845608028654f;

    for (size_t t = 0; t < REAL_GATE_I; t++) {
        for (size_t j = 0; j < REAL_GATE_J; j++) {

            // Dequantize actual propagated Gemmini outputs.
            float x =
                (float)runtime_gate_proj[t][j] *
                (float)REAL_GATE_OUTPUT_SCALE;

            float up =
                (float)runtime_up_proj[t][j] *
                (float)REAL_UP_OUTPUT_SCALE;

            // Exact GELU-tanh structure from decoder_cpu.c.
            float x3 = x * x * x;

            float inner =
                gelu_coeff *
                (x + 0.044715f * x3);

            float gelu =
                0.5f *
                x *
                (1.0f + tanhf(inner));

            // CPU baseline has its BF16 boundary here.
            float product =
                round_to_bf16(gelu * up);

            runtime_gate_up_product[t][j] =
                product;

            // Quantize for Gemmini Down projection.
            int32_t q =
                (int32_t)roundf(
                    product /
                    (float)REAL_DOWN_X_SCALE
                );

            if (q > 127)
                q = 127;
            else if (q < -127)
                q = -127;

            runtime_down_input[t][j] =
                (elem_t)q;
        }
    }

    unsigned long end = read_cycles();
    return end - start;
}


static size_t verify_runtime_down_input(void)
{
    size_t product_mismatches = 0;
    size_t input_mismatches = 0;

    long input_checksum = 0;

    for (size_t t = 0; t < REAL_DOWN_I; t++) {
        for (size_t j = 0; j < REAL_DOWN_K; j++) {

            if (
                runtime_gate_up_product[t][j] !=
                DECODER_RUNTIME_GATE_UP_PRODUCT_EXPECTED[t][j]
            ) {
                if (product_mismatches < 10) {
                    printf(
                        "Gate-Up product mismatch "
                        "t=%lu j=%lu\n",
                        (unsigned long)t,
                        (unsigned long)j
                    );
                }

                product_mismatches++;
            }

            elem_t actual =
                runtime_down_input[t][j];

            elem_t expected =
                DECODER_RUNTIME_DOWN_INPUT_EXPECTED[t][j];

            input_checksum += (long)actual;

            if (actual != expected) {
                if (input_mismatches < 10) {
                    printf(
                        "Runtime Down input mismatch "
                        "t=%lu j=%lu: actual=%d expected=%d\n",
                        (unsigned long)t,
                        (unsigned long)j,
                        (int)actual,
                        (int)expected
                    );
                }

                input_mismatches++;
            }
        }
    }

    printf(
        "Runtime GELU x Up mismatches: %lu\n",
        (unsigned long)product_mismatches
    );

    printf(
        "Runtime Down input checksum: %ld "
        "(expected -32)\n",
        input_checksum
    );

    printf(
        "Runtime Down input mismatches: %lu\n",
        (unsigned long)input_mismatches
    );

    return product_mismatches + input_mismatches;
}



// ============================================================
// Runtime Gemmini Down projection
//
// [14,6144] @ [6144,1536] -> [14,1536]
// ============================================================

static elem_t runtime_down_proj[REAL_DOWN_I][REAL_DOWN_J]
    row_align(1);


static unsigned long run_runtime_down_projection(void)
{
    unsigned long start = read_cycles();

    tiled_matmul_auto(
        REAL_DOWN_I,
        REAL_DOWN_J,
        REAL_DOWN_K,

        (elem_t *)runtime_down_input,
        (elem_t *)REAL_DOWN_W_TRANSPOSED,
        NULL,
        (elem_t *)runtime_down_proj,

        REAL_DOWN_K,
        REAL_DOWN_J,
        REAL_DOWN_J,
        REAL_DOWN_J,

        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,

        NO_ACTIVATION,
        REAL_DOWN_C_SCALE,
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


static size_t verify_runtime_down_projection(void)
{
    size_t mismatches = 0;
    long checksum = 0;

    for (size_t t = 0; t < REAL_DOWN_I; t++) {
        for (size_t j = 0; j < REAL_DOWN_J; j++) {

            elem_t actual =
                runtime_down_proj[t][j];

            elem_t expected =
                DECODER_RUNTIME_DOWN_EXPECTED[t][j];

            checksum += (long)actual;

            if (actual != expected) {
                if (mismatches < 10) {
                    printf(
                        "Runtime Down mismatch "
                        "t=%lu j=%lu: actual=%d expected=%d\n",
                        (unsigned long)t,
                        (unsigned long)j,
                        (int)actual,
                        (int)expected
                    );
                }

                mismatches++;
            }
        }
    }

    printf(
        "Runtime Down checksum: %ld "
        "(expected -2789)\n",
        checksum
    );

    printf(
        "Runtime Down mismatches: %lu\n",
        (unsigned long)mismatches
    );

    return mismatches;
}




// ============================================================
// Runtime post-FFN path
//
// runtime_down_proj
//   -> dequantize
//   -> post-FFN RMSNorm + BF16
//   -> add runtime_first_residual + BF16
//   -> quantize for per-layer Gate
// ============================================================

static float runtime_post_ffn_norm[14][1536];
static float runtime_second_residual[14][1536];

static elem_t runtime_per_layer_gate_x[
    RUNTIME_PL_GATE_I
][
    RUNTIME_PL_GATE_K
] row_align(1);


static unsigned long prepare_runtime_post_ffn(void)
{
    unsigned long start = read_cycles();

    for (size_t t = 0; t < 14; t++) {

        // RMSNorm denominator for dequantized Down output.
        float mean_sq = 0.0f;

        for (size_t d = 0; d < 1536; d++) {
            float x =
                (float)runtime_down_proj[t][d] *
                REAL_DOWN_OUTPUT_SCALE;

            mean_sq += x * x;
        }

        mean_sq /= 1536.0f;

        float inv_rms =
            1.0f / sqrtf(mean_sq + 1.0e-6f);

        // post-FFN RMSNorm, residual, and quantization for
        // the per-layer Gate.
        for (size_t d = 0; d < 1536; d++) {

            float x =
                (float)runtime_down_proj[t][d] *
                REAL_DOWN_OUTPUT_SCALE;

            float norm =
                x *
                inv_rms *
                DECODER_POST_FFN_NORM_WEIGHT[d];

            norm = round_to_bf16(norm);

            runtime_post_ffn_norm[t][d] = norm;

            float residual =
                runtime_first_residual[t][d] + norm;

            residual = round_to_bf16(residual);

            runtime_second_residual[t][d] =
                residual;

            float q =
                roundf(
                    residual /
                    RUNTIME_PL_GATE_X_SCALE
                );

            if (q > 127.0f)
                q = 127.0f;

            if (q < -127.0f)
                q = -127.0f;

            runtime_per_layer_gate_x[t][d] =
                (elem_t)q;
        }
    }

    unsigned long end = read_cycles();
    return end - start;
}


static size_t verify_runtime_post_ffn(void)
{
    size_t post_norm_mismatches = 0;
    size_t residual_mismatches = 0;
    size_t x_mismatches = 0;

    long x_checksum = 0;

    for (size_t t = 0; t < 14; t++) {
        for (size_t d = 0; d < 1536; d++) {

            if (
                runtime_post_ffn_norm[t][d] !=
                DECODER_RUNTIME_POST_FFN_NORM_EXPECTED[t][d]
            ) {
                post_norm_mismatches++;
            }

            if (
                runtime_second_residual[t][d] !=
                DECODER_RUNTIME_SECOND_RESIDUAL_EXPECTED[t][d]
            ) {
                residual_mismatches++;
            }

            elem_t actual_x =
                runtime_per_layer_gate_x[t][d];

            elem_t expected_x =
                DECODER_RUNTIME_PER_LAYER_GATE_X[t][d];

            x_checksum += (long)actual_x;

            if (actual_x != expected_x) {
                if (x_mismatches < 10) {
                    printf(
                        "Runtime post-FFN X mismatch "
                        "t=%lu d=%lu: actual=%d expected=%d\n",
                        (unsigned long)t,
                        (unsigned long)d,
                        (int)actual_x,
                        (int)expected_x
                    );
                }

                x_mismatches++;
            }
        }
    }

    printf(
        "Runtime post-FFN norm mismatches: %lu\n",
        (unsigned long)post_norm_mismatches
    );

    printf(
        "Runtime second residual mismatches: %lu\n",
        (unsigned long)residual_mismatches
    );

    printf(
        "Runtime per-layer Gate X checksum: %ld "
        "(expected 342)\n",
        x_checksum
    );

    printf(
        "Runtime per-layer Gate X mismatches: %lu\n",
        (unsigned long)x_mismatches
    );

    return
        post_norm_mismatches +
        residual_mismatches +
        x_mismatches;
}


// ============================================================
// Runtime per-layer Gate projection
//
// [14,1536] @ [1536,256] -> [14,256]
// ============================================================

static elem_t runtime_per_layer_gate[
    RUNTIME_PL_GATE_I
][
    RUNTIME_PL_GATE_J
] row_align(1);


static unsigned long run_runtime_per_layer_gate(void)
{
    unsigned long start = read_cycles();

    tiled_matmul_auto(
        RUNTIME_PL_GATE_I,
        RUNTIME_PL_GATE_J,
        RUNTIME_PL_GATE_K,

        (elem_t *)runtime_per_layer_gate_x,
        (elem_t *)RUNTIME_PL_GATE_W_TRANSPOSED,
        NULL,
        (elem_t *)runtime_per_layer_gate,

        RUNTIME_PL_GATE_K,
        RUNTIME_PL_GATE_J,
        RUNTIME_PL_GATE_J,
        RUNTIME_PL_GATE_J,

        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,

        NO_ACTIVATION,
        RUNTIME_PL_GATE_C_SCALE,
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


static size_t verify_runtime_per_layer_gate(void)
{
    size_t mismatches = 0;
    long checksum = 0;

    for (size_t t = 0; t < RUNTIME_PL_GATE_I; t++) {
        for (size_t j = 0; j < RUNTIME_PL_GATE_J; j++) {

            elem_t actual =
                runtime_per_layer_gate[t][j];

            elem_t expected =
                DECODER_RUNTIME_PER_LAYER_GATE_EXPECTED[t][j];

            checksum += (long)actual;

            if (actual != expected) {
                if (mismatches < 10) {
                    printf(
                        "Runtime per-layer Gate mismatch "
                        "t=%lu j=%lu: actual=%d expected=%d\n",
                        (unsigned long)t,
                        (unsigned long)j,
                        (int)actual,
                        (int)expected
                    );
                }

                mismatches++;
            }
        }
    }

    printf(
        "Runtime per-layer Gate checksum: %ld "
        "(expected 6429)\n",
        checksum
    );

    printf(
        "Runtime per-layer Gate mismatches: %lu\n",
        (unsigned long)mismatches
    );

    return mismatches;
}




// ============================================================
// Runtime per-layer product
//
// actual per-layer Gate
//   -> dequantize
//   -> GELU-tanh
//   -> multiply per_layer_input
//   -> BF16
//   -> quantize for final Gemmini projection
// ============================================================

static float runtime_per_layer_product[14][256];

static elem_t runtime_per_layer_proj_x[
    RUNTIME_PL_PROJ_I
][
    RUNTIME_PL_PROJ_K
] row_align(1);


static unsigned long prepare_runtime_per_layer_product(void)
{
    unsigned long start = read_cycles();

    for (size_t t = 0; t < 14; t++) {
        for (size_t j = 0; j < 256; j++) {

            float x =
                (float)runtime_per_layer_gate[t][j] *
                RUNTIME_PL_GATE_OUTPUT_SCALE;

            float x3 = x * x * x;

            float inner =
                0.7978845608028654f *
                (x + 0.044715f * x3);

            float gelu =
                0.5f * x *
                (1.0f + tanhf(inner));

            float product =
                gelu *
                DECODER_PER_LAYER_INPUT[t][j];

            product = round_to_bf16(product);

            runtime_per_layer_product[t][j] =
                product;

            float q =
                roundf(
                    product /
                    RUNTIME_PL_PROJ_X_SCALE
                );

            if (q > 127.0f)
                q = 127.0f;

            if (q < -127.0f)
                q = -127.0f;

            runtime_per_layer_proj_x[t][j] =
                (elem_t)q;
        }
    }

    unsigned long end = read_cycles();
    return end - start;
}


static size_t verify_runtime_per_layer_product(void)
{
    size_t x_mismatches = 0;
    long checksum = 0;

    for (size_t t = 0; t < 14; t++) {
        for (size_t j = 0; j < 256; j++) {

            elem_t actual =
                runtime_per_layer_proj_x[t][j];

            elem_t expected =
                DECODER_RUNTIME_PER_LAYER_PROJ_X[t][j];

            checksum += (long)actual;

            if (actual != expected) {
                if (x_mismatches < 10) {
                    printf(
                        "Runtime per-layer product X mismatch "
                        "t=%lu j=%lu: actual=%d expected=%d\n",
                        (unsigned long)t,
                        (unsigned long)j,
                        (int)actual,
                        (int)expected
                    );
                }

                x_mismatches++;
            }
        }
    }

    printf(
        "Runtime per-layer projection X checksum: %ld "
        "(expected -237)\n",
        checksum
    );

    printf(
        "Runtime per-layer projection X mismatches: %lu\n",
        (unsigned long)x_mismatches
    );

    return x_mismatches;
}


// ============================================================
// Runtime per-layer projection
//
// [14,256] @ [256,1536] -> [14,1536]
// ============================================================

static elem_t runtime_per_layer_proj[
    RUNTIME_PL_PROJ_I
][
    RUNTIME_PL_PROJ_J
] row_align(1);


static unsigned long run_runtime_per_layer_projection(void)
{
    unsigned long start = read_cycles();

    tiled_matmul_auto(
        RUNTIME_PL_PROJ_I,
        RUNTIME_PL_PROJ_J,
        RUNTIME_PL_PROJ_K,

        (elem_t *)runtime_per_layer_proj_x,
        (elem_t *)RUNTIME_PL_PROJ_W_TRANSPOSED,
        NULL,
        (elem_t *)runtime_per_layer_proj,

        RUNTIME_PL_PROJ_K,
        RUNTIME_PL_PROJ_J,
        RUNTIME_PL_PROJ_J,
        RUNTIME_PL_PROJ_J,

        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,

        NO_ACTIVATION,
        RUNTIME_PL_PROJ_C_SCALE,
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


static size_t verify_runtime_per_layer_projection(void)
{
    size_t mismatches = 0;
    long checksum = 0;

    for (size_t t = 0; t < RUNTIME_PL_PROJ_I; t++) {
        for (size_t j = 0; j < RUNTIME_PL_PROJ_J; j++) {

            elem_t actual =
                runtime_per_layer_proj[t][j];

            elem_t expected =
                DECODER_RUNTIME_PER_LAYER_PROJ_EXPECTED[t][j];

            checksum += (long)actual;

            if (actual != expected) {
                if (mismatches < 10) {
                    printf(
                        "Runtime per-layer projection mismatch "
                        "t=%lu j=%lu: actual=%d expected=%d\n",
                        (unsigned long)t,
                        (unsigned long)j,
                        (int)actual,
                        (int)expected
                    );
                }

                mismatches++;
            }
        }
    }

    printf(
        "Runtime per-layer projection checksum: %ld "
        "(expected 1397)\n",
        checksum
    );

    printf(
        "Runtime per-layer projection mismatches: %lu\n",
        (unsigned long)mismatches
    );

    return mismatches;
}



// ============================================================
// Final runtime decoder tail
//
// runtime_per_layer_proj
//   -> dequantize
//   -> post-per-layer RMSNorm + BF16
//   -> add runtime_second_residual + BF16
//   -> multiply layer scalar + BF16
//   -> final decoder-layer output
// ============================================================

static float runtime_post_per_layer_norm[14][1536];
static float runtime_pre_scalar[14][1536];
static float runtime_final_output[14][1536];


static unsigned long prepare_runtime_final_output(void)
{
    unsigned long start = read_cycles();

    for (size_t t = 0; t < 14; t++) {

        float mean_sq = 0.0f;

        // RMSNorm denominator from dequantized final Gemmini projection.
        for (size_t d = 0; d < 1536; d++) {

            float x =
                (float)runtime_per_layer_proj[t][d] *
                RUNTIME_PL_PROJ_OUTPUT_SCALE;

            mean_sq += x * x;
        }

        mean_sq /= 1536.0f;

        float inv_rms =
            1.0f / sqrtf(mean_sq + 1.0e-6f);

        for (size_t d = 0; d < 1536; d++) {

            float x =
                (float)runtime_per_layer_proj[t][d] *
                RUNTIME_PL_PROJ_OUTPUT_SCALE;

            float norm =
                x *
                inv_rms *
                DECODER_POST_PER_LAYER_NORM_WEIGHT[d];

            norm = round_to_bf16(norm);

            runtime_post_per_layer_norm[t][d] =
                norm;

            // Same semantics as residual_add() in CPU baseline:
            // BF16-round the residual result.
            float pre_scalar =
                runtime_second_residual[t][d] +
                norm;

            pre_scalar =
                round_to_bf16(pre_scalar);

            runtime_pre_scalar[t][d] =
                pre_scalar;

            // Gemma final layer scalar multiply, then BF16.
            runtime_final_output[t][d] =
                round_to_bf16(
                    pre_scalar *
                    DECODER_LAYER_SCALAR
                );
        }
    }

    unsigned long end = read_cycles();
    return end - start;
}


static void print_double_fixed(const char *label, double value)
{
    const unsigned long scale = 1000000000UL;

    if (value < 0.0) {
        printf("%s-", label);
        value = -value;
    } else {
        printf("%s", label);
    }

    unsigned long whole = (unsigned long)value;
    unsigned long frac =
        (unsigned long)((value - (double)whole) * (double)scale + 0.5);

    if (frac >= scale) {
        whole++;
        frac -= scale;
    }

    printf("%lu.%09lu\n", whole, frac);
}


static size_t verify_runtime_final_output(void)
{
    size_t norm_mismatches = 0;
    size_t pre_scalar_mismatches = 0;
    size_t final_mismatches = 0;

    double final_checksum = 0.0;
    double expected_checksum = 0.0;

    float max_error = 0.0f;
    double total_error = 0.0;

    for (size_t t = 0; t < 14; t++) {
        for (size_t d = 0; d < 1536; d++) {

            float norm_actual =
                runtime_post_per_layer_norm[t][d];

            float norm_expected =
                DECODER_RUNTIME_POST_PER_LAYER_NORM_EXPECTED[t][d];

            if (norm_actual != norm_expected)
                norm_mismatches++;


            float pre_actual =
                runtime_pre_scalar[t][d];

            float pre_expected =
                DECODER_RUNTIME_PRE_SCALAR_EXPECTED[t][d];

            if (pre_actual != pre_expected)
                pre_scalar_mismatches++;


            float actual =
                runtime_final_output[t][d];

            float expected =
                DECODER_RUNTIME_FINAL_EXPECTED[t][d];

            final_checksum += (double)actual;
            expected_checksum += (double)expected;

            float error = fabsf(actual - expected);

            total_error += (double)error;

            if (error > max_error)
                max_error = error;

            if (actual != expected) {
                if (final_mismatches < 10) {
                    printf(
                        "Final runtime mismatch t=%lu d=%lu\n",
                        (unsigned long)t,
                        (unsigned long)d
                    );
                    print_double_fixed(
                        "  actual: ",
                        (double)actual
                    );
                    print_double_fixed(
                        "  expected: ",
                        (double)expected
                    );
                }

                final_mismatches++;
            }
        }
    }

    printf(
        "Post-per-layer RMSNorm mismatches: %lu\n",
        (unsigned long)norm_mismatches
    );

    printf(
        "Pre-scalar mismatches: %lu\n",
        (unsigned long)pre_scalar_mismatches
    );

    printf(
        "Final runtime mismatches: %lu / 21504\n",
        (unsigned long)final_mismatches
    );

    print_double_fixed(
        "Final runtime checksum: ",
        final_checksum
    );

    print_double_fixed(
        "Expected runtime checksum: ",
        expected_checksum
    );

    print_double_fixed(
        "Final runtime max abs error: ",
        (double)max_error
    );

    print_double_fixed(
        "Final runtime mean abs error: ",
        total_error / 21504.0
    );

    return
        norm_mismatches +
        pre_scalar_mismatches +
        final_mismatches;
}


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
    printf("\nPreparing runtime input RMSNorm + Q/K/V quantization\n");

    unsigned long qkv_input_cycles =
        prepare_runtime_qkv_input();

    printf(
        "Runtime QKV input preparation cycles: %lu\n",
        qkv_input_cycles
    );

    size_t runtime_qkv_input_mismatches =
        verify_runtime_qkv_input();

    printf(
        "Runtime QKV input mismatch count: %lu\n",
        (unsigned long)runtime_qkv_input_mismatches
    );

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
    // Runtime Q RMSNorm + RoPE
    // -------------------------------------------------------------------------

    printf("\nStarting runtime Q RMSNorm + RoPE\n");

    unsigned long q_norm_rope_cycles =
        run_runtime_q_norm_rope();

    printf(
        "Runtime Q RMSNorm + RoPE cycles: %lu\n",
        q_norm_rope_cycles
    );

    unsigned long runtime_q_mismatches =
        verify_runtime_q_rope();

    printf(
        "Runtime Q RoPE mismatch count: %lu\n\n",
        runtime_q_mismatches
    );



    // -------------------------------------------------------------------------
    // Runtime K RMSNorm + RoPE
    // -------------------------------------------------------------------------

    printf("\nStarting runtime K RMSNorm + RoPE\n");

    unsigned long k_norm_rope_cycles =
        run_runtime_k_norm_rope();

    printf(
        "Runtime K RMSNorm + RoPE cycles: %lu\n",
        k_norm_rope_cycles
    );

    unsigned long runtime_k_mismatches =
        verify_runtime_k_rope();

    printf(
        "Runtime K RoPE mismatch count: %lu\n\n",
        runtime_k_mismatches
    );


    // -------------------------------------------------------------------------
    // Runtime V RMSNorm
    // -------------------------------------------------------------------------

    printf("\nStarting runtime V RMSNorm\n");

    unsigned long v_norm_cycles =
        run_runtime_v_norm();

    printf(
        "Runtime V RMSNorm cycles: %lu\n",
        v_norm_cycles
    );

    unsigned long runtime_v_mismatches =
        verify_runtime_v_norm();

    printf(
        "Runtime V norm mismatch count: %lu\n\n",
        runtime_v_mismatches
    );


    // -------------------------------------------------------------------------
    // QK^T attention score matmul
    // -------------------------------------------------------------------------

    printf("\nStarting QK^T attention matmul\n");

    // Gemmini implementation
    unsigned long qk_gemmini_cycles =
        run_qk_gemmini();

    printf(
        "Gemmini QK^T cycles: %lu\n",
        qk_gemmini_cycles
    );

    // Compare raw accumulator outputs exactly
    size_t qk_mismatches =
        verify_qk();

    printf(
        "QK^T mismatch count: %lu\n",
        (unsigned long)qk_mismatches
    );

    printf(
        "QK^T sample [head 0][0][0]: "
        "Expected=%d Gemmini=%d\n\n",
        (int)DECODER_RUNTIME_QK_EXPECTED[0][0][0],
        (int)qk_gemmini[0][0][0]
    );


    // -------------------------------------------------------------------------
    // Runtime causal softmax
    // -------------------------------------------------------------------------

    printf("\nStarting CPU causal FP32 softmax\n");

    unsigned long softmax_cycles =
        run_runtime_softmax();

    printf(
        "CPU softmax cycles: %lu\n",
        softmax_cycles
    );

    size_t runtime_attn_mismatches =
        verify_runtime_attn();

    printf(
        "Runtime attention mismatch count: %lu\n",
        (unsigned long)runtime_attn_mismatches
    );

    printf(
        "Runtime attention sample [head 0][13]: "
    );

    for (size_t j = 0; j < SEQ_LEN; j++) {
        printf(
            "%d%s",
            (int)runtime_attn[0][13][j],
            j + 1 == SEQ_LEN ? "\n\n" : " "
        );
    }


    // -------------------------------------------------------------------------
    // Attention x V
    // -------------------------------------------------------------------------

    printf("\nStarting Attention x V matmul\n");

    unsigned long attn_v_gemmini_cycles =
        run_attn_v_gemmini();

    printf(
        "Gemmini Attention x V cycles: %lu\n",
        attn_v_gemmini_cycles
    );

    size_t attn_v_mismatches =
        verify_attn_v();

    printf(
        "Attention x V mismatch count: %lu\n",
        (unsigned long)attn_v_mismatches
    );

    printf(
        "Attention x V sample [head 0][0][0]: "
        "Expected=%d Gemmini=%d\n\n",
        (int)DECODER_FULL_RUNTIME_ATTN_V_EXPECTED[0][0][0],
        (int)attn_v_gemmini[0][0][0]
    );


    // -------------------------------------------------------------------------
    // Runtime Attention x V -> O projection input
    // -------------------------------------------------------------------------

    printf("\nPreparing runtime O projection input\n");

    unsigned long o_input_cycles =
        prepare_runtime_o_input();

    printf(
        "Runtime O input preparation cycles: %lu\n",
        o_input_cycles
    );

    size_t runtime_o_input_mismatches =
        verify_runtime_o_input();

    printf(
        "Runtime O input mismatch count: %lu\n",
        (unsigned long)runtime_o_input_mismatches
    );


    // -------------------------------------------------------------------------
    // Gemmini O projection
    // -------------------------------------------------------------------------

    printf("\nStarting Gemmini O projection\n");

    unsigned long o_proj_cycles =
        run_runtime_o_projection();

    printf(
        "Gemmini O projection cycles: %lu\n",
        o_proj_cycles
    );

    size_t runtime_o_mismatches =
        verify_runtime_o_projection();

    printf(
        "Runtime O projection mismatch count: %lu\n",
        (unsigned long)runtime_o_mismatches
    );

    printf(
        "Runtime O projection sample [0][0]: "
        "Expected=%d Gemmini=%d\n\n",
        (int)DECODER_RUNTIME_O_EXPECTED[0][0],
        (int)runtime_o_proj[0][0]
    );


    // -------------------------------------------------------------------------
    // O -> post-attention norm -> residual -> pre-FFN norm
    // -------------------------------------------------------------------------

    printf(
        "\nPreparing runtime post-attention/residual/pre-FFN path\n"
    );

    unsigned long ffn_input_cycles =
        prepare_runtime_ffn_input();

    printf(
        "Runtime FFN input preparation cycles: %lu\n",
        ffn_input_cycles
    );

    size_t runtime_ffn_input_mismatches =
        verify_runtime_ffn_input();

    printf(
        "Runtime FFN-path mismatch count: %lu\n",
        (unsigned long)runtime_ffn_input_mismatches
    );


    // -------------------------------------------------------------------------
    // Runtime Gemmini Gate + Up projections
    // -------------------------------------------------------------------------

    printf("\nRunning runtime Gate projection on Gemmini\n");

    unsigned long runtime_gate_cycles =
        run_runtime_gate_projection();

    printf(
        "Runtime Gate projection cycles: %lu\n",
        runtime_gate_cycles
    );


    printf("\nRunning runtime Up projection on Gemmini\n");

    unsigned long runtime_up_cycles =
        run_runtime_up_projection();

    printf(
        "Runtime Up projection cycles: %lu\n",
        runtime_up_cycles
    );


    size_t runtime_gate_up_mismatches =
        verify_runtime_gate_up();


    // -------------------------------------------------------------------------
    // Runtime GELU(gate) * up -> Down input
    // -------------------------------------------------------------------------

    printf("\nPreparing runtime Down input\n");

    unsigned long runtime_down_input_cycles =
        prepare_runtime_down_input();

    printf(
        "Runtime Down input preparation cycles: %lu\n",
        runtime_down_input_cycles
    );

    size_t runtime_down_input_mismatches =
        verify_runtime_down_input();


    // -------------------------------------------------------------------------
    // Runtime Gemmini Down projection
    // -------------------------------------------------------------------------

    printf("\nRunning runtime Down projection on Gemmini\n");

    unsigned long runtime_down_cycles =
        run_runtime_down_projection();

    printf(
        "Runtime Down projection cycles: %lu\n",
        runtime_down_cycles
    );

    size_t runtime_down_mismatches =
        verify_runtime_down_projection();


    // -------------------------------------------------------------------------
    // Down -> post-FFN norm -> second residual -> per-layer Gate input
    // -------------------------------------------------------------------------

    printf("\nPreparing runtime post-FFN path\n");

    unsigned long runtime_post_ffn_cycles =
        prepare_runtime_post_ffn();

    printf(
        "Runtime post-FFN preparation cycles: %lu\n",
        runtime_post_ffn_cycles
    );

    size_t runtime_post_ffn_mismatches =
        verify_runtime_post_ffn();

    printf(
        "Runtime post-FFN mismatch count: %lu\n",
        (unsigned long)runtime_post_ffn_mismatches
    );


    // -------------------------------------------------------------------------
    // Runtime per-layer Gate projection
    // -------------------------------------------------------------------------

    printf("\nRunning runtime per-layer Gate on Gemmini\n");

    unsigned long runtime_pl_gate_cycles =
        run_runtime_per_layer_gate();

    printf(
        "Runtime per-layer Gate cycles: %lu\n",
        runtime_pl_gate_cycles
    );

    size_t runtime_pl_gate_mismatches =
        verify_runtime_per_layer_gate();


    // -------------------------------------------------------------------------
    // Runtime per-layer GELU x side-input product
    // -------------------------------------------------------------------------

    printf("\nPreparing runtime per-layer product\n");

    unsigned long runtime_pl_product_cycles =
        prepare_runtime_per_layer_product();

    printf(
        "Runtime per-layer product preparation cycles: %lu\n",
        runtime_pl_product_cycles
    );

    size_t runtime_pl_product_mismatches =
        verify_runtime_per_layer_product();

    printf(
        "Runtime per-layer product mismatch count: %lu\n",
        (unsigned long)runtime_pl_product_mismatches
    );


    // -------------------------------------------------------------------------
    // Runtime per-layer projection
    // -------------------------------------------------------------------------

    printf("\nRunning runtime per-layer projection on Gemmini\n");

    unsigned long runtime_pl_proj_cycles =
        run_runtime_per_layer_projection();

    printf(
        "Runtime per-layer projection cycles: %lu\n",
        runtime_pl_proj_cycles
    );

    size_t runtime_pl_proj_mismatches =
        verify_runtime_per_layer_projection();


    // -------------------------------------------------------------------------
    // Final decoder tail
    // -------------------------------------------------------------------------

    printf("\nPreparing final runtime decoder output\n");

    unsigned long runtime_final_cycles =
        prepare_runtime_final_output();

    printf(
        "Final runtime decoder-tail cycles: %lu\n",
        runtime_final_cycles
    );

    size_t runtime_final_mismatches =
        verify_runtime_final_output();

    printf(
        "Final runtime decoder-tail mismatch count: %lu\n",
        (unsigned long)runtime_final_mismatches
    );


    // -------------------------------------------------------------------------
    // Final result
    // -------------------------------------------------------------------------

    if (
        runtime_qkv_input_mismatches == 0 &&
        q_proj_mismatches == 0 &&
        k_proj_mismatches == 0 &&
        v_proj_mismatches == 0 &&
        q_reshape_mismatches == 0 &&
        k_reshape_mismatches == 0 &&
        v_reshape_mismatches == 0 &&
        runtime_q_mismatches == 0 &&
        runtime_k_mismatches == 0 &&
        runtime_v_mismatches == 0 &&
        qk_mismatches == 0 &&
        runtime_attn_mismatches == 0 &&
        attn_v_mismatches == 0 &&
        runtime_o_input_mismatches == 0 &&
        runtime_o_mismatches == 0 &&
        runtime_ffn_input_mismatches == 0 &&
        runtime_gate_up_mismatches == 0 &&
        runtime_down_input_mismatches == 0 &&
        runtime_down_mismatches == 0 &&
        runtime_post_ffn_mismatches == 0 &&
        runtime_pl_gate_mismatches == 0 &&
        runtime_pl_product_mismatches == 0 &&
        runtime_pl_proj_mismatches == 0 &&
        runtime_final_mismatches == 0
    ) {
        printf(
            "\nPASS: integrated Q/K/V -> RMSNorm/RoPE -> "
            "QK^T -> causal softmax -> Attention x V -> "
            "O projection matches quantized-path references.\n"
        );

        return 0;
    }

    printf(
        "\nFAIL: integrated attention/O-projection pipeline "
        "contains mismatches.\n"
    );

    return 1;
}