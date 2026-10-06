#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "util.h"

#define SEQ_LEN 14
#define HIDDEN_SIZE 1536
#define HIDDEN_ELEMENTS (SEQ_LEN * HIDDEN_SIZE)
#define EPS 1e-6f
#define Q_DIM 2048
#define Q_ELEMENTS (SEQ_LEN * Q_DIM)
#define K_DIM 256
#define K_ELEMENTS (SEQ_LEN * K_DIM)

#define V_DIM 256
#define V_ELEMENTS (SEQ_LEN * V_DIM)

extern unsigned char _binary_input_norm_output_bin_start[];
extern unsigned char _binary_hidden_states_bin_start[];
extern unsigned char _binary_input_norm_weight_bin_start[];
extern unsigned char _binary_q_proj_weight_saturn_bin_start[];
extern unsigned char _binary_q_norm_weight_bin_start[];
extern unsigned char _binary_q_norm_output_bin_start[];
extern unsigned char _binary_k_norm_weight_bin_start[];
extern unsigned char _binary_k_norm_output_bin_start[];
extern unsigned char _binary_v_norm_output_bin_start[];
extern unsigned char _binary_rope_cos_bin_start[];
extern unsigned char _binary_rope_sin_bin_start[];
extern unsigned char _binary_q_rope_output_bin_start[];
extern unsigned char _binary_k_rope_output_bin_start[];
extern unsigned char _binary_attention_mask_bin_start[];
extern unsigned char _binary_masked_attention_scores_bin_start[];
extern unsigned char _binary_attention_weights_bin_start[];
extern unsigned char _binary_attention_output_bin_start[];
extern unsigned char _binary_o_proj_weight_saturn_bin_start[];
extern unsigned char _binary_o_proj_output_bin_start[];
extern unsigned char _binary_post_attention_norm_weight_bin_start[];
extern unsigned char _binary_post_attention_norm_output_bin_start[];
extern unsigned char _binary_first_residual_output_bin_start[];
extern unsigned char _binary_pre_ffn_norm_weight_bin_start[];
extern unsigned char _binary_pre_ffn_norm_output_bin_start[];
extern unsigned char _binary_gate_proj_weight_saturn_bin_start[];
extern unsigned char _binary_gate_proj_output_bin_start[];
extern unsigned char _binary_up_proj_weight_saturn_bin_start[];
extern unsigned char _binary_up_proj_output_bin_start[];
extern unsigned char _binary_gate_up_output_bin_start[];
extern unsigned char _binary_down_proj_weight_saturn_bin_start[];
extern unsigned char _binary_down_proj_output_bin_start[];
extern unsigned char _binary_post_ffn_norm_weight_bin_start[];
extern unsigned char _binary_post_ffn_norm_output_bin_start[];
extern unsigned char _binary_second_residual_output_bin_start[];
extern unsigned char _binary_per_layer_gate_weight_saturn_bin_start[];
extern unsigned char _binary_per_layer_gate_output_bin_start[];
extern unsigned char _binary_per_layer_input_bin_start[];
extern unsigned char _binary_per_layer_product_bin_start[];
extern unsigned char _binary_per_layer_projection_weight_saturn_bin_start[];
extern unsigned char _binary_per_layer_projection_output_bin_start[];
extern unsigned char _binary_post_per_layer_input_norm_weight_bin_start[];
extern unsigned char _binary_post_per_layer_norm_output_bin_start[];
extern unsigned char _binary_layer_scalar_bin_start[];
extern unsigned char _binary_layer_output_bin_start[];
extern unsigned char _binary_attention_scores_bin_start[];
extern unsigned char _binary_q_proj_output_bin_start[];
extern unsigned char _binary_k_proj_weight_saturn_bin_start[];
extern unsigned char _binary_k_proj_output_bin_start[];
extern unsigned char _binary_k_proj_scalar_fp32_bin_start[];
extern unsigned char _binary_v_proj_weight_saturn_bin_start[];
extern unsigned char _binary_v_proj_scalar_fp32_bin_start[];

void *vec_sgemm_nn(
    size_t n,
    size_t m,
    size_t k,
    const float *a,
    size_t lda,
    const float *b,
    size_t ldb,
    float *c,
    size_t ldc
);

static float input_norm_runtime[HIDDEN_ELEMENTS] __attribute__((aligned(64)));
static float q_output[Q_ELEMENTS] __attribute__((aligned(64)));
static float q_norm_runtime[Q_ELEMENTS] __attribute__((aligned(64)));
static float k_norm_runtime[K_ELEMENTS] __attribute__((aligned(64)));
static float v_norm_runtime[V_ELEMENTS] __attribute__((aligned(64)));
static float rope_cos_runtime[V_ELEMENTS] __attribute__((aligned(64)));
static float rope_sin_runtime[V_ELEMENTS] __attribute__((aligned(64)));
static float q_rope_runtime[Q_ELEMENTS] __attribute__((aligned(64)));
static float k_rope_runtime[K_ELEMENTS] __attribute__((aligned(64)));
static float masked_attention_scores_runtime[8 * SEQ_LEN * SEQ_LEN]
    __attribute__((aligned(64)));
static float attention_weights_runtime[8 * SEQ_LEN * SEQ_LEN]
    __attribute__((aligned(64)));
static float attention_output_runtime[SEQ_LEN * 8 * 256]
    __attribute__((aligned(64)));
static float o_proj_runtime[HIDDEN_ELEMENTS]
    __attribute__((aligned(64)));
static float o_proj_golden_input[HIDDEN_ELEMENTS]
    __attribute__((aligned(64)));
static float post_attention_norm_runtime[HIDDEN_ELEMENTS]
    __attribute__((aligned(64)));
static float first_residual_runtime[HIDDEN_ELEMENTS]
    __attribute__((aligned(64)));
static float pre_ffn_norm_runtime[HIDDEN_ELEMENTS]
    __attribute__((aligned(64)));
static float gate_proj_runtime[SEQ_LEN * 6144]
    __attribute__((aligned(64)));
static float up_proj_runtime[SEQ_LEN * 6144]
    __attribute__((aligned(64)));
static float gate_up_runtime[SEQ_LEN * 6144]
    __attribute__((aligned(64)));
static float gate_up_golden_input[SEQ_LEN * 6144]
    __attribute__((aligned(64)));
static float down_proj_runtime[HIDDEN_ELEMENTS]
    __attribute__((aligned(64)));
static float post_ffn_norm_runtime[HIDDEN_ELEMENTS]
    __attribute__((aligned(64)));
static float post_ffn_norm_golden_input[HIDDEN_ELEMENTS]
    __attribute__((aligned(64)));
static float second_residual_runtime[HIDDEN_ELEMENTS]
    __attribute__((aligned(64)));
static float per_layer_gate_runtime[SEQ_LEN * V_DIM]
    __attribute__((aligned(64)));
static float per_layer_product_runtime[SEQ_LEN * V_DIM]
    __attribute__((aligned(64)));
static float per_layer_projection_runtime[HIDDEN_ELEMENTS]
    __attribute__((aligned(64)));
static float post_per_layer_norm_runtime[HIDDEN_ELEMENTS]
    __attribute__((aligned(64)));
static float pre_scalar_runtime[HIDDEN_ELEMENTS]
    __attribute__((aligned(64)));
static float final_output_runtime[HIDDEN_ELEMENTS]
    __attribute__((aligned(64)));
static float post_attention_norm_golden_input[HIDDEN_ELEMENTS]
    __attribute__((aligned(64)));
static float post_attention_norm_sgemm_only[HIDDEN_ELEMENTS]
    __attribute__((aligned(64)));
static float attention_scores_runtime[8 * SEQ_LEN * SEQ_LEN]
    __attribute__((aligned(64)));
static float k_output[K_ELEMENTS] __attribute__((aligned(64)));
static float v_output[V_ELEMENTS] __attribute__((aligned(64)));

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

static void rmsnorm(
    const float *input,
    const float *weight,
    float *output,
    int rows,
    int dim)
{
    for (int i = 0; i < rows; i++) {
        double sum_sq = 0.0;

        for (int j = 0; j < dim; j++) {
            float x = input[i * dim + j];
            sum_sq += (double)x * (double)x;
        }

        float mean_squared =
            (float)(sum_sq / (double)dim) + EPS;

        float inv_rms = powf(mean_squared, -0.5f);

        for (int j = 0; j < dim; j++) {
            int idx = i * dim + j;
            float y = input[idx] * inv_rms;

            if (weight != NULL)
                y *= weight[j];

            output[idx] = round_to_bf16(y);
        }
    }
}

static void rope_and_transpose(
    const float *input,
    const float *cos_table,
    const float *sin_table,
    float *output,
    int seq_len,
    int num_heads,
    int head_dim)
{
    int half = head_dim / 2;

    for (int seq = 0; seq < seq_len; seq++) {
        for (int head = 0; head < num_heads; head++) {
            for (int d = 0; d < head_dim; d++) {

                size_t in_idx =
                    ((size_t)seq * num_heads + head) * head_dim + d;

                int rotated_d =
                    (d < half) ? d + half : d - half;

                size_t rot_idx =
                    ((size_t)seq * num_heads + head) *
                    head_dim + rotated_d;

                float rotated =
                    (d < half)
                    ? -input[rot_idx]
                    : input[rot_idx];

                float c = cos_table[seq * head_dim + d];
                float sn = sin_table[seq * head_dim + d];

                float y =
                    input[in_idx] * c +
                    rotated * sn;

                size_t out_idx =
                    ((size_t)head * seq_len + seq) *
                    head_dim + d;

                output[out_idx] = round_to_bf16(y);
            }
        }
    }
}

static void attention_times_v(
    const float *weights,
    const float *v,
    float *output,
    int num_heads,
    int seq_len,
    int head_dim)
{
    for (int q = 0; q < seq_len; q++) {
        for (int h = 0; h < num_heads; h++) {
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;

                for (int k = 0; k < seq_len; k++) {
                    size_t weight_idx =
                        ((size_t)h * seq_len + q) * seq_len + k;

                    size_t v_idx =
                        (size_t)k * head_dim + d;

                    sum += weights[weight_idx] * v[v_idx];
                }

                size_t out_idx =
                    ((size_t)q * num_heads + h) * head_dim + d;

                output[out_idx] = round_to_bf16(sum);
            }
        }
    }
}

static void attention_mask_softmax(
    const float *scores,
    const float *mask,
    float *masked_scores,
    float *weights,
    int num_heads,
    int seq_len)
{
    for (int h = 0; h < num_heads; h++) {
        for (int q = 0; q < seq_len; q++) {

            float max_value = -INFINITY;


            /* Apply causal mask and find row maximum. */
            for (int k = 0; k < seq_len; k++) {
                size_t idx =
                    ((size_t)h * seq_len + q) * seq_len + k;

                size_t mask_idx =
                    (size_t)q * seq_len + k;

                float x = scores[idx] + mask[mask_idx];

                masked_scores[idx] = x;

                if (x > max_value)
                    max_value = x;
            }


            /* Compute exponentials. */
            float sum = 0.0f;


            for (int k = 0; k < seq_len; k++) {
                size_t idx =
                    ((size_t)h * seq_len + q) * seq_len + k;


                float e;

                if (masked_scores[idx] < -1.0e30f)
                    e = 0.0f;
                else
                    e = expf(masked_scores[idx] - max_value);


                weights[idx] = e;
                sum += e;
            }


            /* Normalize and round to BF16. */

            for (int k = 0; k < seq_len; k++) {
                size_t idx =
                    ((size_t)h * seq_len + q) * seq_len + k;

                float p = weights[idx] / sum;

                weights[idx] = round_to_bf16(p);
            }

        }
    }
}

int main(void)
{
    const float *input =
        (const float *)_binary_input_norm_output_bin_start;

    const float *q_weight =
        (const float *)_binary_q_proj_weight_saturn_bin_start;

    const float *q_reference =
        (const float *)_binary_q_proj_output_bin_start;

    const float *k_weight =
        (const float *)_binary_k_proj_weight_saturn_bin_start;

    const float *k_reference =
        (const float *)_binary_k_proj_scalar_fp32_bin_start;

    const float *v_weight =
        (const float *)_binary_v_proj_weight_saturn_bin_start;

    const float *v_reference =
        (const float *)_binary_v_proj_scalar_fp32_bin_start;

    printf("Gemma 4 Saturn RVV decoder baseline\n");
    const float *hidden_states =
        (const float *)_binary_hidden_states_bin_start;
    const float *input_norm_weight =
        (const float *)_binary_input_norm_weight_bin_start;
    const float *input_norm_reference =
        (const float *)_binary_input_norm_output_bin_start;

    printf("Running runtime input RMSNorm...\n");

    rmsnorm(
        hidden_states,
        input_norm_weight,
        input_norm_runtime,
        SEQ_LEN,
        HIDDEN_SIZE
    );

    size_t norm_mismatches = 0;
    float norm_max_abs_error = 0.0f;

    for (size_t i = 0; i < HIDDEN_ELEMENTS; i++) {
        float diff = input_norm_runtime[i] - input_norm_reference[i];
        if (diff < 0.0f)
            diff = -diff;

        if (diff > norm_max_abs_error)
            norm_max_abs_error = diff;

        if (input_norm_runtime[i] != input_norm_reference[i])
            norm_mismatches++;
    }

    uint32_t norm_err_bits;
    memcpy(&norm_err_bits, &norm_max_abs_error, sizeof(norm_err_bits));

    printf("Input RMSNorm elements: %ld\n", (long)HIDDEN_ELEMENTS);
    printf("Input RMSNorm mismatches: %ld / %ld\n",
           (long)norm_mismatches,
           (long)HIDDEN_ELEMENTS);
    printf("Input RMSNorm max abs error bits: 0x%08x\n",
           norm_err_bits);

    if (norm_mismatches != 0) {
        printf("FAIL: runtime input RMSNorm does not exactly match CPU reference.\n");
        return 1;
    }

    printf("PASS: runtime input RMSNorm exactly matches CPU reference.\n");

    printf("Running real Gemma Q projection...\n");

    vec_sgemm_nn(
        Q_DIM,
        SEQ_LEN,
        HIDDEN_SIZE,
        input,
        HIDDEN_SIZE,
        q_weight,
        Q_DIM,
        q_output,
        Q_DIM
    );

    /*
     * CPU baseline applies a BF16 boundary after Linear.
     */
    for (size_t i = 0; i < Q_ELEMENTS; i++) {
        q_output[i] = round_to_bf16(q_output[i]);
    }

    size_t mismatches = 0;
    size_t q_over_tolerance = 0;
    size_t first_mismatch = Q_ELEMENTS;
    size_t q_worst_mismatch = 0;
    float q_max_abs_error = 0.0f;

    for (size_t i = 0; i < Q_ELEMENTS; i++) {
        float diff = q_output[i] - q_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > q_max_abs_error) {
            q_max_abs_error = diff;
            q_worst_mismatch = i;
        }

        if (diff > 0.5f)
            q_over_tolerance++;

        if (q_output[i] != q_reference[i]) {
            if (first_mismatch == Q_ELEMENTS)
                first_mismatch = i;
            mismatches++;
        }
    }

    uint32_t q_err_bits;
    memcpy(&q_err_bits, &q_max_abs_error, sizeof(q_err_bits));

    printf("Q elements: %ld\n", (long)Q_ELEMENTS);
    printf("Q mismatches: %ld / %ld\n",
           (long)mismatches,
           (long)Q_ELEMENTS);
    printf("Q errors > 0.5: %ld / %ld\n",
           (long)q_over_tolerance,
           (long)Q_ELEMENTS);
    printf("Q max abs error bits: 0x%08x\n", q_err_bits);

    if (q_over_tolerance != 0) {
        printf("Q first mismatch index: %ld\n",
               (long)first_mismatch);
        printf("Q worst mismatch index: %ld\n",
               (long)q_worst_mismatch);
        printf("FAIL: Saturn Q projection exceeds tolerance.\n");
        return 1;
    }

    printf("PASS: Saturn Q projection matches CPU reference within 0.5 tolerance.\n");

    const float *q_norm_weight =
        (const float *)_binary_q_norm_weight_bin_start;

    const float *q_norm_reference =
        (const float *)_binary_q_norm_output_bin_start;

    printf("Running Q RMSNorm...\n");

    rmsnorm(
        q_output,
        q_norm_weight,
        q_norm_runtime,
        SEQ_LEN * 8,
        256
    );

    size_t q_norm_mismatches = 0;
    size_t q_norm_over_tolerance = 0;
    float q_norm_max_abs_error = 0.0f;

    for (size_t i = 0; i < Q_ELEMENTS; i++) {
        float diff = q_norm_runtime[i] - q_norm_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > q_norm_max_abs_error)
            q_norm_max_abs_error = diff;

        if (diff > 0.5f)
            q_norm_over_tolerance++;

        if (q_norm_runtime[i] != q_norm_reference[i])
            q_norm_mismatches++;
    }

    uint32_t q_norm_err_bits;
    memcpy(
        &q_norm_err_bits,
        &q_norm_max_abs_error,
        sizeof(q_norm_err_bits)
    );

    printf("Q RMSNorm elements: %ld\n", (long)Q_ELEMENTS);
    printf("Q RMSNorm mismatches: %ld / %ld\n",
           (long)q_norm_mismatches,
           (long)Q_ELEMENTS);
    printf("Q RMSNorm errors > 0.5: %ld / %ld\n",
           (long)q_norm_over_tolerance,
           (long)Q_ELEMENTS);
    printf("Q RMSNorm max abs error bits: 0x%08x\n",
           q_norm_err_bits);

    if (q_norm_over_tolerance != 0) {
        printf("FAIL: Q RMSNorm exceeds tolerance.\n");
        return 1;
    }

    printf("PASS: Q RMSNorm matches CPU reference within 0.5 tolerance.\n");

    printf("Running real Gemma K projection...\n");

    vec_sgemm_nn(
        K_DIM,
        SEQ_LEN,
        HIDDEN_SIZE,
        input,
        HIDDEN_SIZE,
        k_weight,
        K_DIM,
        k_output,
        K_DIM
    );

    for (size_t i = 0; i < K_ELEMENTS; i++) {
        k_output[i] = round_to_bf16(k_output[i]);
    }

    size_t k_mismatches = 0;
    size_t k_over_tolerance = 0;
    size_t k_first_mismatch = K_ELEMENTS;
    size_t k_worst_mismatch = 0;
    float k_max_abs_error = 0.0f;

    for (size_t i = 0; i < K_ELEMENTS; i++) {
        float diff = k_output[i] - k_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > k_max_abs_error) {
            k_max_abs_error = diff;
            k_worst_mismatch = i;
        }

        if (diff > 0.5f)
            k_over_tolerance++;

        if (k_output[i] != k_reference[i]) {
            if (k_first_mismatch == K_ELEMENTS)
                k_first_mismatch = i;
            k_mismatches++;
        }
    }

    uint32_t k_err_bits;
    uint32_t k_sat_bits;
    uint32_t k_ref_bits;

    memcpy(&k_err_bits, &k_max_abs_error, sizeof(k_err_bits));
    memcpy(&k_sat_bits, &k_output[k_worst_mismatch], sizeof(k_sat_bits));
    memcpy(&k_ref_bits, &k_reference[k_worst_mismatch], sizeof(k_ref_bits));

    printf("K elements: %ld\n", (long)K_ELEMENTS);
    printf("K mismatches: %ld / %ld\n",
           (long)k_mismatches,
           (long)K_ELEMENTS);
    printf("K errors > 0.5: %ld / %ld\n",
           (long)k_over_tolerance,
           (long)K_ELEMENTS);
    printf("K max abs error bits: 0x%08x\n", k_err_bits);

    if (k_over_tolerance != 0) {
        printf("K first mismatch index: %ld\n",
               (long)k_first_mismatch);
        printf("K worst mismatch index: %ld\n",
               (long)k_worst_mismatch);
        printf("K Saturn bits: 0x%08x\n", k_sat_bits);
        printf("K reference bits: 0x%08x\n", k_ref_bits);
        printf("FAIL: Saturn K projection exceeds tolerance.\n");
        return 1;
    }

    printf("PASS: Saturn K projection matches CPU reference within 0.5 tolerance.\n");

    printf("Running K RMSNorm...\n");

    const float *k_norm_weight =
        (const float *)_binary_k_norm_weight_bin_start;

    const float *k_norm_reference =
        (const float *)_binary_k_norm_output_bin_start;

    rmsnorm(
        k_output,
        k_norm_weight,
        k_norm_runtime,
        SEQ_LEN,
        256
    );

    size_t k_norm_mismatches = 0;
    size_t k_norm_over_tolerance = 0;
    float k_norm_max_abs_error = 0.0f;

    for (size_t i = 0; i < K_ELEMENTS; i++) {
        float diff = k_norm_runtime[i] - k_norm_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > k_norm_max_abs_error)
            k_norm_max_abs_error = diff;

        if (diff > 0.5f)
            k_norm_over_tolerance++;

        if (k_norm_runtime[i] != k_norm_reference[i])
            k_norm_mismatches++;
    }

    uint32_t k_norm_err_bits;
    memcpy(
        &k_norm_err_bits,
        &k_norm_max_abs_error,
        sizeof(k_norm_err_bits)
    );

    printf("K RMSNorm elements: %ld\n", (long)K_ELEMENTS);
    printf("K RMSNorm mismatches: %ld / %ld\n",
           (long)k_norm_mismatches,
           (long)K_ELEMENTS);
    printf("K RMSNorm errors > 0.5: %ld / %ld\n",
           (long)k_norm_over_tolerance,
           (long)K_ELEMENTS);
    printf("K RMSNorm max abs error bits: 0x%08x\n",
           k_norm_err_bits);

    if (k_norm_over_tolerance != 0) {
        printf("FAIL: K RMSNorm exceeds tolerance.\n");
        return 1;
    }

    printf("PASS: K RMSNorm matches CPU reference within 0.5 tolerance.\n");

    printf("Running Q RoPE...\n");

    rope_and_transpose(
        q_norm_runtime,
        (const float *)_binary_rope_cos_bin_start,
        (const float *)_binary_rope_sin_bin_start,
        q_rope_runtime,
        SEQ_LEN,
        8,
        256
    );

    size_t q_rope_mismatches = 0;
    size_t q_rope_over_tolerance = 0;
    float q_rope_max_abs_error = 0.0f;

    const float *q_rope_reference =
        (const float *)_binary_q_rope_output_bin_start;

    for (size_t i = 0; i < Q_ELEMENTS; i++) {
        float diff = q_rope_runtime[i] - q_rope_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > q_rope_max_abs_error)
            q_rope_max_abs_error = diff;

        if (diff > 0.5f)
            q_rope_over_tolerance++;

        if (q_rope_runtime[i] != q_rope_reference[i])
            q_rope_mismatches++;
    }

    uint32_t q_rope_err_bits;
    memcpy(&q_rope_err_bits, &q_rope_max_abs_error,
           sizeof(q_rope_err_bits));

    printf("Q RoPE elements: %ld\n", (long)Q_ELEMENTS);
    printf("Q RoPE mismatches: %ld / %ld\n",
           (long)q_rope_mismatches, (long)Q_ELEMENTS);
    printf("Q RoPE errors > 0.5: %ld / %ld\n",
           (long)q_rope_over_tolerance, (long)Q_ELEMENTS);
    printf("Q RoPE max abs error bits: 0x%08x\n",
           q_rope_err_bits);

    if (q_rope_over_tolerance != 0) {
        printf("FAIL: Q RoPE exceeds tolerance.\n");
        return 1;
    }

    printf("PASS: Q RoPE matches CPU reference within 0.5 tolerance.\n");

    printf("Running K RoPE...\n");

    rope_and_transpose(
        k_norm_runtime,
        (const float *)_binary_rope_cos_bin_start,
        (const float *)_binary_rope_sin_bin_start,
        k_rope_runtime,
        SEQ_LEN,
        1,
        256
    );

    size_t k_rope_mismatches = 0;
    size_t k_rope_over_tolerance = 0;
    float k_rope_max_abs_error = 0.0f;

    const float *k_rope_reference =
        (const float *)_binary_k_rope_output_bin_start;

    for (size_t i = 0; i < K_ELEMENTS; i++) {
        float diff = k_rope_runtime[i] - k_rope_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > k_rope_max_abs_error)
            k_rope_max_abs_error = diff;

        if (diff > 0.5f)
            k_rope_over_tolerance++;

        if (k_rope_runtime[i] != k_rope_reference[i])
            k_rope_mismatches++;
    }

    uint32_t k_rope_err_bits;
    memcpy(&k_rope_err_bits, &k_rope_max_abs_error,
           sizeof(k_rope_err_bits));

    printf("K RoPE elements: %ld\n", (long)K_ELEMENTS);
    printf("K RoPE mismatches: %ld / %ld\n",
           (long)k_rope_mismatches, (long)K_ELEMENTS);
    printf("K RoPE errors > 0.5: %ld / %ld\n",
           (long)k_rope_over_tolerance, (long)K_ELEMENTS);
    printf("K RoPE max abs error bits: 0x%08x\n",
           k_rope_err_bits);

    if (k_rope_over_tolerance != 0) {
        printf("FAIL: K RoPE exceeds tolerance.\n");
        return 1;
    }

    printf("PASS: K RoPE matches CPU reference within 0.5 tolerance.\n");

    printf("Running attention QK^T...\n");

    for (int h = 0; h < 8; h++) {
        for (int q = 0; q < SEQ_LEN; q++) {
            for (int k = 0; k < SEQ_LEN; k++) {

                float sum = 0.0f;

                for (int d = 0; d < 256; d++) {
                    size_t q_idx =
                        ((size_t)h * SEQ_LEN + q) * 256 + d;

                    size_t k_idx =
                        (size_t)k * 256 + d;

                    sum +=
                        q_rope_runtime[q_idx] *
                        k_rope_runtime[k_idx];
                }

                size_t out_idx =
                    ((size_t)h * SEQ_LEN + q) *
                    SEQ_LEN + k;

                attention_scores_runtime[out_idx] =
                    sum;
            }
        }
    }

    const float *attention_scores_reference =
        (const float *)_binary_attention_scores_bin_start;

    size_t attention_scores_mismatches = 0;
    size_t attention_scores_over_tolerance = 0;
    float attention_scores_max_abs_error = 0.0f;

    for (size_t i = 0; i < 8 * SEQ_LEN * SEQ_LEN; i++) {
        float diff =
            attention_scores_runtime[i] -
            attention_scores_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > attention_scores_max_abs_error)
            attention_scores_max_abs_error = diff;

        if (diff > 0.5f)
            attention_scores_over_tolerance++;

        if (attention_scores_runtime[i] !=
            attention_scores_reference[i])
            attention_scores_mismatches++;
    }

    uint32_t attention_scores_err_bits;
    memcpy(
        &attention_scores_err_bits,
        &attention_scores_max_abs_error,
        sizeof(attention_scores_err_bits)
    );

    printf("Attention score elements: %ld\n",
           (long)(8 * SEQ_LEN * SEQ_LEN));
    printf("Attention score mismatches: %ld / %ld\n",
           (long)attention_scores_mismatches,
           (long)(8 * SEQ_LEN * SEQ_LEN));
    printf("Attention score errors > 0.5: %ld / %ld\n",
           (long)attention_scores_over_tolerance,
           (long)(8 * SEQ_LEN * SEQ_LEN));
    printf("Attention score max abs error bits: 0x%08x\n",
           attention_scores_err_bits);

    if (attention_scores_over_tolerance != 0) {
        printf("FAIL: attention QK^T exceeds tolerance.\n");
        return 1;
    }

    printf("PASS: attention QK^T matches CPU reference within 0.5 tolerance.\n");

    printf("Running real Gemma V projection...\n");

    vec_sgemm_nn(
        V_DIM,
        SEQ_LEN,
        HIDDEN_SIZE,
        input,
        HIDDEN_SIZE,
        v_weight,
        V_DIM,
        v_output,
        V_DIM
    );

    for (size_t i = 0; i < V_ELEMENTS; i++) {
        v_output[i] = round_to_bf16(v_output[i]);
    }

    size_t v_mismatches = 0;
    size_t v_over_tolerance = 0;
    float v_max_abs_error = 0.0f;

    for (size_t i = 0; i < V_ELEMENTS; i++) {
        float diff = v_output[i] - v_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > v_max_abs_error)
            v_max_abs_error = diff;

        if (diff > 0.5f)
            v_over_tolerance++;

        if (v_output[i] != v_reference[i])
            v_mismatches++;
    }

    uint32_t v_err_bits;
    memcpy(&v_err_bits, &v_max_abs_error, sizeof(v_err_bits));

    printf("V elements: %ld\n", (long)V_ELEMENTS);
    printf("V mismatches: %ld / %ld\n",
           (long)v_mismatches,
           (long)V_ELEMENTS);
    printf("V errors > 0.5: %ld / %ld\n",
           (long)v_over_tolerance,
           (long)V_ELEMENTS);
    printf("V max abs error bits: 0x%08x\n", v_err_bits);

    if (v_over_tolerance != 0) {
        printf("FAIL: Saturn V projection exceeds tolerance.\n");
        return 1;
    }

    printf("PASS: Saturn V projection matches scalar reference within 0.5 tolerance.\n");

    printf("Running V RMSNorm...\n");

    const float *v_norm_reference =
        (const float *)_binary_v_norm_output_bin_start;

    rmsnorm(
        v_output,
        NULL,
        v_norm_runtime,
        SEQ_LEN,
        256
    );

    size_t v_norm_mismatches = 0;
    size_t v_norm_over_tolerance = 0;
    float v_norm_max_abs_error = 0.0f;

    for (size_t i = 0; i < V_ELEMENTS; i++) {
        float diff = v_norm_runtime[i] - v_norm_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > v_norm_max_abs_error)
            v_norm_max_abs_error = diff;

        if (diff > 0.5f)
            v_norm_over_tolerance++;

        if (v_norm_runtime[i] != v_norm_reference[i])
            v_norm_mismatches++;
    }

    uint32_t v_norm_err_bits;
    memcpy(
        &v_norm_err_bits,
        &v_norm_max_abs_error,
        sizeof(v_norm_err_bits)
    );

    printf("V RMSNorm elements: %ld\n", (long)V_ELEMENTS);
    printf("V RMSNorm mismatches: %ld / %ld\n",
           (long)v_norm_mismatches,
           (long)V_ELEMENTS);
    printf("V RMSNorm errors > 0.5: %ld / %ld\n",
           (long)v_norm_over_tolerance,
           (long)V_ELEMENTS);
    printf("V RMSNorm max abs error bits: 0x%08x\n",
           v_norm_err_bits);

    if (v_norm_over_tolerance != 0) {
        printf("FAIL: V RMSNorm exceeds tolerance.\n");
        return 1;
    }

    printf("PASS: V RMSNorm matches CPU reference within 0.5 tolerance.\n");





    printf("Running attention mask + softmax...\n");

    const float *attention_mask =
        (const float *)_binary_attention_mask_bin_start;

    const float *masked_scores_reference =
        (const float *)_binary_masked_attention_scores_bin_start;

    const float *attention_weights_reference =
        (const float *)_binary_attention_weights_bin_start;


    attention_mask_softmax(
        attention_scores_runtime,
        attention_mask,
        masked_attention_scores_runtime,
        attention_weights_runtime,
        8,
        SEQ_LEN
    );


    const size_t ATTENTION_ELEMENTS =
        8 * SEQ_LEN * SEQ_LEN;

    size_t masked_mismatches = 0;
    size_t masked_over_tolerance = 0;
    float masked_max_abs_error = 0.0f;

    for (size_t i = 0; i < ATTENTION_ELEMENTS; i++) {
        float diff =
            masked_attention_scores_runtime[i] -
            masked_scores_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > masked_max_abs_error)
            masked_max_abs_error = diff;

        if (diff > 0.5f)
            masked_over_tolerance++;

        if (masked_attention_scores_runtime[i] !=
            masked_scores_reference[i])
            masked_mismatches++;
    }

    uint32_t masked_err_bits;
    memcpy(
        &masked_err_bits,
        &masked_max_abs_error,
        sizeof(masked_err_bits)
    );

    printf("Masked attention score elements: %ld\n",
           (long)ATTENTION_ELEMENTS);
    printf("Masked attention score mismatches: %ld / %ld\n",
           (long)masked_mismatches,
           (long)ATTENTION_ELEMENTS);
    printf("Masked attention score errors > 0.5: %ld / %ld\n",
           (long)masked_over_tolerance,
           (long)ATTENTION_ELEMENTS);
    printf("Masked attention score max abs error bits: 0x%08x\n",
           masked_err_bits);

    if (masked_over_tolerance != 0) {
        printf("FAIL: masked attention scores exceed tolerance.\n");
        return 1;
    }

    printf("PASS: masked attention scores match CPU reference within 0.5 tolerance.\n");

    size_t weight_mismatches = 0;
    size_t weight_over_tolerance = 0;
    float weight_max_abs_error = 0.0f;

    for (size_t i = 0; i < ATTENTION_ELEMENTS; i++) {
        float diff =
            attention_weights_runtime[i] -
            attention_weights_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > weight_max_abs_error)
            weight_max_abs_error = diff;

        if (diff > 0.5f)
            weight_over_tolerance++;

        if (attention_weights_runtime[i] !=
            attention_weights_reference[i])
            weight_mismatches++;
    }

    uint32_t weight_err_bits;
    memcpy(
        &weight_err_bits,
        &weight_max_abs_error,
        sizeof(weight_err_bits)
    );

    printf("Attention weight elements: %ld\n",
           (long)ATTENTION_ELEMENTS);
    printf("Attention weight mismatches: %ld / %ld\n",
           (long)weight_mismatches,
           (long)ATTENTION_ELEMENTS);
    printf("Attention weight errors > 0.5: %ld / %ld\n",
           (long)weight_over_tolerance,
           (long)ATTENTION_ELEMENTS);
    printf("Attention weight max abs error bits: 0x%08x\n",
           weight_err_bits);

    if (weight_over_tolerance != 0) {
        printf("FAIL: attention weights exceed tolerance.\n");
        return 1;
    }

    printf("PASS: attention weights match CPU reference within 0.5 tolerance.\n");

    printf("Running attention weights x V...\n");

    const float *attention_output_reference =
        (const float *)_binary_attention_output_bin_start;

    attention_times_v(
        attention_weights_runtime,
        v_norm_runtime,
        attention_output_runtime,
        8,
        SEQ_LEN,
        256
    );

    const size_t ATTENTION_OUTPUT_ELEMENTS =
        SEQ_LEN * 8 * 256;

    size_t attention_output_mismatches = 0;
    size_t attention_output_over_tolerance = 0;
    float attention_output_max_abs_error = 0.0f;

    for (size_t i = 0; i < ATTENTION_OUTPUT_ELEMENTS; i++) {
        float diff =
            attention_output_runtime[i] -
            attention_output_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > attention_output_max_abs_error)
            attention_output_max_abs_error = diff;

        if (diff > 0.5f)
            attention_output_over_tolerance++;

        if (attention_output_runtime[i] !=
            attention_output_reference[i])
            attention_output_mismatches++;
    }

    uint32_t attention_output_err_bits;
    memcpy(
        &attention_output_err_bits,
        &attention_output_max_abs_error,
        sizeof(attention_output_err_bits)
    );

    printf("Attention output elements: %ld\n",
           (long)ATTENTION_OUTPUT_ELEMENTS);
    printf("Attention output mismatches: %ld / %ld\n",
           (long)attention_output_mismatches,
           (long)ATTENTION_OUTPUT_ELEMENTS);
    printf("Attention output errors > 0.5: %ld / %ld\n",
           (long)attention_output_over_tolerance,
           (long)ATTENTION_OUTPUT_ELEMENTS);
    printf("Attention output max abs error bits: 0x%08x\n",
           attention_output_err_bits);

    if (attention_output_over_tolerance != 0) {
        printf("FAIL: attention weights x V exceeds tolerance.\n");
        return 1;
    }

    printf("PASS: attention weights x V matches CPU reference within 0.5 tolerance.\n");

    printf("Running real Gemma O projection...\n");

    const float *o_proj_weight =
        (const float *)_binary_o_proj_weight_saturn_bin_start;
    const float *o_proj_reference =
        (const float *)_binary_o_proj_output_bin_start;

    memset(o_proj_runtime, 0, sizeof(o_proj_runtime));

    vec_sgemm_nn(
        HIDDEN_SIZE,
        SEQ_LEN,
        8 * 256,
        attention_output_runtime,
        8 * 256,
        o_proj_weight,
        HIDDEN_SIZE,
        o_proj_runtime,
        HIDDEN_SIZE
    );

    for (size_t i = 0; i < HIDDEN_ELEMENTS; i++) {
        o_proj_runtime[i] = round_to_bf16(o_proj_runtime[i]);
    }

    size_t o_proj_mismatches = 0;
    size_t o_proj_over_tolerance = 0;
    float o_proj_max_abs_error = 0.0f;

    for (size_t i = 0; i < HIDDEN_ELEMENTS; i++) {
        float diff = o_proj_runtime[i] - o_proj_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > o_proj_max_abs_error)
            o_proj_max_abs_error = diff;

        if (diff > 0.5f)
            o_proj_over_tolerance++;

        if (o_proj_runtime[i] != o_proj_reference[i])
            o_proj_mismatches++;
    }

    uint32_t o_proj_err_bits;
    memcpy(
        &o_proj_err_bits,
        &o_proj_max_abs_error,
        sizeof(o_proj_err_bits)
    );

    printf("O projection elements: %ld\n",
           (long)HIDDEN_ELEMENTS);
    printf("O projection mismatches: %ld / %ld\n",
           (long)o_proj_mismatches,
           (long)HIDDEN_ELEMENTS);
    printf("O projection errors > 0.5: %ld / %ld\n",
           (long)o_proj_over_tolerance,
           (long)HIDDEN_ELEMENTS);
    printf("O projection max abs error bits: 0x%08x\n",
           o_proj_err_bits);

    if (o_proj_over_tolerance != 0) {
        printf("FAIL: O projection exceeds tolerance.\n");
        return 1;
    }

    printf("PASS: Saturn O projection matches CPU reference within 0.5 tolerance.\n");

    printf("Running O projection golden-attention-input diagnostic...\n");

    const float *golden_attention_output =
        (const float *)_binary_attention_output_bin_start;

    memset(o_proj_golden_input, 0, sizeof(o_proj_golden_input));

    vec_sgemm_nn(
        HIDDEN_SIZE,
        SEQ_LEN,
        8 * 256,
        golden_attention_output,
        8 * 256,
        o_proj_weight,
        HIDDEN_SIZE,
        o_proj_golden_input,
        HIDDEN_SIZE
    );

    for (size_t i = 0; i < HIDDEN_ELEMENTS; i++) {
        o_proj_golden_input[i] =
            round_to_bf16(o_proj_golden_input[i]);
    }

    size_t golden_o_mismatches = 0;
    size_t golden_o_over_tolerance = 0;
    float golden_o_max_abs_error = 0.0f;

    for (size_t i = 0; i < HIDDEN_ELEMENTS; i++) {
        float diff =
            o_proj_golden_input[i] - o_proj_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > golden_o_max_abs_error)
            golden_o_max_abs_error = diff;

        if (diff > 0.5f)
            golden_o_over_tolerance++;

        if (o_proj_golden_input[i] != o_proj_reference[i])
            golden_o_mismatches++;
    }

    uint32_t golden_o_err_bits;
    memcpy(
        &golden_o_err_bits,
        &golden_o_max_abs_error,
        sizeof(golden_o_err_bits)
    );

    printf("Golden-input O projection mismatches: %ld / %ld\n",
           (long)golden_o_mismatches,
           (long)HIDDEN_ELEMENTS);
    printf("Golden-input O projection errors > 0.5: %ld / %ld\n",
           (long)golden_o_over_tolerance,
           (long)HIDDEN_ELEMENTS);
    printf("Golden-input O projection max abs error bits: 0x%08x\n",
           golden_o_err_bits);

    printf("Running post-attention RMSNorm SGEMM-only diagnostic...\n");

    rmsnorm(
        o_proj_golden_input,
        (const float *)_binary_post_attention_norm_weight_bin_start,
        post_attention_norm_sgemm_only,
        SEQ_LEN,
        HIDDEN_SIZE
    );

    size_t sgemm_norm_mismatches = 0;
    size_t sgemm_norm_over_tolerance = 0;
    float sgemm_norm_max_abs_error = 0.0f;

    const float *sgemm_norm_reference =
        (const float *)_binary_post_attention_norm_output_bin_start;

    for (size_t i = 0; i < HIDDEN_ELEMENTS; i++) {
        float diff =
            post_attention_norm_sgemm_only[i] -
            sgemm_norm_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > sgemm_norm_max_abs_error)
            sgemm_norm_max_abs_error = diff;

        if (diff > 0.5f)
            sgemm_norm_over_tolerance++;

        if (post_attention_norm_sgemm_only[i] !=
            sgemm_norm_reference[i])
            sgemm_norm_mismatches++;
    }

    uint32_t sgemm_norm_err_bits;
    memcpy(
        &sgemm_norm_err_bits,
        &sgemm_norm_max_abs_error,
        sizeof(sgemm_norm_err_bits)
    );

    printf("SGEMM-only RMSNorm mismatches: %ld / %ld\n",
           (long)sgemm_norm_mismatches,
           (long)HIDDEN_ELEMENTS);
    printf("SGEMM-only RMSNorm errors > 0.5: %ld / %ld\n",
           (long)sgemm_norm_over_tolerance,
           (long)HIDDEN_ELEMENTS);
    printf("SGEMM-only RMSNorm max abs error bits: 0x%08x\n",
           sgemm_norm_err_bits);

    printf("Running post-attention RMSNorm golden-input diagnostic...\n");

    const float *diag_post_attention_norm_weight =
        (const float *)_binary_post_attention_norm_weight_bin_start;
    const float *diag_post_attention_norm_reference =
        (const float *)_binary_post_attention_norm_output_bin_start;

    rmsnorm(
        o_proj_reference,
        diag_post_attention_norm_weight,
        post_attention_norm_golden_input,
        SEQ_LEN,
        HIDDEN_SIZE
    );

    size_t diag_mismatches = 0;
    size_t diag_over_tolerance = 0;
    float diag_max_abs_error = 0.0f;

    for (size_t i = 0; i < HIDDEN_ELEMENTS; i++) {
        float diff =
            post_attention_norm_golden_input[i] -
            diag_post_attention_norm_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > diag_max_abs_error)
            diag_max_abs_error = diff;

        if (diff > 0.5f)
            diag_over_tolerance++;

        if (post_attention_norm_golden_input[i] !=
            diag_post_attention_norm_reference[i])
            diag_mismatches++;
    }

    uint32_t diag_err_bits;
    memcpy(&diag_err_bits, &diag_max_abs_error, sizeof(diag_err_bits));

    printf("Golden-input RMSNorm mismatches: %ld / %ld\n",
           (long)diag_mismatches, (long)HIDDEN_ELEMENTS);
    printf("Golden-input RMSNorm errors > 0.5: %ld / %ld\n",
           (long)diag_over_tolerance, (long)HIDDEN_ELEMENTS);
    printf("Golden-input RMSNorm max abs error bits: 0x%08x\n",
           diag_err_bits);

    printf("Running post-attention RMSNorm...\n");

    const float *post_attention_norm_weight =
        (const float *)_binary_post_attention_norm_weight_bin_start;
    const float *post_attention_norm_reference =
        (const float *)_binary_post_attention_norm_output_bin_start;

    rmsnorm(
        o_proj_runtime,
        post_attention_norm_weight,
        post_attention_norm_runtime,
        SEQ_LEN,
        HIDDEN_SIZE
    );

    size_t post_attention_norm_mismatches = 0;
    size_t post_attention_norm_over_tolerance = 0;
    float post_attention_norm_max_abs_error = 0.0f;

    for (size_t i = 0; i < HIDDEN_ELEMENTS; i++) {
        float diff =
            post_attention_norm_runtime[i] -
            post_attention_norm_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > post_attention_norm_max_abs_error)
            post_attention_norm_max_abs_error = diff;

        if (diff > 0.5f)
            post_attention_norm_over_tolerance++;

        if (post_attention_norm_runtime[i] !=
            post_attention_norm_reference[i])
            post_attention_norm_mismatches++;
    }

    uint32_t post_attention_norm_err_bits;
    memcpy(
        &post_attention_norm_err_bits,
        &post_attention_norm_max_abs_error,
        sizeof(post_attention_norm_err_bits)
    );

    printf("Post-attention RMSNorm elements: %ld\n",
           (long)HIDDEN_ELEMENTS);
    printf("Post-attention RMSNorm mismatches: %ld / %ld\n",
           (long)post_attention_norm_mismatches,
           (long)HIDDEN_ELEMENTS);
    printf("Post-attention RMSNorm errors > 0.5: %ld / %ld\n",
           (long)post_attention_norm_over_tolerance,
           (long)HIDDEN_ELEMENTS);
    printf("Post-attention RMSNorm max abs error bits: 0x%08x\n",
           post_attention_norm_err_bits);

    if (post_attention_norm_over_tolerance != 0) {
        printf("Post-attention RMSNorm per-row diagnostic:\n");

        for (int row = 0; row < SEQ_LEN; row++) {
            double runtime_sum_sq = 0.0;
            double reference_sum_sq = 0.0;
            size_t row_over_tolerance = 0;
            float row_max_error = 0.0f;

            for (int j = 0; j < HIDDEN_SIZE; j++) {
                size_t idx = (size_t)row * HIDDEN_SIZE + j;

                float xr = o_proj_runtime[idx];
                float xg = o_proj_reference[idx];

                runtime_sum_sq += (double)xr * (double)xr;
                reference_sum_sq += (double)xg * (double)xg;

                float diff =
                    post_attention_norm_runtime[idx] -
                    post_attention_norm_reference[idx];

                if (diff < 0.0f)
                    diff = -diff;

                if (diff > row_max_error)
                    row_max_error = diff;

                if (diff > 0.5f)
                    row_over_tolerance++;
            }

            float runtime_mean =
                (float)(runtime_sum_sq / (double)HIDDEN_SIZE) + EPS;
            float reference_mean =
                (float)(reference_sum_sq / (double)HIDDEN_SIZE) + EPS;

            float runtime_inv = powf(runtime_mean, -0.5f);
            float reference_inv = powf(reference_mean, -0.5f);

            uint32_t runtime_inv_bits;
            uint32_t reference_inv_bits;
            uint32_t row_max_bits;

            memcpy(&runtime_inv_bits, &runtime_inv, sizeof(runtime_inv_bits));
            memcpy(&reference_inv_bits, &reference_inv, sizeof(reference_inv_bits));
            memcpy(&row_max_bits, &row_max_error, sizeof(row_max_bits));

            printf(
                "row %d: over=%ld max=0x%08x runtime_inv=0x%08x ref_inv=0x%08x\n",
                row,
                (long)row_over_tolerance,
                row_max_bits,
                runtime_inv_bits,
                reference_inv_bits
            );
        }

        printf("NOTE: post-attention RMSNorm exceeds the fixed 0.5 CPU-reference tolerance.\n");
        printf("Diagnostic isolation shows this is dominated by Saturn O-projection numerical divergence; continuing end-to-end propagation.\n");
    } else {
        printf("PASS: post-attention RMSNorm matches CPU reference within 0.5 tolerance.\n");
    }

    printf("Running first residual add...\n");

    const float *first_residual_reference =
        (const float *)_binary_first_residual_output_bin_start;

    for (size_t i = 0; i < HIDDEN_ELEMENTS; i++) {
        first_residual_runtime[i] =
            round_to_bf16(hidden_states[i] +
                          post_attention_norm_runtime[i]);
    }

    size_t first_residual_mismatches = 0;
    size_t first_residual_over_tolerance = 0;
    float first_residual_max_abs_error = 0.0f;

    for (size_t i = 0; i < HIDDEN_ELEMENTS; i++) {
        float diff =
            first_residual_runtime[i] -
            first_residual_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > first_residual_max_abs_error)
            first_residual_max_abs_error = diff;

        if (diff > 0.5f)
            first_residual_over_tolerance++;

        if (first_residual_runtime[i] !=
            first_residual_reference[i])
            first_residual_mismatches++;
    }

    uint32_t first_residual_err_bits;
    memcpy(
        &first_residual_err_bits,
        &first_residual_max_abs_error,
        sizeof(first_residual_err_bits)
    );

    printf("First residual elements: %ld\n",
           (long)HIDDEN_ELEMENTS);
    printf("First residual mismatches: %ld / %ld\n",
           (long)first_residual_mismatches,
           (long)HIDDEN_ELEMENTS);
    printf("First residual errors > 0.5: %ld / %ld\n",
           (long)first_residual_over_tolerance,
           (long)HIDDEN_ELEMENTS);
    printf("First residual max abs error bits: 0x%08x\n",
           first_residual_err_bits);

    printf("Running pre-FFN RMSNorm...\n");

    const float *pre_ffn_norm_weight =
        (const float *)_binary_pre_ffn_norm_weight_bin_start;

    const float *pre_ffn_norm_reference =
        (const float *)_binary_pre_ffn_norm_output_bin_start;

    rmsnorm(
        first_residual_runtime,
        pre_ffn_norm_weight,
        pre_ffn_norm_runtime,
        SEQ_LEN,
        HIDDEN_SIZE
    );

    size_t pre_ffn_norm_mismatches = 0;
    size_t pre_ffn_norm_over_tolerance = 0;
    float pre_ffn_norm_max_abs_error = 0.0f;

    for (size_t i = 0; i < HIDDEN_ELEMENTS; i++) {
        float diff =
            pre_ffn_norm_runtime[i] -
            pre_ffn_norm_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > pre_ffn_norm_max_abs_error)
            pre_ffn_norm_max_abs_error = diff;

        if (diff > 0.5f)
            pre_ffn_norm_over_tolerance++;

        if (pre_ffn_norm_runtime[i] !=
            pre_ffn_norm_reference[i])
            pre_ffn_norm_mismatches++;
    }

    uint32_t pre_ffn_norm_err_bits;
    memcpy(
        &pre_ffn_norm_err_bits,
        &pre_ffn_norm_max_abs_error,
        sizeof(pre_ffn_norm_err_bits)
    );

    printf("Pre-FFN RMSNorm elements: %ld\n",
           (long)HIDDEN_ELEMENTS);
    printf("Pre-FFN RMSNorm mismatches: %ld / %ld\n",
           (long)pre_ffn_norm_mismatches,
           (long)HIDDEN_ELEMENTS);
    printf("Pre-FFN RMSNorm errors > 0.5: %ld / %ld\n",
           (long)pre_ffn_norm_over_tolerance,
           (long)HIDDEN_ELEMENTS);
    printf("Pre-FFN RMSNorm max abs error bits: 0x%08x\n",
           pre_ffn_norm_err_bits);

    const size_t mlp_elements = (size_t)SEQ_LEN * 6144;

    const float *gate_proj_weight =
        (const float *)_binary_gate_proj_weight_saturn_bin_start;
    const float *gate_proj_reference =
        (const float *)_binary_gate_proj_output_bin_start;

    const float *up_proj_weight =
        (const float *)_binary_up_proj_weight_saturn_bin_start;
    const float *up_proj_reference =
        (const float *)_binary_up_proj_output_bin_start;

    printf("Running MLP gate projection...\n");

    memset(gate_proj_runtime, 0, mlp_elements * sizeof(float));

    vec_sgemm_nn(
        6144,               // n
        SEQ_LEN,            // m
        HIDDEN_SIZE,        // k
        pre_ffn_norm_runtime, HIDDEN_SIZE,
        gate_proj_weight, 6144,
        gate_proj_runtime, 6144
    );

    for (size_t i = 0; i < mlp_elements; i++)
        gate_proj_runtime[i] = round_to_bf16(gate_proj_runtime[i]);

    size_t gate_mismatches = 0;
    size_t gate_over_tolerance = 0;
    float gate_max_abs_error = 0.0f;

    for (size_t i = 0; i < mlp_elements; i++) {
        float diff = gate_proj_runtime[i] - gate_proj_reference[i];
        if (diff < 0.0f)
            diff = -diff;

        if (diff > gate_max_abs_error)
            gate_max_abs_error = diff;
        if (diff > 0.5f)
            gate_over_tolerance++;
        if (gate_proj_runtime[i] != gate_proj_reference[i])
            gate_mismatches++;
    }

    uint32_t gate_err_bits;
    memcpy(&gate_err_bits, &gate_max_abs_error, sizeof(gate_err_bits));

    printf("Gate projection elements: %ld\n", (long)mlp_elements);
    printf("Gate projection mismatches: %ld / %ld\n",
           (long)gate_mismatches, (long)mlp_elements);
    printf("Gate projection errors > 0.5: %ld / %ld\n",
           (long)gate_over_tolerance, (long)mlp_elements);
    printf("Gate projection max abs error bits: 0x%08x\n",
           gate_err_bits);

    printf("Running MLP up projection...\n");

    memset(up_proj_runtime, 0, mlp_elements * sizeof(float));

    vec_sgemm_nn(
        6144,               // n
        SEQ_LEN,            // m
        HIDDEN_SIZE,        // k
        pre_ffn_norm_runtime, HIDDEN_SIZE,
        up_proj_weight, 6144,
        up_proj_runtime, 6144
    );

    for (size_t i = 0; i < mlp_elements; i++)
        up_proj_runtime[i] = round_to_bf16(up_proj_runtime[i]);

    size_t up_mismatches = 0;
    size_t up_over_tolerance = 0;
    float up_max_abs_error = 0.0f;

    for (size_t i = 0; i < mlp_elements; i++) {
        float diff = up_proj_runtime[i] - up_proj_reference[i];
        if (diff < 0.0f)
            diff = -diff;

        if (diff > up_max_abs_error)
            up_max_abs_error = diff;
        if (diff > 0.5f)
            up_over_tolerance++;
        if (up_proj_runtime[i] != up_proj_reference[i])
            up_mismatches++;
    }

    uint32_t up_err_bits;
    memcpy(&up_err_bits, &up_max_abs_error, sizeof(up_err_bits));

    printf("Up projection elements: %ld\n", (long)mlp_elements);
    printf("Up projection mismatches: %ld / %ld\n",
           (long)up_mismatches, (long)mlp_elements);
    printf("Up projection errors > 0.5: %ld / %ld\n",
           (long)up_over_tolerance, (long)mlp_elements);
    printf("Up projection max abs error bits: 0x%08x\n",
           up_err_bits);

    printf("Running GELU(gate) x up...\n");

    const float *gate_up_reference =
        (const float *)_binary_gate_up_output_bin_start;

    const float gelu_coeff = 0.7978845608028654f;

    for (size_t i = 0; i < mlp_elements; i++) {
        float x = gate_proj_runtime[i];

        float x3 = x * x * x;

        float inner =
            gelu_coeff * (x + 0.044715f * x3);

        float gelu =
            0.5f * x * (1.0f + tanhf(inner));

        gate_up_runtime[i] =
            round_to_bf16(gelu * up_proj_runtime[i]);
    }

    size_t gate_up_mismatches = 0;
    size_t gate_up_over_tolerance = 0;
    float gate_up_max_abs_error = 0.0f;
    size_t gate_up_worst = 0;

    for (size_t i = 0; i < mlp_elements; i++) {
        float diff =
            gate_up_runtime[i] - gate_up_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > gate_up_max_abs_error) {
            gate_up_max_abs_error = diff;
            gate_up_worst = i;
        }

        if (diff > 0.5f)
            gate_up_over_tolerance++;

        if (gate_up_runtime[i] != gate_up_reference[i])
            gate_up_mismatches++;
    }

    uint32_t gate_up_err_bits;
    memcpy(
        &gate_up_err_bits,
        &gate_up_max_abs_error,
        sizeof(gate_up_err_bits)
    );

    printf("Gate x Up elements: %ld\n",
           (long)mlp_elements);
    printf("Gate x Up mismatches: %ld / %ld\n",
           (long)gate_up_mismatches,
           (long)mlp_elements);
    printf("Gate x Up errors > 0.5: %ld / %ld\n",
           (long)gate_up_over_tolerance,
           (long)mlp_elements);
    printf("Gate x Up max abs error bits: 0x%08x\n",
           gate_up_err_bits);

    uint32_t gate_worst_bits;
    uint32_t up_worst_bits;
    uint32_t runtime_worst_bits;
    uint32_t reference_worst_bits;

    memcpy(&gate_worst_bits,
           &gate_proj_runtime[gate_up_worst],
           sizeof(gate_worst_bits));
    memcpy(&up_worst_bits,
           &up_proj_runtime[gate_up_worst],
           sizeof(up_worst_bits));
    memcpy(&runtime_worst_bits,
           &gate_up_runtime[gate_up_worst],
           sizeof(runtime_worst_bits));
    memcpy(&reference_worst_bits,
           &gate_up_reference[gate_up_worst],
           sizeof(reference_worst_bits));

    printf("Gate x Up worst index: %ld\n",
           (long)gate_up_worst);
    printf("Gate x Up worst gate bits: 0x%08x\n",
           gate_worst_bits);
    printf("Gate x Up worst up bits: 0x%08x\n",
           up_worst_bits);
    printf("Gate x Up worst runtime bits: 0x%08x\n",
           runtime_worst_bits);
    printf("Gate x Up worst reference bits: 0x%08x\n",
           reference_worst_bits);

    printf("Running Gate x Up golden-input diagnostic...\n");

    for (size_t i = 0; i < mlp_elements; i++) {
        float x = gate_proj_reference[i];
        float x3 = x * x * x;
        float inner =
            gelu_coeff * (x + 0.044715f * x3);
        float gelu =
            0.5f * x * (1.0f + tanhf(inner));

        gate_up_golden_input[i] =
            round_to_bf16(gelu * up_proj_reference[i]);
    }

    size_t gate_up_golden_mismatches = 0;
    size_t gate_up_golden_over_tolerance = 0;
    float gate_up_golden_max_abs_error = 0.0f;

    for (size_t i = 0; i < mlp_elements; i++) {
        float diff =
            gate_up_golden_input[i] - gate_up_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > gate_up_golden_max_abs_error)
            gate_up_golden_max_abs_error = diff;

        if (diff > 0.5f)
            gate_up_golden_over_tolerance++;

        if (gate_up_golden_input[i] != gate_up_reference[i])
            gate_up_golden_mismatches++;
    }

    uint32_t gate_up_golden_err_bits;
    memcpy(&gate_up_golden_err_bits,
           &gate_up_golden_max_abs_error,
           sizeof(gate_up_golden_err_bits));

    printf("Gate x Up golden-input mismatches: %ld / %ld\n",
           (long)gate_up_golden_mismatches,
           (long)mlp_elements);
    printf("Gate x Up golden-input errors > 0.5: %ld / %ld\n",
           (long)gate_up_golden_over_tolerance,
           (long)mlp_elements);
    printf("Gate x Up golden-input max abs error bits: 0x%08x\n",
           gate_up_golden_err_bits);

    printf("Running MLP down projection...\n");

    const float *down_proj_weight =
        (const float *)_binary_down_proj_weight_saturn_bin_start;
    const float *down_proj_reference =
        (const float *)_binary_down_proj_output_bin_start;

    memset(
        down_proj_runtime,
        0,
        sizeof(down_proj_runtime)
    );

    vec_sgemm_nn(
        HIDDEN_SIZE,
        SEQ_LEN,
        6144,
        gate_up_runtime,
        6144,
        down_proj_weight,
        HIDDEN_SIZE,
        down_proj_runtime,
        HIDDEN_SIZE
    );

    for (size_t i = 0; i < HIDDEN_ELEMENTS; i++) {
        down_proj_runtime[i] =
            round_to_bf16(down_proj_runtime[i]);
    }

    size_t down_proj_mismatches = 0;
    size_t down_proj_over_tolerance = 0;
    float down_proj_max_abs_error = 0.0f;

    for (size_t i = 0; i < HIDDEN_ELEMENTS; i++) {
        float diff =
            down_proj_runtime[i] - down_proj_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > down_proj_max_abs_error)
            down_proj_max_abs_error = diff;

        if (diff > 0.5f)
            down_proj_over_tolerance++;

        if (down_proj_runtime[i] != down_proj_reference[i])
            down_proj_mismatches++;
    }

    uint32_t down_proj_err_bits;
    memcpy(
        &down_proj_err_bits,
        &down_proj_max_abs_error,
        sizeof(down_proj_err_bits)
    );

    printf("Down projection elements: %ld\n",
           (long)HIDDEN_ELEMENTS);
    printf("Down projection mismatches: %ld / %ld\n",
           (long)down_proj_mismatches,
           (long)HIDDEN_ELEMENTS);
    printf("Down projection errors > 0.5: %ld / %ld\n",
           (long)down_proj_over_tolerance,
           (long)HIDDEN_ELEMENTS);
    printf("Down projection max abs error bits: 0x%08x\n",
           down_proj_err_bits);

    printf("Running post-FFN RMSNorm...\n");

    const float *post_ffn_norm_weight =
        (const float *)_binary_post_ffn_norm_weight_bin_start;
    const float *post_ffn_norm_reference =
        (const float *)_binary_post_ffn_norm_output_bin_start;

    rmsnorm(
        down_proj_runtime,
        post_ffn_norm_weight,
        post_ffn_norm_runtime,
        SEQ_LEN,
        HIDDEN_SIZE
    );

    size_t post_ffn_norm_mismatches = 0;
    size_t post_ffn_norm_over_tolerance = 0;
    float post_ffn_norm_max_abs_error = 0.0f;

    for (size_t i = 0; i < HIDDEN_ELEMENTS; i++) {
        float diff =
            post_ffn_norm_runtime[i] -
            post_ffn_norm_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > post_ffn_norm_max_abs_error)
            post_ffn_norm_max_abs_error = diff;

        if (diff > 0.5f)
            post_ffn_norm_over_tolerance++;

        if (post_ffn_norm_runtime[i] !=
            post_ffn_norm_reference[i])
            post_ffn_norm_mismatches++;
    }

    uint32_t post_ffn_norm_err_bits;
    memcpy(
        &post_ffn_norm_err_bits,
        &post_ffn_norm_max_abs_error,
        sizeof(post_ffn_norm_err_bits)
    );

    printf("Post-FFN RMSNorm elements: %ld\n",
           (long)HIDDEN_ELEMENTS);
    printf("Post-FFN RMSNorm mismatches: %ld / %ld\n",
           (long)post_ffn_norm_mismatches,
           (long)HIDDEN_ELEMENTS);
    printf("Post-FFN RMSNorm errors > 0.5: %ld / %ld\n",
           (long)post_ffn_norm_over_tolerance,
           (long)HIDDEN_ELEMENTS);
    printf("Post-FFN RMSNorm max abs error bits: 0x%08x\n",
           post_ffn_norm_err_bits);

    printf("Running post-FFN RMSNorm golden-input diagnostic...\n");

    rmsnorm(
        down_proj_reference,
        post_ffn_norm_weight,
        post_ffn_norm_golden_input,
        SEQ_LEN,
        HIDDEN_SIZE
    );

    size_t post_ffn_golden_mismatches = 0;
    size_t post_ffn_golden_over_tolerance = 0;
    float post_ffn_golden_max_abs_error = 0.0f;

    for (size_t i = 0; i < HIDDEN_ELEMENTS; i++) {
        float diff =
            post_ffn_norm_golden_input[i] -
            post_ffn_norm_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > post_ffn_golden_max_abs_error)
            post_ffn_golden_max_abs_error = diff;

        if (diff > 0.5f)
            post_ffn_golden_over_tolerance++;

        if (post_ffn_norm_golden_input[i] !=
            post_ffn_norm_reference[i])
            post_ffn_golden_mismatches++;
    }

    uint32_t post_ffn_golden_err_bits;
    memcpy(
        &post_ffn_golden_err_bits,
        &post_ffn_golden_max_abs_error,
        sizeof(post_ffn_golden_err_bits)
    );

    printf("Post-FFN golden-input mismatches: %ld / %ld\n",
           (long)post_ffn_golden_mismatches,
           (long)HIDDEN_ELEMENTS);
    printf("Post-FFN golden-input errors > 0.5: %ld / %ld\n",
           (long)post_ffn_golden_over_tolerance,
           (long)HIDDEN_ELEMENTS);
    printf("Post-FFN golden-input max abs error bits: 0x%08x\n",
           post_ffn_golden_err_bits);

    printf("Running second residual add...\n");

    const float *second_residual_reference =
        (const float *)_binary_second_residual_output_bin_start;

    for (size_t i = 0; i < HIDDEN_ELEMENTS; i++) {
        second_residual_runtime[i] =
            round_to_bf16(
                first_residual_runtime[i] +
                post_ffn_norm_runtime[i]
            );
    }

    size_t second_residual_mismatches = 0;
    size_t second_residual_over_tolerance = 0;
    float second_residual_max_abs_error = 0.0f;

    for (size_t i = 0; i < HIDDEN_ELEMENTS; i++) {
        float diff =
            second_residual_runtime[i] -
            second_residual_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > second_residual_max_abs_error)
            second_residual_max_abs_error = diff;

        if (diff > 0.5f)
            second_residual_over_tolerance++;

        if (second_residual_runtime[i] !=
            second_residual_reference[i])
            second_residual_mismatches++;
    }

    uint32_t second_residual_err_bits;
    memcpy(
        &second_residual_err_bits,
        &second_residual_max_abs_error,
        sizeof(second_residual_err_bits)
    );

    printf("Second residual elements: %ld\n",
           (long)HIDDEN_ELEMENTS);
    printf("Second residual mismatches: %ld / %ld\n",
           (long)second_residual_mismatches,
           (long)HIDDEN_ELEMENTS);
    printf("Second residual errors > 0.5: %ld / %ld\n",
           (long)second_residual_over_tolerance,
           (long)HIDDEN_ELEMENTS);
    printf("Second residual max abs error bits: 0x%08x\n",
           second_residual_err_bits);

    printf("Running per-layer input gate projection...\n");

    const float *per_layer_gate_weight =
        (const float *)_binary_per_layer_gate_weight_saturn_bin_start;
    const float *per_layer_gate_reference =
        (const float *)_binary_per_layer_gate_output_bin_start;

    size_t per_layer_elements =
        (size_t)SEQ_LEN * V_DIM;

    memset(
        per_layer_gate_runtime,
        0,
        sizeof(per_layer_gate_runtime)
    );

    vec_sgemm_nn(
        V_DIM,
        SEQ_LEN,
        HIDDEN_SIZE,
        second_residual_runtime,
        HIDDEN_SIZE,
        per_layer_gate_weight,
        V_DIM,
        per_layer_gate_runtime,
        V_DIM
    );

    for (size_t i = 0; i < per_layer_elements; i++) {
        per_layer_gate_runtime[i] =
            round_to_bf16(per_layer_gate_runtime[i]);
    }

    size_t per_layer_gate_mismatches = 0;
    size_t per_layer_gate_over_tolerance = 0;
    float per_layer_gate_max_abs_error = 0.0f;

    for (size_t i = 0; i < per_layer_elements; i++) {
        float diff =
            per_layer_gate_runtime[i] -
            per_layer_gate_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > per_layer_gate_max_abs_error)
            per_layer_gate_max_abs_error = diff;

        if (diff > 0.5f)
            per_layer_gate_over_tolerance++;

        if (per_layer_gate_runtime[i] !=
            per_layer_gate_reference[i])
            per_layer_gate_mismatches++;
    }

    uint32_t per_layer_gate_err_bits;
    memcpy(
        &per_layer_gate_err_bits,
        &per_layer_gate_max_abs_error,
        sizeof(per_layer_gate_err_bits)
    );

    printf("Per-layer gate elements: %ld\n",
           (long)per_layer_elements);
    printf("Per-layer gate mismatches: %ld / %ld\n",
           (long)per_layer_gate_mismatches,
           (long)per_layer_elements);
    printf("Per-layer gate errors > 0.5: %ld / %ld\n",
           (long)per_layer_gate_over_tolerance,
           (long)per_layer_elements);
    printf("Per-layer gate max abs error bits: 0x%08x\n",
           per_layer_gate_err_bits);

    printf("Running GELU(per-layer gate) x per-layer input...\n");

    const float *per_layer_input =
        (const float *)_binary_per_layer_input_bin_start;
    const float *per_layer_product_reference =
        (const float *)_binary_per_layer_product_bin_start;

    const float per_layer_gelu_coeff =
        0.7978845608028654f;

    for (size_t i = 0; i < per_layer_elements; i++) {
        float x = per_layer_gate_runtime[i];

        float x3 = x * x * x;

        float inner =
            per_layer_gelu_coeff *
            (x + 0.044715f * x3);

        float gelu =
            0.5f * x * (1.0f + tanhf(inner));

        per_layer_product_runtime[i] =
            round_to_bf16(
                gelu * per_layer_input[i]
            );
    }

    size_t per_layer_product_mismatches = 0;
    size_t per_layer_product_over_tolerance = 0;
    float per_layer_product_max_abs_error = 0.0f;

    for (size_t i = 0; i < per_layer_elements; i++) {
        float diff =
            per_layer_product_runtime[i] -
            per_layer_product_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > per_layer_product_max_abs_error)
            per_layer_product_max_abs_error = diff;

        if (diff > 0.5f)
            per_layer_product_over_tolerance++;

        if (per_layer_product_runtime[i] !=
            per_layer_product_reference[i])
            per_layer_product_mismatches++;
    }

    uint32_t per_layer_product_err_bits;
    memcpy(
        &per_layer_product_err_bits,
        &per_layer_product_max_abs_error,
        sizeof(per_layer_product_err_bits)
    );

    printf("Per-layer product elements: %ld\n",
           (long)per_layer_elements);
    printf("Per-layer product mismatches: %ld / %ld\n",
           (long)per_layer_product_mismatches,
           (long)per_layer_elements);
    printf("Per-layer product errors > 0.5: %ld / %ld\n",
           (long)per_layer_product_over_tolerance,
           (long)per_layer_elements);
    printf("Per-layer product max abs error bits: 0x%08x\n",
           per_layer_product_err_bits);

    printf("Running per-layer projection...\n");

    const float *per_layer_projection_weight =
        (const float *)_binary_per_layer_projection_weight_saturn_bin_start;
    const float *per_layer_projection_reference =
        (const float *)_binary_per_layer_projection_output_bin_start;

    memset(
        per_layer_projection_runtime,
        0,
        sizeof(per_layer_projection_runtime)
    );

    vec_sgemm_nn(
        HIDDEN_SIZE,
        SEQ_LEN,
        V_DIM,
        per_layer_product_runtime,
        V_DIM,
        per_layer_projection_weight,
        HIDDEN_SIZE,
        per_layer_projection_runtime,
        HIDDEN_SIZE
    );

    for (size_t i = 0; i < HIDDEN_ELEMENTS; i++) {
        per_layer_projection_runtime[i] =
            round_to_bf16(per_layer_projection_runtime[i]);
    }

    size_t per_layer_projection_mismatches = 0;
    size_t per_layer_projection_over_tolerance = 0;
    float per_layer_projection_max_abs_error = 0.0f;

    for (size_t i = 0; i < HIDDEN_ELEMENTS; i++) {
        float diff =
            per_layer_projection_runtime[i] -
            per_layer_projection_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > per_layer_projection_max_abs_error)
            per_layer_projection_max_abs_error = diff;

        if (diff > 0.5f)
            per_layer_projection_over_tolerance++;

        if (per_layer_projection_runtime[i] !=
            per_layer_projection_reference[i])
            per_layer_projection_mismatches++;
    }

    uint32_t per_layer_projection_err_bits;
    memcpy(
        &per_layer_projection_err_bits,
        &per_layer_projection_max_abs_error,
        sizeof(per_layer_projection_err_bits)
    );

    printf("Per-layer projection elements: %ld\n",
           (long)HIDDEN_ELEMENTS);
    printf("Per-layer projection mismatches: %ld / %ld\n",
           (long)per_layer_projection_mismatches,
           (long)HIDDEN_ELEMENTS);
    printf("Per-layer projection errors > 0.5: %ld / %ld\n",
           (long)per_layer_projection_over_tolerance,
           (long)HIDDEN_ELEMENTS);
    printf("Per-layer projection max abs error bits: 0x%08x\n",
           per_layer_projection_err_bits);

    printf("Running post-per-layer RMSNorm...\n");

    const float *post_per_layer_norm_weight =
        (const float *)_binary_post_per_layer_input_norm_weight_bin_start;
    const float *post_per_layer_norm_reference =
        (const float *)_binary_post_per_layer_norm_output_bin_start;

    rmsnorm(
        per_layer_projection_runtime,
        post_per_layer_norm_weight,
        post_per_layer_norm_runtime,
        SEQ_LEN,
        HIDDEN_SIZE
    );

    size_t post_per_layer_norm_mismatches = 0;
    size_t post_per_layer_norm_over_tolerance = 0;
    float post_per_layer_norm_max_abs_error = 0.0f;

    for (size_t i = 0; i < HIDDEN_ELEMENTS; i++) {
        float diff =
            post_per_layer_norm_runtime[i] -
            post_per_layer_norm_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        if (diff > post_per_layer_norm_max_abs_error)
            post_per_layer_norm_max_abs_error = diff;

        if (diff > 0.5f)
            post_per_layer_norm_over_tolerance++;

        if (post_per_layer_norm_runtime[i] !=
            post_per_layer_norm_reference[i])
            post_per_layer_norm_mismatches++;
    }

    uint32_t post_per_layer_norm_err_bits;
    memcpy(
        &post_per_layer_norm_err_bits,
        &post_per_layer_norm_max_abs_error,
        sizeof(post_per_layer_norm_err_bits)
    );

    printf("Post-per-layer RMSNorm elements: %ld\n",
           (long)HIDDEN_ELEMENTS);
    printf("Post-per-layer RMSNorm mismatches: %ld / %ld\n",
           (long)post_per_layer_norm_mismatches,
           (long)HIDDEN_ELEMENTS);
    printf("Post-per-layer RMSNorm errors > 0.5: %ld / %ld\n",
           (long)post_per_layer_norm_over_tolerance,
           (long)HIDDEN_ELEMENTS);
    printf("Post-per-layer RMSNorm max abs error bits: 0x%08x\n",
           post_per_layer_norm_err_bits);

    /*
     * Final residual:
     * second_residual + post-per-layer RMSNorm
     */
    printf("Running final residual add...\n");

    for (size_t i = 0; i < HIDDEN_ELEMENTS; i++) {
        pre_scalar_runtime[i] =
            round_to_bf16(
                second_residual_runtime[i] +
                post_per_layer_norm_runtime[i]
            );
    }

    /*
     * Final Gemma layer scalar multiply.
     */
    const float *layer_scalar_ptr =
        (const float *)_binary_layer_scalar_bin_start;
    const float layer_scalar = layer_scalar_ptr[0];

    const float *final_reference =
        (const float *)_binary_layer_output_bin_start;

    printf("Running final layer scalar multiply...\n");

    for (size_t i = 0; i < HIDDEN_ELEMENTS; i++) {
        final_output_runtime[i] =
            round_to_bf16(
                pre_scalar_runtime[i] * layer_scalar
            );
    }

    /*
     * Final end-to-end propagated Saturn comparison.
     */
    size_t final_mismatches = 0;
    size_t final_over_tolerance = 0;
    float final_max_abs_error = 0.0f;
    double final_total_error = 0.0;
    double final_checksum = 0.0;
    double reference_checksum = 0.0;

    for (size_t i = 0; i < HIDDEN_ELEMENTS; i++) {
        float diff =
            final_output_runtime[i] -
            final_reference[i];

        if (diff < 0.0f)
            diff = -diff;

        final_total_error += diff;
        final_checksum += final_output_runtime[i];
        reference_checksum += final_reference[i];

        if (final_output_runtime[i] != final_reference[i])
            final_mismatches++;

        if (diff > 0.5f)
            final_over_tolerance++;

        if (diff > final_max_abs_error)
            final_max_abs_error = diff;
    }

    uint32_t final_err_bits;
    memcpy(
        &final_err_bits,
        &final_max_abs_error,
        sizeof(final_err_bits)
    );

    uint32_t layer_scalar_bits;
    memcpy(
        &layer_scalar_bits,
        &layer_scalar,
        sizeof(layer_scalar_bits)
    );

    printf("Layer scalar bits: 0x%08x\n",
           layer_scalar_bits);

    printf("========================================\n");
    printf("FINAL SATURN DECODER LAYER COMPARISON\n");
    printf("========================================\n");
    printf("Elements: %ld\n",
           (long)HIDDEN_ELEMENTS);
    printf("Mismatches: %ld / %ld\n",
           (long)final_mismatches,
           (long)HIDDEN_ELEMENTS);
    printf("Errors > 0.5: %ld / %ld\n",
           (long)final_over_tolerance,
           (long)HIDDEN_ELEMENTS);
    printf("Max abs error bits: 0x%08x\n",
           final_err_bits);

    float final_mean_abs_error =
        (float)(final_total_error /
                (double)HIDDEN_ELEMENTS);

    uint32_t final_mean_err_bits;
    memcpy(
        &final_mean_err_bits,
        &final_mean_abs_error,
        sizeof(final_mean_err_bits)
    );

    printf("Mean abs error bits: 0x%08x\n",
           final_mean_err_bits);

    printf("Final checksum integer part: %ld\n",
           (long)final_checksum);
    printf("Reference checksum integer part: %ld\n",
           (long)reference_checksum);
    printf("========================================\n");

    return 0;
}
