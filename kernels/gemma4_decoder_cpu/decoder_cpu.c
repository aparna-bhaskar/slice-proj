#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define SEQ_LEN 14
#define HIDDEN_SIZE 1536
#define Q_DIM 2048
#define KV_DIM 256
#define HEAD_DIM 256
#define EPS 1e-6f

#define EXPORT_DIR "/home/aparna/gemma4-trace/decoder_cpu_export"

static void load_f32(const char *path, float *data, size_t count)
{
    FILE *f = fopen(path, "rb");

    if (!f) {
        perror(path);
        exit(1);
    }

    size_t n = fread(data, sizeof(float), count, f);
    fclose(f);

    if (n != count) {
        fprintf(stderr,
                "ERROR: expected %zu floats from %s, got %zu\n",
                count, path, n);
        exit(1);
    }
}

/*
 * Round an FP32 value to BF16 using round-to-nearest-even,
 * then return the BF16 value represented as FP32.
 */
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

            float y =
                input[idx] * inv_rms;

            if (weight != NULL) {
                y *= weight[j];
            }

            output[idx] = round_to_bf16(y);
        }
    }
}


static void linear(
    const float *input,
    const float *weight,
    float *output,
    int rows,
    int in_dim,
    int out_dim)
{
    /*
     * PyTorch Linear weight layout:
     *     weight[out_dim][in_dim]
     *
     * output[i][j] =
     *     sum_k input[i][k] * weight[j][k]
     */
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < out_dim; j++) {

            float sum = 0.0f;

            for (int k = 0; k < in_dim; k++) {
                sum +=
                    input[i * in_dim + k] *
                    weight[j * in_dim + k];
            }

            output[i * out_dim + j] =
                round_to_bf16(sum);
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

    /*
     * input:  [seq][head][dim]
     * output: [head][seq][dim]
     *
     * rotate_half([x1, x2]) = [-x2, x1]
     */
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
                    :  input[rot_idx];

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


static void attention_qk(
    const float *q,
    const float *k,
    float *scores,
    int num_heads,
    int seq_len,
    int head_dim)
{
    /*
     * q:      [head][query_seq][dim]
     * k:      [1][key_seq][dim]
     * scores: [head][query_seq][key_seq]
     *
     * Layer 0 has one KV head, shared by all 8 Q heads.
     * scaling = 1.0.
     */
    for (int h = 0; h < num_heads; h++) {
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {

                float sum = 0.0f;

                for (int d = 0; d < head_dim; d++) {
                    size_t q_idx =
                        ((size_t)h * seq_len + i) *
                        head_dim + d;

                    size_t k_idx =
                        (size_t)j * head_dim + d;

                    sum += q[q_idx] * k[k_idx];
                }

                size_t out_idx =
                    ((size_t)h * seq_len + i) *
                    seq_len + j;

                scores[out_idx] = sum;
            }
        }
    }
}


static void attention_softmax(
    const float *scores,
    const float *mask,
    float *weights,
    int num_heads,
    int seq_len)
{
    /*
     * scores:  [head][query][key]
     * mask:    [query][key], broadcast over heads
     * weights: [head][query][key]
     *
     * Matches:
     *   softmax(scores + mask, dim=-1, dtype=float32)
     *   -> cast to BF16
     */
    for (int h = 0; h < num_heads; h++) {
        for (int q = 0; q < seq_len; q++) {

            float max_value = -INFINITY;

            for (int k = 0; k < seq_len; k++) {
                size_t idx =
                    ((size_t)h * seq_len + q) *
                    seq_len + k;

                size_t mask_idx =
                    (size_t)q * seq_len + k;

                float x =
                    scores[idx] + mask[mask_idx];

                if (x > max_value)
                    max_value = x;
            }

            float sum = 0.0f;

            for (int k = 0; k < seq_len; k++) {
                size_t idx =
                    ((size_t)h * seq_len + q) *
                    seq_len + k;

                size_t mask_idx =
                    (size_t)q * seq_len + k;

                float x =
                    scores[idx] + mask[mask_idx];

                float e = expf(x - max_value);

                weights[idx] = e;
                sum += e;
            }

            for (int k = 0; k < seq_len; k++) {
                size_t idx =
                    ((size_t)h * seq_len + q) *
                    seq_len + k;

                float p = weights[idx] / sum;

                weights[idx] = round_to_bf16(p);
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
    /*
     * weights: [head][query][key]
     * v:       [key][dim] -- one shared KV head
     * output:  [query][head][dim]
     *
     * Matches:
     *   torch.matmul(attn_weights, value_states)
     *   followed by transpose(1, 2)
     */
    for (int h = 0; h < num_heads; h++) {
        for (int q = 0; q < seq_len; q++) {
            for (int d = 0; d < head_dim; d++) {

                float sum = 0.0f;

                for (int k = 0; k < seq_len; k++) {
                    size_t weight_idx =
                        ((size_t)h * seq_len + q) *
                        seq_len + k;

                    size_t v_idx =
                        (size_t)k * head_dim + d;

                    sum +=
                        weights[weight_idx] *
                        v[v_idx];
                }

                size_t out_idx =
                    ((size_t)q * num_heads + h) *
                    head_dim + d;

                output[out_idx] = round_to_bf16(sum);
            }
        }
    }
}


static void residual_add(
    const float *a,
    const float *b,
    float *output,
    size_t count)
{
    /*
     * PyTorch layer executes the residual add in BF16:
     *
     *     hidden_states = residual + hidden_states
     *
     * Both inputs are BF16 tensors, so emulate the resulting
     * BF16 boundary explicitly.
     */
    for (size_t i = 0; i < count; i++) {
        output[i] = round_to_bf16(a[i] + b[i]);
    }
}

int main(void)
{
    const size_t num_elements =
        (size_t)SEQ_LEN * HIDDEN_SIZE;

    float *input =
        malloc(num_elements * sizeof(float));

    float *weight =
        malloc(HIDDEN_SIZE * sizeof(float));

    float *output =
        malloc(num_elements * sizeof(float));

    float *reference =
        malloc(num_elements * sizeof(float));

    if (!input || !weight || !output || !reference) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    load_f32(
        EXPORT_DIR "/hidden_states.bin",
        input,
        num_elements
    );

    load_f32(
        EXPORT_DIR "/input_norm_weight.bin",
        weight,
        HIDDEN_SIZE
    );

    load_f32(
        EXPORT_DIR "/input_norm_output.bin",
        reference,
        num_elements
    );

    printf("Gemma 4 decoder CPU baseline\n");
    printf("Running input RMSNorm...\n");

    rmsnorm(
        input,
        weight,
        output,
        SEQ_LEN,
        HIDDEN_SIZE
    );

    double sum_abs_error = 0.0;
    float max_abs_error = 0.0f;
    size_t max_index = 0;

    for (size_t i = 0; i < num_elements; i++) {
        float error = fabsf(output[i] - reference[i]);

        sum_abs_error += error;

        if (error > max_abs_error) {
            max_abs_error = error;
            max_index = i;
        }
    }

    printf("\nInput RMSNorm comparison\n");
    printf("Elements:       %zu\n", num_elements);
    printf("Max abs error:  %.9g\n", max_abs_error);
    printf("Mean abs error: %.9g\n",
           sum_abs_error / num_elements);

    printf("Worst index:    %zu\n", max_index);
    printf("C output:       %.9g\n", output[max_index]);
    printf("PyTorch:        %.9g\n", reference[max_index]);


    /*
     * Q/K/V projections
     *
     * Feed our computed RMSNorm output directly into the
     * three projections.
     */
    size_t q_elements = (size_t)SEQ_LEN * Q_DIM;
    size_t kv_elements = (size_t)SEQ_LEN * KV_DIM;

    float *q_weight =
        malloc((size_t)Q_DIM * HIDDEN_SIZE * sizeof(float));
    float *k_weight =
        malloc((size_t)KV_DIM * HIDDEN_SIZE * sizeof(float));
    float *v_weight =
        malloc((size_t)KV_DIM * HIDDEN_SIZE * sizeof(float));

    float *q_output = malloc(q_elements * sizeof(float));
    float *k_output = malloc(kv_elements * sizeof(float));
    float *v_output = malloc(kv_elements * sizeof(float));

    float *q_reference = malloc(q_elements * sizeof(float));
    float *k_reference = malloc(kv_elements * sizeof(float));
    float *v_reference = malloc(kv_elements * sizeof(float));

    if (!q_weight || !k_weight || !v_weight ||
        !q_output || !k_output || !v_output ||
        !q_reference || !k_reference || !v_reference) {
        fprintf(stderr, "Q/K/V allocation failed\n");
        return 1;
    }

    load_f32(
        EXPORT_DIR "/q_proj_weight.bin",
        q_weight,
        (size_t)Q_DIM * HIDDEN_SIZE
    );

    load_f32(
        EXPORT_DIR "/k_proj_weight.bin",
        k_weight,
        (size_t)KV_DIM * HIDDEN_SIZE
    );

    load_f32(
        EXPORT_DIR "/v_proj_weight.bin",
        v_weight,
        (size_t)KV_DIM * HIDDEN_SIZE
    );

    load_f32(
        EXPORT_DIR "/q_proj_output.bin",
        q_reference,
        q_elements
    );

    load_f32(
        EXPORT_DIR "/k_proj_output.bin",
        k_reference,
        kv_elements
    );

    load_f32(
        EXPORT_DIR "/v_proj_output.bin",
        v_reference,
        kv_elements
    );

    printf("\nRunning Q projection...\n");
    linear(
        output,
        q_weight,
        q_output,
        SEQ_LEN,
        HIDDEN_SIZE,
        Q_DIM
    );

    printf("Running K projection...\n");
    linear(
        output,
        k_weight,
        k_output,
        SEQ_LEN,
        HIDDEN_SIZE,
        KV_DIM
    );

    printf("Running V projection...\n");
    linear(
        output,
        v_weight,
        v_output,
        SEQ_LEN,
        HIDDEN_SIZE,
        KV_DIM
    );

    const char *names[3] = {
        "Q projection",
        "K projection",
        "V projection"
    };

    float *outputs[3] = {
        q_output,
        k_output,
        v_output
    };

    float *references[3] = {
        q_reference,
        k_reference,
        v_reference
    };

    size_t counts[3] = {
        q_elements,
        kv_elements,
        kv_elements
    };

    for (int t = 0; t < 3; t++) {

        double total_error = 0.0;
        float max_error = 0.0f;
        size_t worst = 0;
        size_t mismatch_count = 0;

        for (size_t i = 0; i < counts[t]; i++) {
            float error =
                fabsf(outputs[t][i] - references[t][i]);

            total_error += error;

            if (error != 0.0f)
                mismatch_count++;

            if (error > max_error) {
                max_error = error;
                worst = i;
            }
        }

        printf("\n%s comparison\n", names[t]);
        printf("Elements:       %zu\n", counts[t]);
        printf("Mismatches:     %zu\n", mismatch_count);
        printf("Max abs error:  %.9g\n", max_error);
        printf("Mean abs error: %.9g\n",
               total_error / counts[t]);
        printf("Worst index:    %zu\n", worst);
        printf("C output:       %.9g\n", outputs[t][worst]);
        printf("PyTorch:        %.9g\n", references[t][worst]);
    }


    /*
     * Q/K/V RMSNorm
     *
     * Q is logically [14, 8, 256], so RMSNorm sees
     * 14*8 independent rows of length 256.
     *
     * K/V are [14, 1, 256], so each has 14 rows.
     */
    float *q_norm_weight =
        malloc(HEAD_DIM * sizeof(float));
    float *k_norm_weight =
        malloc(HEAD_DIM * sizeof(float));

    float *q_norm_output =
        malloc(q_elements * sizeof(float));
    float *k_norm_output =
        malloc(kv_elements * sizeof(float));
    float *v_norm_output =
        malloc(kv_elements * sizeof(float));

    float *q_norm_reference =
        malloc(q_elements * sizeof(float));
    float *k_norm_reference =
        malloc(kv_elements * sizeof(float));
    float *v_norm_reference =
        malloc(kv_elements * sizeof(float));

    if (!q_norm_weight || !k_norm_weight ||
        !q_norm_output || !k_norm_output || !v_norm_output ||
        !q_norm_reference || !k_norm_reference || !v_norm_reference) {
        fprintf(stderr, "Q/K/V norm allocation failed\n");
        return 1;
    }

    load_f32(
        EXPORT_DIR "/q_norm_weight.bin",
        q_norm_weight,
        HEAD_DIM
    );

    load_f32(
        EXPORT_DIR "/k_norm_weight.bin",
        k_norm_weight,
        HEAD_DIM
    );

    load_f32(
        EXPORT_DIR "/q_norm_output.bin",
        q_norm_reference,
        q_elements
    );

    load_f32(
        EXPORT_DIR "/k_norm_output.bin",
        k_norm_reference,
        kv_elements
    );

    load_f32(
        EXPORT_DIR "/v_norm_output.bin",
        v_norm_reference,
        kv_elements
    );

    printf("\nRunning Q RMSNorm...\n");
    rmsnorm(
        q_output,
        q_norm_weight,
        q_norm_output,
        SEQ_LEN * 8,
        HEAD_DIM
    );

    printf("Running K RMSNorm...\n");
    rmsnorm(
        k_output,
        k_norm_weight,
        k_norm_output,
        SEQ_LEN,
        HEAD_DIM
    );

    printf("Running V RMSNorm...\n");
    rmsnorm(
        v_output,
        NULL,
        v_norm_output,
        SEQ_LEN,
        HEAD_DIM
    );

    const char *norm_names[3] = {
        "Q RMSNorm",
        "K RMSNorm",
        "V RMSNorm"
    };

    float *norm_outputs[3] = {
        q_norm_output,
        k_norm_output,
        v_norm_output
    };

    float *norm_refs[3] = {
        q_norm_reference,
        k_norm_reference,
        v_norm_reference
    };

    size_t norm_counts[3] = {
        q_elements,
        kv_elements,
        kv_elements
    };

    for (int t = 0; t < 3; t++) {
        double total_error = 0.0;
        float max_error = 0.0f;
        size_t worst = 0;
        size_t mismatch_count = 0;

        for (size_t i = 0; i < norm_counts[t]; i++) {
            float error =
                fabsf(norm_outputs[t][i] - norm_refs[t][i]);

            total_error += error;

            if (error != 0.0f)
                mismatch_count++;

            if (error > max_error) {
                max_error = error;
                worst = i;
            }
        }

        printf("\n%s comparison\n", norm_names[t]);
        printf("Elements:       %zu\n", norm_counts[t]);
        printf("Mismatches:     %zu\n", mismatch_count);
        printf("Max abs error:  %.9g\n", max_error);
        printf("Mean abs error: %.9g\n",
               total_error / norm_counts[t]);
        printf("Worst index:    %zu\n", worst);
        printf("C output:       %.9g\n",
               norm_outputs[t][worst]);
        printf("PyTorch:        %.9g\n",
               norm_refs[t][worst]);
    }


    /*
     * RoPE + transpose
     *
     * Q: [14,8,256] -> [8,14,256]
     * K: [14,1,256] -> [1,14,256]
     */
    size_t rope_table_elements =
        (size_t)SEQ_LEN * HEAD_DIM;

    float *rope_cos =
        malloc(rope_table_elements * sizeof(float));
    float *rope_sin =
        malloc(rope_table_elements * sizeof(float));

    float *q_rope_output =
        malloc(q_elements * sizeof(float));
    float *k_rope_output =
        malloc(kv_elements * sizeof(float));

    float *q_rope_reference =
        malloc(q_elements * sizeof(float));
    float *k_rope_reference =
        malloc(kv_elements * sizeof(float));

    if (!rope_cos || !rope_sin ||
        !q_rope_output || !k_rope_output ||
        !q_rope_reference || !k_rope_reference) {
        fprintf(stderr, "RoPE allocation failed\n");
        return 1;
    }

    load_f32(
        EXPORT_DIR "/rope_cos.bin",
        rope_cos,
        rope_table_elements
    );

    load_f32(
        EXPORT_DIR "/rope_sin.bin",
        rope_sin,
        rope_table_elements
    );

    load_f32(
        EXPORT_DIR "/q_rope_output.bin",
        q_rope_reference,
        q_elements
    );

    load_f32(
        EXPORT_DIR "/k_rope_output.bin",
        k_rope_reference,
        kv_elements
    );

    printf("\nRunning Q RoPE...\n");
    rope_and_transpose(
        q_norm_output,
        rope_cos,
        rope_sin,
        q_rope_output,
        SEQ_LEN,
        8,
        HEAD_DIM
    );

    printf("Running K RoPE...\n");
    rope_and_transpose(
        k_norm_output,
        rope_cos,
        rope_sin,
        k_rope_output,
        SEQ_LEN,
        1,
        HEAD_DIM
    );

    const char *rope_names[2] = {
        "Q RoPE",
        "K RoPE"
    };

    float *rope_outputs[2] = {
        q_rope_output,
        k_rope_output
    };

    float *rope_refs[2] = {
        q_rope_reference,
        k_rope_reference
    };

    size_t rope_counts[2] = {
        q_elements,
        kv_elements
    };

    for (int t = 0; t < 2; t++) {
        double total_error = 0.0;
        float max_error = 0.0f;
        size_t worst = 0;
        size_t mismatch_count = 0;

        for (size_t i = 0; i < rope_counts[t]; i++) {
            float error =
                fabsf(rope_outputs[t][i] - rope_refs[t][i]);

            total_error += error;

            if (error != 0.0f)
                mismatch_count++;

            if (error > max_error) {
                max_error = error;
                worst = i;
            }
        }

        printf("\n%s comparison\n", rope_names[t]);
        printf("Elements:       %zu\n", rope_counts[t]);
        printf("Mismatches:     %zu\n", mismatch_count);
        printf("Max abs error:  %.9g\n", max_error);
        printf("Mean abs error: %.9g\n",
               total_error / rope_counts[t]);
        printf("Worst index:    %zu\n", worst);
        printf("C output:       %.9g\n",
               rope_outputs[t][worst]);
        printf("PyTorch:        %.9g\n",
               rope_refs[t][worst]);
    }


    /*
     * QK^T attention scores
     * [8,14,256] x [1,14,256] -> [8,14,14]
     */
    size_t score_elements =
        (size_t)8 * SEQ_LEN * SEQ_LEN;

    float *attention_scores =
        malloc(score_elements * sizeof(float));

    float *attention_scores_reference =
        malloc(score_elements * sizeof(float));

    if (!attention_scores ||
        !attention_scores_reference) {
        fprintf(stderr, "Attention score allocation failed\n");
        return 1;
    }

    load_f32(
        EXPORT_DIR "/attention_scores.bin",
        attention_scores_reference,
        score_elements
    );

    printf("\nRunning QK^T attention...\n");

    attention_qk(
        q_rope_output,
        k_rope_output,
        attention_scores,
        8,
        SEQ_LEN,
        HEAD_DIM
    );

    double score_total_error = 0.0;
    float score_max_error = 0.0f;
    size_t score_worst = 0;
    size_t score_mismatches = 0;

    for (size_t i = 0; i < score_elements; i++) {
        float error =
            fabsf(attention_scores[i] -
                  attention_scores_reference[i]);

        score_total_error += error;

        if (error != 0.0f)
            score_mismatches++;

        if (error > score_max_error) {
            score_max_error = error;
            score_worst = i;
        }
    }

    printf("\nQK^T attention comparison\n");
    printf("Elements:       %zu\n", score_elements);
    printf("Mismatches:     %zu\n", score_mismatches);
    printf("Max abs error:  %.9g\n", score_max_error);
    printf("Mean abs error: %.9g\n",
           score_total_error / score_elements);
    printf("Worst index:    %zu\n", score_worst);
    printf("C output:       %.9g\n",
           attention_scores[score_worst]);
    printf("PyTorch ref:    %.9g\n",
           attention_scores_reference[score_worst]);


    /*
     * Attention mask + FP32 softmax
     * scores:  [8,14,14]
     * mask:    [14,14], broadcast across heads
     * weights: [8,14,14]
     */
    size_t mask_elements =
        (size_t)SEQ_LEN * SEQ_LEN;

    float *attention_mask =
        malloc(mask_elements * sizeof(float));

    float *attention_weights =
        malloc(score_elements * sizeof(float));

    float *attention_weights_reference =
        malloc(score_elements * sizeof(float));

    if (!attention_mask ||
        !attention_weights ||
        !attention_weights_reference) {
        fprintf(stderr, "Attention softmax allocation failed\n");
        return 1;
    }

    load_f32(
        EXPORT_DIR "/attention_mask.bin",
        attention_mask,
        mask_elements
    );

    load_f32(
        EXPORT_DIR "/attention_weights.bin",
        attention_weights_reference,
        score_elements
    );

    printf("\nRunning attention mask + softmax...\n");

    attention_softmax(
        attention_scores,
        attention_mask,
        attention_weights,
        8,
        SEQ_LEN
    );

    double weight_total_error = 0.0;
    float weight_max_error = 0.0f;
    size_t weight_worst = 0;
    size_t weight_mismatches = 0;

    for (size_t i = 0; i < score_elements; i++) {
        float error =
            fabsf(attention_weights[i] -
                  attention_weights_reference[i]);

        weight_total_error += error;

        if (error != 0.0f)
            weight_mismatches++;

        if (error > weight_max_error) {
            weight_max_error = error;
            weight_worst = i;
        }
    }

    printf("\nAttention softmax comparison\n");
    printf("Elements:       %zu\n", score_elements);
    printf("Mismatches:     %zu\n", weight_mismatches);
    printf("Max abs error:  %.9g\n", weight_max_error);
    printf("Mean abs error: %.9g\n",
           weight_total_error / score_elements);
    printf("Worst index:    %zu\n", weight_worst);
    printf("C output:       %.9g\n",
           attention_weights[weight_worst]);
    printf("PyTorch ref:    %.9g\n",
           attention_weights_reference[weight_worst]);

    /*
     * Each softmax row should sum to approximately 1.
     * Print the first row as a quick sanity check.
     */
    float first_row_sum = 0.0f;

    for (int k = 0; k < SEQ_LEN; k++)
        first_row_sum += attention_weights[k];

    printf("First row sum:  %.9g\n", first_row_sum);


    /*
     * Attention x V
     *
     * weights: [8,14,14]
     * V:       [14,256] shared across all 8 heads
     * output:  [14,8,256]
     */
    float *attention_output =
        malloc(q_elements * sizeof(float));

    float *attention_output_reference =
        malloc(q_elements * sizeof(float));

    if (!attention_output ||
        !attention_output_reference) {
        fprintf(stderr, "Attention x V allocation failed\n");
        return 1;
    }

    load_f32(
        EXPORT_DIR "/attention_output.bin",
        attention_output_reference,
        q_elements
    );

    printf("\nRunning Attention x V...\n");

    attention_times_v(
        attention_weights,
        v_norm_output,
        attention_output,
        8,
        SEQ_LEN,
        HEAD_DIM
    );

    double av_total_error = 0.0;
    float av_max_error = 0.0f;
    size_t av_worst = 0;
    size_t av_mismatches = 0;

    for (size_t i = 0; i < q_elements; i++) {
        float error =
            fabsf(attention_output[i] -
                  attention_output_reference[i]);

        av_total_error += error;

        if (error != 0.0f)
            av_mismatches++;

        if (error > av_max_error) {
            av_max_error = error;
            av_worst = i;
        }
    }

    printf("\nAttention x V comparison\n");
    printf("Elements:       %zu\n", q_elements);
    printf("Mismatches:     %zu\n", av_mismatches);
    printf("Max abs error:  %.9g\n", av_max_error);
    printf("Mean abs error: %.9g\n",
           av_total_error / q_elements);
    printf("Worst index:    %zu\n", av_worst);
    printf("C output:       %.9g\n",
           attention_output[av_worst]);
    printf("PyTorch ref:    %.9g\n",
           attention_output_reference[av_worst]);


    /*
     * Output projection (o_proj)
     *
     * attention_output is already contiguous as:
     * [14,8,256] == [14,2048]
     *
     * o_proj:
     * [14,2048] x weight[1536,2048]
     * -> [14,1536]
     */
    size_t o_weight_elements =
        (size_t)HIDDEN_SIZE * Q_DIM;

    size_t hidden_elements =
        (size_t)SEQ_LEN * HIDDEN_SIZE;

    float *o_proj_weight =
        malloc(o_weight_elements * sizeof(float));

    float *o_proj_output =
        malloc(hidden_elements * sizeof(float));

    float *o_proj_reference =
        malloc(hidden_elements * sizeof(float));

    if (!o_proj_weight ||
        !o_proj_output ||
        !o_proj_reference) {
        fprintf(stderr, "o_proj allocation failed\n");
        return 1;
    }

    load_f32(
        EXPORT_DIR "/o_proj_weight.bin",
        o_proj_weight,
        o_weight_elements
    );

    load_f32(
        EXPORT_DIR "/o_proj_output.bin",
        o_proj_reference,
        hidden_elements
    );

    printf("\nRunning output projection...\n");

    linear(
        attention_output,
        o_proj_weight,
        o_proj_output,
        SEQ_LEN,
        Q_DIM,
        HIDDEN_SIZE
    );

    double o_total_error = 0.0;
    float o_max_error = 0.0f;
    size_t o_worst = 0;
    size_t o_mismatches = 0;

    for (size_t i = 0; i < hidden_elements; i++) {
        float error =
            fabsf(o_proj_output[i] -
                  o_proj_reference[i]);

        o_total_error += error;

        if (error != 0.0f)
            o_mismatches++;

        if (error > o_max_error) {
            o_max_error = error;
            o_worst = i;
        }
    }

    printf("\nOutput projection comparison\n");
    printf("Elements:       %zu\n", hidden_elements);
    printf("Mismatches:     %zu\n", o_mismatches);
    printf("Max abs error:  %.9g\n", o_max_error);
    printf("Mean abs error: %.9g\n",
           o_total_error / hidden_elements);
    printf("Worst index:    %zu\n", o_worst);
    printf("C output:       %.9g\n",
           o_proj_output[o_worst]);
    printf("PyTorch ref:    %.9g\n",
           o_proj_reference[o_worst]);


    /*
     * Post-attention RMSNorm + first residual connection
     *
     * o_proj_output [14,1536]
     *      ↓ post_attention_layernorm
     * normalized [14,1536]
     *      ↓ + original layer input
     * first_residual [14,1536]
     */
    float *post_attn_norm_weight =
        malloc(HIDDEN_SIZE * sizeof(float));

    float *post_attn_norm_output =
        malloc(hidden_elements * sizeof(float));

    float *post_attn_norm_reference =
        malloc(hidden_elements * sizeof(float));

    float *first_residual_output =
        malloc(hidden_elements * sizeof(float));

    float *first_residual_reference =
        malloc(hidden_elements * sizeof(float));

    if (!post_attn_norm_weight ||
        !post_attn_norm_output ||
        !post_attn_norm_reference ||
        !first_residual_output ||
        !first_residual_reference) {
        fprintf(stderr, "Post-attention allocation failed\n");
        return 1;
    }

    load_f32(
        EXPORT_DIR "/post_attention_norm_weight.bin",
        post_attn_norm_weight,
        HIDDEN_SIZE
    );

    load_f32(
        EXPORT_DIR "/post_attention_norm_output.bin",
        post_attn_norm_reference,
        hidden_elements
    );

    load_f32(
        EXPORT_DIR "/first_residual_output.bin",
        first_residual_reference,
        hidden_elements
    );

    printf("\nRunning post-attention RMSNorm...\n");

    rmsnorm(
        o_proj_output,
        post_attn_norm_weight,
        post_attn_norm_output,
        SEQ_LEN,
        HIDDEN_SIZE
    );

    double pan_total_error = 0.0;
    float pan_max_error = 0.0f;
    size_t pan_worst = 0;
    size_t pan_mismatches = 0;

    for (size_t i = 0; i < hidden_elements; i++) {
        float error =
            fabsf(post_attn_norm_output[i] -
                  post_attn_norm_reference[i]);

        pan_total_error += error;

        if (error != 0.0f)
            pan_mismatches++;

        if (error > pan_max_error) {
            pan_max_error = error;
            pan_worst = i;
        }
    }

    printf("\nPost-attention RMSNorm comparison\n");
    printf("Elements:       %zu\n", hidden_elements);
    printf("Mismatches:     %zu\n", pan_mismatches);
    printf("Max abs error:  %.9g\n", pan_max_error);
    printf("Mean abs error: %.9g\n",
           pan_total_error / hidden_elements);
    printf("Worst index:    %zu\n", pan_worst);
    printf("C output:       %.9g\n",
           post_attn_norm_output[pan_worst]);
    printf("PyTorch ref:    %.9g\n",
           post_attn_norm_reference[pan_worst]);

    printf("\nRunning first residual add...\n");

    residual_add(
        input,
        post_attn_norm_output,
        first_residual_output,
        hidden_elements
    );

    double res_total_error = 0.0;
    float res_max_error = 0.0f;
    size_t res_worst = 0;
    size_t res_mismatches = 0;

    for (size_t i = 0; i < hidden_elements; i++) {
        float error =
            fabsf(first_residual_output[i] -
                  first_residual_reference[i]);

        res_total_error += error;

        if (error != 0.0f)
            res_mismatches++;

        if (error > res_max_error) {
            res_max_error = error;
            res_worst = i;
        }
    }

    printf("\nFirst residual comparison\n");
    printf("Elements:       %zu\n", hidden_elements);
    printf("Mismatches:     %zu\n", res_mismatches);
    printf("Max abs error:  %.9g\n", res_max_error);
    printf("Mean abs error: %.9g\n",
           res_total_error / hidden_elements);
    printf("Worst index:    %zu\n", res_worst);
    printf("C output:       %.9g\n",
           first_residual_output[res_worst]);
    printf("PyTorch ref:    %.9g\n",
           first_residual_reference[res_worst]);


    /*
     * Pre-feedforward RMSNorm
     *
     * first_residual_output [14,1536]
     *          ↓
     * pre_feedforward_layernorm
     *          ↓
     * pre_ffn_norm_output [14,1536]
     */
    float *pre_ffn_norm_weight =
        malloc(HIDDEN_SIZE * sizeof(float));

    float *pre_ffn_norm_output =
        malloc(hidden_elements * sizeof(float));

    float *pre_ffn_norm_reference =
        malloc(hidden_elements * sizeof(float));

    if (!pre_ffn_norm_weight ||
        !pre_ffn_norm_output ||
        !pre_ffn_norm_reference) {
        fprintf(stderr, "Pre-FFN RMSNorm allocation failed\n");
        return 1;
    }

    load_f32(
        EXPORT_DIR "/pre_ffn_norm_weight.bin",
        pre_ffn_norm_weight,
        HIDDEN_SIZE
    );

    load_f32(
        EXPORT_DIR "/pre_ffn_norm_output.bin",
        pre_ffn_norm_reference,
        hidden_elements
    );

    printf("\nRunning pre-FFN RMSNorm...\n");

    rmsnorm(
        first_residual_output,
        pre_ffn_norm_weight,
        pre_ffn_norm_output,
        SEQ_LEN,
        HIDDEN_SIZE
    );

    double pfn_total_error = 0.0;
    float pfn_max_error = 0.0f;
    size_t pfn_worst = 0;
    size_t pfn_mismatches = 0;

    for (size_t i = 0; i < hidden_elements; i++) {
        float error =
            fabsf(pre_ffn_norm_output[i] -
                  pre_ffn_norm_reference[i]);

        pfn_total_error += error;

        if (error != 0.0f)
            pfn_mismatches++;

        if (error > pfn_max_error) {
            pfn_max_error = error;
            pfn_worst = i;
        }
    }

    printf("\nPre-FFN RMSNorm comparison\n");
    printf("Elements:       %zu\n", hidden_elements);
    printf("Mismatches:     %zu\n", pfn_mismatches);
    printf("Max abs error:  %.9g\n", pfn_max_error);
    printf("Mean abs error: %.9g\n",
           pfn_total_error / hidden_elements);
    printf("Worst index:    %zu\n", pfn_worst);
    printf("C output:       %.9g\n",
           pre_ffn_norm_output[pfn_worst]);
    printf("PyTorch ref:    %.9g\n",
           pre_ffn_norm_reference[pfn_worst]);


    /*
     * MLP gate and up projections
     *
     * pre_ffn_norm_output [14,1536]
     *
     * gate_proj -> [14,6144]
     * up_proj   -> [14,6144]
     */
    const int INTERMEDIATE_SIZE = 6144;

    size_t mlp_weight_elements =
        (size_t)INTERMEDIATE_SIZE * HIDDEN_SIZE;

    size_t mlp_elements =
        (size_t)SEQ_LEN * INTERMEDIATE_SIZE;

    float *gate_weight =
        malloc(mlp_weight_elements * sizeof(float));

    float *up_weight =
        malloc(mlp_weight_elements * sizeof(float));

    float *gate_output =
        malloc(mlp_elements * sizeof(float));

    float *up_output =
        malloc(mlp_elements * sizeof(float));

    float *gate_reference =
        malloc(mlp_elements * sizeof(float));

    float *up_reference =
        malloc(mlp_elements * sizeof(float));

    if (!gate_weight ||
        !up_weight ||
        !gate_output ||
        !up_output ||
        !gate_reference ||
        !up_reference) {
        fprintf(stderr, "MLP projection allocation failed\n");
        return 1;
    }

    load_f32(
        EXPORT_DIR "/gate_proj_weight.bin",
        gate_weight,
        mlp_weight_elements
    );

    load_f32(
        EXPORT_DIR "/up_proj_weight.bin",
        up_weight,
        mlp_weight_elements
    );

    load_f32(
        EXPORT_DIR "/gate_proj_output.bin",
        gate_reference,
        mlp_elements
    );

    load_f32(
        EXPORT_DIR "/up_proj_output.bin",
        up_reference,
        mlp_elements
    );

    printf("\nRunning MLP gate projection...\n");

    linear(
        pre_ffn_norm_output,
        gate_weight,
        gate_output,
        SEQ_LEN,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE
    );

    printf("Running MLP up projection...\n");

    linear(
        pre_ffn_norm_output,
        up_weight,
        up_output,
        SEQ_LEN,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE
    );

    double gate_total_error = 0.0;
    float gate_max_error = 0.0f;
    size_t gate_worst = 0;
    size_t gate_mismatches = 0;

    double up_total_error = 0.0;
    float up_max_error = 0.0f;
    size_t up_worst = 0;
    size_t up_mismatches = 0;

    for (size_t i = 0; i < mlp_elements; i++) {
        float error =
            fabsf(gate_output[i] - gate_reference[i]);

        gate_total_error += error;

        if (error != 0.0f)
            gate_mismatches++;

        if (error > gate_max_error) {
            gate_max_error = error;
            gate_worst = i;
        }

        error = fabsf(up_output[i] - up_reference[i]);

        up_total_error += error;

        if (error != 0.0f)
            up_mismatches++;

        if (error > up_max_error) {
            up_max_error = error;
            up_worst = i;
        }
    }

    printf("\nGate projection comparison\n");
    printf("Elements:       %zu\n", mlp_elements);
    printf("Mismatches:     %zu\n", gate_mismatches);
    printf("Max abs error:  %.9g\n", gate_max_error);
    printf("Mean abs error: %.9g\n",
           gate_total_error / mlp_elements);
    printf("Worst index:    %zu\n", gate_worst);
    printf("C output:       %.9g\n",
           gate_output[gate_worst]);
    printf("PyTorch ref:    %.9g\n",
           gate_reference[gate_worst]);

    printf("\nUp projection comparison\n");
    printf("Elements:       %zu\n", mlp_elements);
    printf("Mismatches:     %zu\n", up_mismatches);
    printf("Max abs error:  %.9g\n", up_max_error);
    printf("Mean abs error: %.9g\n",
           up_total_error / mlp_elements);
    printf("Worst index:    %zu\n", up_worst);
    printf("C output:       %.9g\n",
           up_output[up_worst]);
    printf("PyTorch ref:    %.9g\n",
           up_reference[up_worst]);


    /*
     * Gemma MLP gated activation:
     *
     * GELU_tanh(gate_proj_output) * up_proj_output
     *
     * Input/output: [14,6144]
     */
    float *gate_up_output =
        malloc(mlp_elements * sizeof(float));

    float *gate_up_reference =
        malloc(mlp_elements * sizeof(float));

    if (!gate_up_output || !gate_up_reference) {
        fprintf(stderr, "Gate x Up allocation failed\n");
        return 1;
    }

    load_f32(
        EXPORT_DIR "/gate_up_output.bin",
        gate_up_reference,
        mlp_elements
    );

    printf("\nRunning GELU(gate) x up...\n");

    const float gelu_coeff = 0.7978845608028654f;

    for (size_t i = 0; i < mlp_elements; i++) {
        float x = gate_output[i];

        float x3 = x * x * x;

        float inner =
            gelu_coeff * (x + 0.044715f * x3);

        float gelu =
            0.5f * x * (1.0f + tanhf(inner));

        gate_up_output[i] =
            round_to_bf16(gelu * up_output[i]);
    }

    double gu_total_error = 0.0;
    float gu_max_error = 0.0f;
    size_t gu_worst = 0;
    size_t gu_mismatches = 0;

    for (size_t i = 0; i < mlp_elements; i++) {
        float error =
            fabsf(gate_up_output[i] -
                  gate_up_reference[i]);

        gu_total_error += error;

        if (error != 0.0f)
            gu_mismatches++;

        if (error > gu_max_error) {
            gu_max_error = error;
            gu_worst = i;
        }
    }

    printf("\nGate x Up comparison\n");
    printf("Elements:       %zu\n", mlp_elements);
    printf("Mismatches:     %zu\n", gu_mismatches);
    printf("Max abs error:  %.9g\n", gu_max_error);
    printf("Mean abs error: %.9g\n",
           gu_total_error / mlp_elements);
    printf("Worst index:    %zu\n", gu_worst);
    printf("C output:       %.9g\n",
           gate_up_output[gu_worst]);
    printf("PyTorch ref:    %.9g\n",
           gate_up_reference[gu_worst]);


    /*
     * MLP down projection
     *
     * gate_up_output [14,6144]
     *        x
     * down_proj_weight [1536,6144]
     *        ↓
     * down_proj_output [14,1536]
     */
    size_t down_weight_elements =
        (size_t)HIDDEN_SIZE * INTERMEDIATE_SIZE;

    float *down_weight =
        malloc(down_weight_elements * sizeof(float));

    float *down_output =
        malloc(hidden_elements * sizeof(float));

    float *down_reference =
        malloc(hidden_elements * sizeof(float));

    if (!down_weight ||
        !down_output ||
        !down_reference) {
        fprintf(stderr, "Down projection allocation failed\n");
        return 1;
    }

    load_f32(
        EXPORT_DIR "/down_proj_weight.bin",
        down_weight,
        down_weight_elements
    );

    load_f32(
        EXPORT_DIR "/down_proj_output.bin",
        down_reference,
        hidden_elements
    );

    printf("\nRunning MLP down projection...\n");

    linear(
        gate_up_output,
        down_weight,
        down_output,
        SEQ_LEN,
        INTERMEDIATE_SIZE,
        HIDDEN_SIZE
    );

    double down_total_error = 0.0;
    float down_max_error = 0.0f;
    size_t down_worst = 0;
    size_t down_mismatches = 0;

    for (size_t i = 0; i < hidden_elements; i++) {
        float error =
            fabsf(down_output[i] -
                  down_reference[i]);

        down_total_error += error;

        if (error != 0.0f)
            down_mismatches++;

        if (error > down_max_error) {
            down_max_error = error;
            down_worst = i;
        }
    }

    printf("\nDown projection comparison\n");
    printf("Elements:       %zu\n", hidden_elements);
    printf("Mismatches:     %zu\n", down_mismatches);
    printf("Max abs error:  %.9g\n", down_max_error);
    printf("Mean abs error: %.9g\n",
           down_total_error / hidden_elements);
    printf("Worst index:    %zu\n", down_worst);
    printf("C output:       %.9g\n",
           down_output[down_worst]);
    printf("PyTorch ref:    %.9g\n",
           down_reference[down_worst]);


    /*
     * Post-feedforward RMSNorm + second residual
     *
     * down_output [14,1536]
     *       ↓ post_feedforward_layernorm
     * post_ffn_norm_output [14,1536]
     *       ↓ + first_residual_output
     * second_residual_output [14,1536]
     */
    float *post_ffn_norm_weight =
        malloc(HIDDEN_SIZE * sizeof(float));

    float *post_ffn_norm_output =
        malloc(hidden_elements * sizeof(float));

    float *post_ffn_norm_reference =
        malloc(hidden_elements * sizeof(float));

    float *second_residual_output =
        malloc(hidden_elements * sizeof(float));

    float *second_residual_reference =
        malloc(hidden_elements * sizeof(float));

    if (!post_ffn_norm_weight ||
        !post_ffn_norm_output ||
        !post_ffn_norm_reference ||
        !second_residual_output ||
        !second_residual_reference) {
        fprintf(stderr, "Post-FFN allocation failed\n");
        return 1;
    }

    load_f32(
        EXPORT_DIR "/post_ffn_norm_weight.bin",
        post_ffn_norm_weight,
        HIDDEN_SIZE
    );

    load_f32(
        EXPORT_DIR "/post_ffn_norm_output.bin",
        post_ffn_norm_reference,
        hidden_elements
    );

    load_f32(
        EXPORT_DIR "/second_residual_output.bin",
        second_residual_reference,
        hidden_elements
    );

    printf("\nRunning post-FFN RMSNorm...\n");

    rmsnorm(
        down_output,
        post_ffn_norm_weight,
        post_ffn_norm_output,
        SEQ_LEN,
        HIDDEN_SIZE
    );

    double postffn_total_error = 0.0;
    float postffn_max_error = 0.0f;
    size_t postffn_worst = 0;
    size_t postffn_mismatches = 0;

    for (size_t i = 0; i < hidden_elements; i++) {
        float error =
            fabsf(post_ffn_norm_output[i] -
                  post_ffn_norm_reference[i]);

        postffn_total_error += error;

        if (error != 0.0f)
            postffn_mismatches++;

        if (error > postffn_max_error) {
            postffn_max_error = error;
            postffn_worst = i;
        }
    }

    printf("\nPost-FFN RMSNorm comparison\n");
    printf("Elements:       %zu\n", hidden_elements);
    printf("Mismatches:     %zu\n", postffn_mismatches);
    printf("Max abs error:  %.9g\n", postffn_max_error);
    printf("Mean abs error: %.9g\n",
           postffn_total_error / hidden_elements);
    printf("Worst index:    %zu\n", postffn_worst);
    printf("C output:       %.9g\n",
           post_ffn_norm_output[postffn_worst]);
    printf("PyTorch ref:    %.9g\n",
           post_ffn_norm_reference[postffn_worst]);

    printf("\nRunning second residual add...\n");

    residual_add(
        first_residual_output,
        post_ffn_norm_output,
        second_residual_output,
        hidden_elements
    );

    double res2_total_error = 0.0;
    float res2_max_error = 0.0f;
    size_t res2_worst = 0;
    size_t res2_mismatches = 0;

    for (size_t i = 0; i < hidden_elements; i++) {
        float error =
            fabsf(second_residual_output[i] -
                  second_residual_reference[i]);

        res2_total_error += error;

        if (error != 0.0f)
            res2_mismatches++;

        if (error > res2_max_error) {
            res2_max_error = error;
            res2_worst = i;
        }
    }

    printf("\nSecond residual comparison\n");
    printf("Elements:       %zu\n", hidden_elements);
    printf("Mismatches:     %zu\n", res2_mismatches);
    printf("Max abs error:  %.9g\n", res2_max_error);
    printf("Mean abs error: %.9g\n",
           res2_total_error / hidden_elements);
    printf("Worst index:    %zu\n", res2_worst);
    printf("C output:       %.9g\n",
           second_residual_output[res2_worst]);
    printf("PyTorch ref:    %.9g\n",
           second_residual_reference[res2_worst]);

    free(post_ffn_norm_weight);
    free(post_ffn_norm_output);
    free(post_ffn_norm_reference);

    /*
     * Gemma 4 per-layer-input gate
     *
     * second_residual_output [14,1536]
     *             x
     * per_layer_gate_weight [256,1536]
     *             ↓
     * per_layer_gate_output [14,256]
     */
    size_t per_layer_elements =
        (size_t)SEQ_LEN * KV_DIM;

    size_t per_layer_gate_weight_elements =
        (size_t)KV_DIM * HIDDEN_SIZE;

    float *per_layer_gate_weight =
        malloc(per_layer_gate_weight_elements * sizeof(float));

    float *per_layer_gate_output =
        malloc(per_layer_elements * sizeof(float));

    float *per_layer_gate_reference =
        malloc(per_layer_elements * sizeof(float));

    if (!per_layer_gate_weight ||
        !per_layer_gate_output ||
        !per_layer_gate_reference) {
        fprintf(stderr, "Per-layer gate allocation failed\n");
        return 1;
    }

    load_f32(
        EXPORT_DIR "/per_layer_gate_weight.bin",
        per_layer_gate_weight,
        per_layer_gate_weight_elements
    );

    load_f32(
        EXPORT_DIR "/per_layer_gate_output.bin",
        per_layer_gate_reference,
        per_layer_elements
    );

    printf("\nRunning per-layer input gate projection...\n");

    linear(
        second_residual_output,
        per_layer_gate_weight,
        per_layer_gate_output,
        SEQ_LEN,
        HIDDEN_SIZE,
        KV_DIM
    );

    double plg_total_error = 0.0;
    float plg_max_error = 0.0f;
    size_t plg_worst = 0;
    size_t plg_mismatches = 0;

    for (size_t i = 0; i < per_layer_elements; i++) {
        float error =
            fabsf(per_layer_gate_output[i] -
                  per_layer_gate_reference[i]);

        plg_total_error += error;

        if (error != 0.0f)
            plg_mismatches++;

        if (error > plg_max_error) {
            plg_max_error = error;
            plg_worst = i;
        }
    }

    printf("\nPer-layer gate comparison\n");
    printf("Elements:       %zu\n", per_layer_elements);
    printf("Mismatches:     %zu\n", plg_mismatches);
    printf("Max abs error:  %.9g\n", plg_max_error);
    printf("Mean abs error: %.9g\n",
           plg_total_error / per_layer_elements);
    printf("Worst index:    %zu\n", plg_worst);
    printf("C output:       %.9g\n",
           per_layer_gate_output[plg_worst]);
    printf("PyTorch ref:    %.9g\n",
           per_layer_gate_reference[plg_worst]);


    /*
     * Gemma 4 per-layer product:
     *
     * GELU_tanh(per_layer_gate_output) * per_layer_input
     *
     * All tensors: [14,256]
     */
    float *per_layer_input =
        malloc(per_layer_elements * sizeof(float));

    float *per_layer_product =
        malloc(per_layer_elements * sizeof(float));

    float *per_layer_product_reference =
        malloc(per_layer_elements * sizeof(float));

    if (!per_layer_input ||
        !per_layer_product ||
        !per_layer_product_reference) {
        fprintf(stderr, "Per-layer product allocation failed\n");
        return 1;
    }

    load_f32(
        EXPORT_DIR "/per_layer_input.bin",
        per_layer_input,
        per_layer_elements
    );

    load_f32(
        EXPORT_DIR "/per_layer_product.bin",
        per_layer_product_reference,
        per_layer_elements
    );

    printf("\nRunning GELU(per-layer gate) x per-layer input...\n");

    const float per_layer_gelu_coeff =
        0.7978845608028654f;

    for (size_t i = 0; i < per_layer_elements; i++) {
        float x = per_layer_gate_output[i];

        float x3 = x * x * x;

        float inner =
            per_layer_gelu_coeff *
            (x + 0.044715f * x3);

        float gelu =
            0.5f * x * (1.0f + tanhf(inner));

        per_layer_product[i] =
            round_to_bf16(
                gelu * per_layer_input[i]
            );
    }

    double plp_total_error = 0.0;
    float plp_max_error = 0.0f;
    size_t plp_worst = 0;
    size_t plp_mismatches = 0;

    for (size_t i = 0; i < per_layer_elements; i++) {
        float error =
            fabsf(per_layer_product[i] -
                  per_layer_product_reference[i]);

        plp_total_error += error;

        if (error != 0.0f)
            plp_mismatches++;

        if (error > plp_max_error) {
            plp_max_error = error;
            plp_worst = i;
        }
    }

    printf("\nPer-layer product comparison\n");
    printf("Elements:       %zu\n", per_layer_elements);
    printf("Mismatches:     %zu\n", plp_mismatches);
    printf("Max abs error:  %.9g\n", plp_max_error);
    printf("Mean abs error: %.9g\n",
           plp_total_error / per_layer_elements);
    printf("Worst index:    %zu\n", plp_worst);
    printf("C output:       %.9g\n",
           per_layer_product[plp_worst]);
    printf("PyTorch ref:    %.9g\n",
           per_layer_product_reference[plp_worst]);


    /*
     * Gemma 4 per-layer projection
     *
     * per_layer_product [14,256]
     *          x
     * weight [1536,256]
     *          ↓
     * output [14,1536]
     */
    size_t per_layer_proj_weight_elements =
        (size_t)HIDDEN_SIZE * KV_DIM;

    float *per_layer_proj_weight =
        malloc(per_layer_proj_weight_elements * sizeof(float));

    float *per_layer_proj_output =
        malloc(hidden_elements * sizeof(float));

    float *per_layer_proj_reference =
        malloc(hidden_elements * sizeof(float));

    if (!per_layer_proj_weight ||
        !per_layer_proj_output ||
        !per_layer_proj_reference) {
        fprintf(stderr, "Per-layer projection allocation failed\n");
        return 1;
    }

    load_f32(
        EXPORT_DIR "/per_layer_projection_weight.bin",
        per_layer_proj_weight,
        per_layer_proj_weight_elements
    );

    load_f32(
        EXPORT_DIR "/per_layer_projection_output.bin",
        per_layer_proj_reference,
        hidden_elements
    );

    printf("\nRunning per-layer projection...\n");

    linear(
        per_layer_product,
        per_layer_proj_weight,
        per_layer_proj_output,
        SEQ_LEN,
        KV_DIM,
        HIDDEN_SIZE
    );

    double plproj_total_error = 0.0;
    float plproj_max_error = 0.0f;
    size_t plproj_worst = 0;
    size_t plproj_mismatches = 0;

    for (size_t i = 0; i < hidden_elements; i++) {
        float error =
            fabsf(per_layer_proj_output[i] -
                  per_layer_proj_reference[i]);

        plproj_total_error += error;

        if (error != 0.0f)
            plproj_mismatches++;

        if (error > plproj_max_error) {
            plproj_max_error = error;
            plproj_worst = i;
        }
    }

    printf("\nPer-layer projection comparison\n");
    printf("Elements:       %zu\n", hidden_elements);
    printf("Mismatches:     %zu\n", plproj_mismatches);
    printf("Max abs error:  %.9g\n", plproj_max_error);
    printf("Mean abs error: %.9g\n",
           plproj_total_error / hidden_elements);
    printf("Worst index:    %zu\n", plproj_worst);
    printf("C output:       %.9g\n",
           per_layer_proj_output[plproj_worst]);
    printf("PyTorch ref:    %.9g\n",
           per_layer_proj_reference[plproj_worst]);


    /*
     * Final Gemma 4 decoder tail:
     *
     * per_layer_proj_output
     *       ↓ RMSNorm
     * + second_residual_output
     *       ↓
     * × layer_scalar
     *       ↓
     * final layer output
     */

    float *post_per_layer_norm_weight =
        malloc(HIDDEN_SIZE * sizeof(float));

    float *post_per_layer_norm_output =
        malloc(hidden_elements * sizeof(float));

    float *post_per_layer_norm_reference =
        malloc(hidden_elements * sizeof(float));

    float *pre_scalar_output =
        malloc(hidden_elements * sizeof(float));

    float *final_output =
        malloc(hidden_elements * sizeof(float));

    float *final_reference =
        malloc(hidden_elements * sizeof(float));

    float layer_scalar = 0.0f;

    if (!post_per_layer_norm_weight ||
        !post_per_layer_norm_output ||
        !post_per_layer_norm_reference ||
        !pre_scalar_output ||
        !final_output ||
        !final_reference) {
        fprintf(stderr, "Final decoder tail allocation failed\n");
        return 1;
    }

    load_f32(
        EXPORT_DIR "/post_per_layer_input_norm_weight.bin",
        post_per_layer_norm_weight,
        HIDDEN_SIZE
    );

    load_f32(
        EXPORT_DIR "/post_per_layer_norm_output.bin",
        post_per_layer_norm_reference,
        hidden_elements
    );

    load_f32(
        EXPORT_DIR "/layer_scalar.bin",
        &layer_scalar,
        1
    );

    load_f32(
        EXPORT_DIR "/layer_output.bin",
        final_reference,
        hidden_elements
    );

    printf("\nRunning post-per-layer RMSNorm...\n");

    rmsnorm(
        per_layer_proj_output,
        post_per_layer_norm_weight,
        post_per_layer_norm_output,
        SEQ_LEN,
        HIDDEN_SIZE
    );

    double plnorm_total_error = 0.0;
    float plnorm_max_error = 0.0f;
    size_t plnorm_worst = 0;
    size_t plnorm_mismatches = 0;

    for (size_t i = 0; i < hidden_elements; i++) {
        float error =
            fabsf(post_per_layer_norm_output[i] -
                  post_per_layer_norm_reference[i]);

        plnorm_total_error += error;

        if (error != 0.0f)
            plnorm_mismatches++;

        if (error > plnorm_max_error) {
            plnorm_max_error = error;
            plnorm_worst = i;
        }
    }

    printf("\nPost-per-layer RMSNorm comparison\n");
    printf("Elements:       %zu\n", hidden_elements);
    printf("Mismatches:     %zu\n", plnorm_mismatches);
    printf("Max abs error:  %.9g\n", plnorm_max_error);
    printf("Mean abs error: %.9g\n",
           plnorm_total_error / hidden_elements);
    printf("Worst index:    %zu\n", plnorm_worst);
    printf("C output:       %.9g\n",
           post_per_layer_norm_output[plnorm_worst]);
    printf("PyTorch ref:    %.9g\n",
           post_per_layer_norm_reference[plnorm_worst]);

    /*
     * Residual after the per-layer-input block.
     */
    residual_add(
        second_residual_output,
        post_per_layer_norm_output,
        pre_scalar_output,
        hidden_elements
    );

    printf("\nLayer scalar: %.9g\n", layer_scalar);
    printf("Running final layer scalar multiply...\n");

    /*
     * Gemma forward:
     * hidden_states *= self.layer_scalar
     *
     * hidden_states is BF16 in PyTorch, so round the result
     * back to BF16.
     */
    for (size_t i = 0; i < hidden_elements; i++) {
        final_output[i] =
            round_to_bf16(
                pre_scalar_output[i] * layer_scalar
            );
    }

    double final_total_error = 0.0;
    float final_max_error = 0.0f;
    size_t final_worst = 0;
    size_t final_mismatches = 0;

    double final_checksum = 0.0;
    double reference_checksum = 0.0;

    for (size_t i = 0; i < hidden_elements; i++) {
        float error =
            fabsf(final_output[i] -
                  final_reference[i]);

        final_total_error += error;
        final_checksum += final_output[i];
        reference_checksum += final_reference[i];

        if (error != 0.0f)
            final_mismatches++;

        if (error > final_max_error) {
            final_max_error = error;
            final_worst = i;
        }
    }

    printf("\n========================================\n");
    printf("FINAL DECODER LAYER COMPARISON\n");
    printf("========================================\n");
    printf("Elements:          %zu\n", hidden_elements);
    printf("Mismatches:        %zu\n", final_mismatches);
    printf("Max abs error:     %.9g\n", final_max_error);
    printf("Mean abs error:    %.9g\n",
           final_total_error / hidden_elements);
    printf("Worst index:       %zu\n", final_worst);
    printf("C output:          %.9g\n",
           final_output[final_worst]);
    printf("PyTorch ref:       %.9g\n",
           final_reference[final_worst]);
    printf("C checksum:        %.12g\n", final_checksum);
    printf("PyTorch checksum:  %.12g\n", reference_checksum);
    printf("========================================\n");

    const float FINAL_TOL = 0.5f;

    if (final_max_error <= FINAL_TOL) {
        printf("CPU DECODER BASELINE: PASS\n");
    } else {
        printf("CPU DECODER BASELINE: FAIL\n");
    }

    free(post_per_layer_norm_weight);
    free(post_per_layer_norm_output);
    free(post_per_layer_norm_reference);
    free(pre_scalar_output);
    free(final_output);
    free(final_reference);

    free(second_residual_output);

    free(per_layer_proj_weight);
    free(per_layer_proj_output);
    free(per_layer_proj_reference);

    free(per_layer_input);
    free(per_layer_product);
    free(per_layer_product_reference);

    free(per_layer_gate_weight);
    free(per_layer_gate_output);
    free(per_layer_gate_reference);

    /* Keep second_residual_output alive for final per-layer residual. */
    free(second_residual_reference);

    free(down_weight);
    free(down_output);
    free(down_reference);

    free(gate_up_output);
    free(gate_up_reference);

    free(gate_weight);
    free(up_weight);
    free(gate_output);
    free(up_output);
    free(gate_reference);
    free(up_reference);

    free(pre_ffn_norm_weight);
    free(pre_ffn_norm_output);
    free(pre_ffn_norm_reference);

    free(post_attn_norm_weight);
    free(post_attn_norm_output);
    free(post_attn_norm_reference);
    free(first_residual_output);
    free(first_residual_reference);

    free(o_proj_weight);
    free(o_proj_output);
    free(o_proj_reference);

    free(attention_output);
    free(attention_output_reference);

    free(attention_mask);
    free(attention_weights);
    free(attention_weights_reference);

    free(attention_scores);
    free(attention_scores_reference);

    free(rope_cos);
    free(rope_sin);
    free(q_rope_output);
    free(k_rope_output);
    free(q_rope_reference);
    free(k_rope_reference);

    free(q_norm_weight);
    free(k_norm_weight);

    free(q_norm_output);
    free(k_norm_output);
    free(v_norm_output);

    free(q_norm_reference);
    free(k_norm_reference);
    free(v_norm_reference);

    free(q_weight);
    free(k_weight);
    free(v_weight);

    free(q_output);
    free(k_output);
    free(v_output);

    free(q_reference);
    free(k_reference);
    free(v_reference);

    free(input);
    free(weight);
    free(output);
    free(reference);

    return 0;
}
