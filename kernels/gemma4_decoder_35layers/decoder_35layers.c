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



#define NUM_LAYERS 35
#define NUM_HEADS 8
#define PER_LAYER_DIM 256
#define FFN_DIM 6144

#define WEIGHTS_DIR "/scratch/aparna/gemma4_35layers/weights"

typedef struct {
    int layer_idx;
    int head_dim;
    int q_dim;
    int kv_dim;
    int shared_kv;
    int full_attention;
} LayerConfig;

static LayerConfig get_layer_config(int layer_idx)
{
    LayerConfig cfg;

    cfg.layer_idx = layer_idx;

    /* Full attention at layers 4, 9, 14, 19, 24, 29, 34. */
    cfg.full_attention = (layer_idx % 5 == 4);

    cfg.head_dim = cfg.full_attention ? 512 : 256;
    cfg.q_dim = NUM_HEADS * cfg.head_dim;
    cfg.kv_dim = cfg.head_dim;

    /* Layers 15-34 reuse K/V from earlier layers. */
    cfg.shared_kv = (layer_idx >= 15);

    return cfg;
}

/* 35-layer decoder implementation */


static float *load_weight(
    int layer_idx,
    const char *name,
    size_t count)
{
    char path[512];

    snprintf(
        path,
        sizeof(path),
        "%s/layer_%02d/%s.bin",
        WEIGHTS_DIR,
        layer_idx,
        name
    );

    float *weight = malloc(count * sizeof(float));

    if (!weight) {
        fprintf(stderr, "Allocation failed: %s\n", name);
        exit(1);
    }

    load_f32(path, weight, count);
    return weight;
}

/*
 * Initial decoder-layer stages:
 *   input -> RMSNorm -> Q/K/V projections
 *
 * This function currently tests the projection stages only.
 * It does NOT yet produce a complete decoder-layer output.
 */

/* Reusable 35-layer decoder implementation goes here. */

#define DATA_ROOT "/scratch/aparna/gemma4_35layers"

typedef struct {
    float *k;
    float *v;
} SharedKV;

static float *alloc_floats(size_t count)
{
    float *p = malloc(count * sizeof(float));
    if (!p) {
        fprintf(stderr, "Failed to allocate %zu floats\n", count);
        exit(1);
    }
    return p;
}

static float *load_data(const char *path, size_t count)
{
    float *p = alloc_floats(count);
    load_f32(path, p, count);
    return p;
}

static float *load_layer_data(
    const char *directory,
    const char *name,
    int layer,
    size_t count)
{
    char path[512];
    snprintf(path, sizeof(path),
             DATA_ROOT "/%s/%s_%02d.bin",
             directory, name, layer);
    return load_data(path, count);
}

static void check_layer_output(int layer, const float *output)
{
    size_t n = (size_t)SEQ_LEN * HIDDEN_SIZE;

    float *ref = load_layer_data(
        "reference", "layer_output", layer, n);

    double error_sum = 0.0;
    double checksum = 0.0;
    float max_error = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float error = fabsf(output[i] - ref[i]);
        error_sum += error;
        checksum += output[i];
        if (error > max_error)
            max_error = error;
    }

    printf("Layer %02d: checksum=%.9g "
           "max_error=%.9g mean_error=%.9g\n",
           layer, checksum, max_error, error_sum / n);
    fflush(stdout);

    free(ref);
}

static float *run_layer(
    int layer,
    const float *hidden,
    SharedKV shared[2])
{
    LayerConfig cfg = get_layer_config(layer);

    size_t hn = (size_t)SEQ_LEN * HIDDEN_SIZE;
    size_t qn = (size_t)SEQ_LEN * cfg.q_dim;
    size_t kn = (size_t)SEQ_LEN * cfg.kv_dim;
    int ffn_dim = (layer >= 15) ? 2 * FFN_DIM : FFN_DIM;
    size_t mn = (size_t)SEQ_LEN * ffn_dim;
    size_t pn = (size_t)SEQ_LEN * PER_LAYER_DIM;
    size_t sn = (size_t)NUM_HEADS * SEQ_LEN * SEQ_LEN;

    float *w, *a, *b, *c;

    /* Input normalization */
    w = load_weight(layer, "input_norm_weight", HIDDEN_SIZE);
    a = alloc_floats(hn);
    rmsnorm(hidden, w, a, SEQ_LEN, HIDDEN_SIZE);
    free(w);

    /* Query projection and normalization */
    w = load_weight(layer, "q_proj_weight",
                    (size_t)cfg.q_dim * HIDDEN_SIZE);
    b = alloc_floats(qn);
    linear(a, w, b, SEQ_LEN, HIDDEN_SIZE, cfg.q_dim);
    free(w);

    w = load_weight(layer, "q_norm_weight", cfg.head_dim);
    c = alloc_floats(qn);
    rmsnorm(b, w, c, SEQ_LEN * NUM_HEADS, cfg.head_dim);
    free(w);
    free(b);

    float *cos = load_layer_data(
        "attention", "rope_cos", layer, kn);
    float *sin = load_layer_data(
        "attention", "rope_sin", layer, kn);

    float *q = alloc_floats(qn);
    rope_and_transpose(c, cos, sin, q,
                       SEQ_LEN, NUM_HEADS, cfg.head_dim);
    free(c);

    int type = cfg.full_attention ? 1 : 0;
    float *k = NULL;
    float *v = NULL;
    int owns_kv = !cfg.shared_kv;

    if (owns_kv) {
        w = load_weight(layer, "k_proj_weight",
                        (size_t)cfg.kv_dim * HIDDEN_SIZE);
        b = alloc_floats(kn);
        linear(a, w, b, SEQ_LEN, HIDDEN_SIZE, cfg.kv_dim);
        free(w);

        w = load_weight(layer, "k_norm_weight", cfg.head_dim);
        c = alloc_floats(kn);
        rmsnorm(b, w, c, SEQ_LEN, cfg.head_dim);
        free(w);
        free(b);

        k = alloc_floats(kn);
        rope_and_transpose(c, cos, sin, k,
                           SEQ_LEN, 1, cfg.head_dim);
        free(c);

        w = load_weight(layer, "v_proj_weight",
                        (size_t)cfg.kv_dim * HIDDEN_SIZE);
        b = alloc_floats(kn);
        linear(a, w, b, SEQ_LEN, HIDDEN_SIZE, cfg.kv_dim);
        free(w);

        v = alloc_floats(kn);
        rmsnorm(b, NULL, v, SEQ_LEN, cfg.head_dim);
        free(b);

        /* Last independent K/V for each attention type */
        if (layer == 13 || layer == 14) {
            shared[type].k = k;
            shared[type].v = v;
            owns_kv = 0;
        }
    } else {
        if (!shared[type].k || !shared[type].v) {
            fprintf(stderr, "Missing shared KV for layer %d\n", layer);
            exit(1);
        }
        k = shared[type].k;
        v = shared[type].v;
    }

    free(a);
    free(cos);
    free(sin);

    /* Attention */
    float *scores = alloc_floats(sn);
    float *mask = load_layer_data(
        "attention", "attention_mask", layer,
        (size_t)SEQ_LEN * SEQ_LEN);
    float *weights = alloc_floats(sn);

    attention_qk(q, k, scores, NUM_HEADS,
                 SEQ_LEN, cfg.head_dim);
    attention_softmax(scores, mask, weights,
                      NUM_HEADS, SEQ_LEN);

    b = alloc_floats(qn);
    attention_times_v(weights, v, b,
                      NUM_HEADS, SEQ_LEN, cfg.head_dim);

    free(q);
    free(scores);
    free(mask);
    free(weights);

    if (owns_kv) {
        free(k);
        free(v);
    }

    /* Output projection */
    w = load_weight(layer, "o_proj_weight",
                    (size_t)HIDDEN_SIZE * cfg.q_dim);
    a = alloc_floats(hn);
    linear(b, w, a, SEQ_LEN, cfg.q_dim, HIDDEN_SIZE);
    free(w);
    free(b);

    /* Post-attention normalization and residual */
    w = load_weight(layer, "post_attention_norm_weight",
                    HIDDEN_SIZE);
    b = alloc_floats(hn);
    rmsnorm(a, w, b, SEQ_LEN, HIDDEN_SIZE);
    free(w);
    free(a);

    float *residual1 = alloc_floats(hn);
    residual_add(hidden, b, residual1, hn);
    free(b);

    /* Pre-FFN normalization */
    w = load_weight(layer, "pre_ffn_norm_weight", HIDDEN_SIZE);
    a = alloc_floats(hn);
    rmsnorm(residual1, w, a, SEQ_LEN, HIDDEN_SIZE);
    free(w);

    /* Gate projection */
    w = load_weight(layer, "gate_proj_weight",
                    (size_t)ffn_dim * HIDDEN_SIZE);
    b = alloc_floats(mn);
    linear(a, w, b, SEQ_LEN, HIDDEN_SIZE, ffn_dim);
    free(w);

    /* Up projection */
    w = load_weight(layer, "up_proj_weight",
                    (size_t)ffn_dim * HIDDEN_SIZE);
    c = alloc_floats(mn);
    linear(a, w, c, SEQ_LEN, HIDDEN_SIZE, ffn_dim);
    free(w);
    free(a);

    /* GELU(gate) * up */
    const float gelu_coeff = 0.7978845608028654f;

    for (size_t i = 0; i < mn; i++) {
        float x = b[i];
        float x3 = x * x * x;
        float inner = gelu_coeff * (x + 0.044715f * x3);
        float gelu = 0.5f * x * (1.0f + tanhf(inner));
        b[i] = round_to_bf16(gelu * c[i]);
    }
    free(c);

    /* Down projection */
    w = load_weight(layer, "down_proj_weight",
                    (size_t)HIDDEN_SIZE * ffn_dim);
    a = alloc_floats(hn);
    linear(b, w, a, SEQ_LEN, ffn_dim, HIDDEN_SIZE);
    free(w);
    free(b);

    /* Post-FFN normalization and residual */
    w = load_weight(layer, "post_ffn_norm_weight", HIDDEN_SIZE);
    b = alloc_floats(hn);
    rmsnorm(a, w, b, SEQ_LEN, HIDDEN_SIZE);
    free(w);
    free(a);

    float *residual2 = alloc_floats(hn);
    residual_add(residual1, b, residual2, hn);
    free(residual1);
    free(b);

    /* Per-layer input gate */
    w = load_weight(layer, "per_layer_gate_weight",
                    (size_t)PER_LAYER_DIM * HIDDEN_SIZE);
    a = alloc_floats(pn);
    linear(residual2, w, a, SEQ_LEN,
           HIDDEN_SIZE, PER_LAYER_DIM);
    free(w);

    b = load_layer_data(
        "inputs", "per_layer_input", layer, pn);

    for (size_t i = 0; i < pn; i++) {
        float x = a[i];
        float x3 = x * x * x;
        float inner = gelu_coeff * (x + 0.044715f * x3);
        float gelu = 0.5f * x * (1.0f + tanhf(inner));
        a[i] = round_to_bf16(gelu * b[i]);
    }
    free(b);

    /* Per-layer projection */
    w = load_weight(layer, "per_layer_projection_weight",
                    (size_t)HIDDEN_SIZE * PER_LAYER_DIM);
    b = alloc_floats(hn);
    linear(a, w, b, SEQ_LEN,
           PER_LAYER_DIM, HIDDEN_SIZE);
    free(w);
    free(a);

    w = load_weight(layer, "post_per_layer_input_norm_weight",
                    HIDDEN_SIZE);
    a = alloc_floats(hn);
    rmsnorm(b, w, a, SEQ_LEN, HIDDEN_SIZE);
    free(w);
    free(b);

    /* Final residual and layer scalar */
    float *output = alloc_floats(hn);
    residual_add(residual2, a, output, hn);
    free(residual2);
    free(a);

    w = load_weight(layer, "layer_scalar", 1);
    for (size_t i = 0; i < hn; i++)
        output[i] = round_to_bf16(output[i] * w[0]);
    free(w);

    check_layer_output(layer, output);
    return output;
}

int main(int argc, char **argv)
{
    int layers_to_run = 2;

    if (argc > 1)
        layers_to_run = atoi(argv[1]);

    if (layers_to_run < 1 || layers_to_run > NUM_LAYERS) {
        fprintf(stderr, "Expected layer count 1..35\n");
        return 1;
    }

    size_t n = (size_t)SEQ_LEN * HIDDEN_SIZE;

    float *hidden = load_data(
        EXPORT_DIR "/hidden_states.bin", n);

    SharedKV shared[2] = {{NULL, NULL}, {NULL, NULL}};

    for (int i = 0; i < layers_to_run; i++) {
        printf("Running decoder layer %02d\n", i);
        fflush(stdout);

        if (i == 15 && getenv("USE_REFERENCE_LAYER14")) {
            free(hidden);
            hidden = load_layer_data(
                "reference", "layer_output", 14, n);
            printf("Using exact PyTorch layer-14 output\n");
        }

        float *next = run_layer(i, hidden, shared);
        free(hidden);
        hidden = next;
    }

    printf("Completed %d connected decoder layers\n",
           layers_to_run);

    free(hidden);

    for (int i = 0; i < 2; i++) {
        free(shared[i].k);
        free(shared[i].v);
    }

    return 0;
}

#define DATA_ROOT "/scratch/aparna/gemma4_35layers"
