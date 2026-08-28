#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#include "include/gemmini_testutils.h"

#define SEQ_LEN      14
#define HIDDEN_SIZE  1536
#define N            (SEQ_LEN * HIDDEN_SIZE)

static elem_t input_a[N] __attribute__((aligned(64)));
static elem_t input_b[N] __attribute__((aligned(64)));
static elem_t output[N] __attribute__((aligned(64)));
static elem_t expected[N] __attribute__((aligned(64)));

static void initialize_inputs(void) {
    for (size_t i = 0; i < N; i++) {
        // Range [-8, 8]
        input_a[i] = (elem_t)((int)(i % 17) - 8);

        // Range [-6, 6]
        input_b[i] = (elem_t)((int)((i * 3) % 13) - 6);

        // Sum remains safely inside INT8 range
        expected[i] =
            (elem_t)((int)input_a[i] + (int)input_b[i]);
    }
}

static void residual_add_cpu(
    const elem_t *a,
    const elem_t *b,
    elem_t *out,
    size_t n
) {
    for (size_t i = 0; i < n; i++) {
        out[i] =
            (elem_t)((int)a[i] + (int)b[i]);
    }
}

static unsigned long verify_output(void) {
    unsigned long mismatches = 0;

    for (size_t i = 0; i < N; i++) {
        if (output[i] != expected[i]) {
            if (mismatches < 10) {
                printf(
                    "Mismatch index=%lu: "
                    "a=%d b=%d output=%d expected=%d\n",
                    (unsigned long)i,
                    (int)input_a[i],
                    (int)input_b[i],
                    (int)output[i],
                    (int)expected[i]
                );
            }

            mismatches++;
        }
    }

    return mismatches;
}

static long checksum(const elem_t *x) {
    long sum = 0;

    for (size_t i = 0; i < N; i++) {
        sum += (int)x[i];
    }

    return sum;
}

int main(void) {
    printf("Gemma 4 CPU residual-add primitive\n\n");

    printf(
        "Tensor shape: [%d, %d]\n",
        SEQ_LEN,
        HIDDEN_SIZE
    );

    printf(
        "Total elements: %d\n\n",
        N
    );

    initialize_inputs();

    printf("Starting CPU residual add\n");

    unsigned long start = read_cycles();

    residual_add_cpu(
        input_a,
        input_b,
        output,
        N
    );

    unsigned long end = read_cycles();

    printf(
        "CPU residual-add cycles: %lu\n\n",
        end - start
    );

    unsigned long mismatches =
        verify_output();

    printf(
        "Mismatch count: %lu\n",
        mismatches
    );

    printf(
        "Output checksum: %ld\n",
        checksum(output)
    );

    printf("\nSample values\n");

    for (size_t i = 0; i < 8; i++) {
        printf(
            "[%lu] %d + %d = %d\n",
            (unsigned long)i,
            (int)input_a[i],
            (int)input_b[i],
            (int)output[i]
        );
    }

    if (mismatches == 0) {
        printf(
            "\nPASS: CPU residual-add output "
            "matches reference.\n"
        );

        return 0;
    }

    printf(
        "\nFAIL: CPU residual-add mismatch detected.\n"
    );

    return 1;
}
