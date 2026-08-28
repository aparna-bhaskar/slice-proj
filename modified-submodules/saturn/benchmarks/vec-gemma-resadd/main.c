#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <riscv_vector.h>

#include "util.h"

#define SEQ_LEN      14
#define HIDDEN_SIZE  1536
#define N            (SEQ_LEN * HIDDEN_SIZE)


// Same tensor size as our CPU residual-add test:
// [14, 1536] = 21,504 elements

static int8_t input_a[N] __attribute__((aligned(64)));
static int8_t input_b[N] __attribute__((aligned(64)));

static int8_t scalar_output[N] __attribute__((aligned(64)));
static int8_t saturn_output[N] __attribute__((aligned(64)));
static int8_t expected[N] __attribute__((aligned(64)));


// -----------------------------------------------------------------------------
// Same deterministic inputs as our CPU test
// -----------------------------------------------------------------------------

static void initialize_inputs(void) {
    for (size_t i = 0; i < N; i++) {

        // Range: [-8, 8]
        input_a[i] =
            (int8_t)((int)(i % 17) - 8);

        // Range: [-6, 6]
        input_b[i] =
            (int8_t)((int)((i * 3) % 13) - 6);

        // Sum stays safely inside INT8 range
        expected[i] =
            (int8_t)(
                (int)input_a[i] +
                (int)input_b[i]
            );
    }
}


// -----------------------------------------------------------------------------
// Scalar CPU reference
// -----------------------------------------------------------------------------

__attribute__((noinline))
static void residual_add_scalar(
    const int8_t *a,
    const int8_t *b,
    int8_t *out,
    size_t n
) {
    for (size_t i = 0; i < n; i++) {
        out[i] =
            (int8_t)(
                (int)a[i] +
                (int)b[i]
            );
    }
}


// -----------------------------------------------------------------------------
// Saturn / RISC-V Vector implementation
// -----------------------------------------------------------------------------

__attribute__((noinline))
static void residual_add_saturn(
    const int8_t *a,
    const int8_t *b,
    int8_t *out,
    size_t n
) {
    size_t i = 0;

    while (i < n) {

        // Ask Saturn/RVV how many int8 elements it can process
        // in this vector iteration.
        size_t vl =
            __riscv_vsetvl_e8m8(n - i);

        // Vector load A
        vint8m8_t va =
            __riscv_vle8_v_i8m8(
                &a[i],
                vl
            );

        // Vector load B
        vint8m8_t vb =
            __riscv_vle8_v_i8m8(
                &b[i],
                vl
            );

        // Vector elementwise addition
        vint8m8_t vc =
            __riscv_vadd_vv_i8m8(
                va,
                vb,
                vl
            );

        // Store result
        __riscv_vse8_v_i8m8(
            &out[i],
            vc,
            vl
        );

        i += vl;
    }
}


// -----------------------------------------------------------------------------
// Verification
// -----------------------------------------------------------------------------

static unsigned long compare_output(
    const int8_t *actual,
    const int8_t *reference,
    const char *name
) {
    unsigned long mismatches = 0;

    for (size_t i = 0; i < N; i++) {

        if (actual[i] != reference[i]) {

            if (mismatches < 10) {
                printf(
                    "%s mismatch [%lu]: "
                    "actual=%d expected=%d\n",
                    name,
                    (unsigned long)i,
                    (int)actual[i],
                    (int)reference[i]
                );
            }

            mismatches++;
        }
    }

    return mismatches;
}


static long checksum(const int8_t *x) {
    long sum = 0;

    for (size_t i = 0; i < N; i++) {
        sum += (int)x[i];
    }

    return sum;
}


// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(void) {

    printf("Gemma 4 residual-add: CPU vs Saturn RVV\n\n");

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


    // -------------------------------------------------------------------------
    // Scalar CPU
    // -------------------------------------------------------------------------

    printf("Starting scalar CPU residual add\n");

    unsigned long scalar_start =
        read_csr(mcycle);

    residual_add_scalar(
        input_a,
        input_b,
        scalar_output,
        N
    );

    asm volatile("fence" ::: "memory");

    unsigned long scalar_end =
        read_csr(mcycle);


    // -------------------------------------------------------------------------
    // Saturn RVV
    // -------------------------------------------------------------------------

    printf("Starting Saturn RVV residual add\n");

    unsigned long saturn_start =
        read_csr(mcycle);

    residual_add_saturn(
        input_a,
        input_b,
        saturn_output,
        N
    );

    asm volatile("fence" ::: "memory");

    unsigned long saturn_end =
        read_csr(mcycle);


    unsigned long scalar_cycles =
        scalar_end - scalar_start;

    unsigned long saturn_cycles =
        saturn_end - saturn_start;


    printf(
        "\nScalar CPU cycles: %lu\n",
        scalar_cycles
    );

    printf(
        "Saturn RVV cycles: %lu\n\n",
        saturn_cycles
    );


    // -------------------------------------------------------------------------
    // Compare outputs
    // -------------------------------------------------------------------------

    unsigned long scalar_mismatches =
        compare_output(
            scalar_output,
            expected,
            "CPU"
        );

    unsigned long saturn_mismatches =
        compare_output(
            saturn_output,
            expected,
            "Saturn"
        );

    unsigned long cross_mismatches =
        compare_output(
            saturn_output,
            scalar_output,
            "Saturn vs CPU"
        );


    printf(
        "CPU vs expected mismatches: %lu\n",
        scalar_mismatches
    );

    printf(
        "Saturn vs expected mismatches: %lu\n",
        saturn_mismatches
    );

    printf(
        "Saturn vs CPU mismatches: %lu\n",
        cross_mismatches
    );

    printf(
        "Saturn checksum: %ld\n\n",
        checksum(saturn_output)
    );


    // -------------------------------------------------------------------------
    // Sample output
    // -------------------------------------------------------------------------

    printf("Sample values\n");

    for (size_t i = 0; i < 8; i++) {
        printf(
            "[%lu] %d + %d = %d\n",
            (unsigned long)i,
            (int)input_a[i],
            (int)input_b[i],
            (int)saturn_output[i]
        );
    }


    // -------------------------------------------------------------------------
    // Final result
    // -------------------------------------------------------------------------

    if (
        scalar_mismatches == 0 &&
        saturn_mismatches == 0 &&
        cross_mismatches == 0
    ) {

        printf(
            "\nPASS: Saturn RVV residual-add "
            "matches CPU and reference.\n"
        );

        return 0;
    }

    printf(
        "\nFAIL: residual-add mismatch detected.\n"
    );

    return 1;
}