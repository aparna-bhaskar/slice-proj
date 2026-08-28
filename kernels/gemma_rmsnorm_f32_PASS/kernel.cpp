#include <mu_intrinsics.h>
#include <mu_schedule.h>

#include <stdint.h>

#define RMS_NUM_WARPS 1

extern "C" uint32_t __mu_num_warps = RMS_NUM_WARPS;

#include "data"


struct RMSNormArgs {
  __global float* X;
  __global float* W;
  __global float* Y;
  __global float* expected;

  uint32_t rows;
  uint32_t dim;

  float eps;
};


static inline void rmsnorm(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args =
      reinterpret_cast<RMSNormArgs*>(raw_arg);

  uint32_t tid =
      tid_in_threadblock;

#if MU_NUM_CLUSTERS != 1

  tid +=
      threadblock_id *
      threads_per_threadblock;

  const uint32_t thread_stride =
      threads_per_threadblock *
      MU_NUM_CLUSTERS;

#else

  (void)threadblock_id;

  const uint32_t thread_stride =
      threads_per_threadblock;

#endif

  uint32_t local_bad = 0;


  /*
   * One thread handles one or more complete rows.
   *
   * For this test:
   *   rows = 14
   *   dim  = 1536
   *
   * With one 16-lane warp, lanes 0..13 each
   * process one Gemma token.
   */
  #pragma unroll 1
  for (
    uint32_t row = tid;
    row < args->rows;
    row += thread_stride
  ) {

    const uint32_t base =
        row * args->dim;


    /*
     * RMS reduction:
     *
     * mean(x^2)
     */
    float sum_sq = 0.0f;

    #pragma unroll 1
    for (
      uint32_t j = 0;
      j < args->dim;
      ++j
    ) {

      const float x =
          args->X[base + j];

      sum_sq += x * x;
    }


    const float mean_sq =
        sum_sq /
        static_cast<float>(args->dim);


    /*
     * inverse RMS:
     *
     * 1 / sqrt(mean(x^2) + eps)
     */
    const float inv_rms =
        1.0f /
        __builtin_sqrtf(
          mean_sq + args->eps
        );


    /*
     * Normalize + learned Gemma scale.
     */
    #pragma unroll 1
    for (
      uint32_t j = 0;
      j < args->dim;
      ++j
    ) {

      const uint32_t idx =
          base + j;

      const float got =
          args->X[idx] *
          inv_rms *
          args->W[j];

      args->Y[idx] = got;


      /*
       * Compare against independently generated
       * PyTorch reference.
       */
      const float expected =
          args->expected[idx];

      const float diff =
          __builtin_fabsf(
            got - expected
          );

      /*
       * FP32 reduction order on Muon may differ
       * slightly from PyTorch's reduction.
       */
      local_bad |=
          static_cast<uint32_t>(
            diff > 2.0e-4f
          );
    }
  }


  /*
   * Correctness gate.
   *
   * Any wrong thread remains active forever.
   * A successful simulation can terminate only
   * when every assigned RMSNorm result is within
   * tolerance of PyTorch.
   */
  if (local_bad != 0) {

    while (1) {
      asm volatile("nop");
    }
  }
}


RMSNormArgs rms_args = {
  .X = nullptr,
  .W = nullptr,
  .Y = nullptr,
  .expected = nullptr,
  .rows = 0,
  .dim = 0,
  .eps = 0.0f,
};


int main() {

  rms_args.X =
      X_raw;

  rms_args.W =
      W_raw;

  rms_args.Y =
      Y_raw;

  rms_args.expected =
      Y_expected;

  rms_args.rows =
      rows;

  rms_args.dim =
      dim;

  rms_args.eps =
      rms_eps;


  /*
   * Keep the known-good Muon control pattern:
   *
   * exactly one scheduler launch,
   * verification inside scheduled threads,
   * immediately return afterward.
   */
  mu_schedule(
    rmsnorm,
    &rms_args,
    RMS_NUM_WARPS
  );


  return 0;
}
