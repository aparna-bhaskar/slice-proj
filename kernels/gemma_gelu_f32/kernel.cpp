#include <mu_intrinsics.h>
#include <mu_schedule.h>

#include <stdint.h>

#define GELU_NUM_WARPS 4

extern "C"
uint32_t __mu_num_warps =
    GELU_NUM_WARPS;

#include "data"


struct GeluArgs {
  __global float* X;
  __global float* Y;

  __global float* expected;

  __global float* tanh_lut;

  uint32_t total;

  uint32_t lut_size;

  float lut_max;
  float lut_scale;
};


static inline float
tanh_lookup(
  float z,
  __global float* lut,
  uint32_t lut_size,
  float lut_max,
  float lut_scale
) {

  float az;
  float sign;

  if (z < 0.0f) {

    az =
        -z;

    sign =
        -1.0f;

  } else {

    az =
        z;

    sign =
        1.0f;
  }


  float t;

  /*
   * tanh(8) is already effectively 1
   * for our FP32 accuracy target.
   */
  if (az >= lut_max) {

    t =
        1.0f;

  } else {

    const float scaled =
        az *
        lut_scale;

    uint32_t idx =
        static_cast<uint32_t>(
          scaled
        );

    /*
     * Defensive clamp.
     */
    if (idx >= lut_size - 1) {

      idx =
          lut_size - 2;
    }


    const float frac =
        scaled
        -
        static_cast<float>(
          idx
        );


    const float t0 =
        lut[idx];

    const float t1 =
        lut[idx + 1];


    t =
        t0
        +
        frac
        *
        (
          t1 - t0
        );
  }


  return
      sign *
      t;
}


static inline void gelu(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {

  auto* args =
      reinterpret_cast<GeluArgs*>(
        raw_arg
      );


  uint32_t tid =
      tid_in_threadblock;


#if MU_NUM_CLUSTERS != 1

  tid +=
      threadblock_id
      *
      threads_per_threadblock;

  const uint32_t stride =
      threads_per_threadblock
      *
      MU_NUM_CLUSTERS;

#else

  (void)threadblock_id;

  const uint32_t stride =
      threads_per_threadblock;

#endif


  uint32_t local_bad =
      0;


  /*
   * sqrt(2/pi)
   */
  constexpr float kSqrt2OverPi =
      0.7978845608028654f;

  constexpr float kGeluCoeff =
      0.044715f;


  #pragma unroll 1
  for (
    uint32_t idx = tid;
    idx < args->total;
    idx += stride
  ) {

    const float x =
        args->X[idx];


    const float x2 =
        x *
        x;

    const float x3 =
        x2 *
        x;


    const float inner =
        x
        +
        kGeluCoeff
        *
        x3;


    const float z =
        kSqrt2OverPi
        *
        inner;


    const float t =
        tanh_lookup(
          z,
          args->tanh_lut,
          args->lut_size,
          args->lut_max,
          args->lut_scale
        );


    /*
     * PyTorch:
     *
     * gelu(x, approximate="tanh")
     */
    const float got =
        0.5f
        *
        x
        *
        (
          1.0f
          +
          t
        );


    args->Y[idx] =
        got;


    const float expected =
        args->expected[idx];


    const float diff =
        __builtin_fabsf(
          got
          -
          expected
        );


    /*
     * LUT approximation error is only
     * a few e-6; leave additional room
     * for Muon FP32 evaluation order.
     */
    local_bad |=
        static_cast<uint32_t>(
          diff > 5.0e-5f
        );
  }


  /*
   * Exactly like our passing RMSNorm
   * and RoPE tests:
   *
   * any incorrect Muon thread remains
   * active forever.
   */
  if (local_bad != 0) {

    while (1) {

      asm volatile(
        "nop"
      );
    }
  }
}


GeluArgs gelu_args = {

  .X = nullptr,

  .Y = nullptr,

  .expected = nullptr,

  .tanh_lut = nullptr,

  .total = 0,

  .lut_size = 0,

  .lut_max = 0.0f,

  .lut_scale = 0.0f,
};


int main() {

  gelu_args.X =
      X_raw;

  gelu_args.Y =
      Y_raw;

  gelu_args.expected =
      Y_expected;

  gelu_args.tanh_lut =
      TANH_LUT;

  gelu_args.total =
      total;

  gelu_args.lut_size =
      tanh_lut_size;

  gelu_args.lut_max =
      tanh_lut_max;

  gelu_args.lut_scale =
      tanh_lut_scale;


  /*
   * Known-good Muon pattern:
   *
   * ONE schedule
   * verification inside kernel
   * immediate return
   */
  mu_schedule(
    gelu,
    &gelu_args,
    GELU_NUM_WARPS
  );


  return 0;
}
