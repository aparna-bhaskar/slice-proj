#include <mu_intrinsics.h>
#include <mu_schedule.h>

#include <stdint.h>

#define ROPE_NUM_WARPS 1

extern "C" uint32_t __mu_num_warps = ROPE_NUM_WARPS;

#include "data"


struct RopeArgs {
  __global float* Q;
  __global float* K;

  __global float* cos_table;
  __global float* sin_table;

  __global float* Qout;
  __global float* Kout;

  __global float* Qexpected;
  __global float* Kexpected;

  uint32_t seq;
  uint32_t q_heads;
  uint32_t k_heads;
  uint32_t dim;
  uint32_t half;
};


static inline void rope(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args =
      reinterpret_cast<RopeArgs*>(raw_arg);

  uint32_t tid =
      tid_in_threadblock;

#if MU_NUM_CLUSTERS != 1
  tid +=
      threadblock_id *
      threads_per_threadblock;

  const uint32_t stride =
      threads_per_threadblock *
      MU_NUM_CLUSTERS;
#else
  (void)threadblock_id;

  const uint32_t stride =
      threads_per_threadblock;
#endif

  uint32_t local_bad = 0;

  const uint32_t qn =
      args->seq *
      args->q_heads *
      args->dim;

  const uint32_t kn =
      args->seq *
      args->k_heads *
      args->dim;


  /*
   * Q RoPE
   */
  #pragma unroll 1
  for (
    uint32_t idx = tid;
    idx < qn;
    idx += stride
  ) {
    const uint32_t d =
        idx % args->dim;

    const uint32_t tmp =
        idx / args->dim;

    const uint32_t pos =
        tmp / args->q_heads;


    uint32_t partner_d;

    float sign;

    if (d < args->half) {
      partner_d =
          d + args->half;

      sign = -1.0f;
    } else {
      partner_d =
          d - args->half;

      sign = 1.0f;
    }


    const uint32_t base =
        idx - d;

    const float x =
        args->Q[idx];

    const float xr =
        sign *
        args->Q[
          base + partner_d
        ];

    const uint32_t trig_idx =
        pos * args->dim + d;

    const float got =
        x *
        args->cos_table[trig_idx]
        +
        xr *
        args->sin_table[trig_idx];

    args->Qout[idx] =
        got;


    const float expected =
        args->Qexpected[idx];

    const float diff =
        __builtin_fabsf(
          got - expected
        );

    local_bad |=
        static_cast<uint32_t>(
          diff > 2.0e-5f
        );
  }


  /*
   * K RoPE
   */
  #pragma unroll 1
  for (
    uint32_t idx = tid;
    idx < kn;
    idx += stride
  ) {
    const uint32_t d =
        idx % args->dim;

    const uint32_t tmp =
        idx / args->dim;

    const uint32_t pos =
        tmp / args->k_heads;


    uint32_t partner_d;

    float sign;

    if (d < args->half) {
      partner_d =
          d + args->half;

      sign = -1.0f;
    } else {
      partner_d =
          d - args->half;

      sign = 1.0f;
    }


    const uint32_t base =
        idx - d;

    const float x =
        args->K[idx];

    const float xr =
        sign *
        args->K[
          base + partner_d
        ];

    const uint32_t trig_idx =
        pos * args->dim + d;

    const float got =
        x *
        args->cos_table[trig_idx]
        +
        xr *
        args->sin_table[trig_idx];

    args->Kout[idx] =
        got;


    const float expected =
        args->Kexpected[idx];

    const float diff =
        __builtin_fabsf(
          got - expected
        );

    local_bad |=
        static_cast<uint32_t>(
          diff > 2.0e-5f
        );
  }


  /*
   * Any incorrect Q/K RoPE value prevents
   * normal RTL termination.
   */
  if (local_bad != 0) {

    while (1) {
      asm volatile("nop");
    }
  }
}


RopeArgs rope_args = {
  .Q = nullptr,
  .K = nullptr,

  .cos_table = nullptr,
  .sin_table = nullptr,

  .Qout = nullptr,
  .Kout = nullptr,

  .Qexpected = nullptr,
  .Kexpected = nullptr,

  .seq = 0,
  .q_heads = 0,
  .k_heads = 0,
  .dim = 0,
  .half = 0,
};


int main() {

  rope_args.Q =
      Q_raw;

  rope_args.K =
      K_raw;

  rope_args.cos_table =
      COS_raw;

  rope_args.sin_table =
      SIN_raw;

  rope_args.Qout =
      Q_out;

  rope_args.Kout =
      K_out;

  rope_args.Qexpected =
      Q_expected;

  rope_args.Kexpected =
      K_expected;

  rope_args.seq =
      seq_len;

  rope_args.q_heads =
      q_heads;

  rope_args.k_heads =
      k_heads;

  rope_args.dim =
      head_dim;

  rope_args.half =
      half_dim;


  mu_schedule(
    rope,
    &rope_args,
    ROPE_NUM_WARPS
  );


  return 0;
}
