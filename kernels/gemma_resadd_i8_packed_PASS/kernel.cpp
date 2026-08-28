#include <mu_intrinsics.h>
#include <mu_schedule.h>

#include <stdint.h>

#define RESADD_NUM_WARPS 4

extern "C" uint32_t __mu_num_warps = RESADD_NUM_WARPS;

#include "data"


static inline int32_t sext8(uint32_t x) {
  return static_cast<int32_t>(x << 24) >> 24;
}


struct ResAddArgs {
  __global uint32_t* A;
  __global uint32_t* B;
  __global uint32_t* C;
  __global uint32_t* E;
  __global int32_t* expected_partial_checksum;

  uint32_t words;
};


static inline void resadd(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args =
      reinterpret_cast<ResAddArgs*>(raw_arg);

  constexpr uint32_t kWarpWidth =
      MU_NUM_THREADS;

  const uint32_t tid_in_warp =
      tid_in_threadblock % kWarpWidth;

  const uint32_t warp_id =
      tid_in_threadblock / kWarpWidth;

  const uint32_t warps_per_threadblock =
      threads_per_threadblock / kWarpWidth;

  uint32_t idx =
      warp_id * kWarpWidth +
      tid_in_warp;

  uint32_t thread_slot = idx;

  const uint32_t stride =
      warps_per_threadblock *
      kWarpWidth;

#if MU_NUM_CLUSTERS != 1

  idx +=
      threadblock_id *
      threads_per_threadblock;

  thread_slot +=
      threadblock_id *
      threads_per_threadblock;

  const uint32_t global_stride =
      stride * MU_NUM_CLUSTERS;

#else

  (void)threadblock_id;

  const uint32_t global_stride =
      stride;

#endif

  int32_t local_checksum = 0;

  uint32_t local_bad = 0;


  #pragma unroll 1
  for (; idx < args->words;
       idx += global_stride) {

    const uint32_t aw =
        args->A[idx];

    const uint32_t bw =
        args->B[idx];


    uint32_t cw = 0;


    #pragma unroll
    for (uint32_t j = 0;
         j < 4;
         ++j) {

      const uint32_t shift =
          j * 8;

      const int32_t a =
          sext8(aw >> shift);

      const int32_t b =
          sext8(bw >> shift);

      const int32_t sum =
          a + b;

      cw |=
          (static_cast<uint32_t>(sum)
           & 0xffu)
          << shift;
    }


    /*
     * Actual residual output.
     */
    args->C[idx] = cw;


    /*
     * Verify every packed word.
     *
     * If every C word == E word,
     * every one of the 21,504 int8
     * residual results is correct.
     */
    const uint32_t got =
        args->C[idx];

    const uint32_t expected =
        args->E[idx];

    local_bad |=
        got ^ expected;


    /*
     * Signed int8 checksum.
     */
    local_checksum +=
        sext8(got);

    local_checksum +=
        sext8(got >> 8);

    local_checksum +=
        sext8(got >> 16);

    local_checksum +=
        sext8(got >> 24);
  }


  /*
   * Each of the 64 threads owns a
   * deterministic subset of words.
   *
   * Verify both:
   *   1. exact packed outputs
   *   2. that thread's checksum
   *
   * A bad thread never completes,
   * so mu_schedule cannot finish.
   */
  const int32_t expected_checksum =
      args->expected_partial_checksum[
          thread_slot
      ];

  if (local_bad != 0 ||
      local_checksum != expected_checksum) {

    while (1) {
      asm volatile("nop");
    }
  }
}


ResAddArgs resadd_args = {
  .A = nullptr,
  .B = nullptr,
  .C = nullptr,
  .E = nullptr,
  .expected_partial_checksum = nullptr,
  .words = 0,
};


int main() {

  resadd_args.A =
      A_raw;

  resadd_args.B =
      B_raw;

  resadd_args.C =
      C_raw;

  resadd_args.E =
      E_raw;

  resadd_args.expected_partial_checksum =
      expected_partial_checksum;

  resadd_args.words =
      words;


  /*
   * Match the known-working stock
   * Radiance vecadd control structure:
   *
   *   schedule once
   *   return immediately
   */
  mu_schedule(
    resadd,
    &resadd_args,
    RESADD_NUM_WARPS
  );


  return 0;
}
