#include <mu_intrinsics.h>
#include <mu_schedule.h>

#include <stdint.h>

#define RESADD_NUM_WARPS 4

extern "C" uint32_t __mu_num_warps = RESADD_NUM_WARPS;

#include "data"


struct ResAddArgs {
  __global int8_t* A;
  __global int8_t* B;
  __global int8_t* C;
  uint32_t n;
};


struct VerifyArgs {
  __global int8_t* C;
  __global volatile int32_t* checksum;
  __global volatile int32_t* mismatches;
  uint32_t n;
};


static inline void resadd(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args = reinterpret_cast<ResAddArgs*>(raw_arg);

  constexpr uint32_t kWarpWidth = MU_NUM_THREADS;

  const uint32_t tid_in_warp =
      tid_in_threadblock % kWarpWidth;

  const uint32_t warp_id =
      tid_in_threadblock / kWarpWidth;

  const uint32_t warps_per_threadblock =
      threads_per_threadblock / kWarpWidth;

  uint32_t idx =
      warp_id * kWarpWidth + tid_in_warp;

  const uint32_t stride =
      warps_per_threadblock * kWarpWidth;

#if MU_NUM_CLUSTERS != 1
  idx += threadblock_id * threads_per_threadblock;

  const uint32_t global_stride =
      stride * MU_NUM_CLUSTERS;
#else
  (void)threadblock_id;

  const uint32_t global_stride = stride;
#endif

  #pragma unroll 1
  for (; idx < args->n; idx += global_stride) {
    int8_t a = args->A[idx];
    int8_t b = args->B[idx];

    args->C[idx] = static_cast<int8_t>(
        static_cast<int32_t>(a) +
        static_cast<int32_t>(b)
    );
  }
}


static inline void verify_resadd(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threads_per_threadblock;
  (void)threadblock_id;

  auto* args = reinterpret_cast<VerifyArgs*>(raw_arg);

  // Only thread 0 performs the scalar reference check.
  if (tid_in_threadblock != 0) {
    return;
  }

  int32_t checksum = 0;
  int32_t mismatches = 0;

  #pragma unroll 1
  for (uint32_t i = 0; i < args->n; ++i) {
    const int32_t a =
        static_cast<int32_t>(i % 17) - 8;

    const int32_t b =
        static_cast<int32_t>((i * 3) % 13) - 6;

    const int32_t expected = a + b;

    const int32_t got =
        static_cast<int32_t>(args->C[i]);

    checksum += got;

    if (got != expected) {
      ++mismatches;
    }
  }

  args->checksum[0] = checksum;
  args->mismatches[0] = mismatches;
}


ResAddArgs resadd_args = {
  .A = nullptr,
  .B = nullptr,
  .C = nullptr,
  .n = 0,
};


VerifyArgs verify_args = {
  .C = nullptr,
  .checksum = nullptr,
  .mismatches = nullptr,
  .n = 0,
};


int main() {
  resadd_args.A = A_raw;
  resadd_args.B = B_raw;
  resadd_args.C = C_raw;

  resadd_args.n = n;

  // Run the actual Radiance/Muon residual add.
  mu_schedule(
    resadd,
    &resadd_args,
    RESADD_NUM_WARPS
  );


  verify_args.C = C_raw;
  verify_args.checksum = verify_checksum;
  verify_args.mismatches = verify_mismatches;

  verify_args.n = n;

  // Verify the generated output on the Muon core.
  mu_schedule(
    verify_resadd,
    &verify_args,
    RESADD_NUM_WARPS
  );


  // Expected:
  //   mismatches = 0
  //   checksum   = -17
  //
  // If either is wrong, deliberately keep the GPU active.
  // A correct run reaches normal "no more active warps" termination.
  if (verify_args.mismatches[0] != 0 ||
      verify_args.checksum[0] != -17) {

    while (1) {
      asm volatile("nop");
    }
  }

  return 0;
}
