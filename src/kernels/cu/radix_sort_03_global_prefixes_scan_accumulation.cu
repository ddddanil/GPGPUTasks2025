#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    const unsigned int* pow2_sum,
          unsigned int* prefix_sum_accum,
    unsigned int n,
    unsigned int pow2)
{
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const uint ix = i + 1;
    const uint curr_pow = 1 << pow2;
    // if (i == 4) printf("ix %d: curr_pow=%d\n", ix, curr_pow);
    if (ix & curr_pow) {
        const uint sum_ix = (ix >> pow2) - 1;
        prefix_sum_accum[i] += pow2_sum[sum_ix];
        // if (i == 4) printf("  ix %d: sum_ix=%d sum=%d\n", ix, sum_ix, pow2_sum[sum_ix]);
    }
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &pow2_sum, gpu::gpu_mem_32u &prefix_sum_accum, unsigned int n, unsigned int pow2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), prefix_sum_accum.cuptr(), n, pow2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
