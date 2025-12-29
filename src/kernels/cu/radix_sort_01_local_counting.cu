#ifdef CLANGD
#include <__clang_cuda_builtin_vars.h>
#endif
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    const unsigned int* input, // [0..n] input values
          unsigned int* bucket_flags, // [0..n] outputs 00, 01, 10, 11
    unsigned int sh_off, // bit shift offset
    unsigned int bucket_n,
    unsigned int n)
{
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    const uint val = input[i];
    const uint bucket_ix = (val >> sh_off) & 0b11;
    if (bucket_ix == bucket_n) {
        bucket_flags[i] = 1;
    } else {
        bucket_flags[i] = 0;
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input, gpu::gpu_mem_32u &bucket_flags, unsigned int sh_off, unsigned int bucket_n, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input.cuptr(), bucket_flags.cuptr(), sh_off, bucket_n, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
