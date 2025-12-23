#ifdef CLANGD
#include <__clang_cuda_builtin_vars.h>
#endif
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; // 0..w
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y; // 0..h
    const uint i_local = threadIdx.x;
    const uint j_local = threadIdx.y;
     __shared__ float tile_a[GROUP_SIZE_X * GROUP_SIZE_Y];
     __shared__ float tile_b[GROUP_SIZE_X * GROUP_SIZE_Y];
    const uint tile_ix = threadIdx.y * GROUP_SIZE_Y + threadIdx.x;
    const uint tile_stride = (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X;

    float acc = 0;
    for (uint ti = 0; ti < tile_stride; ++ti) {
        uint ki = ti*GROUP_SIZE_X + i_local;
        uint kj = ti*GROUP_SIZE_Y + j_local;
        tile_a[tile_ix] = a[j*k + ki];
        tile_b[tile_ix] = b[kj*w + i];

        __syncthreads();
        for (uint t = 0; t < GROUP_SIZE_X; ++t) {
            acc += tile_a[threadIdx.y * GROUP_SIZE_Y + t] * tile_b[t * GROUP_SIZE_Y + threadIdx.x];
        }
    }

    c[j*w + i] = acc;
}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
