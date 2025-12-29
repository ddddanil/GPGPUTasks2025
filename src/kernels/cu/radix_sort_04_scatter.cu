#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    const unsigned int* input_values,
    const unsigned int* value_idx,
    const unsigned int* value_flag,
          unsigned int* output,
    unsigned int n,
    unsigned int global_off)
{
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    uint local_off = value_idx[i];
    uint output_idx = local_off + global_off - 1;
    uint flag = value_flag[i];
    // if (i == 25) printf("elem %d: value=%d flag=%d ix=%d+%d=%d\n", i, input_values[i], flag, global_off, local_off, output_idx);
    if (flag) {
        // printf("elem %d: value=%d flag=%d ix=%d+%d=%d\n", i, input_values[i], flag, global_off, local_off, output_idx);
        // rassert(output_idx != -1, 1418239198, i, output_idx);
        output[output_idx] = input_values[i];
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &input_values, const gpu::gpu_mem_32u &value_idx, const gpu::gpu_mem_32u &value_flag, gpu::gpu_mem_32u &output, unsigned int n, unsigned int global_off)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_values.cuptr(), value_idx.cuptr(), value_flag.cuptr(), output.cuptr(), n, global_off);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
