#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__device__ uint shift_local(uint ix) {
    uint row = ix / GROUP_SIZE_X;
    uint col = ix % GROUP_SIZE_X;
    uint shifted_col = (col + row) % GROUP_SIZE_X;
    uint shifted_ix = row * GROUP_SIZE_X + shifted_col;
    return shifted_ix;
}

__global__ void matrix_transpose_coalesced_via_local_memory(
                       const float* matrix,            // w x h
                             float* transposed_matrix, // h x w
                             unsigned int w,
                             unsigned int h)
{
    const uint x_index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y_index = blockIdx.y * blockDim.y + threadIdx.y;
    const uint x_local = threadIdx.x;
    const uint y_local = threadIdx.y;
     __shared__ float local_buf[GROUP_SIZE_X * GROUP_SIZE_Y];

    if (x_index >= w || y_index >= h)  {
        return;
    }

    const uint src_ix = y_index * w + x_index;
    const uint src_local_ix = y_local * GROUP_SIZE_Y + x_local;
    const uint sh_src_local_ix = shift_local(src_local_ix);

    const uint dst_row = (blockIdx.x * blockDim.x) + y_local;
    const uint dst_col = (blockIdx.y * blockDim.y) + x_local;
    const uint dst_ix = dst_row * h + dst_col;
    const uint dst_local_ix = x_local * GROUP_SIZE_X + y_local;
    const uint sh_dst_local_ix = shift_local(dst_local_ix);

    // local_buf[src_local_ix] = matrix[src_ix];
    // __syncthreads();
    // transposed_matrix[dst_ix] = local_buf[dst_local_ix];
    local_buf[sh_src_local_ix] = matrix[src_ix];
    __syncthreads();
    transposed_matrix[dst_ix] = local_buf[sh_dst_local_ix];

    // if (x_index == 1 && y_index == 0) {
    //     printf("worker (%d, %d):\n\tmove %d(%d,%d)->%d(%d,%d);\n\tbuf %d->%d, %d->%d\n\tval: %f <-> %f\n\tbuf val: %f <-> %f\n",
    //            x_local, y_local,
    //            src_ix, x_index, y_index, dst_ix, dst_col, dst_row,
    //            src_local_ix, dst_local_ix, src_local_ix, dst_local_ix,
    //            matrix[src_ix], transposed_matrix[dst_ix],
    //            local_buf[src_local_ix], local_buf[dst_local_ix]
    //     );
    // }

}

namespace cuda {
void matrix_transpose_coalesced_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &matrix, gpu::gpu_mem_32f &transposed_matrix, unsigned int w, unsigned int h)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_transpose_coalesced_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(matrix.cuptr(), transposed_matrix.cuptr(), w, h);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
