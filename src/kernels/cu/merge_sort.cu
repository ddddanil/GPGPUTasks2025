#ifdef CLANGD
#include <__clang_cuda_builtin_vars.h>
#endif
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>
#include <utility>

#include "helpers/rassert.cu"
#include "../defines.h"

// number of elements on d-th diagonal in a w x h rect
__device__ int diag_width(int w, int h, int diag) {
    auto minwh = min(w, h);
    auto maxwh = max(w, h);
    if (diag < minwh) return diag + 1;
    if (diag > maxwh) return minwh - (diag - maxwh);
    return minwh;
}

__device__ uint diag_choice(
    uint const* input_l, // [0..n/2]
    uint const* input_r, // [0..n/2]
    int w, int h,
    int diag, int i )
{
    uint l_idx, r_idx;
    if (diag < h) {
        uint o = i / 2;
        l_idx = diag - o;
        r_idx = o;
    } else {
        uint o = (i - 1) / 2;
        l_idx = h - o - 1;
        r_idx = (o + 1) + (diag - h);
    }
    if (l_idx == h) {
        l_idx--;
    }
    if (r_idx == w) {
        r_idx--;
    }
    auto res_ix = (i % 2) ^ (diag >= h);
    auto res = (res_ix) ? input_r[r_idx] : input_l[l_idx];
    printf(" choice diag %d:%d ix=%d; (%3d)[%3d] (%3d)[%3d] => %d\n", diag, i, res_ix, input_l[l_idx], l_idx, input_r[r_idx], r_idx, res);

    return res;
}

__device__ bool diag_pred(
    uint const* input_l, // [0..n/2]
    uint const* input_r, // [0..n/2]
    int w, int h,
    int diag, int i )
{
    if (i == 0) {
        return false;
    }

    uint l_idx, r_idx;
    uint pos = (i % 2) ^ (diag >= h);
    uint o = i / 2;
    if (pos) {
        // check below
        if (diag < h) {
            l_idx = diag - o;
            r_idx = o;
        } else {
            l_idx = h - o;
            r_idx = (diag - h) + o;
        }
    } else {
        // check left
        if (diag < h) {
            l_idx = diag - o;
            r_idx = o - 1;
        } else {
            l_idx = h - o - 1;
            r_idx = (diag - h) + o;
        }
    }

    printf(" pred: diag=%d:%d cmp [%3d] <=> [%3d]\n", diag, i, l_idx, r_idx);
    auto [l_val, r_val] = std::pair{input_l[l_idx], input_r[r_idx]};
    printf(" pred: diag=%d:%d cmp (%3d) <=> (%3d)\n", diag, i, l_val, r_val);
    auto pred = l_val <= r_val;
    return pred;
}

__device__ uint bin_search_diag(
    uint const* input_l, // [0..n/2]
    uint const* input_r, // [0..n/2]
    int w, int h,
    int diag_n, // diagonal number
    int l, int r )
{
    while (l < r - 1) {
        int m = (l + r) / 2;
        printf("   binsrch: diag=%d l=%d r=%d m=%d\n", diag_n, l, r, m);

        // auto [l_i, r_i] = input_idx(w, h, diag_n, m);
        // if (l_i == h) {
        //     l_i--;
        // }
        // if (r_i == w) {
        //     r_i--;
        // }
        // printf("   binsrch: diag=%d cmp [%3d] <=> [%3d]\n", diag_n, l_i, r_i);
        // auto [l_val, r_val] = std::pair{input_l[l_i], input_r[r_i]};
        // printf("   binsrch: diag=%d cmp (%3d) <=> (%3d)\n", diag_n, l_val, r_val);
        auto pred = diag_pred(input_l, input_r, w, h, diag_n, m);
        printf("   binsrch: diag=%d pred %d\n", diag_n, int(pred));

        if (pred) {
            r = m;
        } else {
            l = m;
        }
    }
    // printf("   binsrch: diag=%d res=%d-%d\n", diag_n, l, r);
    return l;
}

// __global__ void merge_sort_coarse(
//     uint const* input_data, // [0..n/2]+[n/2..n] two halves
//     uint      * output_data, // [0..n] output
//     uint      * output_windows, // [0..t] window output
//      int        sorted_k,
//      int        n)
// {
//     const int t = blockIdx.x * blockDim.x + threadIdx.x; // long indexing
//     const int i = GROUP_SIZE * t; // output index

//     const int start_l = 2*t;
//     const int start_r = 2*t+1;
//     const int end_l = 2*t+2;
//     const int end_r = 2*t+3;

//     if (t == 0) { // fill start
//         output_windows[start_l] = 0;
//         output_windows[start_r] = 0;
//     }

//     auto input_l = input_data;
//     auto input_r = input_data + (n / 2);

//     auto [res_l, res_r] = bin_search_diag(input_l, input_r, n, i, 0, n);

//     output_windows[end_l] = res_l;
//     output_windows[end_r] = res_r;
//     output_data = input_l[res_l];
// }

__global__ void merge_sort_fine(
    uint const* input_data, // [0..n/2]+[n/2..n] two halves
    uint      * output_data, // [0..n] output
    uint const* input_windows, // [0..t] window output
     int        sorted_k, // len of merges
     int        n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x; // array ix
    if (i >= n) return;
    // const int t = blockIdx.x; // window ix

    // const int start_l = 2*t;
    // const int start_r = 2*t+1;
    // const int end_l = 2*t+2;
    // const int end_r = 2*t+3;

    const int block_ix = i / sorted_k;
    const int block_off = i % sorted_k;
    auto h = sorted_k / 2, w = min((n - (block_ix * sorted_k)) - h, h);
    printf("block %d:%d hw %dx%d\n", block_ix, block_off, h, w);
    if (w <= 0) {
        output_data[i] = input_data[i];
        return;
    }
    
    auto input_l = input_data + (block_ix * sorted_k);
    auto input_r = input_l + h;
    auto diag_w = diag_width(w, h, block_off);

    printf("pos %d: block %d:%d sorted=%d diag_w=%d\n", i, block_ix, block_off, sorted_k, diag_w);
    printf("pos %d: input_l=[%d, %d, ..] off=%ld\n", i, input_l[0], input_l[1], (input_l - input_data));
    printf("pos %d: input_r=[%d, %d, ..] off=%ld\n", i, input_r[0], input_r[1], (input_r - input_data));

    auto res_diag = bin_search_diag(
        input_l, input_r,
        w, h,
        block_off,
        0, 2*diag_w
    );
    auto res = diag_choice(input_l, input_r, w, h, block_off, res_diag);
    output_data[i] = res;
}

namespace cuda {
// void merge_sort_coarse(const gpu::WorkSize &workSize,
//             const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, gpu::gpu_mem_32u &output_windows, int sorted_k, int n)
// {
//     gpu::Context context;
//     rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
//     cudaStream_t stream = context.cudaStream();
//     ::merge_sort_coarse<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), output_windows.cuptr(), sorted_k, n);
//     CUDA_CHECK_KERNEL(stream);
// }

void merge_sort_fine(const gpu::WorkSize &workSize,
            gpu::gpu_mem_32u const& input_data, gpu::gpu_mem_32u & output_data, gpu::gpu_mem_32u const& input_windows, int sorted_k, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort_fine<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), input_windows.cuptr(), sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
