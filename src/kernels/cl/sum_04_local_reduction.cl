#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define WARP_SIZE 32

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_04_local_reduction(__global const uint* a,
                                     __global       uint* b,
                                            unsigned int  n)
{
    // Подсказки:
    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    const uint group_index = index / GROUP_SIZE;
    // const uint batch_sz = n/GROUP_SIZE;
    __local uint local_data[GROUP_SIZE];


    const uint start_ix = (group_index * GROUP_SIZE * LOAD_K_VALUES_PER_ITEM);
    // if (local_index == 0) printf("Arr size: %d\nGroup size: %d\nGroup index: %d\nStart from: %d\nGlobal ix: %d\n", n, GROUP_SIZE, group_index, start_ix, index);

    uint my_sum = 0;
    for (uint k = 0; k < LOAD_K_VALUES_PER_ITEM; ++k) {
        uint batch_off = start_ix + (k * GROUP_SIZE);
        uint i = batch_off + local_index;
        if (i < n) {
            // printf("worker %d: load from %d = %d\n", local_index, i, a[i]);
            my_sum += a[i];
        }
    }
    // printf("worker %d: sum %d\n", local_index, my_sum);
    local_data[local_index] = my_sum;

    // if (index < n) {
    //     local_data[local_index] = a[index];
    // } else {
    //     local_data[local_index] = 0;
    // }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index == 0) {
        uint group_sum = 0;
        for (int i = 0; i < GROUP_SIZE; i++) {
            group_sum += local_data[i];
        }

        int b_index = index / (GROUP_SIZE);
        // printf("Output ix for %d: %d\n", index, b_index);
        b[b_index] = group_sum;
    }
}
