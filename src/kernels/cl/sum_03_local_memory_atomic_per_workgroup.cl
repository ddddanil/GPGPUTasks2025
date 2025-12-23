#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_03_local_memory_atomic_per_workgroup(__global const uint* a,
                                                       __global       uint* sum,
                                                       const unsigned int n)
{
    // Подсказки:
    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    const uint batch_size =  n/LOAD_K_VALUES_PER_ITEM;
    __local uint local_data[GROUP_SIZE];

    uint my_sum = 0;

    if (index < batch_size) {
        for (uint k = 0; k < LOAD_K_VALUES_PER_ITEM; ++k) {
            uint batch_off = k * batch_size;
            my_sum += a[batch_off + index];
        }
        local_data[local_index] = my_sum;
    } else {
        local_data[local_index] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index == 0) {
        uint group_sum = 0;
        for (int i = 0; i < GROUP_SIZE; i++) {
            group_sum += local_data[i];
        }
        atomic_add(sum, group_sum);
    }
}
