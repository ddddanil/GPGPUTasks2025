#include <iomanip>
#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include "debug.h" // TODO очень советую использовать debug::prettyBits(...) для отладки
#include "libbase/runtime_assert.h"
#include "libgpu/work_size.h"

#include <iostream>
#include <ostream>
#include <vector>

void run_prefix_sum(
    gpu::gpu_mem_32u const& input,
    gpu::gpu_mem_32u &pow2_buf1,
    gpu::gpu_mem_32u &pow2_buf2,
    gpu::gpu_mem_32u &accum_buf,
    uint n
)
{
    uint pow2 = 0;
    // std::cout << "do pow " << pow2 << " size " << n << "->" << (n/2) << std::endl;
    cuda::radix_sort_03_global_prefixes_scan_accumulation(gpu::WorkSize(GROUP_SIZE, n), input, accum_buf, n, pow2);
    cuda::radix_sort_02_global_prefixes_scan_sum_reduction(gpu::WorkSize(GROUP_SIZE, n/2), input, pow2_buf1, n);
    // {
    //     std::vector<unsigned int> inputs = input.readVector();
    //     std::cout << "curr sum: ";
    //     for (int i = 0; i < n; ++i) std::cout << std::setw(3) << inputs[i] << " ";
    //     std::cout << std::endl;
    //     std::vector<unsigned int> gpu_prefix_sum = accum_buf.readVector();
    //     std::cout << "curr sum: ";
    //     for (int i = 0; i < n; ++i) std::cout << std::setw(3) << gpu_prefix_sum[i] << " ";
    //     std::cout << std::endl;
    //     std::vector<unsigned int> gpu_pow_sum = pow2_buf1.readVector();
    //     std::cout << "curr pow: ";
    //     for (int i = 0; i < n; ++i) std::cout << std::setw(3) << gpu_pow_sum[i] << " ";
    //     std::cout << std::endl;
    // }

    for (uint pow_n = (n + 1) / 2; pow_n > 1; pow_n = (pow_n + 1) / 2) {
        pow2++;
        uint next_pow_n = (pow_n + 1) / 2;
        // std::cout << "do pow " << pow2 << " size " << pow_n << "->" << next_pow_n << std::endl;
        if (pow2 & 1) {
            cuda::radix_sort_03_global_prefixes_scan_accumulation(gpu::WorkSize(GROUP_SIZE, n), pow2_buf1, accum_buf, n, pow2);
            cuda::radix_sort_02_global_prefixes_scan_sum_reduction(gpu::WorkSize(GROUP_SIZE, next_pow_n), pow2_buf1, pow2_buf2, pow_n);
            // {
            //     std::vector<unsigned int> gpu_prefix_sum = accum_buf.readVector();
            //     std::cout << "curr sum: ";
            //     for (int i = 0; i < n; ++i) std::cout << std::setw(3) << gpu_prefix_sum[i] << " ";
            //     std::cout << std::endl;
            //     std::vector<unsigned int> gpu_pow_sum = pow2_buf2.readVector();
            //     std::cout << "curr pow: ";
            //     for (int i = 0; i < n; ++i) std::cout << std::setw(3) << gpu_pow_sum[i] << " ";
            //     std::cout << std::endl;
            // }
        } else {
            cuda::radix_sort_03_global_prefixes_scan_accumulation(gpu::WorkSize(GROUP_SIZE, n), pow2_buf2, accum_buf, n, pow2);
            cuda::radix_sort_02_global_prefixes_scan_sum_reduction(gpu::WorkSize(GROUP_SIZE, next_pow_n), pow2_buf2, pow2_buf1, pow_n);
            // {
            //     std::vector<unsigned int> gpu_prefix_sum = accum_buf.readVector();
            //     std::cout << "curr sum: ";
            //     for (int i = 0; i < n; ++i) std::cout << std::setw(3) << gpu_prefix_sum[i] << " ";
            //     std::cout << std::endl;
            //     std::vector<unsigned int> gpu_pow_sum = pow2_buf1.readVector();
            //     std::cout << "curr pow: ";
            //     for (int i = 0; i < n; ++i) std::cout << std::setw(3) << gpu_pow_sum[i] << " ";
            //     std::cout << std::endl;
            // }
        }
    }
}

void run(int argc, char** argv)
{
    // chooseGPUVkDevices:
    // - Если не доступо ни одного устройства - кинет ошибку
    // - Если доступно ровно одно устройство - вернет это устройство
    // - Если доступно N>1 устройства:
    //   - Если аргументов запуска нет или переданное число не находится в диапазоне от 0 до N-1 - кинет ошибку
    //   - Если аргумент запуска есть и он от 0 до N-1 - вернет устройство под указанным номером
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    // TODO 000 сделайте здесь свой выбор API - если он отличается от OpenCL то в этой строке нужно заменить TypeOpenCL на TypeCUDA или TypeVulkan
    // TODO 000 после этого изучите этот код, запустите его, изучите соответсвующий вашему выбору кернел - src/kernels/<ваш выбор>/aplusb.<ваш выбор>
    // TODO 000 P.S. если вы выбрали CUDA - не забудьте установить CUDA SDK и добавить -DCUDA_SUPPORT=ON в CMake options
    // TODO 010 P.S. так же в случае CUDA - добавьте в CMake options (НЕ меняйте сами CMakeLists.txt чтобы не менять окружение тестирования):
    // TODO 010 "-DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo" (первое - чтобы включить поддержку WMMA, второе - чтобы compute-sanitizer и профилировщик знали номера строк кернела)
    gpu::Context context = activateContext(device, gpu::Context::TypeCUDA);
    // OpenCL - рекомендуется как вариант по умолчанию, можно выполнять на CPU, есть printf, есть аналог valgrind/cuda-memcheck - https://github.com/jrprice/Oclgrind
    // CUDA   - рекомендуется если у вас NVIDIA видеокарта, есть printf, т.к. в таком случае вы сможете пользоваться профилировщиком (nsight-compute) и санитайзером (compute-sanitizer, это бывший cuda-memcheck)
    // Vulkan - не рекомендуется, т.к. писать код (compute shaders) на шейдерном языке GLSL на мой взгляд менее приятно чем в случае OpenCL/CUDA
    //          если же вас это не останавливает - профилировщик (nsight-systems) при запуске на NVIDIA тоже работает (хоть и менее мощный чем nsight-compute)
    //          кроме того есть debugPrintfEXT(...) для вывода в консоль с видеокарты
    //          кроме того используемая библиотека поддерживает rassert-проверки (своеобразные инварианты с уникальным числом) на видеокарте для Vulkan

    ocl::KernelSource ocl_fillBufferWithZeros(ocl::getFillBufferWithZeros());
    ocl::KernelSource ocl_radixSort01LocalCounting(ocl::getRadixSort01LocalCounting());
    ocl::KernelSource ocl_radixSort02GlobalPrefixesScanSumReduction(ocl::getRadixSort02GlobalPrefixesScanSumReduction());
    ocl::KernelSource ocl_radixSort03GlobalPrefixesScanAccumulation(ocl::getRadixSort03GlobalPrefixesScanAccumulation());
    ocl::KernelSource ocl_radixSort04Scatter(ocl::getRadixSort04Scatter());

    avk2::KernelSource vk_fillBufferWithZeros(avk2::getFillBufferWithZeros());
    avk2::KernelSource vk_radixSort01LocalCounting(avk2::getRadixSort01LocalCounting());
    avk2::KernelSource vk_radixSort02GlobalPrefixesScanSumReduction(avk2::getRadixSort02GlobalPrefixesScanSumReduction());
    avk2::KernelSource vk_radixSort03GlobalPrefixesScanAccumulation(avk2::getRadixSort03GlobalPrefixesScanAccumulation());
    avk2::KernelSource vk_radixSort04Scatter(avk2::getRadixSort04Scatter());

    FastRandom r;

    int n = 100*1000*1000; // TODO при отладке используйте минимальное n (например n=5 или n=10) при котором воспроизводится бага
    // const int n = 1000;
    int max_value = std::numeric_limits<int>::max(); // TODO при отладке используйте минимальное max_value (например max_value=8) при котором воспроизводится бага
    // const int max_value = 128;
    std::vector<unsigned int> as(n, 0);
    std::vector<unsigned int> sorted(n, 0);
    for (size_t i = 0; i < n; ++i) {
        as[i] = r.next(0, max_value);
    }
    std::cout << "n=" << n << " max_value=" << max_value << std::endl;

    {
        // убедимся что в массиве есть хотя бы несколько повторяющихся значений
        size_t force_duplicates_attempts = 3;
        bool all_attempts_missed = true;
        for (size_t k = 0; k < force_duplicates_attempts; ++k) {
            size_t i = r.next(0, n - 1);
            size_t j = r.next(0, n - 1);
            if (i != j) {
                as[j] = as[i];
                all_attempts_missed = false;
            }
        }
        rassert(!all_attempts_missed, 4353245123412);
    }

    {
        sorted = as;
        std::cout << "sorting on CPU..." << std::endl;
        timer t;
        std::sort(sorted.begin(), sorted.end());
        // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
        double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
        std::cout << "CPU std::sort finished in " << t.elapsed() << " sec" << std::endl;
        std::cout << "CPU std::sort effective RAM bandwidth: " << memory_size_gb / t.elapsed() << " GB/s (" << float(n) / 1000 / 1000 / t.elapsed() << " uint millions/s)" << std::endl;
    }

    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32u input_gpu(n);
    gpu::gpu_mem_32u map_flags_gpu(n), local_idx_gpu(n), psum_buf1_gpu(n), psum_buf2_gpu(n); // TODO это просто шаблонка, можете переименовать эти буферы, сделать другого размера/типа, удалить часть, добавить новые
    gpu::gpu_mem_32u output1_gpu(n), output2_gpu(n);

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);
    // Советую занулить (или еще лучше - заполнить какой-то уникальной константой, например 255) все буферы
    // В некоторых случаях это ускоряет отладку, но обратите внимание, что fill реализован через копию множества нулей по PCI-E, то есть он очень медленный
    // Если вам нужно занулять буферы в процессе вычислений - используйте кернел который это сделает (см. кернел fill_buffer_with_zeros)
    map_flags_gpu.fill(255);
    psum_buf2_gpu.fill(255);
    local_idx_gpu.fill(255);
    psum_buf1_gpu.fill(255);
    output1_gpu.fill(255);
    output2_gpu.fill(255);

    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) { // TODO при отладке запускайте одну итерацию
        timer t;

        // Запускаем кернел, с указанием размера рабочего пространства и передачей всех аргументов
        // Если хотите - можете удалить ветвление здесь и оставить только тот код который соответствует вашему выбору API
        if (context.type() == gpu::Context::TypeOpenCL) {
            // TODO
            throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
            // ocl_fillBufferWithZeros.exec();
            // ocl_radixSort01LocalCounting.exec();
            // ocl_radixSort02GlobalPrefixesScanSumReduction.exec();
            // ocl_radixSort03GlobalPrefixesScanAccumulation.exec();
            // ocl_radixSort04Scatter.exec();
        } else if (context.type() == gpu::Context::TypeCUDA) {
            // {
            //     std::vector<unsigned int> input = input_gpu.readVector();
            //     std::cout << "input: ";
            //     for (int i = 0; i < n; ++i) std::cout << input[i] << " ";
            //     std::cout << std::endl;
            // }
            std::vector<uint> bit_offs = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
            std::vector<uint> buckets = {0b00, 0b01, 0b10, 0b11};
            // std::vector<uint> buckets = {0b01};

            for (auto bit_off : bit_offs) {
                // std::cout << "=== Offset " << bit_off << " ===" << std::endl;

                auto& input_vals_gpu = (bit_off == 0) ? input_gpu : (bit_off % 4) ? output2_gpu : output1_gpu;
                auto& output_vals_gpu = (bit_off % 4) ? output1_gpu : output2_gpu;

                uint global_off = 0;
                for (auto bucket : buckets) {
                    // std::cout << "== Bucket " << bucket << " ==" << std::endl;
                    cuda::radix_sort_01_local_counting(gpu::WorkSize(GROUP_SIZE, n), input_vals_gpu, map_flags_gpu, bit_off, bucket, n);
                    // {
                    //     std::vector<unsigned int> input = input_vals_gpu.readVector();
                    //     std::cout << " input:      ";
                    //     for (int i = 0; i < n; ++i) std::cout << std::setw(8) << input[i] << " ";
                    //     std::cout << std::endl;
                    //     auto input_bits = debug::prettyBits(input_vals_gpu.readVector(), max_value << 4, bit_off, 2);
                    //     std::cout << " input vals: ";
                    //     for (int i = 0; i < n; ++i) std::cout << std::setw(8) << input_bits[i] << " ";
                    //     std::cout << std::endl;
                    //     std::vector<unsigned int> map_flags = map_flags_gpu.readVector();
                    //     std::cout << " map flags:  ";
                    //     for (int i = 0; i < n; ++i) std::cout << std::setw(8) << map_flags[i] << " ";
                    //     std::cout << std::endl;
                    // }
                    cuda::fill_buffer_with_zeros(gpu::WorkSize(GROUP_SIZE, n), psum_buf1_gpu, n);
                    cuda::fill_buffer_with_zeros(gpu::WorkSize(GROUP_SIZE, n), psum_buf2_gpu, n);
                    cuda::fill_buffer_with_zeros(gpu::WorkSize(GROUP_SIZE, n), local_idx_gpu, n);
                    run_prefix_sum(map_flags_gpu, psum_buf1_gpu, psum_buf2_gpu, local_idx_gpu, n);
                    uint local_off = 0;
                    local_idx_gpu.readN(&local_off, 1, n-1);
                    // {
                    //     std::vector<unsigned int> local_idx = local_idx_gpu.readVector();
                    //     std::cout << " local idx:  ";
                    //     for (int i = 0; i < n; ++i) std::cout << std::setw(8) << local_idx[i] << " ";
                    //     std::cout << std::endl;
                    //     std::cout << "local last off: " << local_off << std::endl;
                    // }
                    cuda::radix_sort_04_scatter(gpu::WorkSize(GROUP_SIZE, n), input_vals_gpu, local_idx_gpu, map_flags_gpu, output_vals_gpu, n, global_off);
                    // {
                    //     std::vector<unsigned int> output = output_vals_gpu.readVector();
                    //     std::cout << " output:     ";
                    //     for (int i = 0; i < n; ++i) std::cout << std::setw(8) << output[i] << " ";
                    //     std::cout << std::endl;
                    // }
                    global_off += local_off;
                }
                rassert(global_off == n, 142800184, global_off, n);
            }
            // cuda::radix_sort_02_global_prefixes_scan_sum_reduction();
            // cuda::radix_sort_03_global_prefixes_scan_accumulation();
            // cuda::radix_sort_04_scatter();
        } else if (context.type() == gpu::Context::TypeVulkan) {
            // TODO
            throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
            // vk_fillBufferWithZeros.exec();
            // vk_radixSort01LocalCounting.exec();
            // vk_radixSort02GlobalPrefixesScanSumReduction.exec();
            // vk_radixSort03GlobalPrefixesScanAccumulation.exec();
            // vk_radixSort04Scatter.exec();
        } else {
            rassert(false, 4531412341, context.type());
        }

        times.push_back(t.elapsed());
    }
    std::cout << "GPU radix-sort times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "GPU radix-sort median effective VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s (" << float(n) / 1000 / 1000 / stats::median(times) << " uint millions/s)" << std::endl;

    // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
    std::vector<unsigned int> gpu_sorted = output1_gpu.readVector();

    // {
    //     std::cout << "CPU output: ";
    //     for (int i = 0; i < n; ++i) std::cout << std::setw(3) << sorted[i] << " ";
    //     std::cout << std::endl;
    //     std::cout << "GPU output: ";
    //     for (int i = 0; i < n; ++i) std::cout << std::setw(3) << gpu_sorted[i] << " ";
    //     std::cout << std::endl;
    // }

    // Сверяем результат
    for (size_t i = 0; i < n; ++i) {
        rassert(sorted[i] == gpu_sorted[i], 566324523452323, sorted[i], gpu_sorted[i], i);
    }

    // Проверяем что входные данные остались нетронуты (ведь мы их переиспользуем от итерации к итерации)
    std::vector<unsigned int> input_values = input_gpu.readVector();
    for (size_t i = 0; i < n; ++i) {
        rassert(input_values[i] == as[i], 6573452432, input_values[i], as[i]);
    }
}

int main(int argc, char** argv)
{
    try {
        run(argc, argv);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (e.what() == DEVICE_NOT_SUPPORT_API) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за выбора CUDA API (его нет на процессоре - т.е. в случае CI на GitHub Actions)
            return 0;
        } if (e.what() == CODE_IS_NOT_IMPLEMENTED) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за того что задание еще не выполнено
            return 0;
        } else {
            // Выставляем ненулевой exit code, чтобы сообщить, что случилась ошибка
            return 1;
        }
    }

    return 0;
}
