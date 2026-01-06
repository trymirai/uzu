// SPDX-License-Identifier: MIT

#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

template <
    typename DataType,
    typename AccumulatorType,
    ushort SIMDGROUPS_PER_ROW,
    ushort SIMDGROUPS_PER_REDUCTION,
    ushort THREADS_PER_SIMDGROUP_ROW,
    ushort THREADS_PER_SIMDGROUP_COL,
    ushort ELEMENTS_PER_THREAD_ROW,
    ushort ELEMENTS_PER_THREAD_COL,
    bool HAS_BIAS>
struct GemvKernel {

    static constant constexpr ushort TOTAL_THREADS_ROW = SIMDGROUPS_PER_ROW * THREADS_PER_SIMDGROUP_ROW;
    static constant constexpr ushort TOTAL_THREADS_COL = SIMDGROUPS_PER_REDUCTION * THREADS_PER_SIMDGROUP_COL;
    static constant constexpr ushort OUTPUT_ELEMENTS_PER_THREADGROUP = TOTAL_THREADS_ROW * ELEMENTS_PER_THREAD_ROW;
    static constant constexpr ushort INPUT_ELEMENTS_PER_ITERATION = TOTAL_THREADS_COL * ELEMENTS_PER_THREAD_COL;
    static constant constexpr bool NEEDS_THREADGROUP_REDUCTION = SIMDGROUPS_PER_REDUCTION > 1;
    static constant constexpr ushort THREADGROUP_MEMORY_SIZE = NEEDS_THREADGROUP_REDUCTION
        ? SIMDGROUPS_PER_REDUCTION * (OUTPUT_ELEMENTS_PER_THREADGROUP + ELEMENTS_PER_THREAD_ROW)
        : 0;

    static_assert(THREADS_PER_SIMDGROUP_ROW * THREADS_PER_SIMDGROUP_COL == 32,
                  "simdgroup must have exactly 32 threads");

    template <typename DestinationType>
    static METAL_FUNC void load_input_elements_unchecked(
        const device DataType* source,
        thread DestinationType destination[ELEMENTS_PER_THREAD_COL],
        int source_offset
    ) {
        #pragma unroll(ELEMENTS_PER_THREAD_COL)
        for (ushort element_index = 0; element_index < ELEMENTS_PER_THREAD_COL; ++element_index) {
            destination[element_index] = static_cast<DestinationType>(source[source_offset + element_index]);
        }
    }

    template <typename DestinationType>
    static METAL_FUNC void load_input_elements_checked(
        const device DataType* source,
        thread DestinationType destination[ELEMENTS_PER_THREAD_COL],
        int source_offset,
        int source_size
    ) {
        #pragma unroll(ELEMENTS_PER_THREAD_COL)
        for (ushort element_index = 0; element_index < ELEMENTS_PER_THREAD_COL; ++element_index) {
            int global_index = source_offset + element_index;
            destination[element_index] = (global_index < source_size)
                ? static_cast<DestinationType>(source[global_index])
                : DestinationType(0);
        }
    }

    static METAL_FUNC void execute(
        const device DataType* weight_matrix,
        const device DataType* input_vector,
        const device DataType* bias_vector,
        device DataType* output_vector,
        constant int& input_dimension,
        constant int& output_dimension,
        constant int& weight_row_stride,
        constant int& input_batch_stride,
        constant int& output_batch_stride,
        threadgroup AccumulatorType* shared_memory,
        uint3 threadgroup_position,
        uint simdgroup_index,
        uint thread_index_in_simdgroup
    ) {
        thread AccumulatorType accumulated_results[ELEMENTS_PER_THREAD_ROW] = {0};
        thread DataType weight_elements[ELEMENTS_PER_THREAD_COL];
        thread AccumulatorType input_elements[ELEMENTS_PER_THREAD_COL];

        int thread_row_in_simdgroup = (THREADS_PER_SIMDGROUP_COL != 32)
            ? (thread_index_in_simdgroup / THREADS_PER_SIMDGROUP_COL)
            : 0;
        int thread_col_in_simdgroup = (THREADS_PER_SIMDGROUP_COL != 32)
            ? (thread_index_in_simdgroup % THREADS_PER_SIMDGROUP_COL)
            : int(thread_index_in_simdgroup);

        int simdgroup_reduction_index = NEEDS_THREADGROUP_REDUCTION
            ? (simdgroup_index % SIMDGROUPS_PER_REDUCTION)
            : 0;

        int simdgroup_row_offset = NEEDS_THREADGROUP_REDUCTION
            ? THREADS_PER_SIMDGROUP_ROW * (simdgroup_index / SIMDGROUPS_PER_REDUCTION)
            : int(THREADS_PER_SIMDGROUP_ROW * simdgroup_index);

        int simdgroup_col_offset = NEEDS_THREADGROUP_REDUCTION
            ? THREADS_PER_SIMDGROUP_COL * simdgroup_reduction_index
            : 0;

        int thread_output_offset = (simdgroup_row_offset + thread_row_in_simdgroup) * ELEMENTS_PER_THREAD_ROW;
        int thread_input_offset = (simdgroup_col_offset + thread_col_in_simdgroup) * ELEMENTS_PER_THREAD_COL;

        int batch_index = threadgroup_position.z;
        int output_row_start = threadgroup_position.x * OUTPUT_ELEMENTS_PER_THREADGROUP + thread_output_offset;

        if (output_row_start >= output_dimension) {
    return;
  }

        output_row_start = (output_row_start + ELEMENTS_PER_THREAD_ROW <= output_dimension)
            ? output_row_start
            : (output_dimension - ELEMENTS_PER_THREAD_ROW);

        const device DataType* batch_input = input_vector + batch_index * input_batch_stride;
        device DataType* batch_output = output_vector + batch_index * output_batch_stride;
        const device DataType* weight_rows = weight_matrix + output_row_start * weight_row_stride;

        int full_iterations = input_dimension / INPUT_ELEMENTS_PER_ITERATION;
        int remaining_elements = input_dimension - (full_iterations * INPUT_ELEMENTS_PER_ITERATION);

        int current_input_offset = thread_input_offset;

        for (int iteration = 0; iteration < full_iterations; ++iteration) {
            load_input_elements_unchecked<AccumulatorType>(batch_input, input_elements, current_input_offset);

            int weight_offset = 0;
            #pragma unroll(ELEMENTS_PER_THREAD_ROW)
            for (ushort row = 0; row < ELEMENTS_PER_THREAD_ROW; ++row) {
                load_input_elements_unchecked<DataType>(weight_rows, weight_elements, weight_offset + current_input_offset);

                #pragma unroll(ELEMENTS_PER_THREAD_COL)
                for (ushort col = 0; col < ELEMENTS_PER_THREAD_COL; ++col) {
                    accumulated_results[row] += static_cast<AccumulatorType>(weight_elements[col]) * input_elements[col];
                }

                weight_offset += weight_row_stride;
            }

            current_input_offset += INPUT_ELEMENTS_PER_ITERATION;
        }

        if (remaining_elements > 0) {
            load_input_elements_checked<AccumulatorType>(batch_input, input_elements, current_input_offset, input_dimension);

            #pragma unroll(ELEMENTS_PER_THREAD_ROW)
            for (ushort row = 0; row < ELEMENTS_PER_THREAD_ROW; ++row) {
                load_input_elements_checked<DataType>(
                    weight_rows + row * weight_row_stride,
                    weight_elements,
                    current_input_offset,
                    input_dimension
                );

                #pragma unroll(ELEMENTS_PER_THREAD_COL)
                for (ushort col = 0; col < ELEMENTS_PER_THREAD_COL; ++col) {
                    accumulated_results[row] += static_cast<AccumulatorType>(weight_elements[col]) * input_elements[col];
                }
            }
        }

        #pragma unroll(ELEMENTS_PER_THREAD_ROW)
        for (ushort row = 0; row < ELEMENTS_PER_THREAD_ROW; ++row) {
            #pragma unroll
            for (ushort shuffle_offset = (THREADS_PER_SIMDGROUP_COL / 2); shuffle_offset >= 1; shuffle_offset >>= 1) {
                accumulated_results[row] += simd_shuffle_down(accumulated_results[row], shuffle_offset);
            }
        }

        if (NEEDS_THREADGROUP_REDUCTION) {
            threadgroup AccumulatorType* reduction_memory =
                shared_memory + simdgroup_reduction_index * (OUTPUT_ELEMENTS_PER_THREADGROUP + ELEMENTS_PER_THREAD_ROW)
                + thread_output_offset;

            if (thread_col_in_simdgroup == 0) {
                #pragma unroll(ELEMENTS_PER_THREAD_ROW)
                for (ushort row = 0; row < ELEMENTS_PER_THREAD_ROW; ++row) {
                    reduction_memory[row] = accumulated_results[row];
                }

                threadgroup_barrier(mem_flags::mem_none);

                if (simdgroup_reduction_index == 0) {
                    #pragma unroll
                    for (ushort other_simdgroup = 1; other_simdgroup < SIMDGROUPS_PER_REDUCTION; ++other_simdgroup) {
                        #pragma unroll(ELEMENTS_PER_THREAD_ROW)
                        for (ushort row = 0; row < ELEMENTS_PER_THREAD_ROW; ++row) {
                            accumulated_results[row] +=
                                reduction_memory[other_simdgroup * (OUTPUT_ELEMENTS_PER_THREADGROUP + ELEMENTS_PER_THREAD_ROW) + row];
                        }
                    }
                }
            }
        }

        bool is_first_thread_in_reduction = (simdgroup_col_offset == 0) && (thread_col_in_simdgroup == 0);

        if (is_first_thread_in_reduction) {
            #pragma unroll(ELEMENTS_PER_THREAD_ROW)
            for (ushort row = 0; row < ELEMENTS_PER_THREAD_ROW; ++row) {
                int output_index = output_row_start + row;
                if (HAS_BIAS) {
                    batch_output[output_index] =
                        static_cast<DataType>(accumulated_results[row]) + bias_vector[output_index];
                } else {
                    batch_output[output_index] = static_cast<DataType>(accumulated_results[row]);
                }
            }
        }
    }
};

#define INSTANTIATE_GEMV_KERNEL(                                                              \
    kernel_name, data_type, accumulator_type,                                                 \
    simdgroups_per_row, simdgroups_per_reduction,                                             \
    threads_per_simdgroup_row, threads_per_simdgroup_col,                                     \
    elements_per_thread_row, elements_per_thread_col, has_bias)                               \
                                                                                              \
[[kernel, max_total_threads_per_threadgroup(                                                  \
    simdgroups_per_row * simdgroups_per_reduction * 32)]]                                     \
void kernel_name(                                                                             \
    const device data_type* weight_matrix [[buffer(0)]],                                      \
    const device data_type* input_vector [[buffer(1)]],                                       \
    const device data_type* bias_vector [[buffer(2)]],                                        \
    device data_type* output_vector [[buffer(3)]],                                            \
    constant int& input_dimension [[buffer(4)]],                                              \
    constant int& output_dimension [[buffer(5)]],                                             \
    constant int& weight_row_stride [[buffer(6)]],                                            \
    constant int& input_batch_stride [[buffer(7)]],                                           \
    constant int& output_batch_stride [[buffer(8)]],                                          \
    uint3 threadgroup_position [[threadgroup_position_in_grid]],                              \
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],                                  \
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]                              \
) {                                                                                           \
    using Kernel = GemvKernel<                                                                \
        data_type, accumulator_type,                                                          \
        simdgroups_per_row, simdgroups_per_reduction,                                         \
        threads_per_simdgroup_row, threads_per_simdgroup_col,                                 \
        elements_per_thread_row, elements_per_thread_col, has_bias>;                          \
                                                                                              \
    threadgroup accumulator_type shared_memory[                                               \
        Kernel::THREADGROUP_MEMORY_SIZE == 0 ? 1 : Kernel::THREADGROUP_MEMORY_SIZE];          \
                                                                                              \
    Kernel::execute(                                                                          \
        weight_matrix,                                                                        \
        input_vector,                                                                         \
        bias_vector,                                                                          \
        output_vector,                                                                        \
        input_dimension,                                                                      \
        output_dimension,                                                                     \
        weight_row_stride,                                                                    \
        input_batch_stride,                                                                   \
        output_batch_stride,                                                                  \
        shared_memory,                                                                        \
        threadgroup_position,                                                                 \
        simdgroup_index,                                                                      \
        thread_index_in_simdgroup                                                             \
    );                                                                                        \
}

#define INSTANTIATE_GEMV_CONFIG(dtype, acc_type, rows, reduction, thr_row, thr_col, elem_row, elem_col) \
    INSTANTIATE_GEMV_KERNEL(                                                                            \
        gemv_##dtype##_rows##rows##_reduction##reduction##_elements##elem_row,                          \
        dtype, acc_type, rows, reduction, thr_row, thr_col, elem_row, elem_col, false)                  \
    INSTANTIATE_GEMV_KERNEL(                                                                            \
        gemv_##dtype##_rows##rows##_reduction##reduction##_elements##elem_row##_bias,                   \
        dtype, acc_type, rows, reduction, thr_row, thr_col, elem_row, elem_col, true)

#define INSTANTIATE_ALL_CONFIGS_FOR_DTYPE(dtype, acc_type)                                              \
    INSTANTIATE_GEMV_CONFIG(dtype, acc_type, 1, 8, 1, 32, 4, 4)                                          \
    INSTANTIATE_GEMV_CONFIG(dtype, acc_type, 1, 8, 1, 32, 1, 4)                                          \
    INSTANTIATE_GEMV_CONFIG(dtype, acc_type, 1, 1, 8,  4, 4, 4)                                          \
    INSTANTIATE_GEMV_CONFIG(dtype, acc_type, 1, 1, 8,  4, 1, 4)                                          \
    INSTANTIATE_GEMV_CONFIG(dtype, acc_type, 4, 1, 1, 32, 1, 4)                                          \
    INSTANTIATE_GEMV_CONFIG(dtype, acc_type, 4, 1, 1, 32, 4, 4)                                          \
    INSTANTIATE_GEMV_CONFIG(dtype, acc_type, 8, 1, 1, 32, 4, 4)

INSTANTIATE_ALL_CONFIGS_FOR_DTYPE(half, float)
INSTANTIATE_ALL_CONFIGS_FOR_DTYPE(bfloat, float)
INSTANTIATE_ALL_CONFIGS_FOR_DTYPE(float, float)
