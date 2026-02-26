use uzu::{
    DataType,
    backends::{
        common::{
            gpu_types::GEMMSpiltKParams,
            kernel::matmul::{
                MatmulArguments, MatmulDispatchDescriptor, GridSize, gemm, gemm_mixed_types_simple, gemm_mpp,
                gemv::{DispatchDescriptor as GemvDescriptor, OutputSource as GemvOutputSource, Specialization as GemvSpecialization},
                split_k::{DispatchDescriptor as SplitKDescriptor, TileConfiguration},
            },
        },
        metal::{Metal, MetalContext},
    },
};

use super::combos::DtypeCombo;

pub fn try_all_descriptors(
    context: &MetalContext,
    combo: &DtypeCombo,
    arguments: &MatmulArguments<Metal>,
) -> Vec<(&'static str, MatmulDispatchDescriptor)> {
    let mut descriptors = Vec::new();

    if let Ok(descriptor) = gemm_mpp::DispatchDescriptor::new(combo.output_dtype, arguments) {
        descriptors.push(("GemmMpp", MatmulDispatchDescriptor::GemmMpp(descriptor)));
    }

    if gemm_mixed_types_simple::supports_combo(combo.a_dtype, combo.b_dtype, combo.output_dtype) {
        if let Ok(descriptor) = gemm_mixed_types_simple::DispatchDescriptor::new(combo.output_dtype, arguments) {
            descriptors.push(("MixedTypesSimpleGemm", MatmulDispatchDescriptor::GemmMixedTypesSimple(descriptor)));
        }
    }

    let is_same_type = combo.a_dtype == combo.b_dtype && combo.b_dtype == combo.output_dtype;
    if is_same_type && matches!(combo.output_dtype, DataType::F16 | DataType::BF16 | DataType::F32) {
        if let Ok(descriptor) = gemm::DispatchDescriptor::new(context, combo.output_dtype, arguments) {
            descriptors.push(("Gemm", MatmulDispatchDescriptor::Gemm(descriptor)));
        }

        if let Some(descriptor) = force_gemv_descriptor(combo.output_dtype, arguments) {
            descriptors.push(("Gemv", MatmulDispatchDescriptor::Gemv(descriptor)));
        }

        if let Some(descriptor) = force_splitk_descriptor(arguments) {
            descriptors.push(("SplitK", MatmulDispatchDescriptor::SplitK(descriptor)));
        }
    }

    descriptors
}

fn force_gemv_descriptor(
    data_type: DataType,
    arguments: &MatmulArguments<Metal>,
) -> Option<GemvDescriptor> {
    if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
        return None;
    }
    if !arguments.transpose_b {
        return None;
    }

    let n = arguments.output_dim;
    if n == 1 && arguments.batch != 1 {
        return None;
    }

    let matrix_is_rhs = n != 1;
    let transpose_matrix = if matrix_is_rhs { !arguments.transpose_b } else { false };

    let output_source = if arguments.bias.is_some() {
        GemvOutputSource::Bias
    } else {
        GemvOutputSource::None
    };

    let (apply_output_scale_and_accumulate, alpha, _beta, bias_stride) = match output_source {
        GemvOutputSource::None => (false, 1.0f32, 0.0f32, 0),
        GemvOutputSource::Bias => (true, 1.0f32, 1.0f32, 1),
    };

    let output_dimension = if matrix_is_rhs { arguments.output_dim } else { arguments.batch };

    let specialization = GemvSpecialization::select(
        transpose_matrix,
        arguments.input_dim,
        output_dimension,
        apply_output_scale_and_accumulate,
    );

    let input_dimension = arguments.input_dim;
    let matrix_leading_dim = if matrix_is_rhs { arguments.ldb } else { arguments.lda };

    let batch_shape = [if arguments.batch_count > 1 { arguments.batch_count } else { 1 }];

    let elements_per_matrix_a = (arguments.batch as i64) * (arguments.lda as i64);
    let elements_per_matrix_b = if arguments.transpose_b {
        (arguments.output_dim as i64) * (arguments.ldb as i64)
    } else {
        (arguments.input_dim as i64) * (arguments.ldb as i64)
    };

    let vector_batch_stride = [if matrix_is_rhs { elements_per_matrix_a } else { elements_per_matrix_b }];
    let matrix_batch_stride = [if matrix_is_rhs { elements_per_matrix_b } else { elements_per_matrix_a }];
    let bias_batch_stride =
        [if arguments.batch_count > 1 { (output_dimension as i64) * (arguments.ldd as i64) } else { 0 }];

    Some(GemvDescriptor {
        specialization,
        matrix_is_rhs,
        output_source,
        input_dimension,
        output_dimension,
        matrix_leading_dim,
        alpha,
        beta: if arguments.bias.is_some() { 1.0 } else { 0.0 },
        batch_shape,
        vector_batch_stride,
        matrix_batch_stride,
        bias_batch_stride,
        bias_stride,
        batch_rows: arguments.batch,
    })
}

fn force_splitk_descriptor(arguments: &MatmulArguments<Metal>) -> Option<SplitKDescriptor> {
    if !arguments.transpose_b || arguments.batch_count != 1 {
        return None;
    }

    let m = arguments.batch;
    let n = arguments.output_dim;
    let k = arguments.input_dim;

    if m <= 0 || n <= 0 || k <= 0 {
        return None;
    }

    let tile = TileConfiguration {
        tile_rows: 16,
        tile_cols: 32,
        tile_depth: 16,
        warps_per_row: 2,
        warps_per_col: 2,
    };

    let k_aligned = (k % tile.tile_depth) == 0;
    if !k_aligned {
        return None;
    }

    let k_tiles = k / 16;
    let partition_count = if k_tiles < 16 {
        2
    } else if k_tiles < 32 {
        4
    } else if k_tiles < 64 {
        8
    } else {
        16
    };

    let tile_count_m = (m + tile.tile_rows - 1) / tile.tile_rows;
    let tile_count_n = (n + tile.tile_cols - 1) / tile.tile_cols;
    let gemm_k_iterations = (k / tile.tile_depth) / partition_count;
    let k_elements_per_partition = gemm_k_iterations * tile.tile_depth;
    let output_elements_per_partition = m * n;
    let accumulator_element_count = partition_count * output_elements_per_partition * arguments.batch_count;
    let accumulator_bytes = (accumulator_element_count as usize) * std::mem::size_of::<f32>();

    Some(SplitKDescriptor {
        params: GEMMSpiltKParams {
            M: m,
            N: n,
            K: k,
            lda: arguments.lda,
            ldb: arguments.ldb,
            ldc: n,
            tiles_n: tile_count_n,
            tiles_m: tile_count_m,
            split_k_partitions: partition_count,
            split_k_partition_stride: output_elements_per_partition,
            split_k_partition_size: k_elements_per_partition,
            gemm_k_iterations_aligned: gemm_k_iterations,
        },
        partition_count,
        output_elements_per_partition,
        accumulator_bytes,
        partial_threadgroups: GridSize {
            x: tile_count_n as usize,
            y: tile_count_m as usize,
            z: partition_count as usize,
        },
        accum_total_threads: GridSize {
            x: n as usize,
            y: m as usize,
            z: 1,
        },
    })
}
