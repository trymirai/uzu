use super::{super::grid_size::GridSize, specialization::GemmSpecialization};
use crate::{
    DataType,
    backends::common::{
        Backend,
        gpu_types::GemmParams,
        kernel::matmul::{MatmulArguments, MatmulError},
    },
};

#[derive(Debug, Clone)]
pub struct GemmDispatchDescriptor {
    pub specialization: GemmSpecialization,
    pub params: GemmParams,
    pub threadgroups: GridSize,
}

impl GemmDispatchDescriptor {
    pub fn try_new<B: Backend>(
        context: &B::Context,
        data_type: DataType,
        arguments: &MatmulArguments<B>,
    ) -> Result<Self, MatmulError<B>> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(MatmulError::UnsupportedDataType(data_type));
        }

        let config = GemmSpecialization::select::<B>(context, data_type, arguments);

        let m = arguments.batch;
        let n = arguments.output_dim;
        let k = arguments.input_dim;

        let threadgroups_per_row = (n + config.block_cols - 1) / config.block_cols;
        let threadgroups_per_column = (m + config.block_rows - 1) / config.block_rows;
        let swizzle_log = config.swizzle_log2;

        let swizzle_stride = 1_i32 << swizzle_log;
        let tm_swizzled = (threadgroups_per_column + swizzle_stride - 1) / swizzle_stride;
        let tn_swizzled = threadgroups_per_row * swizzle_stride;

        let params = GemmParams {
            M: m,
            N: n,
            K: k,
            leading_dimension_a: arguments.leading_dimension_a,
            leading_dimension_b: arguments.leading_dimension_b,
            leading_dimension_d: arguments.leading_dimension_d,
            threadgroups_per_row,
            threadgroups_per_column,
            swizzle_log,
            aligned_inner_iterations: k / config.block_depth,
        };

        let threadgroups = GridSize {
            x: tn_swizzled as usize,
            y: tm_swizzled as usize,
            z: 1,
        };

        Ok(Self {
            specialization: config,
            params,
            threadgroups,
        })
    }
}
