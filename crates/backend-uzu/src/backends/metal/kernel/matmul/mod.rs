pub mod gemm;
pub mod gemv;

pub use self::gemm::{GemmDispatchPath, GemmKernel};
use self::gemv::{GemvDispatch, GemvSpecialization};
use crate::{
    backends::{
        common::{
            BufferArg, Encoder,
            kernel::matmul::{MatmulArguments, MatmulError, MatmulKernel},
        },
        metal::{Metal, context::MetalContext, error::MetalError, metal_extensions::DeviceExt},
    },
    data_type::DataType,
};

pub struct MatmulMetalKernel {
    gemv: GemvDispatch,
    pub gemm: GemmKernel,
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
}

impl MatmulKernel for MatmulMetalKernel {
    type Backend = Metal;

    fn new(
        context: &MetalContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, MetalError> {
        for data_type in [weights_data_type, input_data_type, output_data_type] {
            if !matches!(data_type, DataType::BF16 | DataType::F32) {
                return Err(MatmulError::<Metal>::UnsupportedDataType(data_type).into());
            }
        }

        let gemm = GemmKernel::new(context, weights_data_type, input_data_type, output_data_type)?;
        let gemv = GemvDispatch::new(weights_data_type, input_data_type, output_data_type);

        Ok(Self {
            gemv,
            gemm,
            weights_data_type,
            input_data_type,
            output_data_type,
        })
    }

    fn supports_int8_symmetric_a(
        &self,
        context: &MetalContext,
        m: u32,
        n: u32,
        k: u32,
        group_size: u32,
    ) -> bool {
        if !context.supports_mxu()
            || group_size == 0
            || !group_size.is_multiple_of(32)
            || !k.is_multiple_of(group_size)
            || ![self.weights_data_type, self.input_data_type, self.output_data_type]
                .into_iter()
                .all(|data_type| data_type == DataType::BF16)
        {
            return false;
        }

        let tiling = gemm::select_mxu_quant_tiling(m, n, group_size);
        k.is_multiple_of(tiling.block_k())
    }

    fn encode<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let skip_gemv = encoder.context().device.supports_mxu() && self.gemm.should_skip_gemv_for_mxu(&arguments);
        if !skip_gemv
            && let Some(gemv) = GemvSpecialization::select(
                &arguments,
                self.weights_data_type,
                self.input_data_type,
                self.output_data_type,
                encoder.context().device_tier(),
            )
        {
            return self.gemv.encode(arguments, gemv, encoder).map_err(MetalError::from);
        }
        self.gemm.encode(arguments, encoder)
    }
}
