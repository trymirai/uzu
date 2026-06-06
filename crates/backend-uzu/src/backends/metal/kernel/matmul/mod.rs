pub mod gemm;
pub mod gemv;

pub use self::gemm::{GemmDispatchPath, GemmKernel};
use self::gemv::{GemvDispatch, GemvSpecialization};
use crate::{
    backends::{
        common::{
            AsBufferRangeRef, Buffer, Encoder,
            gpu_types::gemm::GemmDTransform,
            kernel::matmul::{MatmulArguments, MatmulError, MatmulKernel, MatmulQuantCombo},
        },
        metal::{Metal, context::MetalContext, error::MetalError},
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
        let gemv = GemvDispatch::new(context, weights_data_type, input_data_type, output_data_type)
            .map_err(MetalError::from)?;

        Ok(Self {
            gemv,
            gemm,
            weights_data_type,
            input_data_type,
            output_data_type,
        })
    }

    fn encode<TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        arguments: MatmulArguments<Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        match GemvSpecialization::select(
            &arguments,
            self.weights_data_type,
            self.input_data_type,
            self.output_data_type,
        ) {
            Some(spec) => self.gemv.encode(arguments, spec, encoder).map_err(MetalError::from),
            None => self.gemm.encode(arguments, encoder),
        }
    }

    fn preheat_quant_combo(
        &mut self,
        context: &MetalContext,
        combo: MatmulQuantCombo,
        output_dim: u32,
        input_dim: u32,
    ) -> Result<(), MetalError> {
        self.gemv.preheat_quant_combo(context, combo, input_dim).map_err(MetalError::from)?;
        self.gemm.preheat_quant_combo(context, combo, output_dim, input_dim)
    }

    fn preheat_full_precision(
        &mut self,
        context: &MetalContext,
        output_dim: u32,
        input_dim: u32,
        has_bias: bool,
    ) -> Result<(), MetalError> {
        let output_transform = if has_bias {
            GemmDTransform::BIAS
        } else {
            GemmDTransform::empty()
        };
        self.gemm.preheat_full_precision(context, output_dim, input_dim, output_transform)
    }
}
