pub mod gemm;
pub mod gemv;

pub use self::gemm::{GemmDispatchPath, GemmKernel};
use self::gemv::{GemvDispatch, GemvSpecialization, max_gemv_batch_threshold};
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

        // TODO: remove after GatherGEMM is supported
        if arguments.gather_indices.is_some() {
            return Err(MetalError::KernelDispatchFailed(
                format!(
                    "gathered readout requires the GEMV path, but shape (m={}, n={}) routes to GEMM",
                    arguments.m, arguments.n
                )
                .into(),
            ));
        }
        self.gemm.encode(arguments, encoder)
    }
}

impl MatmulMetalKernel {
    pub fn try_encode_gemv<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<bool, MetalError> {
        let Some(gemv) = GemvSpecialization::select_with_max_m(
            &arguments,
            self.weights_data_type,
            self.input_data_type,
            self.output_data_type,
            encoder.context().device_tier(),
            max_gemv_batch_threshold(),
        ) else {
            return Ok(false);
        };
        self.gemv.encode(arguments, gemv, encoder).map_err(MetalError::from)?;
        Ok(true)
    }
}
