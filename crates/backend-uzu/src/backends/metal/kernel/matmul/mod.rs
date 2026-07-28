pub mod gemm;
pub mod gemv;

pub use self::gemm::{GemmDispatchPath, GemmKernel};
use self::gemv::{GemvDispatch, GemvSpecialization};
use crate::{
    backends::{
        common::{
            BufferArg, Encoder,
            gpu_types::gemm::GemmBPrologueKind,
            kernel::matmul::{MatmulArguments, MatmulError, MatmulKernel, MatmulPath, MatmulShape},
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

    fn select_path(
        &self,
        shape: &MatmulShape,
        b_prologue: GemmBPrologueKind,
        context: &MetalContext,
    ) -> MatmulPath {
        let skip_gemv = context.device.supports_mxu() && self.gemm.should_skip_gemv_for_mxu_shape(shape, b_prologue);
        if !skip_gemv
            && GemvSpecialization::select_shape(
                shape,
                b_prologue,
                self.weights_data_type,
                self.input_data_type,
                self.output_data_type,
                context.device_tier(),
            )
            .is_some()
        {
            return MatmulPath::Gemv;
        }
        MatmulPath::Gemm
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
