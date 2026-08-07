pub mod gemm;
pub mod gemv;

pub use self::gemm::GemmKernel;
use self::{
    gemm::{GemmPlan, GemmProblem},
    gemv::{GemvDispatch, GemvSpecialization},
};
use crate::{
    backends::{
        common::{
            BufferArg, Encoder,
            kernel::matmul::{MatmulArguments, MatmulError, MatmulKernel, MatmulPath, MatmulShape},
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

enum MatmulDispatch {
    Gemv(GemvSpecialization),
    Gemm(GemmPlan),
}

impl MatmulMetalKernel {
    fn select_dispatch(
        &self,
        shape: &MatmulShape,
        context: &MetalContext,
    ) -> MatmulDispatch {
        let gemv = GemvSpecialization::select_shape(
            shape,
            self.weights_data_type,
            self.input_data_type,
            self.output_data_type,
            context.device_tier(),
        );
        let problem = GemmProblem::new(
            *shape,
            self.weights_data_type,
            self.input_data_type,
            self.output_data_type,
            context.supports_mxu(),
        );
        match problem.plan_for_dispatch(gemv.is_some()) {
            Some(plan) => MatmulDispatch::Gemm(plan),
            None => MatmulDispatch::Gemv(gemv.expect("GEMV must be available when GEMM is not selected")),
        }
    }
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
        context: &MetalContext,
    ) -> MatmulPath {
        match self.select_dispatch(shape, context) {
            MatmulDispatch::Gemv(_) => MatmulPath::Gemv,
            MatmulDispatch::Gemm(_) => MatmulPath::Gemm,
        }
    }

    fn encode<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let shape = MatmulShape::from_arguments(&arguments);
        let plan = match self.select_dispatch(&shape, encoder.context()) {
            MatmulDispatch::Gemv(gemv) => {
                return self.gemv.encode(arguments, gemv, encoder).map_err(MetalError::from);
            },
            MatmulDispatch::Gemm(plan) => plan,
        };

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
        self.gemm.encode_plan(arguments, plan, encoder)
    }
}
