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
            gpu_types::gemm::GemmTiling,
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
    fn prefer_gemm_over_gemv(
        shape: MatmulShape,
        plan: GemmPlan,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> bool {
        if shape.gathered || plan.engine != gemm::GemmEngine::Mxu {
            return false;
        }
        match (shape.m, shape.n == shape.k, (weights_data_type, input_data_type, output_data_type)) {
            (4, true, (DataType::F32, DataType::F32, DataType::F32))
            | (5, _, (DataType::BF16, DataType::BF16, DataType::BF16)) => return false,
            _ => {},
        }
        match shape.m {
            0..=3 => return false,
            4 => {
                let small_enough_for_mxu = shape.n <= 6144 && shape.k <= 9728;
                let k_dominates = shape.k > 3_u32.saturating_mul(shape.n);
                if !(small_enough_for_mxu || k_dominates) {
                    return false;
                }
            },
            _ => {},
        }
        matches!(plan.tiling, GemmTiling::Tile16x32x256_Simdgroups1x1 | GemmTiling::Tile16x128x256_Simdgroups1x4)
    }

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
            context.device_tier,
        );
        let problem = GemmProblem::new(*shape, self.weights_data_type, self.output_data_type, context.supports_mxu());
        let plan = problem.select_plan();
        match gemv {
            None => MatmulDispatch::Gemm(plan),
            Some(_)
                if Self::prefer_gemm_over_gemv(
                    *shape,
                    plan,
                    self.weights_data_type,
                    self.input_data_type,
                    self.output_data_type,
                ) =>
            {
                MatmulDispatch::Gemm(plan)
            },
            Some(gemv) => MatmulDispatch::Gemv(gemv),
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
        let plan = match self.select_dispatch(&shape, encoder.context) {
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
