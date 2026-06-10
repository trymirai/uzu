pub mod gemm;
pub mod gemv;

pub use self::gemm::{GemmDispatchPath, GemmKernel};
use self::{
    gemm::GemmDispatchPlan,
    gemv::{GemvDispatch, GemvSpecialization},
};
use crate::{
    backends::{
        common::{
            AsBufferRangeRef, Buffer, Encoder,
            kernel::matmul::{MatmulArguments, MatmulError, MatmulKernel, MatmulTask},
        },
        metal::{Metal, context::MetalContext, error::MetalError, metal_extensions::DeviceExt},
    },
    data_type::DataType,
};

enum MatmulPlan {
    Gemv(GemvSpecialization),
    Gemm(GemmDispatchPlan),
}

pub struct MatmulMetalKernel {
    gemv: GemvDispatch,
    pub gemm: GemmKernel,
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
}

impl MatmulMetalKernel {
    fn plan(
        &self,
        task: &MatmulTask,
        mxu_supported: bool,
    ) -> Result<MatmulPlan, MetalError> {
        match GemvSpecialization::select(task, self.weights_data_type, self.input_data_type, self.output_data_type) {
            Some(spec) => Ok(MatmulPlan::Gemv(spec)),
            None => {
                let path = self.gemm.pick_path(task, mxu_supported);
                Ok(MatmulPlan::Gemm(self.gemm.plan(task, path)?))
            },
        }
    }

    fn resolve(
        &mut self,
        context: &MetalContext,
        plan: &MatmulPlan,
    ) -> Result<(), MetalError> {
        match plan {
            MatmulPlan::Gemv(spec) => {
                self.gemv.get_or_create(context, *spec).map_err(MetalError::from)?;
            },
            MatmulPlan::Gemm(gemm_plan) => {
                self.gemm.resolve(context, gemm_plan)?;
            },
        }
        Ok(())
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
        let mxu_supported = encoder.context().device.supports_mxu();
        match self.plan(&arguments.task(), mxu_supported)? {
            MatmulPlan::Gemv(spec) => self.gemv.encode(arguments, spec, encoder).map_err(MetalError::from),
            MatmulPlan::Gemm(plan) => self.gemm.dispatch(arguments, plan, encoder),
        }
    }

    fn precompile(
        &mut self,
        context: &MetalContext,
        task: &MatmulTask,
        batch_sizes: &[u32],
    ) -> Result<(), MetalError> {
        let mxu_supported = context.device.supports_mxu();
        for &m in batch_sizes {
            let plan = self.plan(&task.with_m(m), mxu_supported)?;
            self.resolve(context, &plan)?;
        }
        Ok(())
    }
}
