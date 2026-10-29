#[cfg(test)]
use crate::backends::common::CommandBufferEncoding;
use crate::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CpuBuffer,
            kernel::matmul::{ActivationFormat, MatmulArguments, MatmulKernel, MatmulShape},
        },
        metal::{
            Metal,
            command_buffer::MetalCommandBufferEncoding,
            context::MetalContext,
            error::MetalError,
            gpu_types::gemm::GemmBLayout,
            kernel::matmul::{
                GemmKernel, MatmulMetalError,
                gemv::GemvKernel,
                matmul_output_work::MatmulOutputWork,
                selection::{MatmulDispatch, MatmulSelector},
            },
        },
    },
    data_type::DataType,
};

pub struct MatmulMetalKernel {
    gemv: GemvKernel,
    pub gemm: GemmKernel,
    output_work: MatmulOutputWork,
    selector: MatmulSelector,
}

impl MatmulKernel for MatmulMetalKernel {
    type Backend = Metal;
    type Error = MatmulMetalError;
    type WeightsLayout = GemmBLayout;

    fn new(
        context: &MetalContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, Self::Error> {
        let output_work = MatmulOutputWork::new(context, weights_data_type, output_data_type)?;
        let gemm = GemmKernel::new(weights_data_type, input_data_type, output_data_type);
        let gemv = GemvKernel::new(weights_data_type, input_data_type, output_data_type);

        Ok(Self {
            gemv,
            gemm,
            output_work,
            selector: MatmulSelector {
                gpu_core_count: context.gpu_core_count,
                apple_gpu_family: context.apple_gpu_family,
                supports_mxu: context.supports_mxu,
                data_types: (weights_data_type, input_data_type, output_data_type),
            },
        })
    }

    fn activation_preparation(
        &self,
        shape: &MatmulShape,
        layout: GemmBLayout,
        _context: &MetalContext,
    ) -> ActivationFormat {
        self.selector.activation_preparation(shape, layout)
    }

    fn select_activation_format(
        &self,
        shape: &MatmulShape,
        layout: GemmBLayout,
        prepared: ActivationFormat,
        _context: &MetalContext,
    ) -> ActivationFormat {
        self.selector.select(shape, layout, prepared).activation_format
    }

    fn optimize_weights_layout(
        &self,
        a8_shape: &MatmulShape,
        weights: &mut <Metal as Backend>::GlobalGpuBuffer,
        context: &MetalContext,
    ) -> Result<(GemmBLayout, bool), MetalError> {
        let prepared_shape = MatmulShape {
            signed_codes: a8_shape.signed_codes || a8_shape.b_bits == Some(8),
            ..*a8_shape
        };
        if !matches!(
            self.selector.activation_preparation(&prepared_shape, GemmBLayout::RowMajor),
            ActivationFormat::Int8(_)
        ) {
            return Ok((GemmBLayout::RowMajor, a8_shape.signed_codes));
        }
        if !a8_shape.signed_codes && prepared_shape.signed_codes {
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    weights.cpu_ptr().as_ptr().cast::<u8>(),
                    crate::backends::common::Buffer::size(weights),
                )
            };
            let (prefix, words, suffix) = bytemuck::pod_align_to_mut::<u8, u64>(bytes);
            words.iter_mut().for_each(|word| *word ^= 0x8080_8080_8080_8080);
            prefix.iter_mut().chain(suffix.iter_mut()).for_each(|code| *code ^= 0x80);
        }
        GemmKernel::prepare_gemm_b_layout(
            weights,
            a8_shape.n,
            a8_shape.k,
            a8_shape.b_bits.expect("quantized weights"),
            context,
        )?;
        Ok((GemmBLayout::MxuTiled, prepared_shape.signed_codes))
    }

    fn encode(
        &mut self,
        arguments: MatmulArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferMut<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<(), Self::Error> {
        let shape = MatmulShape::from_arguments(&arguments);
        let plan = match self.selector.select(&shape, arguments.b.weights_layout(), ActivationFormat::Bf16).dispatch {
            MatmulDispatch::Gemv(gemv) => {
                return self.gemv.encode(arguments, gemv, &self.output_work, command_buffer);
            },
            MatmulDispatch::Gemm(plan) => plan,
        };
        self.gemm.encode_plan(arguments, plan, &self.output_work, command_buffer)
    }
}

#[cfg(test)]
impl MatmulMetalKernel {
    pub fn encode_with_gemm_engine(
        &mut self,
        arguments: MatmulArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferMut<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        engine: super::gemm::GemmEngine,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<(), MatmulMetalError> {
        let shape = MatmulShape::from_arguments(&arguments);
        let plan = self.gemm.select_plan_for_engine(&shape, engine, command_buffer.context())?;
        self.gemm.encode_plan(arguments, plan, &self.output_work, command_buffer)
    }
}
