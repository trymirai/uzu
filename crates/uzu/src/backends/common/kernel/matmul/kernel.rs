use std::ops::DerefMut;

use super::{
    MatmulError, dispatch_descriptor::MatmulDispatchDescriptor, gemm_mpp::GemmMppKernel,
    gemv::GemvKernel, matmul_arguments::MatmulArguments,
};
use crate::{
    DataType,
    backends::common::{Backend, CommandBuffer, Kernels, kernel::TensorAddBiasKernel},
};

fn is_valid_dtype_combo(
    a: DataType,
    b: DataType,
    out: DataType,
) -> bool {
    matches!(
        (a, b, out),
        (DataType::F16, DataType::F16, DataType::F16)
            | (DataType::BF16, DataType::BF16, DataType::BF16)
    )
}

pub struct MatmulKernel<B: Backend> {
    pub(crate) a_dtype: DataType,
    pub(crate) b_dtype: DataType,
    pub(crate) output_dtype: DataType,
    gemv: Option<GemvKernel<B>>,
    gemm_mpp: Option<GemmMppKernel<B>>,
    bias_add: Option<<B::Kernels as Kernels>::TensorAddBiasKernel>,
}

impl<B: Backend> MatmulKernel<B> {
    pub fn new(data_type: DataType) -> Result<Self, MatmulError<B>> {
        Self::new_mixed(data_type, data_type, data_type)
    }

    pub fn new_mixed(
        a_dtype: DataType,
        b_dtype: DataType,
        output_dtype: DataType,
    ) -> Result<Self, MatmulError<B>> {
        if !is_valid_dtype_combo(a_dtype, b_dtype, output_dtype) {
            return Err(MatmulError::UnsupportedDataType(output_dtype));
        }

        Ok(Self {
            a_dtype,
            b_dtype,
            output_dtype,
            gemv: None,
            gemm_mpp: None,
            bias_add: None,
        })
    }

    pub fn precompile(
        &mut self,
        context: &B::Context,
    ) -> Result<(), MatmulError<B>> {
        let gemv = self.get_or_create_gemv()?;
        gemv.precompile(context)?;

        Ok(())
    }

    fn get_or_create_gemv(&mut self) -> Result<&mut GemvKernel<B>, MatmulError<B>> {
        if self.gemv.is_none() {
            self.gemv = Some(GemvKernel::<B>::new(self.output_dtype)?);
        }
        Ok(self.gemv.as_mut().unwrap())
    }

    fn get_or_create_gemm_mpp(&mut self) -> Result<&mut GemmMppKernel<B>, MatmulError<B>> {
        if self.gemm_mpp.is_none() {
            self.gemm_mpp = Some(
                GemmMppKernel::<B>::new(self.output_dtype)
                    .map_err(MatmulError::BackendError)?,
            );
        }
        Ok(self.gemm_mpp.as_mut().unwrap())
    }

    fn encode_dispatch_descriptor(
        &mut self,
        context: &B::Context,
        arguments: &mut MatmulArguments<B>,
        dispatch_descriptor: &MatmulDispatchDescriptor,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), MatmulError<B>> {
        match dispatch_descriptor {
            MatmulDispatchDescriptor::Gemv(d) => {
                let gemv = self.get_or_create_gemv()?;
                gemv.encode(context, arguments, d, command_buffer)
            },
            MatmulDispatchDescriptor::GemmMpp(d) => {
                let gemm_mpp = self.get_or_create_gemm_mpp()?;
                gemm_mpp.encode(context, arguments, d, command_buffer)
            },
        }
    }

    pub fn encode_with_descriptor(
        &mut self,
        context: &B::Context,
        mut arguments: MatmulArguments<B>,
        dispatch_descriptor: &MatmulDispatchDescriptor,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), MatmulError<B>> {
        self.encode_dispatch_descriptor(context, &mut arguments, dispatch_descriptor, command_buffer)?;

        if let Some(bias) = arguments.bias {
            if !dispatch_descriptor.bias_is_fused() {
                self.apply_bias_add(context, &mut arguments, bias, command_buffer)?;
            }
        }

        Ok(())
    }

    fn apply_bias_add(
        &mut self,
        context: &B::Context,
        arguments: &mut MatmulArguments<B>,
        bias: &B::Buffer,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), MatmulError<B>> {
        let m = arguments.batch as usize;
        let n = arguments.output_dim as usize;
        let batch_count = arguments.batch_count as usize;
        let total_len = m * n * batch_count;
        if total_len == 0 {
            return Ok(());
        }

        if self.bias_add.is_none() {
            self.bias_add = Some(
                <B::Kernels as Kernels>::TensorAddBiasKernel::new(context, self.output_dtype, true)
                    .map_err(MatmulError::BackendError)?,
            );
        }
        let bias_add = self.bias_add.as_ref().unwrap();
        bias_add.encode(None::<&B::Buffer>, bias, arguments.d.deref_mut(), n as u32, total_len as u32, command_buffer);
        Ok(())
    }

    pub fn apply_batch_collapse(arguments: &mut MatmulArguments<B>) {
        if arguments.batch_count <= 1 {
            return;
        }
        if arguments.leading_dim_a == arguments.input_dim && arguments.transpose_b {
            arguments.batch *= arguments.batch_count;
            arguments.batch_count = 1;
        }
    }
}
