use std::ops::DerefMut;

use super::{
    dispatch_descriptor::MatmulDispatchDescriptor, gemm::GemmKernel, gemm_mpp::GemmMppKernel,
    gemm_scalar_int::GemmScalarIntKernel, gemv::GemvKernel, matmul_arguments::MatmulArguments,
    split_k::SplitKKernel,
};
use crate::{
    DataType,
    backends::common::{Backend, CommandBuffer, Kernels, kernel::TensorAddBiasKernel},
};

fn is_valid_dtype_combo(a: DataType, b: DataType, out: DataType) -> bool {
    matches!(
        (a, b, out),
        (DataType::F16, DataType::F16, DataType::F16)
            | (DataType::BF16, DataType::BF16, DataType::BF16)
            | (DataType::F32, DataType::F32, DataType::F32)
            | (DataType::I8, DataType::I8, DataType::I32)
            | (DataType::I8, DataType::BF16, DataType::BF16)
    )
}

pub struct MatmulKernel<B: Backend> {
    pub(crate) a_dtype: DataType,
    pub(crate) b_dtype: DataType,
    pub(crate) output_dtype: DataType,
    gemm: Option<GemmKernel<B>>,
    gemv: Option<GemvKernel<B>>,
    splitk: Option<SplitKKernel<B>>,
    gemm_mpp: Option<GemmMppKernel<B>>,
    gemm_scalar_int: Option<GemmScalarIntKernel<B>>,
    bias_add: Option<<B::Kernels as Kernels>::TensorAddBiasKernel>,
}

impl<B: Backend> MatmulKernel<B>
where
    B::Error: From<String>,
{
    pub fn new(data_type: DataType) -> Result<Self, B::Error> {
        Self::new_mixed(data_type, data_type, data_type)
    }

    pub fn new_mixed(a_dtype: DataType, b_dtype: DataType, output_dtype: DataType) -> Result<Self, B::Error> {
        if !is_valid_dtype_combo(a_dtype, b_dtype, output_dtype) {
            return Err(B::Error::from(format!(
                "Unsupported dtype combo for MatmulKernel: {a_dtype:?} * {b_dtype:?} -> {output_dtype:?}"
            )));
        }

        Ok(Self {
            a_dtype,
            b_dtype,
            output_dtype,
            gemm: None,
            gemv: None,
            splitk: None,
            gemm_mpp: None,
            gemm_scalar_int: None,
            bias_add: None,
        })
    }

    pub fn precompile(
        &mut self,
        context: &B::Context,
    ) -> Result<(), B::Error> {
        let gemm = self.get_or_create_gemm()?;
        gemm.precompile(context)?;

        let gemv = self.get_or_create_gemv()?;
        gemv.precompile(context)?;

        let splitk = self.get_or_create_splitk()?;
        splitk.precompile(context)?;

        Ok(())
    }

    fn get_or_create_gemm(&mut self) -> Result<&mut GemmKernel<B>, B::Error> {
        if self.gemm.is_none() {
            self.gemm = Some(GemmKernel::<B>::new(self.output_dtype)?);
        }
        Ok(self.gemm.as_mut().unwrap())
    }

    fn get_or_create_gemv(&mut self) -> Result<&mut GemvKernel<B>, B::Error> {
        if self.gemv.is_none() {
            self.gemv = Some(GemvKernel::<B>::new(self.output_dtype)?);
        }
        Ok(self.gemv.as_mut().unwrap())
    }

    fn get_or_create_splitk(&mut self) -> Result<&mut SplitKKernel<B>, B::Error> {
        if self.splitk.is_none() {
            self.splitk = Some(SplitKKernel::<B>::new(self.output_dtype)?);
        }
        Ok(self.splitk.as_mut().unwrap())
    }

    fn get_or_create_gemm_mpp(&mut self) -> Result<&mut GemmMppKernel<B>, B::Error> {
        if self.gemm_mpp.is_none() {
            self.gemm_mpp = Some(GemmMppKernel::<B>::new(self.a_dtype)?);
        }
        Ok(self.gemm_mpp.as_mut().unwrap())
    }

    fn get_or_create_gemm_scalar_int(&mut self) -> Result<&mut GemmScalarIntKernel<B>, B::Error> {
        if self.gemm_scalar_int.is_none() {
            self.gemm_scalar_int = Some(GemmScalarIntKernel::<B>::new(self.a_dtype, self.b_dtype)?);
        }
        Ok(self.gemm_scalar_int.as_mut().unwrap())
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
            MatmulDispatchDescriptor::SplitK(d) => {
                let splitk = self.get_or_create_splitk()?;
                splitk.encode(context, arguments, d, command_buffer)
            },
            MatmulDispatchDescriptor::Gemm(d) => {
                let gemm = self.get_or_create_gemm()?;
                gemm.encode(context, arguments, d, command_buffer)
            },
            MatmulDispatchDescriptor::GemmMpp(d) => {
                let gemm_mpp = self.get_or_create_gemm_mpp()?;
                gemm_mpp.encode(context, arguments, d, encoder)
            },
            MatmulDispatchDescriptor::GemmScalarInt(d) => {
                let gemm_scalar_int = self.get_or_create_gemm_scalar_int()?;
                gemm_scalar_int.encode(context, arguments, d, encoder)
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
            self.bias_add = Some(<B::Kernels as Kernels>::TensorAddBiasKernel::new(context, self.data_type, true)?);
        }
        let bias_add = self.bias_add.as_ref().unwrap();
        bias_add.encode(None::<&B::Buffer>, bias, arguments.d.deref_mut(), n as u32, total_len as u32, command_buffer);
        Ok(())
    }

    pub fn apply_batch_collapse(arguments: &mut MatmulArguments<B>) {
        if arguments.batch_count <= 1 {
            return;
        }
        if arguments.lda == arguments.input_dim && arguments.transpose_b {
            arguments.batch *= arguments.batch_count;
            arguments.batch_count = 1;
        }
    }
}
