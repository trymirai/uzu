use super::{
    dispatch_descriptor::MatmulDispatchDescriptor, gemm::GemmKernel, gemv::GemvKernel,
    matmul_arguments::MatmulArguments, split_k::SplitKKernel,
};
use crate::{
    DataType,
    backends::common::{Backend, Kernels, kernel::TensorAddBiasKernel},
};

pub struct MatmulKernel<B: Backend> {
    pub(crate) data_type: DataType,
    gemm: Option<GemmKernel<B>>,
    gemv: Option<GemvKernel<B>>,
    splitk: Option<SplitKKernel<B>>,
    bias_add: Option<<B::Kernels as Kernels>::TensorAddBiasKernel>,
}

impl<B: Backend> MatmulKernel<B>
where
    B::Error: From<String>,
{
    pub fn new(data_type: DataType) -> Result<Self, B::Error> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(B::Error::from(format!("Unsupported dtype for MatmulKernel: {data_type:?}")));
        }

        Ok(Self {
            data_type,
            gemm: None,
            gemv: None,
            splitk: None,
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
            self.gemm = Some(GemmKernel::<B>::new(self.data_type)?);
        }
        Ok(self.gemm.as_mut().unwrap())
    }

    fn get_or_create_gemv(&mut self) -> Result<&mut GemvKernel<B>, B::Error> {
        if self.gemv.is_none() {
            self.gemv = Some(GemvKernel::<B>::new(self.data_type)?);
        }
        Ok(self.gemv.as_mut().unwrap())
    }

    fn get_or_create_splitk(&mut self) -> Result<&mut SplitKKernel<B>, B::Error> {
        if self.splitk.is_none() {
            self.splitk = Some(SplitKKernel::<B>::new(self.data_type)?);
        }
        Ok(self.splitk.as_mut().unwrap())
    }

    fn encode_dispatch_descriptor(
        &mut self,
        context: &B::Context,
        arguments: &MatmulArguments<B>,
        dispatch_descriptor: &MatmulDispatchDescriptor,
        encoder: &B::ComputeEncoder,
    ) -> Result<(), B::Error> {
        match dispatch_descriptor {
            MatmulDispatchDescriptor::Gemv(d) => {
                let gemv = self.get_or_create_gemv()?;
                gemv.encode(context, arguments, d, encoder)
            },
            MatmulDispatchDescriptor::SplitK(d) => {
                let splitk = self.get_or_create_splitk()?;
                splitk.encode(context, arguments, d, encoder)
            },
            MatmulDispatchDescriptor::Gemm(d) => {
                let gemm = self.get_or_create_gemm()?;
                gemm.encode(context, arguments, d, encoder)
            },
        }
    }

    pub fn encode_with_descriptor(
        &mut self,
        context: &B::Context,
        arguments: MatmulArguments<B>,
        dispatch_descriptor: &MatmulDispatchDescriptor,
        encoder: &B::ComputeEncoder,
    ) -> Result<(), B::Error> {
        self.encode_dispatch_descriptor(context, &arguments, dispatch_descriptor, encoder)?;

        if let Some(bias) = arguments.bias {
            if !dispatch_descriptor.bias_is_fused() {
                self.apply_bias_add(context, &arguments, bias, encoder)?;
            }
        }

        Ok(())
    }

    fn apply_bias_add(
        &mut self,
        context: &B::Context,
        arguments: &MatmulArguments<B>,
        bias: &B::NativeBuffer,
        encoder: &B::ComputeEncoder,
    ) -> Result<(), B::Error> {
        let m = arguments.batch as usize;
        let n = arguments.output_dim as usize;
        let batch_count = arguments.batch_count as usize;
        let total_len = m * n * batch_count;
        if total_len == 0 {
            return Ok(());
        }

        if self.bias_add.is_none() {
            self.bias_add = Some(<B::Kernels as Kernels>::TensorAddBiasKernel::new(context, self.data_type)?);
        }
        let bias_add = self.bias_add.as_ref().unwrap();
        bias_add.encode(arguments.d, bias, arguments.d, n as u32, total_len as u32, encoder);
        Ok(())
    }

    pub fn apply_batch_collapse(arguments: &mut MatmulArguments<B>) {
        if arguments.transpose_a {
            return;
        }
        if arguments.batch_count <= 1 {
            return;
        }
        if arguments.lda == arguments.input_dim && arguments.transpose_b {
            arguments.batch *= arguments.batch_count;
            arguments.batch_count = 1;
        }
    }
}
