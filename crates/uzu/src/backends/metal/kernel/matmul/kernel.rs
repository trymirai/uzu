use super::{
    common::MatmulArguments,
    dispatch_descriptor::{
        MatmulDispatchDescriptor, choose_dispatch_descriptor,
    },
    gemm,
    gemv::GemvKernel,
    split_k::SplitKGemm,
};
use crate::{
    DataType,
    backends::{
        common::kernel::TensorAddBiasKernel as _,
        metal::{
            MTLBuffer, MTLComputeCommandEncoder, MTLContext, MTLError,
            ProtocolObject, kernel::dsl::TensorAddBiasKernel,
        },
    },
};

pub struct MatmulKernel {
    data_type: DataType,
    gemm: Option<gemm::GemmKernel>,
    gemv: Option<GemvKernel>,
    splitk: Option<SplitKGemm>,
    bias_add: Option<TensorAddBiasKernel>,
}

impl MatmulKernel {
    pub fn new(data_type: DataType) -> Result<Self, MTLError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32)
        {
            return Err(MTLError::Generic(format!(
                "Unsupported dtype for MatmulKernel: {data_type:?}"
            )));
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
        context: &MTLContext,
    ) -> Result<(), MTLError> {
        let gemm = self.get_or_create_gemm()?;
        gemm.precompile(context)?;

        let gemv = self.get_or_create_gemv()?;
        gemv.precompile(context)?;

        let splitk = self.get_or_create_splitk()?;
        splitk.precompile(context)?;

        Ok(())
    }

    fn get_or_create_gemm(
        &mut self
    ) -> Result<&mut gemm::GemmKernel, MTLError> {
        if self.gemm.is_none() {
            self.gemm = Some(gemm::GemmKernel::new(self.data_type)?);
        }
        Ok(self.gemm.as_mut().unwrap())
    }

    fn get_or_create_gemv(&mut self) -> Result<&mut GemvKernel, MTLError> {
        if self.gemv.is_none() {
            self.gemv = Some(GemvKernel::new(self.data_type)?);
        }
        Ok(self.gemv.as_mut().unwrap())
    }

    fn get_or_create_splitk(&mut self) -> Result<&mut SplitKGemm, MTLError> {
        if self.splitk.is_none() {
            self.splitk = Some(SplitKGemm::new(self.data_type)?);
        }
        Ok(self.splitk.as_mut().unwrap())
    }

    fn encode_dispatch_descriptor(
        &mut self,
        context: &MTLContext,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        arguments: &MatmulArguments,
        descriptor: &MatmulDispatchDescriptor,
    ) -> Result<bool, MTLError> {
        match descriptor {
            MatmulDispatchDescriptor::Gemv(descriptor) => {
                let gemv = self.get_or_create_gemv()?;
                gemv.encode_descriptor(context, encoder, arguments, descriptor)
            },
            MatmulDispatchDescriptor::SplitK(descriptor) => {
                let splitk = self.get_or_create_splitk()?;
                splitk
                    .encode_descriptor(context, encoder, arguments, descriptor)
            },
            MatmulDispatchDescriptor::Gemm(descriptor) => {
                let gemm = self.get_or_create_gemm()?;
                gemm.encode_descriptor(context, encoder, arguments, descriptor)
            },
        }
    }

    pub fn encode(
        &mut self,
        context: &MTLContext,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        mut arguments: MatmulArguments,
    ) -> Result<(), MTLError> {
        Self::apply_batch_collapse(&mut arguments);

        let descriptor =
            choose_dispatch_descriptor(context, self.data_type, &arguments)?;

        let bias_fused = self.encode_dispatch_descriptor(
            context,
            encoder,
            &arguments,
            &descriptor,
        )?;

        if let Some(bias) = arguments.bias {
            if !bias_fused {
                self.apply_bias_add(context, encoder, &arguments, bias)?;
            }
        }

        Ok(())
    }

    fn apply_bias_add(
        &mut self,
        context: &MTLContext,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        arguments: &MatmulArguments,
        bias: &ProtocolObject<dyn MTLBuffer>,
    ) -> Result<(), MTLError> {
        let m = arguments.batch as usize;
        let n = arguments.output_dim as usize;
        let batch_count = arguments.batch_count as usize;
        let total_len = m * n * batch_count;
        if total_len == 0 {
            return Ok(());
        }

        if self.bias_add.is_none() {
            self.bias_add =
                Some(TensorAddBiasKernel::new(context, self.data_type)?);
        }
        let bias_add = self.bias_add.as_ref().unwrap();
        bias_add.encode(
            arguments.d,
            bias,
            arguments.d,
            n as u32,
            total_len as u32,
            encoder,
        );
        Ok(())
    }

    fn apply_batch_collapse(arguments: &mut MatmulArguments) {
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
