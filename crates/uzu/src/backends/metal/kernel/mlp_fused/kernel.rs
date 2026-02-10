use crate::backends::metal::{MTLComputeCommandEncoder, ProtocolObject};

use super::{
    common::MlpFusedArguments,
    dispatch_descriptor::{MlpFusedDispatchDescriptor, choose_dispatch_descriptor},
    gemm::GemmKernel,
    gemv::GemvKernel,
    split_k::SplitKKernel,
};
use crate::{
    DataType,
    backends::metal::{MTLContext, MTLError},
};

pub struct MlpFusedKernel {
    data_type: DataType,
    weights_transposed: bool,
    gemm: Option<GemmKernel>,
    gemv: Option<GemvKernel>,
    splitk: Option<SplitKKernel>,
}

impl MlpFusedKernel {
    pub fn new(
        data_type: DataType,
        weights_transposed: bool,
    ) -> Result<Self, MTLError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(MTLError::Generic(format!("Unsupported dtype for MlpFusedKernel: {data_type:?}")));
        }

        Ok(Self {
            data_type,
            weights_transposed,
            gemm: None,
            gemv: None,
            splitk: None,
        })
    }

    fn get_or_create_gemm(&mut self) -> Result<&mut GemmKernel, MTLError> {
        if self.gemm.is_none() {
            self.gemm = Some(GemmKernel::new(self.data_type, self.weights_transposed)?);
        }
        Ok(self.gemm.as_mut().unwrap())
    }

    fn get_or_create_gemv(&mut self) -> Result<&mut GemvKernel, MTLError> {
        if self.gemv.is_none() {
            self.gemv = Some(GemvKernel::new(self.data_type)?);
        }
        Ok(self.gemv.as_mut().unwrap())
    }

    fn get_or_create_splitk(&mut self) -> Result<&mut SplitKKernel, MTLError> {
        if self.splitk.is_none() {
            self.splitk = Some(SplitKKernel::new(self.data_type)?);
        }
        Ok(self.splitk.as_mut().unwrap())
    }

    fn encode_dispatch_descriptor(
        &mut self,
        context: &MTLContext,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        arguments: &MlpFusedArguments,
        descriptor: &MlpFusedDispatchDescriptor,
    ) -> Result<(), MTLError> {
        match descriptor {
            MlpFusedDispatchDescriptor::Gemv(descriptor) => {
                let gemv = self.get_or_create_gemv()?;
                gemv.encode_descriptor(context, encoder, arguments, descriptor)
            },
            MlpFusedDispatchDescriptor::SplitK(descriptor) => {
                let splitk = self.get_or_create_splitk()?;
                splitk.encode_descriptor(context, encoder, arguments, descriptor)
            },
            MlpFusedDispatchDescriptor::Gemm(descriptor) => {
                let gemm = self.get_or_create_gemm()?;
                gemm.encode_descriptor(context, encoder, arguments, descriptor)
            },
        }
    }

    pub fn encode(
        &mut self,
        context: &MTLContext,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        arguments: &MlpFusedArguments,
    ) -> Result<(), MTLError> {
        let descriptor = choose_dispatch_descriptor(context, self.data_type, self.weights_transposed, arguments)?;

        self.encode_dispatch_descriptor(context, encoder, arguments, &descriptor)
    }
}
