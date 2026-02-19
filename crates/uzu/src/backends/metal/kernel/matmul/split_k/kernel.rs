use metal::MTLComputeCommandEncoder;
use objc2::runtime::ProtocolObject;

use super::{DispatchDescriptor, dsl_kernel::DslKernel};
use crate::{
    DataType,
    backends::metal::{MetalContext, MetalError, kernel::matmul::common::MatmulArguments},
};

pub struct Kernel {
    data_type: DataType,
    dsl: Option<DslKernel>,
}

impl Kernel {
    pub fn new(data_type: DataType) -> Result<Self, MetalError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(MetalError::Generic(format!("Unsupported dtype for Split-K: {:?}", data_type)));
        }
        Ok(Self {
            data_type,
            dsl: None,
        })
    }

    pub fn precompile(
        &mut self,
        context: &MetalContext,
    ) -> Result<(), MetalError> {
        self.get_or_create_dsl()?.precompile(context)
    }

    fn get_or_create_dsl(&mut self) -> Result<&mut DslKernel, MetalError> {
        if self.dsl.is_none() {
            self.dsl = Some(DslKernel::new(self.data_type)?);
        }
        Ok(self.dsl.as_mut().unwrap())
    }

    pub(crate) fn encode_descriptor(
        &mut self,
        context: &MetalContext,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        arguments: &MatmulArguments,
        descriptor: &DispatchDescriptor,
    ) -> Result<bool, MetalError> {
        self.get_or_create_dsl()?.encode_descriptor(context, encoder, arguments, descriptor)
    }
}
