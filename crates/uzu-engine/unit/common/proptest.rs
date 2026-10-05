use std::sync::Arc;

use proptest::prelude::*;

#[cfg(backend = "amdgpu")]
use crate::backends::amdgpu::Amdgpu;
#[cfg(backend = "metal")]
use crate::backends::metal::Metal;
use crate::{
    backends::{
        common::{Backend, Context},
        cpu::Cpu,
    },
    data_type::DataType,
};

pub fn kernel_data_type() -> impl Strategy<Value = DataType> {
    prop_oneof![Just(DataType::BF16), Just(DataType::F32)]
}

pub struct TestContextes {
    pub cpu: Arc<<Cpu as Backend>::Context>,
    #[cfg(backend = "metal")]
    pub metal: Arc<<Metal as Backend>::Context>,
    #[cfg(backend = "amdgpu")]
    pub amdgpu: Arc<<Amdgpu as Backend>::Context>,
}

impl TestContextes {
    pub fn new() -> TestContextes {
        TestContextes {
            cpu: <Cpu as Backend>::Context::new().expect("Failed to create Cpu context"),
            #[cfg(backend = "metal")]
            metal: <Metal as Backend>::Context::new().expect("Failed to create Metal context"),
            #[cfg(backend = "amdgpu")]
            amdgpu: <Amdgpu as Backend>::Context::new().expect("Failed to create AMDGPU context"),
        }
    }
}

pub struct TestResults<T> {
    pub cpu: T,
    #[cfg(backend = "metal")]
    pub metal: T,
    #[cfg(backend = "amdgpu")]
    pub amdgpu: T,
}

macro_rules! for_each_context {
    ($CONTEXTES:ident, |$CONTEXT_NAME:ident: $CONTEXT_TYPE:ident| $body:expr) => {
        crate::tests::proptest::TestResults {
            cpu: ({
                type $CONTEXT_TYPE = <crate::backends::cpu::Cpu as crate::backends::common::Backend>::Context;
                let $CONTEXT_NAME = $CONTEXTES.cpu.as_ref();
                $body
            })?,
            #[cfg(backend = "metal")]
            metal: ({
                type $CONTEXT_TYPE = <crate::backends::metal::Metal as crate::backends::common::Backend>::Context;
                let $CONTEXT_NAME = $CONTEXTES.metal.as_ref();
                $body
            })?,
            #[cfg(backend = "amdgpu")]
            amdgpu: ({
                type $CONTEXT_TYPE = <crate::backends::amdgpu::Amdgpu as crate::backends::common::Backend>::Context;
                let $CONTEXT_NAME = $CONTEXTES.amdgpu.as_ref();
                $body
            })?,
        }
    };
}
pub(crate) use for_each_context;

pub trait ComparableTestResults {
    fn compare(
        backend: &str,
        actual: &Self,
        reference: &Self,
    ) -> Result<(), TestCaseError>;
}

impl<T: ComparableTestResults> TestResults<T> {
    pub fn compare_results(&self) -> Result<(), TestCaseError> {
        #[cfg(backend = "metal")]
        T::compare("metal", &self.metal, &self.cpu)?;
        #[cfg(backend = "amdgpu")]
        T::compare("amdgpu", &self.amdgpu, &self.cpu)?;

        Ok(())
    }
}
