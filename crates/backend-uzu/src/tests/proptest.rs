use std::rc::Rc;

#[cfg(metal_backend)]
use backend_uzu::backends::metal::Metal;
use backend_uzu::{
    backends::{
        common::{Backend, Context},
        cpu::Cpu,
    },
    data_type::DataType,
};
use proptest::prelude::*;

pub fn kernel_data_type() -> impl Strategy<Value = DataType> {
    prop_oneof![Just(DataType::BF16), Just(DataType::F32)]
}

pub struct TestContextes {
    pub cpu: Rc<<Cpu as Backend>::Context>,
    #[cfg(metal_backend)]
    pub metal: Rc<<Metal as Backend>::Context>,
}

impl TestContextes {
    pub fn new() -> TestContextes {
        TestContextes {
            cpu: <Cpu as Backend>::Context::new().expect("Failed to create Cpu context"),
            #[cfg(metal_backend)]
            metal: <Metal as Backend>::Context::new().expect("Failed to create Metal context"),
        }
    }
}

pub struct TestResults<T> {
    pub cpu: T,
    #[cfg(metal_backend)]
    pub metal: T,
}

macro_rules! for_each_context {
    ($CONTEXTES:ident, |$CONTEXT_NAME:ident: $CONTEXT_TYPE:ident| $body:expr) => {
        crate::tests::proptest::TestResults {
            cpu: ({
                type $CONTEXT_TYPE =
                    <backend_uzu::backends::cpu::Cpu as backend_uzu::backends::common::Backend>::Context;
                let $CONTEXT_NAME = $CONTEXTES.cpu.as_ref();
                $body
            })?,
            #[cfg(metal_backend)]
            metal: ({
                type $CONTEXT_TYPE =
                    <backend_uzu::backends::metal::Metal as backend_uzu::backends::common::Backend>::Context;
                let $CONTEXT_NAME = $CONTEXTES.metal.as_ref();
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
        #[cfg(metal_backend)]
        T::compare("metal", &self.metal, &self.cpu)?;

        Ok(())
    }
}
