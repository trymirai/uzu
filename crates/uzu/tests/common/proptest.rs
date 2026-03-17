use std::rc::Rc;

use proptest::prelude::*;
#[cfg(metal_backend)]
use uzu::backends::metal::Metal;
use uzu::{
    DataType,
    backends::{
        common::{Backend, Context},
        cpu::Cpu,
    },
};

pub fn kernel_data_type() -> impl Strategy<Value = DataType> {
    prop_oneof![Just(DataType::F16), Just(DataType::BF16), Just(DataType::F32)]
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

#[macro_export]
macro_rules! for_each_context {
    ($CONTEXTES:ident, |$CONTEXT_NAME:ident: $CONTEXT_TYPE:ident| $body:expr) => {
        crate::common::proptest::TestResults {
            cpu: ({
                type $CONTEXT_TYPE = <uzu::backends::cpu::Cpu as uzu::backends::common::Backend>::Context;
                let $CONTEXT_NAME = $CONTEXTES.cpu.as_ref();
                $body
            })?,
            #[cfg(metal_backend)]
            metal: ({
                type $CONTEXT_TYPE = <uzu::backends::metal::Metal as uzu::backends::common::Backend>::Context;
                let $CONTEXT_NAME = $CONTEXTES.metal.as_ref();
                $body
            })?,
        }
    };
}

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

#[macro_export]
macro_rules! dispatch_dtype {
    (|()| $body:expr) => { $body };
    (|($T:ident : $dt:expr $(, $rest_T:ident : $rest_dt:expr)*)| $body:expr) => {
        match $dt {
            uzu::DataType::F16 => { type $T = half::f16; dispatch_dtype!(|($($rest_T : $rest_dt),*)| $body) }
            uzu::DataType::BF16 => { type $T = half::bf16; dispatch_dtype!(|($($rest_T : $rest_dt),*)| $body) }
            uzu::DataType::F32 => { type $T = f32; dispatch_dtype!(|($($rest_T : $rest_dt),*)| $body) }
            other => panic!("unsupported dtype in test: {:?}", other),
        }
    };
}
