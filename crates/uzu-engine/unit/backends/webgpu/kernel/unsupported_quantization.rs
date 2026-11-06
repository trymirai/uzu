use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Context,
            kernel::{
                ActivationQuantization, ActivationTransform, GatedActMul, GatedActMulSettings,
                matmul::{ActivationFormat, Int8CodeLayout},
            },
        },
        webgpu::{WebGPU, context::WebGPUContext, error::WebGPUError},
    },
    data_type::DataType,
};

#[uzu_test]
fn quantized_activations_are_rejected() {
    let context = block_on(WebGPUContext::new()).unwrap();
    for emit_group_sums in [false, true] {
        let quantization = ActivationFormat::Int8(
            ActivationQuantization::new(128, 32, emit_group_sums, Int8CodeLayout::GroupedByNibble).unwrap(),
        );
        assert!(matches!(
            ActivationTransform::<WebGPU>::quantize(&context, DataType::BF16, quantization),
            Err(WebGPUError::UnsupportedKernelVariant(_))
        ));
        assert!(matches!(
            GatedActMul::<WebGPU>::quantized(&context, DataType::BF16, quantization, GatedActMulSettings::default()),
            Err(WebGPUError::UnsupportedKernelVariant(_))
        ));
    }
}
