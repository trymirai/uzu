use uzu_engine_macros::uzu_test;

use super::*;
use crate::tests::{helpers::create_context, matmul::quantized_matmul_shape};

#[uzu_test]
fn int8_activations_require_the_a8_contract() {
    let context = create_context::<Vulkan>();
    let kernel = VkMatmulKernel::new(&context, DataType::BF16, DataType::BF16, DataType::BF16).expect("matmul kernel");
    for (k, bits, group, b_transpose, expected) in [
        (256, 4, 64, true, true),
        (256, 8, 64, true, true),
        (256, 4, 32, false, false),
        (320, 4, 64, true, false),
        (256, 4, 16, true, false),
    ] {
        let shape = MatmulShape {
            b_transpose,
            ..quantized_matmul_shape(16, 96, k, bits, group)
        };
        let selected = kernel.select_activation_quantization(&shape, &context).is_some();
        assert_eq!(selected, expected, "k={k} W{bits} gs={group} b_transpose={b_transpose}");
    }
}
