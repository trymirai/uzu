use std::{ops::Range, sync::Arc};

use uzu_engine_macros::uzu_test;

use super::kernel_fixture::KernelFixture;
use crate::backends::vulkan::{
    VkBuffer,
    vk_kernels::{
        TestSpecializationAVulkanKernel, TestSpecializationBVulkanKernel, TestSpecializationPlainVulkanKernel,
    },
};

const SENTINEL: u32 = 0xdead_beef;

fn range((buffer, range): &(Arc<VkBuffer>, Range<u64>)) -> (&Arc<VkBuffer>, Range<u64>) {
    (buffer, range.clone())
}

/// Kernels A and B of one SPIR-V module both specialize on `flag`. Every command buffer chains plain, A, plain, B
/// through guarded buffers, for every flag pair and twice over, so each pipeline must apply only its own flag.
#[uzu_test]
fn sibling_kernels_own_their_specializations() {
    let fixture = KernelFixture::new();
    let plain = TestSpecializationPlainVulkanKernel::new(&fixture.context).expect("TestSpecializationPlain");
    let a = [false, true]
        .map(|flag| TestSpecializationAVulkanKernel::new(&fixture.context, flag).expect("TestSpecializationA"));
    let b = [false, true]
        .map(|flag| TestSpecializationBVulkanKernel::new(&fixture.context, flag).expect("TestSpecializationB"));
    let input = (0..257u32).map(|index| index.wrapping_mul(0x9e37_79b9)).collect::<Vec<_>>();
    let size = input.len() as u32;
    let source = fixture.guarded(&input, SENTINEL);
    for round in 0..2 {
        for (a_flag, b_flag) in [(false, false), (false, true), (true, false), (true, true)] {
            let stages = [(); 4].map(|_| fixture.guarded(&vec![SENTINEL; input.len()], SENTINEL));
            let mut encoding = fixture.encoding();
            // SAFETY: every range holds `size` aligned elements, the kernels index only `idx < size` and no output
            // aliases another argument.
            unsafe {
                plain.encode(range(&source), range(&stages[0]), size, &mut encoding);
                a[usize::from(a_flag)].encode(range(&stages[0]), range(&stages[1]), size, &mut encoding);
                plain.encode(range(&stages[1]), range(&stages[2]), size, &mut encoding);
                b[usize::from(b_flag)].encode(range(&stages[2]), range(&stages[3]), size, &mut encoding);
            }
            KernelFixture::complete(encoding);
            let increments = [1, [0x20, 0x10][usize::from(a_flag)], 1, [0x200, 0x100][usize::from(b_flag)]];
            let mut expected = input.clone();
            // SAFETY: the only command buffer using these buffers has completed.
            unsafe {
                KernelFixture::assert_unchanged(&source, SENTINEL, &input, "input");
                for (stage, (index, increment)) in stages.iter().zip(increments.into_iter().enumerate()) {
                    expected.iter_mut().for_each(|value| *value = value.wrapping_add(increment));
                    let actual = KernelFixture::read_guarded(stage, SENTINEL);
                    assert!(actual == expected, "round {round}, A {a_flag}, B {b_flag}: stage {index} differs");
                }
            }
        }
    }
    fixture.assert_clean();
}
