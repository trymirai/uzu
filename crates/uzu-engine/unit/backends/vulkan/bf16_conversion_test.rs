use std::{fmt::Debug, mem::size_of, ops::Range, sync::Arc};

use bytemuck::{AnyBitPattern, NoUninit};
use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::kernel_fixture::KernelFixture;
use crate::backends::vulkan::{
    VkBuffer, VkCommandBufferEncoding,
    vk_kernels::{TestBf16LoadVulkanKernel, TestBf16StoreVulkanKernel},
};

/// Sentinel elements before and after every range, so both ranges start at a nonzero offset.
const GUARD: usize = 64;

/// Runs one conversion dispatch over guarded ranges and returns the converted range after checking that the
/// output guards are untouched.
fn convert<I: NoUninit + AnyBitPattern, O: NoUninit + AnyBitPattern + PartialEq + Debug>(
    fixture: &KernelFixture,
    input: &[I],
    (input_sentinel, output_sentinel): (I, O),
    encode: impl FnOnce((&Arc<VkBuffer>, Range<u64>), (&Arc<VkBuffer>, Range<u64>), u32, &mut VkCommandBufferEncoding),
) -> Vec<O> {
    let range = |size: usize| (GUARD * size) as u64..((GUARD + input.len()) * size) as u64;
    let input_buffer =
        fixture.buffer(&[vec![input_sentinel; GUARD], input.to_vec(), vec![input_sentinel; GUARD]].concat());
    let output_buffer = fixture.buffer(&vec![output_sentinel; input.len() + 2 * GUARD]);
    let mut encoding = fixture.encoding();
    encode(
        (&input_buffer, range(size_of::<I>())),
        (&output_buffer, range(size_of::<O>())),
        input.len() as u32,
        &mut encoding,
    );
    KernelFixture::complete(encoding);
    // SAFETY: the only command buffer writing `output_buffer` has completed.
    let output = unsafe { KernelFixture::read::<O>(&output_buffer) };
    let (head, rest) = output.split_at(GUARD);
    let (body, tail) = rest.split_at(input.len());
    assert!(head.iter().chain(tail).all(|value| *value == output_sentinel), "output guards were written");
    body.to_vec()
}

/// Asserts `actual` equals `expected` bit for bit, after printing the count and the first mismatches.
fn assert_exact(
    case: &str,
    inputs: &[u32],
    expected: &[u32],
    actual: &[u32],
) {
    let mismatches = (0..inputs.len()).filter(|&index| expected[index] != actual[index]).collect::<Vec<_>>();
    eprintln!("{case}: {} patterns, {} bit mismatches", inputs.len(), mismatches.len());
    for &index in mismatches.iter().take(8) {
        eprintln!("  input {:#010x}: half {:#010x}, Vulkan {:#010x}", inputs[index], expected[index], actual[index]);
    }
    assert!(mismatches.is_empty(), "{case}: {} bit mismatches", mismatches.len());
}

#[uzu_test]
fn loads_every_bf16_pattern() {
    let fixture = KernelFixture::new();
    let kernel = TestBf16LoadVulkanKernel::new(&fixture.context).expect("TestBf16Load");
    let input = (0..=u16::MAX).collect::<Vec<_>>();
    let output = convert(&fixture, &input, (0u16, 0xdead_beef_u32), |input, output, size, encoding| {
        // SAFETY: both ranges hold `size` aligned elements and the kernel indexes only `idx < size`.
        unsafe { kernel.encode(input, output, size, encoding) }
    });
    let values = input.iter().map(|&bits| bf16::from_bits(bits)).collect::<Vec<_>>();
    for (value, &actual) in values.iter().zip(&output) {
        assert_eq!(value.is_nan(), f32::from_bits(actual).is_nan(), "BF16 {:#06x} classification", value.to_bits());
    }
    let (nans, subnormals) = (
        values.iter().filter(|value| value.is_nan()).count(),
        values.iter().filter(|value| value.is_subnormal()).count(),
    );
    eprintln!("BF16 loads cover {nans} NaN and {subnormals} subnormal patterns");
    let inputs = input.iter().map(|&bits| u32::from(bits)).collect::<Vec<_>>();
    let expected = values.iter().map(|value| value.to_f32().to_bits()).collect::<Vec<_>>();
    assert_exact("BF16 -> FP32 load", &inputs, &expected, &output);
    fixture.assert_clean();
}

#[uzu_test]
fn stores_fp32_boundaries_and_ties() {
    let fixture = KernelFixture::new();
    let kernel = TestBf16StoreVulkanKernel::new(&fixture.context).expect("TestBf16Store");
    let input = (0..=u32::from(u16::MAX))
        .flat_map(|upper| [0, 1, 0x7fff, 0x8000, 0x8001, 0xffff].map(|lower| upper << 16 | lower))
        .collect::<Vec<_>>();
    let output = convert(&fixture, &input, (0u32, 0xbeef_u16), |input, output, size, encoding| {
        // SAFETY: both ranges hold `size` aligned elements and the kernel indexes only `idx < size`.
        unsafe { kernel.encode(input, output, size, encoding) }
    });
    let values = input.iter().map(|&bits| f32::from_bits(bits)).collect::<Vec<_>>();
    let (nans, subnormals, ties) = (
        values.iter().filter(|value| value.is_nan()).count(),
        values.iter().filter(|value| value.is_subnormal()).count(),
        input.iter().filter(|&&bits| bits & 0xffff == 0x8000).count(),
    );
    eprintln!("FP32 stores cover {nans} NaN, {subnormals} subnormal and {ties} exact-tie patterns");
    let expected = values.iter().map(|&value| u32::from(bf16::from_f32(value).to_bits())).collect::<Vec<_>>();
    let actual = output.iter().map(|&bits| u32::from(bits)).collect::<Vec<_>>();
    assert_exact("FP32 -> BF16 store", &input, &expected, &actual);
    fixture.assert_clean();
}
