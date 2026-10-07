use std::{ops::Range, sync::Arc};

use bytemuck::{AnyBitPattern, NoUninit};
use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::kernel_fixture::KernelFixture;
use crate::backends::vulkan::{
    VkBuffer, VkCommandBufferEncoding,
    vk_kernels::{
        TestBf16LoadVulkanKernel, TestBf16StoreVulkanKernel, TestF16LoadVulkanKernel, TestF16StoreVulkanKernel,
    },
};

/// Runs one conversion dispatch over guarded ranges and returns the converted range after checking that the
/// output guards are untouched.
fn convert<I: NoUninit + AnyBitPattern, O: NoUninit + AnyBitPattern>(
    fixture: &KernelFixture,
    input: &[I],
    (input_sentinel, output_sentinel): (I, O),
    encode: impl FnOnce((&Arc<VkBuffer>, Range<u64>), (&Arc<VkBuffer>, Range<u64>), u32, &mut VkCommandBufferEncoding),
) -> Vec<O> {
    let input_buffer = fixture.guarded(input, input_sentinel);
    let output_buffer = fixture.guarded(&vec![output_sentinel; input.len()], output_sentinel);
    let mut encoding = fixture.encoding();
    encode(
        (&input_buffer.0, input_buffer.1.clone()),
        (&output_buffer.0, output_buffer.1.clone()),
        input.len() as u32,
        &mut encoding,
    );
    KernelFixture::complete(encoding);
    // SAFETY: the only command buffer using these buffers has completed.
    unsafe {
        let read = KernelFixture::read_guarded(&input_buffer, input_sentinel);
        assert_eq!(bytemuck::cast_slice::<I, u8>(&read), bytemuck::cast_slice::<I, u8>(input), "input changed");
        KernelFixture::read_guarded(&output_buffer, output_sentinel)
    }
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

/// Asserts NaN occurs exactly where `nan` puts it, since conversions promise NaN classification but not payloads, then
/// asserts every other result bit for bit.
fn assert_exact_or_nan(
    case: &str,
    inputs: &[u32],
    expected: &[u32],
    actual: &[u32],
    nan: impl Fn(u32) -> bool,
) {
    let mut ordered = Vec::new();
    for index in 0..inputs.len() {
        assert_eq!(
            nan(expected[index]),
            nan(actual[index]),
            "{case}: input {:#010x} NaN classification",
            inputs[index]
        );
        if !nan(expected[index]) {
            ordered.push(index);
        }
    }
    eprintln!("{case}: {} NaN results classified", inputs.len() - ordered.len());
    let pick = |values: &[u32]| ordered.iter().map(|&index| values[index]).collect::<Vec<_>>();
    assert_exact(case, &pick(inputs), &pick(expected), &pick(actual));
}

#[uzu_test]
fn loads_every_f16_pattern() {
    let fixture = KernelFixture::new();
    let kernel = TestF16LoadVulkanKernel::new(&fixture.context).expect("TestF16Load");
    let input = (0..=u16::MAX).collect::<Vec<_>>();
    let output = convert(&fixture, &input, (0u16, 0xdead_beef_u32), |input, output, size, encoding| {
        // SAFETY: both ranges hold `size` aligned elements and the kernel indexes only `idx < size`.
        unsafe { kernel.encode(input, output, size, encoding) }
    });
    let values = input.iter().map(|&bits| f16::from_bits(bits)).collect::<Vec<_>>();
    eprintln!("F16 loads cover {} subnormal patterns", values.iter().filter(|value| value.is_subnormal()).count());
    let inputs = input.iter().map(|&bits| u32::from(bits)).collect::<Vec<_>>();
    let expected = values.iter().map(|value| value.to_f32().to_bits()).collect::<Vec<_>>();
    assert_exact_or_nan("F16 -> FP32 load", &inputs, &expected, &output, |bits| f32::from_bits(bits).is_nan());
    fixture.assert_clean();
}

/// Both values of every adjacent pair of finite F16 values, including the largest finite value and the overflow
/// threshold 65536, their exact midpoint and the FP32 values on either side of it, all with both signs; then every FP32
/// upper half with each of the low halves [0, 1, 0xfff, 0x1000, 0x1001, 0x1fff, 0x2000, 0xffff], where 0x1000 is an F16
/// rounding midpoint of normal values. Results match half's round-to-nearest-even bit for bit, except NaN payloads.
#[uzu_test]
fn stores_fp32_to_f16_boundaries_and_ties() {
    let fixture = KernelFixture::new();
    let kernel = TestF16StoreVulkanKernel::new(&fixture.context).expect("TestF16Store");
    let pairs = (0..0x7bffu16)
        .map(|bits| (f16::from_bits(bits).to_f32(), f16::from_bits(bits + 1).to_f32()))
        .chain([(f16::MAX.to_f32(), 65536.0)]);
    let neighborhoods = pairs.flat_map(|(low, high)| {
        let middle = (low + high) / 2.0;
        [low, high, middle, f32::from_bits(middle.to_bits() - 1), f32::from_bits(middle.to_bits() + 1)]
    });
    let input =
        neighborhoods
            .flat_map(|value| [value.to_bits(), (-value).to_bits()])
            .chain((0..=u32::from(u16::MAX)).flat_map(|upper| {
                [0, 1, 0xfff, 0x1000, 0x1001, 0x1fff, 0x2000, 0xffff].map(|lower| upper << 16 | lower)
            }))
            .collect::<Vec<_>>();
    let output = convert(&fixture, &input, (0u32, 0xbeef_u16), |input, output, size, encoding| {
        // SAFETY: both ranges hold `size` aligned elements and the kernel indexes only `idx < size`.
        unsafe { kernel.encode(input, output, size, encoding) }
    });
    eprintln!("FP32 -> F16 store corpus: {} inputs", input.len());
    let expected =
        input.iter().map(|&bits| u32::from(f16::from_f32(f32::from_bits(bits)).to_bits())).collect::<Vec<_>>();
    let actual = output.iter().map(|&bits| u32::from(bits)).collect::<Vec<_>>();
    let nan = |bits: u32| f16::from_bits(bits as u16).is_nan();
    assert_exact_or_nan("FP32 -> F16 store", &input, &expected, &actual, nan);
    fixture.assert_clean();
}
