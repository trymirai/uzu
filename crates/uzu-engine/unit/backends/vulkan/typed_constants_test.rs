use std::{
    any::Any,
    ops::Range,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
};

use uzu_engine_macros::uzu_test;

use super::kernel_fixture::KernelFixture;
use crate::backends::{
    common::gpu_types::ActivationType,
    vulkan::{
        VkBuffer,
        vk_kernels::{TestByteStorageVulkanKernel, TestPipelineVariantsVulkanKernel, TestTypedConstantsVulkanKernel},
    },
};

const SENTINEL: u32 = 0xdead_beef;
const ABSENT: u32 = 0xa5a5_a5a5;
const BYTE_SENTINEL: u8 = 0x5a;
const ACTIVATIONS: [ActivationType; 5] = [
    ActivationType::SILU,
    ActivationType::GELUApprox,
    ActivationType::GELUExact,
    ActivationType::IDENTITY,
    ActivationType::SOFTPLUS,
];
const WORDS: [u32; 4] = [0, 1, 257, u32::MAX];
const SIZES: [u32; 3] = [1, 33, 257];

fn range((buffer, range): &(Arc<VkBuffer>, Range<u64>)) -> (&Arc<VkBuffer>, Range<u64>) {
    (buffer, range.clone())
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(message) => *message,
        Err(payload) => payload.downcast::<&str>().map(|message| message.to_string()).expect("panic message"),
    }
}

/// Constructor arguments `(mode, word, has_offset, group, secondary)`: every activation specialized with and without
/// the offset, every word, both present groups and secondaries spanning every activation.
fn configurations() -> Vec<(ActivationType, u32, bool, Option<u32>, Option<ActivationType>)> {
    (0..10)
        .map(|index| {
            let (mode, has_offset) = (ACTIVATIONS[index / 2], index % 2 == 1);
            let group = has_offset.then_some([0, 257][index / 2 % 2]);
            let secondary = (mode != ActivationType::SILU).then_some(ACTIVATIONS[(index + 3) % 5]);
            (mode, WORDS[index % 4], has_offset, group, secondary)
        })
        .collect()
}

/// The eight words of each element in each of the two rows; rows the dispatch does not reach keep the sentinel. The
/// uniform flags reach word 3 as bits 1 and 2.
fn expected(
    (mode, word, has_offset, group, secondary): (ActivationType, u32, bool, Option<u32>, Option<ActivationType>),
    size: u32,
    activation: ActivationType,
    (flag, optional_flag): (bool, Option<bool>),
    offset: Option<u32>,
    scale: Option<f32>,
) -> Vec<u32> {
    let rows = mode as u32 % 2 + 1;
    let flags = u32::from(has_offset) | u32::from(flag) << 1 | u32::from(optional_flag == Some(true)) << 2;
    (0..2)
        .flat_map(|row| {
            (0..size).flat_map(move |index| match row < rows {
                true => [
                    activation as u32,
                    mode as u32,
                    word,
                    flags,
                    offset.map_or(ABSENT, |offset| offset.wrapping_add(index)),
                    scale.map_or(ABSENT, f32::to_bits),
                    group.unwrap_or(ABSENT),
                    secondary.map_or(ABSENT, |secondary| secondary as u32),
                ],
                false => [SENTINEL; 8],
            })
        })
        .collect()
}

/// Ten pipelines of one module, differing in every specialization, record interleaved into one command buffer per
/// size, in both orders, with every uniform activation and each optional uniform present exactly when required. The
/// bool uniforms alternate between consecutive dispatches of each pipeline, so a stale or misplaced word shows.
#[uzu_test]
fn typed_constants_reach_the_kernel() {
    let fixture = KernelFixture::new();
    let configurations = configurations();
    let kernels = configurations
        .iter()
        .map(|&(mode, word, has_offset, group, secondary)| {
            TestTypedConstantsVulkanKernel::new(&fixture.context, mode, word, has_offset, group, secondary)
                .expect("TestTypedConstants")
        })
        .collect::<Vec<_>>();
    for size in SIZES {
        for order in [false, true] {
            let mut dispatches = Vec::new();
            let mut encoding = fixture.encoding();
            for (activation_index, activation) in ACTIVATIONS.into_iter().enumerate() {
                for index in 0..kernels.len() {
                    let index = if order {
                        kernels.len() - 1 - index
                    } else {
                        index
                    };
                    let (mode, _, has_offset, ..) = configurations[index];
                    let offset = has_offset.then_some([u32::MAX - 3, 257, 0][(index + activation_index) % 3]);
                    let scale = (mode == ActivationType::IDENTITY)
                        .then_some([-0.0, 3.5, f32::from_bits(1)][(index + activation_index) % 3]);
                    let flags = (activation_index % 2 == 1, has_offset.then_some(activation_index % 2 == 0));
                    let output = fixture.guarded(&vec![SENTINEL; 2 * 8 * size as usize], SENTINEL);
                    // SAFETY: the output holds two rows of `size` eight-word elements, which bounds every index the
                    // kernel writes, and aliases nothing.
                    unsafe {
                        kernels[index].encode(
                            range(&output),
                            size,
                            activation,
                            flags.0,
                            offset,
                            flags.1,
                            scale,
                            &mut encoding,
                        )
                    };
                    dispatches.push((index, activation, flags, offset, scale, output));
                }
            }
            KernelFixture::complete(encoding);
            for (index, activation, flags, offset, scale, output) in dispatches {
                // SAFETY: the only command buffer using this buffer has completed.
                let actual = unsafe { KernelFixture::read_guarded(&output, SENTINEL) };
                let expected = expected(configurations[index], size, activation, flags, offset, scale);
                assert!(
                    actual == expected,
                    "size {size}, reversed {order}, {:?}, uniform {activation:?}: words differ",
                    configurations[index]
                );
            }
        }
    }
    fixture.assert_clean();
}

/// Three pipeline-variant kernels differing in both regular specializations record every activation each, in both
/// orders, into one command buffer per size: each element echoes the selected pipeline's activation, its kernel's
/// specializations and the uniform after the selector, and the words past the span keep their sentinel; an empty
/// dispatch records nothing.
#[uzu_test]
fn pipeline_variants_select_each_pipeline() {
    let fixture = KernelFixture::new();
    let configurations = [(false, 0), (true, 257), (true, u32::MAX)];
    let kernels = configurations.map(|(flag, group)| {
        TestPipelineVariantsVulkanKernel::new(&fixture.context, flag, group).expect("TestPipelineVariants")
    });
    let count = kernels.len() * ACTIVATIONS.len();
    for size in [0, 1, 33, 257] {
        for order in [false, true] {
            let mut encoding = fixture.encoding();
            let mut dispatches = Vec::new();
            for index in 0..count {
                let index = if order {
                    count - 1 - index
                } else {
                    index
                };
                let (kernel, activation) = (index / ACTIVATIONS.len(), ACTIVATIONS[index % ACTIVATIONS.len()]);
                let word = 1000 * index as u32;
                let output = fixture.guarded(&vec![SENTINEL; 4 * size as usize + 4], SENTINEL);
                // SAFETY: the output holds `size` four-word elements, which bounds every index the kernel writes, and
                // aliases nothing.
                unsafe { kernels[kernel].encode(range(&output), size, activation, word, &mut encoding) };
                dispatches.push((kernel, activation, word, output));
            }
            KernelFixture::complete(encoding);
            for (kernel, activation, word, output) in dispatches {
                let (flag, group) = configurations[kernel];
                let elements = (0..size).flat_map(|i| [activation as u32, u32::from(flag), group, word + i]);
                let expected = elements.chain([SENTINEL; 4]).collect::<Vec<_>>();
                // SAFETY: the only command buffer using this buffer has completed.
                let actual = unsafe { KernelFixture::read_guarded(&output, SENTINEL) };
                assert!(
                    actual == expected,
                    "size {size}, reversed {order}, kernel {kernel}, {activation:?}: words differ"
                );
            }
        }
    }
    fixture.assert_clean();
}

/// The constructor rejects optional specializations whose presence contradicts their conditions, and `encode`
/// rejects optional uniforms likewise, even for an empty dispatch, before recording anything: the same command buffer
/// then records and completes a valid dispatch while the rejected outputs stay untouched.
#[uzu_test]
fn typed_constants_presence_is_checked_first() {
    let fixture = KernelFixture::new();
    let new = |mode, has_offset, group, secondary| {
        catch_unwind(AssertUnwindSafe(|| {
            TestTypedConstantsVulkanKernel::new(&fixture.context, mode, 7, has_offset, group, secondary)
        }))
    };
    for (mode, has_offset, group, secondary, argument) in [
        (ActivationType::SILU, false, Some(0), None, "group"),
        (ActivationType::GELUExact, true, None, Some(ActivationType::SILU), "group"),
        (ActivationType::SILU, true, Some(257), Some(ActivationType::IDENTITY), "secondary"),
        (ActivationType::IDENTITY, false, None, None, "secondary"),
    ] {
        let Err(payload) = new(mode, has_offset, group, secondary) else {
            panic!("{argument}: constructor accepted");
        };
        let message = panic_message(payload);
        assert!(
            message.contains(&format!("argument '{argument}' must be present exactly when")),
            "{argument}: {message}"
        );
    }

    // Both conditions true, then both false: each optional uniform must be present exactly when its condition holds.
    let present = (ActivationType::IDENTITY, 7, true, Some(0), Some(ActivationType::GELUApprox));
    let absent = (ActivationType::SILU, 7, false, None, None);
    let kernels = [present, absent].map(|(mode, _, has_offset, group, secondary)| {
        new(mode, has_offset, group, secondary).expect("constructor panicked").expect("TestTypedConstants")
    });
    let mut encoding = fixture.encoding();
    let mut rejected = Vec::new();
    for (kernel, size, offset, optional_flag, scale, argument) in [
        (&kernels[0], 33, None, Some(true), Some(1.0), "offset"),
        (&kernels[0], 0, None, Some(true), Some(1.0), "offset"),
        (&kernels[0], 33, Some(1), None, Some(1.0), "optional_flag"),
        (&kernels[0], 0, Some(1), None, Some(1.0), "optional_flag"),
        (&kernels[0], 33, Some(1), Some(true), None, "scale"),
        (&kernels[0], 0, Some(1), Some(true), None, "scale"),
        (&kernels[1], 33, Some(1), None, None, "offset"),
        (&kernels[1], 0, Some(1), None, None, "offset"),
        (&kernels[1], 33, None, Some(false), None, "optional_flag"),
        (&kernels[1], 0, None, Some(false), None, "optional_flag"),
        (&kernels[1], 33, None, None, Some(1.0), "scale"),
        (&kernels[1], 0, None, None, Some(1.0), "scale"),
    ] {
        let output = fixture.guarded(&vec![SENTINEL; 2 * 8 * 33], SENTINEL);
        let result = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: the output holds two rows of 33 eight-word elements and aliases nothing.
            unsafe {
                kernel.encode(
                    range(&output),
                    size,
                    ActivationType::SILU,
                    true,
                    offset,
                    optional_flag,
                    scale,
                    &mut encoding,
                )
            }
        }));
        let message = panic_message(result.expect_err("encode accepted"));
        assert!(message.contains(&format!("argument '{argument}' must be present exactly when")), "{message}");
        rejected.push(output);
    }
    let accepted = [(present, (true, Some(true)), Some(5), Some(2.0)), (absent, (false, None), None, None)].map(
        |(configuration, flags, offset, scale)| {
            let output = fixture.guarded(&vec![SENTINEL; 2 * 8 * 33], SENTINEL);
            (configuration, flags, offset, scale, output)
        },
    );
    for (kernel, (_, flags, offset, scale, output)) in kernels.iter().zip(&accepted) {
        // SAFETY: as above.
        unsafe {
            kernel.encode(range(output), 33, ActivationType::SOFTPLUS, flags.0, *offset, flags.1, *scale, &mut encoding)
        };
    }
    KernelFixture::complete(encoding);
    // SAFETY: the only command buffer using these buffers has completed.
    unsafe {
        for output in &rejected {
            KernelFixture::assert_unchanged(output, SENTINEL, &vec![SENTINEL; 2 * 8 * 33], "rejected output");
        }
        for (configuration, flags, offset, scale, output) in &accepted {
            let expected = expected(*configuration, 33, ActivationType::SOFTPLUS, *flags, *offset, *scale);
            assert!(KernelFixture::read_guarded(output, SENTINEL) == expected, "{configuration:?}: words differ");
        }
    }
    fixture.assert_clean();
}

/// Signed and unsigned bytes load, wrap and store exactly at odd lengths from nonzero offsets, the longest input
/// holding all 256 patterns of each.
#[uzu_test]
fn byte_storage_round_trips() {
    let fixture = KernelFixture::new();
    let kernel = TestByteStorageVulkanKernel::new(&fixture.context).expect("TestByteStorage");
    for size in SIZES {
        let signed = (0..size).map(|index| (index as u8).wrapping_mul(97) as i8).collect::<Vec<_>>();
        let unsigned = (0..size).map(|index| (index as u8).wrapping_mul(37).wrapping_add(11)).collect::<Vec<_>>();
        let signed_input = fixture.guarded(&signed, BYTE_SENTINEL as i8);
        let unsigned_input = fixture.guarded(&unsigned, BYTE_SENTINEL);
        let signed_output = fixture.guarded(&vec![BYTE_SENTINEL as i8; size as usize], BYTE_SENTINEL as i8);
        let unsigned_output = fixture.guarded(&vec![BYTE_SENTINEL; size as usize], BYTE_SENTINEL);
        let mut encoding = fixture.encoding();
        // SAFETY: every range holds `size` bytes, the kernel indexes only `index < size` and no output aliases another
        // argument.
        unsafe {
            kernel.encode(
                range(&signed_input),
                range(&unsigned_input),
                range(&signed_output),
                range(&unsigned_output),
                size,
                &mut encoding,
            )
        };
        KernelFixture::complete(encoding);
        let expected_signed = signed
            .iter()
            .zip(&unsigned)
            .map(|(&s, &u)| (i32::from(s) * 3 + i32::from(u) + i32::from(s < 0)) as i8)
            .collect::<Vec<_>>();
        let expected_unsigned = signed
            .iter()
            .zip(&unsigned)
            .map(|(&s, &u)| u32::from(u).wrapping_mul(7).wrapping_add(i32::from(s) as u32) as u8)
            .collect::<Vec<_>>();
        // SAFETY: the only command buffer using these buffers has completed.
        unsafe {
            KernelFixture::assert_unchanged(&signed_input, BYTE_SENTINEL as i8, &signed, "signed input");
            KernelFixture::assert_unchanged(&unsigned_input, BYTE_SENTINEL, &unsigned, "unsigned input");
            let actual_signed = KernelFixture::read_guarded(&signed_output, BYTE_SENTINEL as i8);
            let differing = actual_signed.iter().zip(&expected_signed).filter(|(actual, expected)| actual != expected);
            assert!(
                actual_signed == expected_signed,
                "{size}: {} signed bytes differ, first (Vulkan, expected) {:?}",
                differing.clone().count(),
                differing.clone().next()
            );
            assert_eq!(KernelFixture::read_guarded(&unsigned_output, BYTE_SENTINEL), expected_unsigned, "{size}");
        }
    }
    fixture.assert_clean();
}
