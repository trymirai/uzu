use std::{fmt::Debug, mem::size_of, ops::Range, sync::Arc};

use bytemuck::{AnyBitPattern, NoUninit};
use uzu_engine_macros::uzu_test;

use super::{arg, kernel_fixture::KernelFixture};
use crate::{
    backends::vulkan::{
        VkBuffer, VkCommandBufferEncoding,
        vk_kernels::{TestStorageAddressCopyVulkanKernel, TestStorageAddressWordsVulkanKernel},
    },
    data_type::DataType,
};

/// (rows, columns, stride): one element, padded rows, whole and partial 64-thread groups, odd columns, and no work.
const SHAPES: [(u32, u32, u32); 6] = [(1, 1, 1), (3, 5, 7), (2, 64, 64), (5, 33, 40), (7, 65, 66), (0, 3, 4)];

/// FP32 zeros, subnormals, infinity and NaN payloads.
const F32_SPECIALS: [u32; 8] = [0, 0x8000_0000, 1, 0x807f_ffff, 0x7f80_0000, 0x7f80_0001, 0xffc0_1234, 0xffff_ffff];

/// F16 and BF16 zeros, subnormals, infinity and NaN payloads.
const F16_SPECIALS: [u16; 9] = [0, 0x8000, 1, 0x83ff, 0x7c00, 0x7c01, 0xfd55, 0x7f81, 0xffc1];

/// Synthetic (row, stride, column, element bytes, base low, base high): zero; a high-bit and the largest single index;
/// product, column, byte and base carries; the largest index 2^64 - 2^32, whose byte offset wraps; an address wrapping
/// past 2^64; odd, zero and maximum element bytes; and mixed bits.
const CASES: [[u32; 6]; 14] = [
    [0, 0, 0, 4, 0, 0],
    [0, 0, 0x8000_0000, 2, 0x1000, 0],
    [0, 7, u32::MAX, 4, 0x10, 1],
    [0x1_0000, 0x1_0000, 0, 2, 0, 0],
    [u32::MAX, 1, 1, 4, 0, 0],
    [0x4000_0000, 1, 0, 4, 0, 0],
    [0, 0, 4, 4, 0xffff_fff0, 0],
    [u32::MAX, u32::MAX, u32::MAX, 4, 0, 0],
    [0, 0, 8, 4, 0xffff_fff0, u32::MAX],
    [3, 0x5555_5555, 0x1234_5678, 3, 0xdead_beef, 0x0bad_f00d],
    [0, 0, u32::MAX, 0, u32::MAX, u32::MAX],
    [u32::MAX, u32::MAX, u32::MAX - 1, u32::MAX, u32::MAX, u32::MAX],
    [0xffff_fffe, 0x8000_0001, 0x7fff_ffff, 2, 0x8000_0000, 0x7fff_ffff],
    [0x1234_5678, 0x9abc_def0, 0xfedc_ba98, 4, 0x0f0f_0f0f, 0xf0f0_f0f0],
];

const SENTINEL: u32 = 0xa5a5_a5a5;

/// For every shape, copies A to B and B to C in one command buffer through `encode`, over guarded ranges ending at the
/// last copied element, then checks the copied elements bit for bit and the padding, the guards and A unchanged.
fn copies_chain<B: NoUninit + AnyBitPattern + PartialEq + Debug>(
    fixture: &KernelFixture,
    specials: &[B],
    mut encode: impl FnMut(
        (&Arc<VkBuffer>, Range<u64>),
        (&Arc<VkBuffer>, Range<u64>),
        (u32, u32, u32),
        &mut VkCommandBufferEncoding,
    ),
) {
    let bits = |seed: usize| -> B {
        let hashed = (seed as u32).wrapping_mul(0x9e37_79b9).rotate_left(13).to_ne_bytes();
        bytemuck::pod_read_unaligned(&hashed[..size_of::<B>()])
    };
    let (sentinel, pad) = (bits(1), bits(2));
    for (rows, columns, stride) in SHAPES {
        let len = rows.checked_sub(1).map_or(0, |last| (last * stride + columns) as usize);
        let a = (0..len).map(|i| specials.get(i).copied().unwrap_or_else(|| bits(i + 3))).collect::<Vec<_>>();
        let expected = (0..len)
            .map(|i| {
                if (i as u32) % stride < columns {
                    a[i]
                } else {
                    pad
                }
            })
            .collect::<Vec<_>>();
        let buffers = [a.clone(), vec![pad; len], vec![pad; len]].map(|values| fixture.guarded(&values, sentinel));
        let mut encoding = fixture.encoding();
        encode(arg(&buffers[0]), arg(&buffers[1]), (rows, columns, stride), &mut encoding);
        encode(arg(&buffers[1]), arg(&buffers[2]), (rows, columns, stride), &mut encoding);
        KernelFixture::complete(encoding);
        // SAFETY: the only command buffer using these buffers has completed.
        unsafe {
            KernelFixture::assert_unchanged(&buffers[0], sentinel, &a, "A");
            for (name, buffer) in [("B", &buffers[1]), ("C", &buffers[2])] {
                let actual = KernelFixture::read_guarded(buffer, sentinel);
                assert_eq!(actual, expected, "{name} of {rows} x {columns}, stride {stride}");
            }
        }
    }
}

/// The element index, byte offset and address words of a case, from exact u128 arithmetic reduced modulo 2^64.
fn oracle([row, stride, column, bytes, low, high]: [u32; 6]) -> [u32; 6] {
    let index = u128::from(row) * u128::from(stride) + u128::from(column);
    let offset = index * u128::from(bytes) % (1 << 64);
    let address = (u128::from(high) << 32 | u128::from(low)) + offset;
    let [index, offset, address] = [index, offset, address].map(|value| value as u64);
    [index as u32, (index >> 32) as u32, offset as u32, (offset >> 32) as u32, address as u32, (address >> 32) as u32]
}

#[uzu_test]
fn copies_every_type_through_rebuilt_pointers() {
    let fixture = KernelFixture::new();
    for ty in [DataType::F32, DataType::F16, DataType::BF16] {
        let kernel = TestStorageAddressCopyVulkanKernel::new(&fixture.context, ty).expect("TestStorageAddressCopy");
        let encode = |input: (&Arc<VkBuffer>, Range<u64>),
                      output: (&Arc<VkBuffer>, Range<u64>),
                      (rows, columns, stride): (u32, u32, u32),
                      encoding: &mut VkCommandBufferEncoding| {
            // SAFETY: each range holds the (rows - 1) * stride + columns aligned elements the kernel indexes, and the
            // output does not alias the input.
            unsafe { kernel.encode(input, output, rows * columns, columns, stride, encoding) }
        };
        match ty {
            DataType::F32 => copies_chain(&fixture, &F32_SPECIALS, encode),
            _ => copies_chain(&fixture, &F16_SPECIALS, encode),
        }
    }
    let words = TestStorageAddressWordsVulkanKernel::new(&fixture.context).expect("TestStorageAddressWords");
    let mut groups = Vec::new();
    copies_chain(&fixture, &F32_SPECIALS, |input, output, (rows, columns, stride), encoding| {
        let (cases, results) = (fixture.guarded::<u32>(&[], 0), fixture.guarded(&[SENTINEL], SENTINEL));
        // SAFETY: as above; with count 0 no case is read and results holds only the NumWorkgroups word.
        unsafe {
            words.encode(input, output, arg(&cases), arg(&results), rows * columns, columns, stride, 0, encoding)
        };
        groups.push((results, (rows * columns).div_ceil(64)));
    });
    for (results, groups) in &groups {
        let expected = if *groups == 0 {
            SENTINEL
        } else {
            *groups
        };
        // SAFETY: every command buffer using these buffers has completed.
        let actual = unsafe { KernelFixture::read_guarded(results, SENTINEL) };
        assert_eq!(actual, [expected], "NumWorkgroups.x of {groups} groups");
    }
    fixture.assert_clean();
}

#[uzu_test]
fn computes_full_width_address_words() {
    let fixture = KernelFixture::new();
    let words = TestStorageAddressWordsVulkanKernel::new(&fixture.context).expect("TestStorageAddressWords");
    let count = CASES.len() as u32;
    let input = (0..count).map(|i| i.wrapping_mul(0x9e37_79b9)).collect::<Vec<_>>();
    let flat = CASES.as_flattened();
    let buffers = [&input[..], &vec![SENTINEL; input.len()][..], flat, &vec![SENTINEL; flat.len() + 1][..]]
        .map(|values| fixture.guarded(values, SENTINEL));
    let [input_buffer, output, cases, results] = &buffers;
    let mut encoding = fixture.encoding();
    // SAFETY: one row of `count` elements; cases holds 6 words and results 6 words per case plus one; no aliasing.
    unsafe {
        words.encode(
            arg(input_buffer),
            arg(output),
            arg(cases),
            arg(results),
            count,
            count,
            count,
            count,
            &mut encoding,
        )
    };
    KernelFixture::complete(encoding);
    // SAFETY: the only command buffer using these buffers has completed.
    unsafe {
        KernelFixture::assert_unchanged(input_buffer, SENTINEL, &input, "input");
        KernelFixture::assert_unchanged(cases, SENTINEL, flat, "cases");
        assert_eq!(KernelFixture::read_guarded(output, SENTINEL), input, "uint copy");
        let actual = KernelFixture::read_guarded::<u32>(results, SENTINEL);
        for (case, actual) in CASES.iter().zip(actual.as_chunks::<6>().0) {
            assert_eq!(*actual, oracle(*case), "case {case:#x?}");
        }
        assert_eq!(actual[flat.len()], 1, "NumWorkgroups.x of 1 group");
    }
    fixture.assert_clean();
}
