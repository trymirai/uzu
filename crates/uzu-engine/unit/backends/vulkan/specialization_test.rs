use std::{fmt::Debug, ops::Range, sync::Arc};

use ash::vk;
use bytemuck::{AnyBitPattern, NoUninit};
use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::kernel_fixture::KernelFixture;
use crate::{
    array::ArrayElement,
    backends::vulkan::{
        VkBuffer, VkComputePipeline, VkShader,
        vk_kernels::{
            TestSpecializationAVulkanKernel, TestSpecializationArrayVulkanKernel, TestSpecializationBVulkanKernel,
            TestSpecializationPlainVulkanKernel,
        },
    },
    data_type::DataType,
};

const SENTINEL: u32 = 0xdead_beef;
/// The module the generated bindings load, for creating TestSpecializationArray with no specialization map entry.
const SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vulkan/tests/specialization.spv"));
/// TestSpecializationArray's state_size SpecId values; `None` passes no map entry, keeping the module default 64.
const STATE_SIZES: [Option<u32>; 10] =
    [None, Some(0), Some(1), Some(63), Some(64), Some(65), Some(128), Some(256), Some(1024), Some(4096)];
/// One workgroup of 64 invocations: invocation v owns row v.
const INVOCATIONS: usize = 64;

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

fn sentinel<T: Float>() -> T {
    T::from(-7.0).unwrap()
}

fn untouched<T: NoUninit + Float>(slack: &[T]) -> bool {
    slack.iter().all(|value| bytemuck::bytes_of(value) == bytemuck::bytes_of(&sentinel::<T>()))
}

/// TestSpecializationArray in scalar f32: per invocation v, row[i] = f32(in[v R + i]) for i < R; per token t,
/// a = f32(in[t]), acc = +0, then row[i] = row[i] a + 1 and acc += row[i] for i < n, then row[bits(acc) mod R] = acc;
/// finally out[v R + i] = T(row[i]). The tests' inputs keep every product and sum an exact integer, so fusion is moot.
fn reference<T: Float>(
    input: &[T],
    capacity: usize,
    n: usize,
    tokens: usize,
) -> Vec<T> {
    let f = |value: T| value.to_f32().unwrap();
    (0..INVOCATIONS)
        .flat_map(|v| {
            let mut row = input[v * capacity..(v + 1) * capacity].iter().map(|&value| f(value)).collect::<Vec<_>>();
            for t in 0..tokens {
                let (a, mut acc) = (f(input[t]), 0.0f32);
                for i in 0..n {
                    row[i] = row[i] * a + 1.0;
                    acc += row[i];
                }
                row[acc.to_bits() as usize % capacity] = acc;
            }
            row.into_iter().map(|value| T::from(value).unwrap())
        })
        .collect()
}

/// One dispatch of q tokens: through the generated binding with `state_size`, or with `None` through the raw owners on
/// the same module with an empty specialization map. Guarded ranges of at least 64 rows of 64 keep a pipeline ignoring
/// the specialization in bounds and visible. Asserts the input and guards unchanged; returns the whole output range.
fn dispatch<T: ArrayElement + NoUninit + AnyBitPattern + Float>(
    fixture: &KernelFixture,
    state_size: Option<u32>,
    capacity: usize,
    tokens: u32,
    input: &[T],
) -> Vec<T> {
    let label = format!("{:?} state_size {state_size:?} q {tokens}", T::data_type());
    let source = fixture.guarded(input, sentinel());
    let output = fixture.guarded(&vec![sentinel::<T>(); INVOCATIONS * capacity.max(64)], sentinel());
    let mut encoding = fixture.encoding();
    // SAFETY (both arms): the kernel accesses max(64 R, q) input and 64 R output elements, inside the guarded ranges.
    match state_size {
        Some(size) => unsafe {
            TestSpecializationArrayVulkanKernel::new(&fixture.context, T::data_type(), size)
                .unwrap_or_else(|error| panic!("{label}: {error:?}"))
                .encode(range(&source), range(&output), tokens, &mut encoding)
        },
        None => {
            let entry = match T::data_type() {
                DataType::F32 => "__dsl_23TestSpecializationArray_5float",
                DataType::F16 => "__dsl_23TestSpecializationArray_4half",
                DataType::BF16 => "__dsl_23TestSpecializationArray_4bf16",
                other => panic!("{other:?} has no TestSpecializationArray entry point"),
            };
            let shader = VkShader::new(fixture.context.clone(), SPIRV).expect("shader module");
            let pipeline = VkComputePipeline::new(
                fixture.context.clone(),
                shader.module(),
                &[],
                24,
                entry,
                &vk::SpecializationInfo::default(),
            )
            .unwrap_or_else(|error| panic!("{label}: {error:?}"));
            let address = |(buffer, range): &(Arc<VkBuffer>, Range<u64>)| buffer.device_address() + range.start;
            // The entry point's 24-byte block: both range start addresses, then q.
            let push =
                [&address(&source).to_ne_bytes()[..], &address(&output).to_ne_bytes(), &tokens.to_ne_bytes(), &[0; 4]];
            unsafe {
                encoding.encode_dispatch(
                    &Arc::new(pipeline),
                    &push.concat(),
                    [1, 1, 1],
                    [range(&source)],
                    [range(&output)],
                )
            }
            .unwrap_or_else(|error| panic!("{label}: {error:?}"));
        },
    }
    KernelFixture::complete(encoding);
    // SAFETY: the only command buffer using these buffers has completed.
    unsafe {
        KernelFixture::assert_unchanged(&source, sentinel(), input, &format!("{label}: input"));
        KernelFixture::read_guarded(&output, sentinel())
    }
}

/// A module-owned specialization sizes TestSpecializationArray's private row: for the generated binding's SpecId and the
/// module default (no map entry), every `STATE_SIZES` x q in 0..3 case of T matches the reference bit for bit with the
/// slack beyond 64 R untouched. Values 1..7 with in[0] = 1, in[1] = 2 make every token multiplier 1 or 2.
fn module_owned_array<T: ArrayElement + NoUninit + AnyBitPattern + Float + Debug>() {
    let fixture = KernelFixture::new();
    for state_size in STATE_SIZES {
        let n = state_size.map_or(64, |size| size as usize);
        let capacity = n.max(1);
        for tokens in 0..3 {
            let input = (0..INVOCATIONS * capacity.max(64)).map(|k| T::from(k % 7 + 1).unwrap()).collect::<Vec<_>>();
            let output = dispatch(&fixture, state_size, capacity, tokens, &input);
            let label = format!("{:?} state_size {state_size:?} R {capacity} q {tokens}", T::data_type());
            let (payload, slack) = output.split_at(INVOCATIONS * capacity);
            KernelFixture::assert_bits(&reference(&input, capacity, n, tokens as usize), payload, &label);
            assert!(untouched(slack), "{label}: slack written");
        }
    }
    fixture.assert_clean();
}

#[uzu_test]
fn module_owned_array_sizes_private_rows() {
    module_owned_array::<f32>();
    module_owned_array::<f16>();
    module_owned_array::<bf16>();
}

/// N 1, q 2 with in[0] = in[1] = 1 (the token multipliers) and every other row v >= 2 starting at 2048 (F16) or 256
/// (BF16), where T's spacing is 2: the FP32 row ends at 2050 or 258, a row rounded to T after each token at 2048 or 256
/// (2049 and 257 round to even). Rows 0 and 1 follow the reference.
#[uzu_test]
fn module_owned_array_retains_fp32_rows() {
    fn check<T: ArrayElement + NoUninit + AnyBitPattern + Float + Debug>(start: f32) {
        let fixture = KernelFixture::new();
        let input =
            (0..INVOCATIONS * 64).map(|k| T::from([start, 1.0][usize::from(k < 2)]).unwrap()).collect::<Vec<_>>();
        let output = dispatch(&fixture, Some(1), 1, 2, &input);
        let (payload, slack) = output.split_at(INVOCATIONS);
        KernelFixture::assert_bits(&reference(&input, 1, 1, 2), payload, &format!("{:?} retention", T::data_type()));
        let typed = (0..2).fold(T::from(start).unwrap(), |row, _| T::from(row.to_f32().unwrap() + 1.0).unwrap());
        for &row in &payload[2..] {
            assert_eq!(row.to_f32(), Some(start + 2.0), "{:?}: retained row", T::data_type());
            assert_ne!(row.to_f32(), typed.to_f32(), "{:?}: equals the per-token T alternative", T::data_type());
        }
        assert!(untouched(slack), "{:?}: slack written", T::data_type());
        fixture.assert_clean();
    }
    check::<f16>(2048.0);
    check::<bf16>(256.0);
}
