use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    time::{Duration, Instant},
};

use bytemuck::{AnyBitPattern, NoUninit};
use half::bf16;
use uzu_engine_macros::uzu_test;

use super::kernel_fixture::KernelFixture;
use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Context, Kernels, gpu_types::Copy, kernel::KVCacheUpdateKernel},
        cpu::Cpu,
        vulkan::{Error, KVCacheUpdateVulkanKernel, VkCommandBufferEncoding},
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, submit_command_buffer},
};

/// FP32 patterns whose upper halves are BF16 patterns of the same kinds: signaling and quiet NaNs with payloads,
/// infinities, signed zeros and subnormals down to the smallest, which a copy through arithmetic would change.
const SPECIAL: [u32; 9] =
    [0x7f81_0001, 0xffa1_2345, 0x7fc0_0001, 0xff80_0000, 0x8000_0000, 0x0000_0001, 0x0001_0000, 0x8001_8000, 0];

fn kernel<T: ArrayElement>(fixture: &KernelFixture) -> KVCacheUpdateVulkanKernel {
    KVCacheUpdateVulkanKernel::new(&fixture.context, T::data_type()).expect("KVCacheUpdate")
}

/// The storage element of `bits`: all of them for FP32, the upper half for BF16.
fn element<T: AnyBitPattern>(bits: u32) -> T {
    match size_of::<T>() {
        2 => bytemuck::pod_read_unaligned(&((bits >> 16) as u16).to_ne_bytes()),
        _ => bytemuck::pod_read_unaligned(&bits.to_ne_bytes()),
    }
}

/// `rows` rows of `element_dim` hashed bit patterns, every fourth a special one.
fn plane<T: AnyBitPattern>(
    rows: u32,
    element_dim: u32,
    seed: u32,
) -> Vec<T> {
    (0..rows * element_dim)
        .map(|index| {
            let hash = (index ^ seed).wrapping_mul(0x9e37_79b9).rotate_left(13).wrapping_mul(0x85eb_ca6b);
            element(match hash % 4 {
                0 => SPECIAL[index.wrapping_add(seed) as usize % SPECIAL.len()],
                _ => hash,
            })
        })
        .collect()
}

fn copies(pairs: &[(u32, u32)]) -> Vec<Copy> {
    pairs
        .iter()
        .map(|&(source, destination)| Copy {
            source,
            destination,
        })
        .collect()
}

/// Keys, values, copies, copy_count and element_dim of one dispatch.
fn dispatch<T: AnyBitPattern>(
    rows: u32,
    element_dim: u32,
    pairs: &[(u32, u32)],
    copy_count: u32,
) -> (Vec<T>, Vec<T>, Vec<Copy>, u32, u32) {
    let seed = rows * 131 + element_dim * 7 + copy_count;
    (plane(rows, element_dim, seed), plane(rows, element_dim, !seed), copies(pairs), copy_count, element_dim)
}

/// The copies applied one whole row after another, a formulation independent of the CPU's column order.
fn row_oracle<T: Clone>(
    plane: &[T],
    copies: &[Copy],
    element_dim: u32,
) -> Vec<T> {
    let mut result = plane.to_vec();
    let dim = element_dim as usize;
    for copy in copies {
        let row = result[copy.source as usize * dim..][..dim].to_vec();
        result[copy.destination as usize * dim..][..dim].clone_from_slice(&row);
    }
    result
}

/// The CPU kernel through the shared trait. CPU buffers cannot be empty.
fn cpu_output<T: ArrayElement>(
    (keys, values, copies, copy_count, element_dim): &(Vec<T>, Vec<T>, Vec<Copy>, u32, u32)
) -> (Vec<T>, Vec<T>) {
    if keys.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let context = create_context::<Cpu>();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::KVCacheUpdateKernel::new(&context, T::data_type())
        .expect("CPU KVCacheUpdate");
    let mut keys = create_buffer_with_data::<Cpu, T>(&context, keys);
    let mut values = create_buffer_with_data::<Cpu, T>(&context, values);
    let mut command_buffer = context.create_command_buffer(None, None).expect("CPU command buffer");
    kernel.encode(&mut keys, &mut values, copies, *copy_count, *element_dim, &mut command_buffer);
    submit_command_buffer(command_buffer);
    (buffer_to_vec::<Cpu, T>(&keys), buffer_to_vec::<Cpu, T>(&values))
}

/// Records every dispatch into `encoding` over guarded keys and values, then drops the host copies before completing it.
/// Returns the keys and values after checking their guards.
fn gpu_outputs<T: ArrayElement + NoUninit + AnyBitPattern>(
    fixture: &KernelFixture,
    kernel: &KVCacheUpdateVulkanKernel,
    dispatches: &[(Vec<T>, Vec<T>, Vec<Copy>, u32, u32)],
    mut encoding: VkCommandBufferEncoding,
) -> Vec<(Vec<T>, Vec<T>)> {
    let sentinel = element::<T>(0x7fa5_5aa5);
    let buffers = dispatches
        .iter()
        .map(|(keys, values, ..)| (fixture.guarded(keys, sentinel), fixture.guarded(values, sentinel)))
        .collect::<Vec<_>>();
    for ((_, _, copies, copy_count, element_dim), (keys, values)) in dispatches.iter().zip(&buffers) {
        let copies = copies.clone();
        // SAFETY: keys and values hold every row the copies name, aligned, and alias nothing.
        unsafe {
            kernel.encode(
                (&keys.0, keys.1.clone()),
                (&values.0, values.1.clone()),
                &copies,
                *copy_count,
                *element_dim,
                &mut encoding,
            )
        };
    }
    KernelFixture::complete(encoding);
    // SAFETY: the only command buffer using these buffers has completed.
    buffers
        .iter()
        .map(|(keys, values)| unsafe {
            (KernelFixture::read_guarded(keys, sentinel), KernelFixture::read_guarded(values, sentinel))
        })
        .collect()
}

/// Vulkan, the CPU kernel and the row oracle agree bit for bit on keys and values.
fn check<T: ArrayElement + NoUninit + AnyBitPattern>(
    fixture: &KernelFixture,
    dispatches: &[(Vec<T>, Vec<T>, Vec<Copy>, u32, u32)],
    encoding: VkCommandBufferEncoding,
) {
    let outputs = gpu_outputs(fixture, &kernel::<T>(fixture), dispatches, encoding);
    for (dispatch, (keys, values)) in dispatches.iter().zip(outputs) {
        let (expected_keys, expected_values, copies, copy_count, element_dim) = dispatch;
        let applied = &copies[..*copy_count as usize];
        let case = format!("KVCacheUpdate {:?} dim {element_dim} copies {applied:?}", T::data_type());
        let (cpu_keys, cpu_values) = cpu_output(dispatch);
        for (name, input, cpu, gpu) in
            [("keys", expected_keys, cpu_keys, keys), ("values", expected_values, cpu_values, values)]
        {
            let oracle = row_oracle(input, applied, *element_dim);
            let bytes = bytemuck::cast_slice::<T, u8>;
            assert_eq!(bytes(&cpu), bytes(&oracle), "{case}: CPU {name} differ from the row oracle");
            assert_eq!(bytes(&gpu), bytes(&oracle), "{case}: Vulkan {name} differ");
        }
    }
}

/// Dependent copies in order: forward and reverse chains, a swap without and with a spare row, repeated destinations,
/// self-copies, a prefix of the slice, no copies and an empty slice; across columns of one, tails and whole groups, and no
/// columns. All in one command buffer, each dispatch with its own upload.
fn ordered_copies<T: ArrayElement + NoUninit + AnyBitPattern>() {
    let fixture = KernelFixture::new();
    let cases: [(&[(u32, u32)], u32); 10] = [
        (&[(0, 1), (1, 2)], 2),
        (&[(1, 2), (0, 1)], 2),
        (&[(0, 1), (1, 0)], 2),
        (&[(0, 7), (1, 0), (7, 1)], 3),
        (&[(2, 5), (3, 5), (5, 6)], 3),
        (&[(4, 4), (4, 6), (6, 6)], 3),
        (&[(0, 3), (1, 3), (2, 3)], 1),
        (&[(0, 1)], 0),
        (&[], 0),
        (&[(7, 0), (6, 1), (5, 2), (4, 3), (3, 4), (2, 5), (1, 6), (0, 7)], 8),
    ];
    let mut dispatches = Vec::new();
    for element_dim in [0, 1, 3, 255, 256, 257, 1000] {
        for (pairs, copy_count) in cases {
            dispatches.push(dispatch::<T>(8, element_dim, pairs, copy_count));
        }
    }
    check(&fixture, &dispatches, fixture.encoding());
    fixture.assert_clean();
}

#[uzu_test]
fn ordered_copies_all_types() {
    ordered_copies::<f32>();
    ordered_copies::<bf16>();
}

/// Many rows of a wide cache with hashed copies among them, chains and repeats included.
fn hashed_copies<T: ArrayElement + NoUninit + AnyBitPattern>() {
    let fixture = KernelFixture::new();
    let mut dispatches = Vec::new();
    for (rows, element_dim, count) in [(64, 4103, 100), (1100, 576, 1024)] {
        let pairs =
            (0..count).map(|i: u32| (i.wrapping_mul(2_654_435_761) % rows, (i * 7 + 3) % rows)).collect::<Vec<_>>();
        dispatches.push(dispatch::<T>(rows, element_dim, &pairs, count - 1));
    }
    check(&fixture, &dispatches, fixture.encoding());
    fixture.assert_clean();
}

#[uzu_test]
fn hashed_copies_all_types() {
    hashed_copies::<f32>();
    hashed_copies::<bf16>();
}

/// Each dispatch reads the copies as they were when it was encoded: a slice changed and encoded again, and one dropped,
/// both before submission; the kernel and the encoding are gone by then too. An encoding dropped unsubmitted after an
/// upload releases it cleanly.
#[uzu_test]
fn uploads_are_snapshots() {
    let fixture = KernelFixture::new();
    let sentinel = element::<f32>(0x7fa5_5aa5);
    let kernel = kernel::<f32>(&fixture);
    let planes = (0..6).map(|seed| plane::<f32>(4, 33, seed)).collect::<Vec<_>>();
    let buffers = planes.iter().map(|plane| fixture.guarded(plane, sentinel)).collect::<Vec<_>>();
    let encode =
        |kernel: &KVCacheUpdateVulkanKernel, encoding: &mut VkCommandBufferEncoding, index: usize, copies: &[Copy]| {
            let ((keys, key_range), (values, value_range)) = (&buffers[2 * index], &buffers[2 * index + 1]);
            // SAFETY: keys and values hold the 4 rows of 33 the copies name and alias nothing.
            unsafe { kernel.encode((keys, key_range.clone()), (values, value_range.clone()), copies, 1, 33, encoding) };
        };
    let mut encoding = fixture.encoding();
    let mut changing = copies(&[(0, 1)]);
    encode(&kernel, &mut encoding, 0, &changing);
    changing[0] = Copy {
        source: 2,
        destination: 3,
    };
    encode(&kernel, &mut encoding, 1, &changing);
    drop(changing);
    let dropped = copies(&[(3, 0)]);
    encode(&kernel, &mut encoding, 2, &dropped);
    drop(dropped);
    let mut unsubmitted = fixture.encoding();
    encode(&kernel, &mut unsubmitted, 0, &copies(&[(1, 2)]));
    drop(unsubmitted);
    let executable = encoding.end_encoding().expect("end encoding");
    drop(kernel);
    executable.submit().wait_until_completed().expect("completion");
    for (index, pairs) in [[(0, 1)], [(2, 3)], [(3, 0)]].iter().enumerate() {
        for plane in [2 * index, 2 * index + 1] {
            // SAFETY: the command buffer has completed.
            let result = unsafe { KernelFixture::read_guarded(&buffers[plane], sentinel) };
            let expected = row_oracle(&planes[plane], &copies(pairs), 33);
            let bytes = bytemuck::cast_slice::<f32, u8>;
            assert_eq!(bytes(&result), bytes(&expected), "{pairs:?} plane {plane}");
        }
    }
    fixture.assert_clean();
}

/// Construction rejects F16 and I32. `encode` rejects more copies than the slice holds before recording or uploading
/// anything; the same command buffer then completes valid work and the rejected keys stay untouched.
#[uzu_test]
fn rejects_invalid_contracts() {
    let fixture = KernelFixture::new();
    for data_type in [DataType::F16, DataType::I32] {
        assert!(
            matches!(
                KVCacheUpdateVulkanKernel::new(&fixture.context, data_type),
                Err(Error::KernelVariant {
                    kernel: "KVCacheUpdate",
                    ..
                })
            ),
            "{data_type:?}"
        );
    }
    let kernel = kernel::<f32>(&fixture);
    let untouched = fixture.buffer(&[0u32; 1024]);
    let mut encoding = fixture.encoding();
    for (pairs, copy_count) in [(&[][..], 1), (&[(0, 1)][..], 2), (&[(0, 1), (1, 2)][..], u32::MAX)] {
        let copies = copies(pairs);
        let result = catch_unwind(AssertUnwindSafe(|| unsafe {
            // SAFETY: never dispatched: the precondition fails before recording.
            kernel.encode((&untouched, 0..2048), (&untouched, 2048..4096), &copies, copy_count, 4, &mut encoding)
        }));
        let payload = result.expect_err("encode accepted");
        let message = payload.downcast_ref::<String>().expect("precondition message");
        assert!(message.contains("KVCacheUpdate: precondition"), "{pairs:?} {copy_count}: {message}");
    }
    check(&fixture, &[dispatch::<f32>(4, 17, &[(0, 2), (2, 3)], 2)], encoding);
    // SAFETY: the completed command buffer recorded only the valid dispatch.
    assert!(unsafe { KernelFixture::read::<u32>(&untouched) }.iter().all(|&word| word == 0), "a rejected call wrote");
    fixture.assert_clean();
}

/// Run alone: `cargo test ... kv_cache_update_test::throughput -- --ignored --nocapture`. Accepting speculative
/// tokens of caches of 4096 rows of 2 to 8 KV heads of 128: copy counts up to the suffix capacity of 1024, each copy
/// from row 2i to row i past a prefix of 2048. Encoding, which includes the upload, is timed apart from the GPU.
#[uzu_test]
#[ignore]
fn throughput() {
    fn measure<T: ArrayElement + NoUninit + AnyBitPattern>(fixture: &KernelFixture) {
        let kernel = kernel::<T>(fixture);
        for element_dim in [256u32, 512, 1024] {
            let (keys, values) =
                (fixture.buffer(&plane::<T>(4096, element_dim, 1)), fixture.buffer(&plane::<T>(4096, element_dim, 2)));
            for count in [1u32, 16, 128, 1024] {
                let copies = copies(&(0..count).map(|i| (2048 + 2 * i, 2048 + i)).collect::<Vec<_>>());
                let mut encodes = Vec::new();
                let (gpu, wall) = fixture.median_times(|encoding| {
                    let start = Instant::now();
                    // SAFETY: both planes hold 4096 rows of `element_dim`; they do not alias.
                    unsafe {
                        kernel.encode(
                            (&keys, 0..keys.size()),
                            (&values, 0..values.size()),
                            &copies,
                            count,
                            element_dim,
                            encoding,
                        )
                    };
                    encodes.push(start.elapsed());
                });
                encodes.sort();
                let encode: Duration = encodes[encodes.len() / 2];
                let bytes = 4 * u64::from(count * element_dim) * size_of::<T>() as u64;
                eprintln!(
                    "MEASURE KVCacheUpdate {:?} dim {element_dim} copies {count}: {bytes} B; median of 10 after 3 warm-up: GPU {gpu:?} ({:.1} GB/s), encode+upload {encode:?}, wall {wall:?}",
                    T::data_type(),
                    bytes as f64 / gpu.as_secs_f64() / 1e9
                );
            }
        }
    }
    let fixture = KernelFixture::new();
    measure::<f32>(&fixture);
    measure::<bf16>(&fixture);
    fixture.assert_clean();
}
