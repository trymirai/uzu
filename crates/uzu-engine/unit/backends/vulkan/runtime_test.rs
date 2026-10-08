use std::{
    ffi::CStr,
    sync::{Arc, Weak},
    time::Duration,
};

use ash::vk;
use uzu_engine_macros::uzu_test;

use super::kernel_fixture::KernelFixture;
use crate::backends::vulkan::{
    Error, VkBuffer, VkCommandBufferEncoding, VkComputePipeline, VkContext, VkShader, VkTimestampQueryPool,
    vk_kernels::TestKernelAxisFloatVulkanKernel,
};

const GROUP_SIZE: u32 = 64;

/// `output = input_0 + input_1` over `size` floats through the shared test kernel.
fn add_pipeline_with_arguments(
    context: &Arc<VkContext>,
    push_constant_size: u32,
) -> Result<VkComputePipeline, Error> {
    let shader = VkShader::new(context.clone(), include_bytes!(concat!(env!("OUT_DIR"), "/vulkan/test_kernel.spv")))?;
    VkComputePipeline::new(
        context.clone(),
        shader.module(),
        &[],
        push_constant_size,
        "__dsl_19TestKernelAxisFloat",
        &vk::SpecializationInfo::default(),
    )
}

fn add_pipeline(context: &Arc<VkContext>) -> Arc<VkComputePipeline> {
    Arc::new(add_pipeline_with_arguments(context, 28).expect("pipeline"))
}

fn encode_add(
    encoding: &mut VkCommandBufferEncoding,
    pipeline: &Arc<VkComputePipeline>,
    [input_0, input_1, output]: [&Arc<VkBuffer>; 3],
    size: u32,
) -> Result<(), Error> {
    let push_constants = [input_0, input_1, output]
        .iter()
        .flat_map(|buffer| buffer.device_address().to_ne_bytes())
        .chain(size.to_ne_bytes())
        .collect::<Vec<_>>();
    let bytes = 0..size as u64 * 4;
    // SAFETY: test_kernel_axis_float takes three device addresses and `size`, and touches only
    // indices < size of input_0/input_1 (read) and output (written), all declared below.
    unsafe {
        encoding.encode_dispatch(
            pipeline,
            &push_constants,
            [size.div_ceil(GROUP_SIZE), 1, 1],
            [(input_0, bytes.clone()), (input_1, bytes.clone())],
            [(output, bytes)],
        )
    }
}

/// The test kernel's generated binding records the same addition as the raw pipeline, over guarded ranges.
#[uzu_test]
fn test_kernel_binding_adds() {
    let fixture = KernelFixture::new();
    let size = 1003u32;
    let inputs = [1.0f32, 2.0].map(|scale| (0..size).map(|index| index as f32 * scale).collect::<Vec<_>>());
    let [input_0, input_1] = inputs.each_ref().map(|values| fixture.guarded(values, -7.0f32));
    let output = fixture.guarded(&vec![-1234.0f32; size as usize], -7.0);
    let kernel = TestKernelAxisFloatVulkanKernel::new(&fixture.context).expect("binding");
    let mut encoding = fixture.encoding();
    // SAFETY: the guarded ranges hold `size` floats each; the output aliases nothing.
    unsafe {
        kernel.encode(
            (&input_0.0, input_0.1.clone()),
            (&input_1.0, input_1.1.clone()),
            (&output.0, output.1.clone()),
            size,
            &mut encoding,
        )
    };
    KernelFixture::complete(encoding);
    // SAFETY: the only command buffer using the output has completed.
    let result = unsafe { KernelFixture::read_guarded(&output, -7.0f32) };
    assert_eq!(result, inputs[0].iter().zip(&inputs[1]).map(|(a, b)| a + b).collect::<Vec<_>>());
    fixture.assert_clean();
}

#[uzu_test]
fn copy_and_fill_round_trip() {
    let fixture = KernelFixture::new();
    let context = &fixture.context;
    let pattern = (0..4096).map(|index| (index % 251) as u8).collect::<Vec<_>>();
    let source = fixture.buffer(&pattern);
    let destination = fixture.buffer(&[0u8; 4096]);
    let mut encoding = VkCommandBufferEncoding::new(context.clone()).unwrap();
    encoding.encode_fill(&destination, 0..4096, 0xab).unwrap();
    encoding.encode_copy(&source, 256..1280, &destination, 512).unwrap();
    encoding.encode_fill(&destination, 2048..2052, 0x01).unwrap();
    KernelFixture::complete(encoding);

    let mut expected = vec![0xab; 4096];
    expected[512..1536].copy_from_slice(&pattern[256..1280]);
    expected[2048..2052].fill(0x01);
    // SAFETY: the only command buffer writing `destination` has completed.
    assert_eq!(unsafe { destination.get_bytes() }.as_ref(), expected.as_slice());
    fixture.assert_clean();
}

#[uzu_test]
fn invalid_ranges_return_errors_without_recording() {
    let fixture = KernelFixture::new();
    let context = &fixture.context;
    let pipeline = add_pipeline(context);
    let first = fixture.buffer(&[7u8; 4096]);
    let second = fixture.buffer(&[0u8; 4096]);
    let mut encoding = VkCommandBufferEncoding::new(context.clone()).unwrap();
    let range_error = |result| matches!(result, Err(Error::BufferRange { .. }));
    assert!(range_error(encoding.encode_copy(&first, 0..0, &second, 0)));
    assert!(range_error(encoding.encode_copy(&first, 4000..4100, &second, 0)));
    assert!(range_error(encoding.encode_copy(&first, 0..200, &second, 4000)));
    assert!(range_error(encoding.encode_copy(&first, 0..200, &second, u64::MAX - 1)));
    assert!(matches!(encoding.encode_copy(&first, 0..100, &first, 50), Err(Error::CopyOverlap)));
    assert!(matches!(encoding.encode_fill(&second, 1..9, 0), Err(Error::FillAlignment { .. })));
    assert!(matches!(encoding.encode_fill(&second, 0..6, 0), Err(Error::FillAlignment { .. })));
    assert!(range_error(encoding.encode_fill(&second, 4096..4100, 0)));
    let limit = context.physical_device().properties.limits.max_push_constants_size;
    for size in [6, limit + 4] {
        assert!(matches!(add_pipeline_with_arguments(context, size), Err(Error::PushConstants { .. })));
    }
    // SAFETY: every call below fails its checks before recording, so the shader never runs.
    unsafe {
        for push_constants in [vec![0; 3], vec![0; 24], vec![0; 32], vec![0; limit as usize]] {
            let result = encoding.encode_dispatch(&pipeline, &push_constants, [1, 1, 1], [], []);
            assert!(matches!(
                result,
                Err(Error::PushConstantsMismatch {
                    expected: 28,
                    ..
                })
            ));
        }
        let result = encoding.encode_dispatch(&pipeline, &[0; 28], [u32::MAX, 1, 1], [], []);
        assert!(matches!(result, Err(Error::DispatchGroups { .. })));
        assert!(range_error(encoding.encode_dispatch(&pipeline, &[0; 28], [1, 1, 1], [(&first, 0..4097)], [])));
        #[allow(clippy::reversed_empty_ranges)]
        for range in [10..5, 4097..4097, 4096..4097] {
            assert!(range_error(encoding.encode_dispatch(
                &pipeline,
                &[0; 28],
                [1, 1, 1],
                [(&first, range.clone())],
                []
            )));
            assert!(range_error(encoding.encode_dispatch(&pipeline, &[0; 28], [1, 1, 1], [], [(&second, range)])));
        }
    }

    encoding.encode_copy(&first, 0..100, &first, 100).unwrap();
    encoding.encode_copy(&first, 0..200, &second, 0).unwrap();
    KernelFixture::complete(encoding);
    // SAFETY: the only command buffer writing `second` has completed.
    assert_eq!(&unsafe { second.get_bytes() }[..200], &[7; 200]);
    assert!(matches!(VkBuffer::new(context.clone(), 0), Err(Error::EmptyBuffer)));
    fixture.assert_clean();
}

#[uzu_test]
fn foreign_context_objects_are_rejected() {
    let other = KernelFixture::new();
    let other_context = &other.context;
    let fixture = KernelFixture::new();
    let context = &fixture.context;
    let size = 64u32;
    let input = vec![1.0f32; size as usize];
    let [local_a, local_b, local_output] = [&input, &input, &input].map(|values| fixture.buffer(values));
    let foreign = other.buffer(&input);
    let local_pipeline = add_pipeline(context);
    let foreign_pipeline = add_pipeline(other_context);
    let foreign_error = |result| matches!(result, Err(Error::ForeignContext));

    let mut encoding = VkCommandBufferEncoding::new(context.clone()).unwrap();
    assert!(foreign_error(encoding.encode_copy(&foreign, 0..64, &local_output, 0)));
    assert!(foreign_error(encoding.encode_copy(&local_a, 0..64, &foreign, 0)));
    assert!(foreign_error(encoding.encode_fill(&foreign, 0..64, 0)));
    assert!(foreign_error(encode_add(&mut encoding, &foreign_pipeline, [&local_a, &local_b, &local_output], size)));
    assert!(foreign_error(encode_add(&mut encoding, &local_pipeline, [&local_a, &foreign, &local_output], size)));
    assert!(foreign_error(encode_add(&mut encoding, &local_pipeline, [&local_a, &local_b, &foreign], size)));
    // SAFETY: rejected before recording.
    let empty_foreign =
        unsafe { encoding.encode_dispatch(&local_pipeline, &[0; 28], [1, 1, 1], [(&foreign, 0..0)], []) };
    assert!(foreign_error(empty_foreign));

    encode_add(&mut encoding, &local_pipeline, [&local_a, &local_b, &local_output], size).unwrap();
    KernelFixture::complete(encoding);
    // SAFETY: the command buffer writing `local_output` has completed and none writes `foreign`.
    let (local, foreign) = unsafe { (KernelFixture::read::<f32>(&local_output), KernelFixture::read::<f32>(&foreign)) };
    assert!(local.iter().all(|&value| value == 2.0));
    assert!(foreign.iter().all(|&value| value == 1.0));
    fixture.assert_clean();
    other.assert_clean();
}

/// A dispatch may declare empty read and write spans, including one at offset == size: they are bounds-checked and
/// retained until completion, and the shader never touches them.
#[uzu_test]
fn dispatch_accepts_empty_spans() {
    let fixture = KernelFixture::new();
    let pipeline = add_pipeline(&fixture.context);
    let size = 64u32;
    let [a, b, output] = [(); 3].map(|_| fixture.buffer(&[1.5f32; 64]));
    let untouched = fixture.buffer(&[3u8; 256]);
    let pending = {
        let spare = fixture.buffer(&[3u8; 256]);
        let push_constants = [&a, &b, &output]
            .iter()
            .flat_map(|buffer| buffer.device_address().to_ne_bytes())
            .chain(size.to_ne_bytes())
            .collect::<Vec<_>>();
        let mut encoding = fixture.encoding();
        // SAFETY: the kernel reads 64 floats of a and b and writes 64 of output; the empty spans are never
        // dereferenced.
        unsafe {
            encoding.encode_dispatch(
                &pipeline,
                &push_constants,
                [1, 1, 1],
                [(&a, 0..256), (&b, 0..256), (&spare, 0..0), (&untouched, 256..256)],
                [(&output, 0..256), (&spare, 256..256), (&untouched, 128..128)],
            )
        }
        .unwrap();
        let weak = Arc::downgrade(&spare);
        (encoding.end_encoding().unwrap().submit(), weak)
    };
    assert!(pending.1.upgrade().is_some(), "an empty span's buffer is retained until completion");
    pending.0.wait_until_completed().unwrap();
    assert!(pending.1.upgrade().is_none());
    // SAFETY: the command buffer has completed.
    unsafe {
        assert_eq!(KernelFixture::read::<f32>(&output), [3.0; 64]);
        assert_eq!(KernelFixture::read::<u8>(&untouched), [3; 256]);
    }
    fixture.assert_clean();
}

#[uzu_test]
fn transfer_and_dispatch_chain_is_ordered() {
    let fixture = KernelFixture::new();
    let context = &fixture.context;
    let pipeline = add_pipeline(context);
    let size = 1024u32;
    let input = (0..size).map(|index| index as f32).collect::<Vec<_>>();
    let [a, b] = [&input, &input].map(|values| fixture.buffer(values));
    let [sum, copied, result] = [(); 3].map(|_| fixture.buffer(&vec![-1.0f32; size as usize]));
    let bytes = 0..size as u64 * 4;

    let mut encoding = VkCommandBufferEncoding::new(context.clone()).unwrap();
    encoding.encode_fill(&sum, bytes.clone(), 0).unwrap();
    encode_add(&mut encoding, &pipeline, [&a, &b, &sum], size).unwrap();
    encoding.encode_copy(&sum, bytes.clone(), &copied, 0).unwrap();
    encoding.encode_fill(&a, bytes, 0).unwrap();
    encode_add(&mut encoding, &pipeline, [&a, &copied, &result], size).unwrap();
    KernelFixture::complete(encoding);

    // SAFETY: the only command buffer writing `a` and `result` has completed.
    let (a, result) = unsafe { (KernelFixture::read::<f32>(&a), KernelFixture::read::<f32>(&result)) };
    assert!(a.iter().all(|&value| value == 0.0));
    assert_eq!(result, input.iter().map(|value| value * 2.0).collect::<Vec<_>>());
    fixture.assert_clean();
}

#[uzu_test]
fn multiple_command_buffers_in_flight() {
    let fixture = KernelFixture::new();
    let context = &fixture.context;
    let destination = fixture.buffer(&[0u8; 8 * 256]);
    let pending = (0..8u8)
        .map(|index| {
            let source = fixture.buffer(&[index + 1; 256]);
            let mut encoding = VkCommandBufferEncoding::new(context.clone()).unwrap();
            encoding.encode_copy(&source, 0..256, &destination, index as u64 * 256).unwrap();
            encoding.end_encoding().unwrap().submit()
        })
        .collect::<Vec<_>>();
    for pending in pending.into_iter().rev() {
        pending.wait_until_completed().unwrap();
    }
    // SAFETY: all eight command buffers writing `destination` have completed.
    let bytes = unsafe { destination.get_bytes() };
    for (index, chunk) in bytes.chunks(256).enumerate() {
        assert!(chunk.iter().all(|&value| value == index as u8 + 1), "chunk {index}");
    }
    fixture.assert_clean();
}

#[uzu_test]
fn retains_resources_until_completion() {
    let fixture = KernelFixture::new();
    let context = &fixture.context;
    let size = 1003u32;
    let input = (0..size).map(|index| index as f32).collect::<Vec<_>>();
    let output = fixture.buffer(&vec![0.0f32; size as usize]);
    let mut retained = Vec::<Weak<dyn Send + Sync>>::new();
    let pending = {
        let pipeline = add_pipeline(context);
        let [a, b] = [&input, &input].map(|values| fixture.buffer(values));
        let mut encoding = VkCommandBufferEncoding::new(context.clone()).unwrap();
        encode_add(&mut encoding, &pipeline, [&a, &b, &output], size).unwrap();
        retained.push(Arc::downgrade(&pipeline) as Weak<dyn Send + Sync>);
        retained.extend([&a, &b].map(|buffer| Arc::downgrade(buffer) as Weak<dyn Send + Sync>));
        encoding.end_encoding().unwrap().submit()
    };
    assert!(retained.iter().all(|resource| resource.upgrade().is_some()));
    pending.wait_until_completed().unwrap();
    assert!(retained.iter().all(|resource| resource.upgrade().is_none()));
    // SAFETY: the command buffer writing `output` has completed; the next one is never submitted.
    let output_values = unsafe { KernelFixture::read::<f32>(&output) };
    assert_eq!(output_values, input.iter().map(|value| value * 2.0).collect::<Vec<_>>());

    let source = fixture.buffer(&[1u8; 64]);
    let weak_source = Arc::downgrade(&source);
    let mut encoding = VkCommandBufferEncoding::new(context.clone()).unwrap();
    encoding.encode_copy(&source, 0..64, &output, 0).unwrap();
    drop(source);
    drop(encoding.end_encoding().unwrap());
    assert!(weak_source.upgrade().is_none());
    fixture.assert_clean();
}

#[uzu_test]
fn dropped_pending_waits_for_gpu() {
    let fixture = KernelFixture::new();
    let context = &fixture.context;
    let pattern = (0..4 << 20).map(|index: u32| (index % 253) as u8).collect::<Vec<_>>();
    let source = fixture.buffer(&pattern);
    let weak_source = Arc::downgrade(&source);
    let destination = fixture.buffer(&vec![0u8; pattern.len()]);
    let mut encoding = VkCommandBufferEncoding::new(context.clone()).unwrap();
    encoding.encode_copy(&source, 0..pattern.len() as u64, &destination, 0).unwrap();
    drop(source);
    drop(encoding.end_encoding().unwrap().submit());
    assert!(weak_source.upgrade().is_none());
    // SAFETY: dropping the pending command buffer waited for the copy into `destination`.
    assert_eq!(unsafe { destination.get_bytes() }.as_ref(), pattern.as_slice());
    fixture.assert_clean();
}

#[uzu_test]
fn command_resources_survive_thousand_reuse_cycles() {
    let fixture = KernelFixture::new();
    let context = &fixture.context;
    let destination = fixture.buffer(&[0u8; 256]);
    let mut gpu_time = Duration::ZERO;
    for cycle in 0..1000u32 {
        if cycle.is_multiple_of(10) {
            let mut abandoned = VkCommandBufferEncoding::new(context.clone()).unwrap();
            abandoned.encode_fill(&destination, 0..256, 0xff).unwrap();
            if cycle.is_multiple_of(20) {
                drop(abandoned.end_encoding().unwrap());
            }
        }
        let mut encoding = VkCommandBufferEncoding::new(context.clone()).unwrap();
        encoding.encode_fill(&destination, 0..256, cycle as u8).unwrap();
        gpu_time += KernelFixture::complete(encoding).gpu_execution_time();
        // SAFETY: this cycle's command buffer completed; abandoned ones were never submitted.
        let bytes = unsafe { destination.get_bytes() };
        assert!(bytes.iter().all(|&value| value == cycle as u8), "cycle {cycle}");
    }
    assert!(gpu_time > Duration::ZERO);
    fixture.assert_clean();
}

#[uzu_test]
fn four_threads_submit_concurrently() {
    let fixture = KernelFixture::new();
    let context = &fixture.context;
    std::thread::scope(|scope| {
        for thread in 0..4u8 {
            let (context, fixture) = (context.clone(), &fixture);
            scope.spawn(move || {
                let source = fixture.buffer(&[0u8; 1024]);
                let destination = fixture.buffer(&[0u8; 1024]);
                for submit in 0..32u8 {
                    let value = thread * 32 + submit;
                    let mut encoding = VkCommandBufferEncoding::new(context.clone()).unwrap();
                    encoding.encode_fill(&source, 0..1024, value).unwrap();
                    encoding.encode_copy(&source, 0..1024, &destination, 0).unwrap();
                    KernelFixture::complete(encoding);
                    // SAFETY: this thread's buffers are written only by its own completed command buffer.
                    assert!(unsafe { destination.get_bytes() }.iter().all(|&byte| byte == value));
                }
            });
        }
    });
    fixture.assert_clean();
}

#[uzu_test]
fn timestamps_measure_each_submission() {
    let fixture = KernelFixture::new();
    let context = &fixture.context;
    assert!(matches!(VkTimestampQueryPool::new(context.clone(), 0), Err(Error::TimestampRange)));
    let pool = VkTimestampQueryPool::new(context.clone(), 2).unwrap();
    assert!(matches!(pool.get_duration_nanos(0), Err(Error::TimestampRange)));
    let source = fixture.buffer(&vec![3u8; 16 << 20]);
    let destination = fixture.buffer(&vec![0u8; 16 << 20]);
    for _ in 0..3 {
        let mut encoding = VkCommandBufferEncoding::new(context.clone()).unwrap();
        encoding.encode_copy(&source, 0..16 << 20, &destination, 0).unwrap();
        assert!(KernelFixture::complete(encoding).gpu_execution_time() > Duration::ZERO);
    }
    fixture.assert_clean();
}

/// Run explicitly on the target machine: `cargo test ... selected_device_is_hardware -- --ignored`.
#[uzu_test]
#[ignore]
fn selected_device_is_hardware() {
    let fixture = KernelFixture::new();
    let context = &fixture.context;
    let properties = &context.physical_device().properties;
    let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }.to_string_lossy();
    assert_ne!(properties.device_type, vk::PhysicalDeviceType::CPU, "software Vulkan device {name}");
    assert!(name.starts_with("Apple M2"), "unexpected Vulkan device {name}");
    // The selection requirements of the 16-bit round-to-nearest-even execution mode every kernel declares.
    let physical_device = context.physical_device();
    assert!(physical_device.shader_rounding_mode_rte_float16);
    assert_eq!(physical_device.rounding_mode_independence, vk::ShaderFloatControlsIndependence::ALL);
    fixture.assert_clean();
}
