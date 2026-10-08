use std::{ffi::CStr, sync::Arc, time::Instant};

use ash::vk;
use uzu_engine::backends::vulkan::{
    Error, VkBuffer, VkBufferError, VkCommandBufferCompleted, VkCommandBufferEncoding, VkComputePipeline, VkContext,
    VkContextCreateInfo, VkShader,
};

const WARMUP: usize = 3;
const SAMPLES: usize = 10;
const COPY_SIZES: [u64; 3] = [16 << 20, 64 << 20, 256 << 20];
const DISPATCH_ELEMENTS: u32 = 1 << 20;
const DISPATCH_COUNTS: [u32; 2] = [1, 64];

fn main() -> Result<(), Error> {
    let context = Arc::new(VkContext::new(VkContextCreateInfo::default())?);
    let properties = &context.physical_device().properties;
    let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) };
    let run_kind = if properties.device_type == vk::PhysicalDeviceType::CPU {
        "software diagnostic run, not hardware performance"
    } else {
        "hardware run"
    };
    println!("Vulkan context created: {} ({:?}, {run_kind})", name.to_string_lossy(), properties.device_type);

    let data = (0..4096).map(|index| (index % 100) as u8).collect::<Box<[u8]>>();
    let mut buffer = VkBuffer::new(context.clone(), data.len() as u64)?;
    buffer.fill(&data)?;
    // SAFETY: no command buffer has accessed `buffer`.
    assert_eq!(unsafe { buffer.get_bytes() }.as_ref(), data.as_ref());
    assert!(matches!(buffer.fill(&[0; 4097]), Err(VkBufferError::SizeOutOfBounds { .. })));
    println!("Vulkan buffer round-trip verified: {} bytes", data.len());

    std::thread::scope(|scope| {
        let threads = (0..4)
            .map(|_| {
                let context = context.clone();
                scope.spawn(move || -> Result<(), Error> {
                    for _ in 0..32 {
                        drop(VkCommandBufferEncoding::new(context.clone())?);
                    }
                    Ok(())
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().expect("command resource worker panicked")?;
        }
        Ok::<_, Error>(())
    })?;

    let size = 1003u32;
    let group_size = 64u32;
    let lanes = size.div_ceil(group_size) * group_size;
    let bytes = 0..lanes as u64 * 4;
    let input_0 = (0..lanes).map(|i| i as f32).collect::<Vec<_>>();
    let input_1 = (0..lanes).map(|i| (i * 2) as f32).collect::<Vec<_>>();
    let mut buffers =
        (0..3).map(|_| VkBuffer::new(context.clone(), bytes.end).map(Arc::new)).collect::<Result<Vec<_>, _>>()?;
    for (buffer, input) in buffers.iter_mut().zip([&input_0, &input_1]) {
        Arc::get_mut(buffer).expect("input is not shared").fill(bytemuck::cast_slice(input))?;
    }
    let push_constants = buffers
        .iter()
        .flat_map(|buffer| buffer.device_address().to_ne_bytes())
        .chain(size.to_ne_bytes())
        .collect::<Vec<_>>();
    let shader = VkShader::new(context.clone(), include_bytes!(concat!(env!("OUT_DIR"), "/vulkan/test_kernel.spv")))?;
    let pipeline = Arc::new(VkComputePipeline::new(
        context.clone(),
        shader.module(),
        &[],
        push_constants.len() as u32,
        "__dsl_19TestKernelAxisFloat",
        &vk::SpecializationInfo::default(),
    )?);

    for iteration in 0..2 {
        Arc::get_mut(&mut buffers[2])
            .expect("no pending command buffer retains the output")
            .fill(bytemuck::cast_slice(&vec![-1234f32; lanes as usize]))?;
        let mut encoding = VkCommandBufferEncoding::new(context.clone())?;
        // SAFETY: test_kernel_axis_float reads input_0/input_1 and writes output at indices < size,
        // which the push constants pass as the three declared buffers' addresses followed by size.
        unsafe {
            encoding.encode_dispatch(
                &pipeline,
                &push_constants,
                [size.div_ceil(group_size), 1, 1],
                [(&buffers[0], bytes.clone()), (&buffers[1], bytes.clone())],
                [(&buffers[2], bytes.clone())],
            )?;
        }
        let completed = encoding.end_encoding()?.submit().wait_until_completed()?;
        // SAFETY: the only command buffer writing the output has completed.
        let result = unsafe { buffers[2].get_bytes() };
        for (index, bytes) in result.as_chunks::<4>().0.iter().enumerate() {
            let actual = f32::from_ne_bytes(*bytes);
            let expected = if index < size as usize {
                input_0[index] + input_1[index]
            } else {
                -1234f32
            };
            assert_eq!(actual, expected, "output[{index}] on iteration {iteration}");
        }
        let nanos = completed.gpu_execution_time().as_nanos();
        println!(
            "Vulkan dispatch {iteration} verified: {size} floats, {} guarded lanes, {nanos} ns timestamp delta (not validated for performance)",
            lanes - size
        );
    }

    println!("Measurements ({run_kind}): median of {SAMPLES} samples after {WARMUP} warm-up submissions each");
    let mut copy_times = Vec::new();
    for copy_bytes in COPY_SIZES {
        let source = Arc::new(VkBuffer::new(context.clone(), copy_bytes)?);
        let destination = Arc::new(VkBuffer::new(context.clone(), copy_bytes)?);
        let (gpu, wall) = measure(|| {
            let mut encoding = VkCommandBufferEncoding::new(context.clone())?;
            encoding.encode_copy(&source, 0..copy_bytes, &destination, 0)?;
            encoding.end_encoding()?.submit().wait_until_completed()
        })?;
        let rate = |seconds: f64| copy_bytes as f64 / seconds / 1e9;
        println!(
            "  copy {:>3} MiB: GPU {:>9.1} us, wall {:>9.1} us ({:.1} GB/s including encode, submit and wait)",
            copy_bytes >> 20,
            gpu * 1e6,
            wall * 1e6,
            rate(wall)
        );
        copy_times.push((gpu, rate(gpu)));
    }
    if scales(copy_times[0].0, copy_times[2].0) {
        let rates = copy_times.iter().map(|(_, rate)| format!("{rate:.1}")).collect::<Vec<_>>().join(" / ");
        println!("  GPU copy throughput for 16/64/256 MiB: {rates} GB/s (timestamps scale with copy size)");
    } else {
        println!("  GPU copy timestamps do not scale with copy size; GPU copy throughput is not reported");
    }

    let elements = DISPATCH_ELEMENTS as usize;
    let chain_bytes = 0..DISPATCH_ELEMENTS as u64 * 4;
    let upload = |values: &[f32]| -> Result<Arc<VkBuffer>, Error> {
        let mut buffer = VkBuffer::new(context.clone(), chain_bytes.end)?;
        buffer.fill(bytemuck::cast_slice(values))?;
        Ok(Arc::new(buffer))
    };
    let ramp_values = (0..elements).map(|index| index as f32).collect::<Vec<_>>();
    let (ramp, ones) = (upload(&ramp_values)?, upload(&vec![1.0; elements])?);
    let (ping, pong) = (upload(&ramp_values)?, upload(&ramp_values)?);
    let mut dispatch_times = Vec::new();
    for dispatches in DISPATCH_COUNTS {
        let (gpu, wall) = measure(|| {
            let mut encoding = VkCommandBufferEncoding::new(context.clone())?;
            // Dispatch d writes ping (even d) or pong (odd d) from the previous output, so each one
            // depends on the last through a declared read-after-write or write-after-read hazard.
            for dispatch in 0..dispatches {
                let input = match dispatch {
                    0 => &ramp,
                    _ if dispatch % 2 == 1 => &ping,
                    _ => &pong,
                };
                let output = if dispatch % 2 == 0 {
                    &ping
                } else {
                    &pong
                };
                let arguments = [input, &ones, output]
                    .iter()
                    .flat_map(|buffer| buffer.device_address().to_ne_bytes())
                    .chain(DISPATCH_ELEMENTS.to_ne_bytes())
                    .collect::<Vec<_>>();
                // SAFETY: test_kernel_axis_float reads input/ones and writes output at indices
                // < DISPATCH_ELEMENTS, the three declared buffers whose addresses are in `arguments`.
                unsafe {
                    encoding.encode_dispatch(
                        &pipeline,
                        &arguments,
                        [DISPATCH_ELEMENTS.div_ceil(group_size), 1, 1],
                        [(input, chain_bytes.clone()), (&ones, chain_bytes.clone())],
                        [(output, chain_bytes.clone())],
                    )?;
                }
            }
            encoding.end_encoding()?.submit().wait_until_completed()
        })?;
        let last = if dispatches % 2 == 1 {
            &ping
        } else {
            &pong
        };
        // SAFETY: every command buffer writing the chain buffers has completed.
        let result = unsafe { last.get_bytes() };
        let result = bytemuck::cast_slice::<u8, f32>(&result);
        assert!(result.iter().enumerate().all(|(index, &value)| value == (index as u32 + dispatches) as f32));
        println!(
            "  {dispatches:>2} dependent add dispatches over {elements} floats: GPU {:>9.1} us, wall {:>9.1} us",
            gpu * 1e6,
            wall * 1e6
        );
        dispatch_times.push(gpu);
    }
    match scales(dispatch_times[0], dispatch_times[1]) {
        true => println!("  GPU dispatch timestamps scale with dispatch count"),
        false => {
            println!("  GPU dispatch timestamps do not scale with dispatch count; dispatch GPU times are not reliable")
        },
    }
    Ok(())
}

/// Returns the median GPU timestamp delta and median wall time of one submission, in seconds.
fn measure(mut submission: impl FnMut() -> Result<VkCommandBufferCompleted, Error>) -> Result<(f64, f64), Error> {
    let mut samples = Vec::with_capacity(SAMPLES);
    for sample in 0..WARMUP + SAMPLES {
        let start = Instant::now();
        let gpu = submission()?.gpu_execution_time().as_secs_f64();
        if sample >= WARMUP {
            samples.push((gpu, start.elapsed().as_secs_f64()));
        }
    }
    let median = |mut values: Vec<f64>| {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    Ok((
        median(samples.iter().map(|sample| sample.0).collect()),
        median(samples.iter().map(|sample| sample.1).collect()),
    ))
}

/// GPU times are only trusted when 16x (copies) or 64x (dispatches) more work takes at least 4x longer.
fn scales(
    small: f64,
    large: f64,
) -> bool {
    large >= small * 4.0
}
