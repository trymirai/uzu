use std::sync::Arc;

use ash::vk;
use uzu_engine::backends::vulkan::{
    Error, VkBuffer, VkBufferCreateInfo, VkBufferError, VkCommandBuffers, VkComputePipeline,
    VkComputeShaderLayoutBuffer, VkComputeShaderLayoutSet, VkContext, VkContextCreateInfo, VkShader,
    VkTimestampQueryPool,
};

fn main() -> Result<(), Error> {
    let context = Arc::new(VkContext::new(VkContextCreateInfo::default())?);
    let name = unsafe { std::ffi::CStr::from_ptr(context.physical_device().properties.device_name.as_ptr()) };
    println!("Vulkan context created: {}", name.to_string_lossy());
    let data = (0..4096).map(|index| (index % 100) as u8).collect::<Box<[u8]>>();
    let info = VkBufferCreateInfo::new(data.len() as u64, true, true);
    let mut buffer = VkBuffer::new_with_info(context.clone(), &info)?;
    buffer.fill(&data)?;
    assert_eq!(buffer.get_bytes()?.as_ref(), data.as_ref());
    assert!(matches!(buffer.fill(&[0; 4097]), Err(VkBufferError::SizeOutOfBounds { .. })));
    println!("Vulkan buffer round-trip verified: {} bytes", data.len());

    std::thread::scope(|scope| {
        let threads = (0..4)
            .map(|_| {
                let context = context.clone();
                scope.spawn(move || -> Result<(), Error> {
                    for _ in 0..32 {
                        drop(VkCommandBuffers::new(context.clone(), true, 1)?);
                    }
                    Ok(())
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().expect("command pool worker panicked")?;
        }
        Ok::<_, Error>(())
    })?;
    let layout = VkComputeShaderLayoutSet::new(
        context.clone(),
        Box::new([VkComputeShaderLayoutBuffer {
            buffer,
            binding: 0,
        }]),
    )?;

    let size = 1003u32;
    let group_size = 64u32;
    let lanes = size.div_ceil(group_size) * group_size;
    let input_0 = (0..lanes).map(|i| i as f32).collect::<Vec<_>>();
    let input_1 = (0..lanes).map(|i| (i * 2) as f32).collect::<Vec<_>>();
    let mut buffers = (0..3)
        .map(|_| VkBuffer::new_with_info(context.clone(), &VkBufferCreateInfo::new(lanes as u64 * 4, true, true)))
        .collect::<Result<Vec<_>, _>>()?;
    buffers[0].fill(bytemuck::cast_slice(&input_0))?;
    buffers[1].fill(bytemuck::cast_slice(&input_1))?;
    let push_constants = buffers
        .iter()
        .flat_map(|buffer| buffer.device_address().to_ne_bytes())
        .chain(size.to_ne_bytes())
        .collect::<Vec<_>>();
    let shader = VkShader::new(context.clone(), concat!(env!("OUT_DIR"), "/vulkan/test_kernel.spv"))?;
    let entries = [vk::SpecializationMapEntry::default().constant_id(0).offset(0).size(4)];
    let group_bytes = group_size.to_ne_bytes();
    let specialization = vk::SpecializationInfo::default().map_entries(&entries).data(&group_bytes);
    let pipeline = VkComputePipeline::new(
        context.clone(),
        shader.module(),
        &[layout.descriptor_set_layout()],
        push_constants.len() as u32,
        "__dsl_22test_kernel_axis_float",
        &specialization,
    )?;
    let commands = VkCommandBuffers::new(context.clone(), true, 1)?;
    let command = commands.command_buffers()[0];
    let mut timestamps = VkTimestampQueryPool::new(context.clone(), 2)?;
    let device = context.device();

    for iteration in 0..2 {
        buffers[2].fill(bytemuck::cast_slice(&vec![-1234f32; lanes as usize]))?;
        // Hold the pool lock for every operation recording or resetting its buffers.
        let pool = context.command_pool();
        unsafe {
            device.reset_command_buffer(command, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())?;
            timestamps.write(command)?;
            assert!(matches!(timestamps.get_duration_nanos(0), Err(Error::TimestampRange)));
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline());
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.pipeline_layout(),
                0,
                &[layout.descriptor_set()],
                &[],
            );
            device.cmd_push_constants(
                command,
                pipeline.pipeline_layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push_constants,
            );
            device.cmd_dispatch(command, size.div_ceil(group_size), 1, 1);
            timestamps.write(command)?;
            assert!(matches!(timestamps.write(command), Err(Error::TimestampCapacity)));
            let barriers = [buffers[2].get_memory_barrier(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::HOST_READ)];
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &barriers,
                &[],
            );
            device.end_command_buffer(command)?;
        }
        drop(pool);
        assert!(matches!(timestamps.get_duration_nanos(0), Err(Error::Vulkan(vk::Result::NOT_READY))));
        unsafe {
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let command_list = [command];
            let submission = [vk::SubmitInfo::default().command_buffers(&command_list)];
            let submitted = device.queue_submit(context.queue(), &submission, fence);
            let completed = submitted.and_then(|_| device.wait_for_fences(&[fence], true, u64::MAX));
            device.destroy_fence(fence, None);
            completed?;
        }
        let result = buffers[2].get_bytes()?;
        for (index, bytes) in result.as_chunks::<4>().0.iter().enumerate() {
            let actual = f32::from_ne_bytes(*bytes);
            let expected = if index < size as usize {
                input_0[index] + input_1[index]
            } else {
                -1234f32
            };
            assert_eq!(actual, expected, "output[{index}] on iteration {iteration}");
        }
        let nanos = timestamps.get_duration_nanos(0)?;
        assert!(nanos.is_finite() && nanos >= 0.0);
        println!("Vulkan dispatch {iteration} verified: {size} floats, {} guarded lanes, {nanos} ns", lanes - size);
        unsafe {
            timestamps.reset();
        }
    }
    println!("Vulkan command pool concurrency and timestamp reuse verified");
    Ok(())
}
