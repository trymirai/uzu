use std::sync::Arc;

use futures::executor::block_on;
use uzu_engine::backends::vulkan::{
    CommandBufferCompleted, CommandBufferExecutable, CommandBufferPending, Context, Error,
    TestKernelAxisFloatVulkanKernel, VkBuffer, VkBufferError, VkContext, VkContextCreateInfo,
};

fn main() -> Result<(), Error> {
    let context = VkContext::with_create_info(VkContextCreateInfo::default())?;
    println!("Vulkan context created: {}", context.device_name().unwrap_or("unknown"));

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
                scope.spawn(|| -> Result<(), Error> {
                    for _ in 0..32 {
                        drop(context.create_command_buffer(None, None)?);
                    }
                    Ok(())
                })
            })
            .collect::<Vec<_>>();
        threads.into_iter().try_for_each(|thread| thread.join().expect("command buffer worker panicked"))
    })?;
    println!("Vulkan command buffer concurrency verified");

    // The kernel's 64-lane groups cover 1024 floats, of which the guarded last 21 stay untouched.
    let (size, lanes) = (1003u32, 1024usize);
    let bytes = 0..lanes as u64 * 4;
    let filled = |values: &[f32]| -> Result<Arc<VkBuffer>, Error> {
        let mut buffer = VkBuffer::new(context.clone(), bytes.end)?;
        buffer.fill(bytemuck::cast_slice(values))?;
        Ok(Arc::new(buffer))
    };
    let input_0 = (0..lanes).map(|index| index as f32).collect::<Vec<_>>();
    let input_1 = (0..lanes).map(|index| (index * 2) as f32).collect::<Vec<_>>();
    let inputs = [filled(&input_0)?, filled(&input_1)?];
    let kernel = TestKernelAxisFloatVulkanKernel::new(&context)?;
    // Two command buffers, the second reusing the first's recycled resources and timestamps.
    for iteration in 0..2 {
        let output = filled(&vec![-1234.0; lanes])?;
        let mut encoding = context.create_command_buffer(None, None)?;
        // SAFETY: every range holds `lanes` floats, of which the kernel reads and writes the first `size`.
        unsafe {
            kernel.encode(
                (&inputs[0], bytes.clone()),
                (&inputs[1], bytes.clone()),
                (&output, bytes.clone()),
                size,
                &mut encoding,
            );
        }
        let completed = block_on(encoding.try_end_encoding()?.submit().wait_until_completed())?;
        // SAFETY: the only command buffer writing the output has completed.
        let result = unsafe { output.get_bytes() };
        for (index, bytes) in result.as_chunks::<4>().0.iter().enumerate() {
            let expected = if index < size as usize {
                input_0[index] + input_1[index]
            } else {
                -1234.0
            };
            assert_eq!(f32::from_ne_bytes(*bytes), expected, "output[{index}] on iteration {iteration}");
        }
        println!("Vulkan dispatch {iteration} verified in {:?}", completed.gpu_execution_time());
    }
    Ok(())
}
