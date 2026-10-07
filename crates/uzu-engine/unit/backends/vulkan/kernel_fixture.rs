use std::{ffi::CStr, fmt::Debug, mem::size_of, ops::Range, sync::Arc};

use bytemuck::{AnyBitPattern, NoUninit};
use num_traits::Float;

use super::validation_logger::ValidationLogger;
use crate::backends::vulkan::{
    VkBuffer, VkCommandBufferCompleted, VkCommandBufferEncoding, VkContext, VkContextCreateInfo,
};

/// A validated Vulkan context with the buffer and submission helpers shared by the Vulkan tests.
pub struct KernelFixture {
    pub context: Arc<VkContext>,
    logger: ValidationLogger,
}

impl KernelFixture {
    /// Sentinel elements before and after every guarded range, so guarded ranges start at a nonzero offset.
    pub const GUARD: usize = 64;

    pub fn new() -> Self {
        let logger = ValidationLogger::default();
        let context = VkContext::new(VkContextCreateInfo {
            with_validation: true,
            logger: Box::new(logger.clone()),
        })
        .expect("Vulkan context");
        let physical_device = context.physical_device();
        let name = unsafe { CStr::from_ptr(physical_device.properties.device_name.as_ptr()) };
        eprintln!(
            "Vulkan test device: {} ({:?}, subgroup size {})",
            name.to_string_lossy(),
            physical_device.properties.device_type,
            physical_device.subgroup_properties.size
        );
        Self {
            context: Arc::new(context),
            logger,
        }
    }

    pub fn buffer<T: NoUninit>(
        &self,
        values: &[T],
    ) -> Arc<VkBuffer> {
        let bytes = bytemuck::cast_slice::<T, u8>(values);
        let mut buffer = VkBuffer::new(self.context.clone(), bytes.len() as u64).expect("buffer");
        buffer.fill(bytes).expect("host fill");
        Arc::new(buffer)
    }

    /// A buffer holding `values` between `GUARD` sentinels on each side, with the byte range of `values`.
    pub fn guarded<T: NoUninit>(
        &self,
        values: &[T],
        sentinel: T,
    ) -> (Arc<VkBuffer>, Range<u64>) {
        let buffer = self.buffer(&[vec![sentinel; Self::GUARD], values.to_vec(), vec![sentinel; Self::GUARD]].concat());
        (buffer, (Self::GUARD * size_of::<T>()) as u64..((Self::GUARD + values.len()) * size_of::<T>()) as u64)
    }

    pub fn encoding(&self) -> VkCommandBufferEncoding {
        VkCommandBufferEncoding::new(self.context.clone()).expect("command buffer")
    }

    pub fn complete(encoding: VkCommandBufferEncoding) -> VkCommandBufferCompleted {
        encoding.end_encoding().expect("end encoding").submit().wait_until_completed().expect("completion")
    }

    /// # Safety
    /// Same contract as `VkBuffer::get_bytes`.
    pub unsafe fn read<T: NoUninit + AnyBitPattern>(buffer: &VkBuffer) -> Vec<T> {
        bytemuck::pod_collect_to_vec(&unsafe { buffer.get_bytes() })
    }

    /// Reads a range made by `guarded` after asserting every guard element still holds `sentinel`, bit for bit.
    ///
    /// # Safety
    /// Same contract as `VkBuffer::get_bytes`.
    pub unsafe fn read_guarded<T: NoUninit + AnyBitPattern>(
        (buffer, range): &(Arc<VkBuffer>, Range<u64>),
        sentinel: T,
    ) -> Vec<T> {
        let values = unsafe { Self::read::<T>(buffer) };
        let (start, end) = (range.start as usize / size_of::<T>(), range.end as usize / size_of::<T>());
        let mut guards = values[..start].iter().chain(&values[end..]);
        assert!(guards.all(|value| bytemuck::bytes_of(value) == bytemuck::bytes_of(&sentinel)), "a guard was written");
        values[start..end].to_vec()
    }

    /// Bit equality, except that any NaN matches any NaN (payloads after FP32 arithmetic are not portable).
    pub fn assert_bits<T: NoUninit + Float + Debug>(
        expected: &[T],
        actual: &[T],
        case: &str,
    ) {
        assert_eq!(expected.len(), actual.len(), "{case}: length");
        for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
            let same = match expected.is_nan() {
                true => actual.is_nan(),
                false => bytemuck::bytes_of(&expected) == bytemuck::bytes_of(&actual),
            };
            assert!(same, "{case}: element {index}: CPU {expected:?}, Vulkan {actual:?}");
        }
    }

    pub fn assert_clean(&self) {
        self.logger.assert_clean();
    }
}
