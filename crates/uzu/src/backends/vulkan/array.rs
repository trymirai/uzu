use core::ffi;
use ash::prelude::VkResult;
use ash::vk;
use crate::backends::vulkan::buffer::VkBuffer;
use crate::{array_size_in_bytes, Array, DataType};

pub struct VkArray {
    buffer: VkBuffer,
    shape: Box<[usize]>,
    data_type: DataType,
    size: usize,
    offset: usize,
    label: String,
}

impl VkArray {
    pub fn new(
        buffer: VkBuffer,
        shape: &[usize],
        data_type: DataType
    ) -> Self {
        Self::new_with_offset_and_label(buffer, shape, data_type, 0, "".to_string())
    }

    pub fn new_with_offset_and_label(
        buffer: VkBuffer,
        shape: &[usize],
        data_type: DataType,
        offset: usize,
        label: String
    ) -> Self {
        let size = array_size_in_bytes(&shape, data_type);
        Self { buffer, shape: shape.into(), data_type, size, offset, label }
    }

    fn map_memory(&self) -> VkResult<*mut ffi::c_void> {
        unsafe {
            self.buffer.device().map_memory(
                self.buffer.memory(),
                self.offset as vk::DeviceSize,
                self.size as vk::DeviceSize,
                vk::MemoryMapFlags::empty()
            )
        }
    }

    fn unmap_memory(&self) {
        unsafe {
            self.buffer.device().unmap_memory(self.buffer.memory())
        }
    }
}

impl Array for VkArray {
    type BackendBuffer = VkBuffer;

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data_type(&self) -> DataType {
        self.data_type
    }

    fn label(&self) -> String {
        "".to_string()
    }

    fn buffer(&self) -> &[u8] {
        unsafe {
            let pts = self.map_memory().unwrap() as *const u8;
            let slice = std::slice::from_raw_parts(pts, self.size);
            self.unmap_memory();
            slice
        }
    }

    fn buffer_mut(&mut self) -> &mut [u8] {
        unsafe {
            let pts = self.map_memory().unwrap() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(pts, self.size);
            self.unmap_memory();
            slice
        }
    }

    fn backend_buffer(&self) -> &Self::BackendBuffer {
        &self.buffer
    }
}