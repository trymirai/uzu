use half::{bf16, f16};
use metal::{Buffer as MTLBuffer, MTLResourceOptions};
use mpsgraph::TensorData;
use objc2::rc::Retained;

use super::utils::mps_shape;
use crate::{Array, ArrayElement, DataType, array::array_size_in_bytes};

/// Represents an n-dimensional array for Metal computation
#[derive(Debug, Clone)]
pub struct MetalArray {
    /// Metal buffer containing the array data
    buffer: MTLBuffer,
    /// Shape of the array (dimensions)
    shape: Box<[usize]>,
    /// Data type of the elements
    data_type: DataType,
}

impl Array for MetalArray {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data_type(&self) -> DataType {
        self.data_type
    }

    fn buffer(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.buffer.contents() as *const u8,
                self.size_in_bytes(),
            )
        }
    }

    fn buffer_mut(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.buffer.contents() as *mut u8,
                self.size_in_bytes(),
            )
        }
    }
}

impl MetalArray {
    /// Create a new Array
    pub unsafe fn new(
        buffer: MTLBuffer,
        shape: &[usize],
        data_type: DataType,
    ) -> Self {
        let required_bytes = array_size_in_bytes(shape, data_type);
        assert!(
            required_bytes <= buffer.length() as usize,
            "Shape {:?} with data type {:?} requires {} bytes, but buffer length is {} bytes",
            shape,
            data_type,
            required_bytes,
            buffer.length()
        );
        Self {
            buffer,
            shape: shape.into(),
            data_type,
        }
    }

    /// Wraps the underlying MTLBuffer into MPSTensorData for use with MPSGraph.
    pub unsafe fn to_mps_tensor_data(&mut self) -> Retained<TensorData> {
        TensorData::from_buffer(
            &self.buffer,
            &mps_shape(&self.shape),
            self.data_type.into(),
        )
    }

    /// Returns the underlying MTLBuffer.
    pub unsafe fn mtl_buffer(&mut self) -> &MTLBuffer {
        &self.buffer
    }

    /// Splits the array into multiple `MetalArray` objects along the first dimension (rows) only.
    /// This is a zero-copy operation that creates child buffers pointing to the same memory.
    ///
    /// # Arguments
    /// * `row_lengths` - Array of row counts for each split. Must sum to total rows.
    ///
    /// # Example
    /// ```
    /// use uzu::backends::metal::MetalArray;
    /// use uzu::DataType;
    /// use metal::MTLResourceOptions;
    ///
    /// # let device = metal::Device::system_default().unwrap();
    /// # let shape = [10usize, 6usize];
    /// # let num_elems = shape[0] * shape[1];
    /// # let data = vec![0.0f32; num_elems];
    /// # let buffer_size_bytes = (num_elems * std::mem::size_of::<f32>()) as u64;
    /// # let buffer = device.new_buffer_with_data(
    /// #     data.as_ptr() as *const _,
    /// #     buffer_size_bytes,
    /// #     MTLResourceOptions::StorageModeShared,
    /// # );
    /// # let mut array = unsafe { MetalArray::new(buffer, &shape, DataType::F32) };
    ///
    /// // Original: [10, 6] tensor (10 rows × 6 columns)
    /// let splits = unsafe { array.row_split_no_copy(&[2, 5, 3]) };
    /// // Results: [2,6], [5,6], [3,6] - each split has full 6-column width
    /// assert_eq!(splits.len(), 3);
    /// ```
    pub unsafe fn row_split_no_copy(
        &mut self,
        row_lengths: &[usize],
    ) -> Box<[MetalArray]> {
        assert!(!self.shape.is_empty(), "Cannot split scalar or empty tensor");
        assert_eq!(
            row_lengths.iter().sum::<usize>(),
            self.shape[0],
            "Sum of row lengths must equal number of rows"
        );

        let elem_size = self.data_type.size_in_bytes();
        let row_size_bytes =
            self.shape.iter().skip(1).product::<usize>() * elem_size;
        let device = self.buffer.device().to_owned();

        let mut results = Vec::with_capacity(row_lengths.len());
        let mut current_row_offset = 0usize;

        for &num_rows in row_lengths {
            let offset_bytes = current_row_offset * row_size_bytes;
            let required_len = (num_rows * row_size_bytes).max(48);

            // Create no-copy child buffer
            let base_ptr = self.buffer.contents() as *mut u8;
            let slice_ptr =
                unsafe { base_ptr.add(offset_bytes) } as *mut std::ffi::c_void;

            let child_buffer = device.new_buffer_with_bytes_no_copy(
                slice_ptr,
                required_len as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );

            // Create split shape
            let mut split_shape = self.shape.to_vec();
            split_shape[0] = num_rows;

            // Create MetalArray instead of TensorData
            let split_array = unsafe {
                MetalArray::new(child_buffer, &split_shape, self.data_type)
            };

            results.push(split_array);
            current_row_offset += num_rows;
        }

        results.into_boxed_slice()
    }
}

impl MetalArray {
    pub fn copy_from_array(
        &mut self,
        array: &MetalArray,
    ) {
        match self.data_type {
            DataType::BF16 => self.copy_from_array_with_type::<bf16>(array),
            DataType::F16 => self.copy_from_array_with_type::<f16>(array),
            DataType::F32 => self.copy_from_array_with_type::<f32>(array),
            DataType::F64 => self.copy_from_array_with_type::<f64>(array),
            DataType::I8 => self.copy_from_array_with_type::<i8>(array),
            DataType::U8 => self.copy_from_array_with_type::<u8>(array),
            DataType::I16 => self.copy_from_array_with_type::<i16>(array),
            DataType::U16 => self.copy_from_array_with_type::<u16>(array),
            DataType::I32 => self.copy_from_array_with_type::<i32>(array),
            DataType::U32 => self.copy_from_array_with_type::<u32>(array),
            DataType::I64 => self.copy_from_array_with_type::<i64>(array),
            DataType::U64 => self.copy_from_array_with_type::<u64>(array),
            _ => panic!("Unsupported data type"),
        }
    }

    fn copy_from_array_with_type<T: ArrayElement>(
        &mut self,
        array: &MetalArray,
    ) {
        assert_eq!(self.shape, array.shape);
        assert_eq!(self.data_type, array.data_type);
        self.as_slice_mut::<T>()
            .unwrap()
            .copy_from_slice(array.as_slice::<T>().unwrap());
    }
}
