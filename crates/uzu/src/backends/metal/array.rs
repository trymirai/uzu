use std::ops::Range;

use half::{bf16, f16};
use metal::{Buffer as MTLBuffer, MTLResourceOptions};
use mpsgraph::TensorData;
use objc2::rc::Retained;

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
    /// Byte offset into the buffer (for indexed async buffers)
    offset: usize,
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
                (self.buffer.contents() as *const u8).add(self.offset),
                self.size_in_bytes(),
            )
        }
    }

    fn buffer_mut(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                (self.buffer.contents() as *mut u8).add(self.offset),
                self.size_in_bytes(),
            )
        }
    }
}

impl MetalArray {
    /// Create a new Array at offset 0
    pub unsafe fn new(
        buffer: MTLBuffer,
        shape: &[usize],
        data_type: DataType,
    ) -> Self {
        unsafe { Self::new_with_offset(buffer, shape, data_type, 0) }
    }

    /// Create a new Array at a byte offset into the buffer.
    /// Used for indexed async buffers where each pass uses a different offset.
    pub unsafe fn new_with_offset(
        buffer: MTLBuffer,
        shape: &[usize],
        data_type: DataType,
        offset: usize,
    ) -> Self {
        let required_bytes = array_size_in_bytes(shape, data_type);
        assert!(
            offset + required_bytes <= buffer.length() as usize,
            "Shape {:?} with data type {:?} at offset {} requires {} bytes total, but buffer length is {} bytes",
            shape,
            data_type,
            offset,
            offset + required_bytes,
            buffer.length()
        );
        Self {
            buffer,
            shape: shape.into(),
            data_type,
            offset,
        }
    }

    /// Returns the byte offset into the underlying buffer
    pub fn buffer_offset(&self) -> usize {
        self.offset
    }

    /// Wraps the underlying MTLBuffer into MPSTensorData for use with MPSGraph.
    pub unsafe fn to_mps_tensor_data(&mut self) -> Retained<TensorData> {
        TensorData::new_with_mtl_buffer(
            &self.buffer,
            &self.shape,
            self.data_type.into(),
            None,
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

    pub fn copy_slice(
        &mut self,
        source: &Self,
        axis: usize,
        src_range: Range<usize>,
        dst_offset: usize,
    ) {
        assert_eq!(self.shape().len(), source.shape().len(), "Rank mismatch");
        assert!(axis < self.shape().len(), "Axis out of bounds");
        for (i, (a, b)) in self.shape().iter().zip(source.shape()).enumerate() {
            if i != axis {
                assert_eq!(a, b, "Shapes must match on all non-sliced axes");
            }
        }
        assert_eq!(
            self.data_type(),
            source.data_type(),
            "Arrays must have the same data type"
        );

        let elem_size = self.data_type().size_in_bytes();

        // The number of contiguous elements to copy for each slice operation
        let block_size_elems =
            self.shape().iter().skip(axis + 1).product::<usize>();
        let block_size_bytes = block_size_elems * elem_size;

        // The total number of blocks to copy
        let num_blocks: usize = self.shape().iter().take(axis).product();

        // Strides between the start of each block
        let src_stride_bytes = source.shape()[axis] * block_size_bytes;
        let dst_stride_bytes = self.shape()[axis] * block_size_bytes;

        let rows_to_copy = src_range.end - src_range.start;
        assert!(dst_offset + rows_to_copy <= self.shape()[axis]);
        assert!(src_range.end <= source.shape()[axis]);

        let src_buf = source.buffer();
        let dst_buf = self.buffer_mut();
        let copy_bytes = rows_to_copy * block_size_bytes;

        for i in 0..num_blocks {
            let src_block_start = i * src_stride_bytes;
            let dst_block_start = i * dst_stride_bytes;

            let src_start =
                src_block_start + src_range.start * block_size_bytes;
            let dst_start = dst_block_start + dst_offset * block_size_bytes;

            let src_slice = &src_buf[src_start..src_start + copy_bytes];
            let dst_slice = &mut dst_buf[dst_start..dst_start + copy_bytes];
            dst_slice.copy_from_slice(src_slice);
        }
    }
}
