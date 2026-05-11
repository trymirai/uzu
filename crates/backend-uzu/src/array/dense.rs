use std::{ops::Range, os::raw::c_void, ptr::NonNull};

use ndarray::{ArrayView, Dimension, IxDyn};

use crate::{
    Array, ArrayElement,
    backends::common::{AsBufferRangeMut, Backend, DenseBuffer},
};

impl<B: Backend, BufferRange: AsBufferRangeMut<Buffer: DenseBuffer<Backend = B>>> Array<B, BufferRange> {
    pub fn cpu_ptr(&self) -> NonNull<c_void> {
        let buffer_range = self.as_buffer_range_ref();
        let range = buffer_range.range();
        unsafe { buffer_range.buffer().cpu_ptr().add(range.start) }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.cpu_ptr().as_ptr() as *const u8, self.size()) }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        let size = self.size();
        let buffer_range = self.as_buffer_range_mut();
        let range = buffer_range.range();
        unsafe {
            std::slice::from_raw_parts_mut((buffer_range.buffer().cpu_ptr().as_ptr() as *mut u8).add(range.start), size)
        }
    }

    pub fn as_slice<T: ArrayElement>(&self) -> &[T] {
        self.validate_element_type::<T>();
        bytemuck::cast_slice(self.as_bytes())
    }

    pub fn as_slice_mut<T: ArrayElement>(&mut self) -> &mut [T] {
        self.validate_element_type::<T>();
        bytemuck::cast_slice_mut(self.as_bytes_mut())
    }

    pub fn as_view<T: ArrayElement>(&self) -> ArrayView<'_, T, IxDyn> {
        ArrayView::from_shape(IxDyn(self.shape()), self.as_slice::<T>()).expect("Failed to create array view")
    }

    pub fn copy_from_array<
        OtherBackend: Backend,
        OtherBufferRange: AsBufferRangeMut<Buffer: DenseBuffer<Backend = OtherBackend>>,
    >(
        &mut self,
        other: &Array<OtherBackend, OtherBufferRange>,
    ) {
        assert_eq!(self.shape(), other.shape());
        assert_eq!(self.data_type(), other.data_type());

        self.as_bytes_mut().copy_from_slice(other.as_bytes());
    }

    pub fn copy_slice<
        OtherBackend: Backend,
        OtherBufferRange: AsBufferRangeMut<Buffer: DenseBuffer<Backend = OtherBackend>>,
    >(
        &mut self,
        source: &Array<OtherBackend, OtherBufferRange>,
        axis: usize,
        src_range: Range<usize>,
        dst_offset: usize,
    ) {
        assert_eq!(self.shape().len(), source.shape().len(), "Rank mismatch");
        assert!(axis < self.shape().len(), "Axis out of bounds");
        for (dimension_index, (destination_dim, source_dim)) in self.shape().iter().zip(source.shape()).enumerate() {
            if dimension_index != axis {
                assert_eq!(destination_dim, source_dim, "Shapes must match on all non-sliced axes");
            }
        }
        assert_eq!(self.data_type(), source.data_type(), "Arrays must have the same data type");

        let element_size = self.data_type().size_in_bytes();
        let block_size_elements = self.shape().iter().skip(axis + 1).product::<usize>();
        let block_size_bytes = block_size_elements * element_size;
        let num_blocks: usize = self.shape().iter().take(axis).product();
        let source_stride_bytes = source.shape()[axis] * block_size_bytes;
        let destination_stride_bytes = self.shape()[axis] * block_size_bytes;

        let rows_to_copy = src_range.end - src_range.start;
        assert!(dst_offset + rows_to_copy <= self.shape()[axis]);
        assert!(src_range.end <= source.shape()[axis]);
        let source_bytes = source.as_bytes();
        let destination_bytes = self.as_bytes_mut();
        let copy_bytes = rows_to_copy * block_size_bytes;

        for block_index in 0..num_blocks {
            let source_block_start = block_index * source_stride_bytes;
            let destination_block_start = block_index * destination_stride_bytes;

            let source_start = source_block_start + src_range.start * block_size_bytes;
            let destination_start = destination_block_start + dst_offset * block_size_bytes;

            let source_slice = &source_bytes[source_start..source_start + copy_bytes];
            let destination_slice = &mut destination_bytes[destination_start..destination_start + copy_bytes];
            destination_slice.copy_from_slice(source_slice);
        }
    }

    pub fn copy_from_view<T: ArrayElement, D: Dimension>(
        &mut self,
        view: ArrayView<T, D>,
    ) {
        assert_eq!(self.data_type(), T::data_type());

        let destination_slice = self.as_slice_mut::<T>();

        if let Some(source_slice) = view.as_slice_memory_order() {
            assert!(source_slice.len() <= destination_slice.len());
            destination_slice[..source_slice.len()].copy_from_slice(source_slice);
        } else {
            assert!(view.len() <= destination_slice.len());
            for (destination, source) in destination_slice.iter_mut().zip(view.iter()) {
                *destination = *source;
            }
        }
    }

    fn validate_element_type<T: ArrayElement>(&self) {
        assert_eq!(
            T::data_type(),
            self.data_type(),
            "Invalid data type, expected {:?}, actual {:?}",
            T::data_type(),
            self.data_type()
        );
    }
}
