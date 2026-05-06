use std::{ops::Range, os::raw::c_void, ptr::NonNull};

use ndarray::{ArrayView, Dimension, IxDyn};

use crate::{
    ArrayElement,
    array::Array,
    backends::common::{Backend, DenseBuffer},
};

impl<B: Backend, Buf: DenseBuffer<Backend = B>> Array<B, Buf> {
    pub fn cpu_ptr(&self) -> NonNull<c_void> {
        unsafe { self.buffer.borrow().cpu_ptr().add(self.offset) }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.cpu_ptr().as_ptr() as *const u8, self.size()) }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.cpu_ptr().as_ptr() as *mut u8, self.size()) }
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

    pub fn copy_from_array<C: Backend>(
        &mut self,
        other: &Array<C>,
    ) {
        assert_eq!(self.shape, other.shape);
        assert_eq!(self.data_type, other.data_type);

        self.as_bytes_mut().copy_from_slice(other.as_bytes());
    }

    pub fn copy_slice<C: Backend>(
        &mut self,
        source: &Array<C>,
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
        assert_eq!(self.data_type(), source.data_type(), "Arrays must have the same data type");

        let elem_size = self.data_type().size_in_bytes();

        // The number of contiguous elements to copy for each slice operation
        let block_size_elems = self.shape().iter().skip(axis + 1).product::<usize>();
        let block_size_bytes = block_size_elems * elem_size;

        // The total number of blocks to copy
        let num_blocks: usize = self.shape().iter().take(axis).product();

        // Strides between the start of each block
        let src_stride_bytes = source.shape()[axis] * block_size_bytes;
        let dst_stride_bytes = self.shape()[axis] * block_size_bytes;

        let rows_to_copy = src_range.end - src_range.start;
        assert!(dst_offset + rows_to_copy <= self.shape()[axis]);
        assert!(src_range.end <= source.shape()[axis]);
        let src_buf = source.as_bytes();
        let dst_buf = self.as_bytes_mut();
        let copy_bytes = rows_to_copy * block_size_bytes;

        for i in 0..num_blocks {
            let src_block_start = i * src_stride_bytes;
            let dst_block_start = i * dst_stride_bytes;

            let src_start = src_block_start + src_range.start * block_size_bytes;
            let dst_start = dst_block_start + dst_offset * block_size_bytes;

            let src_slice = &src_buf[src_start..src_start + copy_bytes];
            let dst_slice = &mut dst_buf[dst_start..dst_start + copy_bytes];
            dst_slice.copy_from_slice(src_slice);
        }
    }

    pub fn copy_from_view<T: ArrayElement, D: Dimension>(
        &mut self,
        view: ArrayView<T, D>,
    ) {
        assert_eq!(self.data_type(), T::data_type());

        let dst_slice = self.as_slice_mut::<T>();

        if let Some(src_slice) = view.as_slice_memory_order() {
            assert!(src_slice.len() <= dst_slice.len());
            dst_slice[..src_slice.len()].copy_from_slice(src_slice);
        } else {
            assert!(view.len() <= dst_slice.len());
            for (d, s) in dst_slice.iter_mut().zip(view.iter()) {
                *d = *s;
            }
        }
    }

    fn validate_element_type<T: ArrayElement>(&self) {
        assert_eq!(
            T::data_type(),
            self.data_type,
            "Invalid data type, expected {:?}, actual {:?}",
            T::data_type(),
            self.data_type
        );
    }
}
