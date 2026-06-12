use std::{os::raw::c_void, ptr::NonNull};

use ndarray::{ArrayView, Dimension, IxDyn};

use crate::{
    array::{Array, ArrayElement},
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
