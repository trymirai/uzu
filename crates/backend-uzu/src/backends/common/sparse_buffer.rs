use std::{fmt::Debug, os::raw::c_void, ptr::NonNull};

use crate::backends::common::{Backend, Buffer, Context, SparsePages, SparsePagesOperation};

#[derive(Debug)]
pub struct SparseBuffer<B: Backend> {
    pages: Box<dyn SparsePages<Backend = B>>,

    /// Total sum of add_length from `extend`
    length: usize,
}

impl<B: Backend> SparseBuffer<B> {
    pub fn new(pages: Box<impl SparsePages<Backend = B> + 'static>) -> Result<Self, B::Error> {
        Ok(Self {
            pages,
            length: 0,
        })
    }

    pub fn buffer(&self) -> &B::Buffer {
        self.pages.buffer()
    }

    pub fn buffer_mut(&mut self) -> &mut B::Buffer {
        self.pages.buffer_mut()
    }

    pub fn capacity(&self) -> usize {
        self.pages.total_pages() * self.pages.page_size()
    }

    pub fn extend(
        &mut self,
        add_length: usize,
    ) {
        assert!(self.length + add_length <= self.capacity(), "SparseBuffer capacity overflow");

        let page_size = self.pages.page_size();
        let mapped_pages = self.length.div_ceil(page_size);

        let new_length = self.length + add_length;
        let new_mapped_pages = new_length.div_ceil(page_size);
        self.length = new_length;

        if new_mapped_pages == mapped_pages {
            return;
        }

        let operation = SparsePagesOperation {
            map: true,
            pages: mapped_pages..new_mapped_pages,
        };
        self.pages.execute(&[operation]);
    }
}

impl<B: Backend> Buffer for SparseBuffer<B> {
    type Backend = B;

    fn set_label(
        &mut self,
        label: Option<&str>,
    ) {
        self.pages.buffer_mut().set_label(label);
    }

    /// Returns 0, because of private memory
    fn cpu_ptr(&self) -> NonNull<c_void> {
        self.pages.buffer().cpu_ptr()
    }

    fn gpu_ptr(&self) -> usize {
        self.pages.buffer().gpu_ptr()
    }

    fn length(&self) -> usize {
        self.length
    }
}

pub trait SparseBufferContext {
    type Backend: Backend;

    fn create_sparse_buffer(
        &self,
        capacity: usize,
    ) -> Result<SparseBuffer<Self::Backend>, <Self::Backend as Backend>::Error>;
}

impl<C: Context> SparseBufferContext for C {
    type Backend = C::Backend;

    fn create_sparse_buffer(
        &self,
        capacity: usize,
    ) -> Result<SparseBuffer<Self::Backend>, <Self::Backend as Backend>::Error> {
        let pages = self.create_sparse_pages(capacity)?;
        SparseBuffer::<C::Backend>::new(pages)
    }
}
