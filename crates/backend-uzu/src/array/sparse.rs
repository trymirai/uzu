use crate::{
    Array,
    backends::common::{Backend, SparseBuffer},
};

impl<B: Backend, Buf: SparseBuffer<Backend = B>> Array<B, Buf> {}
