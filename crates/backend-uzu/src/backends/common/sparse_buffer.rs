use crate::backends::common::Buffer;

pub trait SparseBuffer: Buffer {
    fn capacity(&self) -> usize;

    fn extend(
        &mut self,
        add_length: usize,
    );
}
