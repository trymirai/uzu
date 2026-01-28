use super::{AllocError, Buffer};

pub trait Device: Sized + Send + Sync {
    type Buffer: Buffer;
    type ResourceOptions: Copy + Send + Sync;

    fn create_buffer(
        &self,
        size: usize,
        options: Self::ResourceOptions,
    ) -> Result<Self::Buffer, AllocError>;
}
