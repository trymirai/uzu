use super::Backend;

pub trait CommandBuffer {
    type Backend: Backend;

    fn with_encoder<T>(
        &self,
        callback: impl FnOnce(&<Self::Backend as Backend>::EncoderRef) -> T,
    ) -> T;
}
