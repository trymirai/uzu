use super::Backend;

pub trait CopyEncoder {
    type Backend: Backend;

    fn encode_copy(
        &self,
        src: &<Self::Backend as Backend>::NativeBuffer,
        dst: &<Self::Backend as Backend>::NativeBuffer,
        size: usize,
    );
}
