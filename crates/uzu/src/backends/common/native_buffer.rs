use super::Backend;

pub trait NativeBuffer: Send + Sync {
    type Backend: Backend;

    fn length(&self) -> usize;
    fn id(&self) -> usize;
}
