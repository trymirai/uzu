pub trait NativeBuffer: Send + Sync {
    fn length(&self) -> usize;
    fn id(&self) -> usize;
}
