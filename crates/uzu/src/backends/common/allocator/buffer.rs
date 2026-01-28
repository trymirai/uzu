pub trait Buffer: Send + Sync {
    fn length(&self) -> usize;
    fn id(&self) -> usize;
}
