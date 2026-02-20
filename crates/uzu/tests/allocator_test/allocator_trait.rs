pub trait AllocatorTrait {
    type Buffer;

    fn alloc(
        &self,
        size: usize,
    ) -> Self::Buffer;
    fn free(
        &self,
        buffer: Self::Buffer,
    );
    fn peak_memory(&self) -> usize;
    fn cache_memory(&self) -> usize;
    fn active_memory(&self) -> usize;
}
