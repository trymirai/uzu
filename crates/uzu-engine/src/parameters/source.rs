/// Borrowed random-access weights. Absolute file offsets remain 64-bit even
/// when the caller and individual tensor allocations use a 32-bit address space.
/// Sources need only live for model construction; the model retains its buffers.
pub trait ParameterReadSource {
    fn len(&self) -> u64;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn read_exact_at(
        &self,
        destination: &mut [u8],
        offset: u64,
    ) -> std::io::Result<()>;
}
