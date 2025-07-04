pub trait Speculator: Send + Sync {
    fn generate_proposals(
        &self,
        prefix: &[u64],
    ) -> Vec<Vec<u64>>;
}
