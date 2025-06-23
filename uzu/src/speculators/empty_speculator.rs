use super::speculator::Speculator;

#[derive(Default)]
pub struct EmptySpeculator;

impl Speculator for EmptySpeculator {
    fn generate_proposals(
        &self,
        _prefix: &[u64],
    ) -> Vec<Vec<u64>> {
        Vec::new()
    }
}
