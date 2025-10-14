use super::speculator::Speculator;
use std::collections::HashMap;

#[derive(Default, Debug, Clone, Copy)]
pub struct EmptySpeculator;

impl Speculator for EmptySpeculator {
    fn speculate(
        &self,
        _prefix: &[u64],
    ) -> HashMap<u64, f32> {
        HashMap::new()
    }
}
