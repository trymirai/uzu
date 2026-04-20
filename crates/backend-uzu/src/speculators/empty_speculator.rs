use std::collections::HashMap;

use super::speculator::Speculator;

#[derive(Default, Debug, Clone, Copy)]
pub struct EmptySpeculator;

impl Speculator for EmptySpeculator {
    fn speculate(
        &self,
        prefix: &[u64],
    ) -> HashMap<u64, f32> {
        let _ = prefix;
        HashMap::new()
    }
}
