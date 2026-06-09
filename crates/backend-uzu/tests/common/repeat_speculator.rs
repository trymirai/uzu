use std::collections::HashMap;

use backend_uzu::prelude::Speculator;

pub struct RepeatSpeculator;

impl Speculator for RepeatSpeculator {
    fn speculate(
        &self,
        prefix: &[u64],
    ) -> HashMap<u64, f32> {
        let mut hm = HashMap::new();

        for (pos, token) in prefix.iter().copied().enumerate() {
            *hm.entry(token).or_insert(0.0) += f32::sqrt((1 + pos) as f32);
        }

        let sum = hm.values().sum::<f32>();

        hm.into_iter().map(|(pos, weight)| (pos, weight / sum)).collect()
    }
}
