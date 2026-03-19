use std::collections::HashMap;

use uzu::prelude::Speculator;

pub struct StaticSpeculator {
    responses: HashMap<Vec<u64>, HashMap<u64, f32>>,
    default_response: Option<HashMap<u64, f32>>,
}

impl StaticSpeculator {
    pub fn new(responses: HashMap<Vec<u64>, HashMap<u64, f32>>) -> Self {
        Self {
            responses,
            default_response: None,
        }
    }

    pub fn with_default_response(default_response: HashMap<u64, f32>) -> Self {
        Self {
            responses: HashMap::new(),
            default_response: Some(default_response),
        }
    }
}

impl Speculator for StaticSpeculator {
    fn speculate(
        &self,
        prefix: &[u64],
    ) -> HashMap<u64, f32> {
        self.responses.get(prefix).cloned().or_else(|| self.default_response.clone()).unwrap_or_default()
    }
}

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
