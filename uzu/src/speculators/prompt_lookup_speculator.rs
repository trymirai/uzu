use super::speculator::Speculator;

pub struct PromptLookupSpeculator {
    n: usize,
    len: usize,
}

impl PromptLookupSpeculator {
    pub fn new() -> Self {
        Self {
            n: 0,
            len: 0,
        }
    }

    pub fn new_with_params(
        n: usize,
        len: usize,
    ) -> Self {
        Self {
            n,
            len,
        }
    }
}

impl Speculator for PromptLookupSpeculator {
    fn generate_proposals(
        &self,
        _prefix: &[u64],
    ) -> Vec<Vec<u64>> {
        Vec::new()
    }
}
