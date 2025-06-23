use super::speculator::Speculator;

pub struct FixedTokensSpeculator {
    tokens: Vec<Vec<u64>>,
}

impl FixedTokensSpeculator {
    pub fn new(tokens: Vec<Vec<u64>>) -> Self {
        Self {
            tokens,
        }
    }
}

impl Speculator for FixedTokensSpeculator {
    fn generate_proposals(
        &self,
        _prefix: &[u64],
    ) -> Vec<Vec<u64>> {
        self.tokens.clone()
    }
}
