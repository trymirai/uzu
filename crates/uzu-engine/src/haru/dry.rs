use std::collections::HashMap;

use super::config::HaruDryDefaults;

/// CPU-side DRY repetition penalty tracker matching `haru_mlx/haru_mlx/sampling.py`.
///
/// Exempt tokens (`PAD`, `EOP`, `EOS`) are never appended to `history` and
/// never penalized.
#[derive(Debug, Clone)]
pub struct DryState {
    history: Vec<u32>,
    exempt: [u32; 3],
    multiplier: f32,
    base: f32,
    allowed_length: usize,
    lookback: usize,
    max_match_length: usize,
    max_penalty: f32,
}

impl DryState {
    pub fn new(pad_token_id: u32, eop_token_id: u32, eos_token_id: u32, cfg: &HaruDryDefaults) -> Self {
        Self {
            history: Vec::with_capacity(cfg.lookback.max(256)),
            exempt: [pad_token_id, eop_token_id, eos_token_id],
            multiplier: cfg.multiplier,
            base: cfg.base,
            allowed_length: cfg.allowed_length,
            lookback: cfg.lookback,
            max_match_length: cfg.max_match_length,
            max_penalty: cfg.max_penalty,
        }
    }

    pub fn is_exempt(&self, token_id: u32) -> bool {
        self.exempt.contains(&token_id)
    }

    pub fn push(&mut self, token_id: u32) {
        if !self.is_exempt(token_id) {
            self.history.push(token_id);
        }
    }

    pub fn extend<I: IntoIterator<Item = u32>>(&mut self, tokens: I) {
        for tok in tokens {
            self.push(tok);
        }
    }

    pub fn clear(&mut self) {
        self.history.clear();
    }

    /// Returns `(token_ids, penalties)` to subtract from `logits`.
    pub fn compute_penalties(&self) -> (Vec<u32>, Vec<f32>) {
        if self.multiplier <= 0.0 || self.history.len() < self.allowed_length + 1 {
            return (Vec::new(), Vec::new());
        }
        let n = self.history.len();
        let stop = n.saturating_sub(self.lookback);
        let mut max_match_for_next: HashMap<u32, usize> = HashMap::new();

        for j in (stop..=n - 2).rev() {
            let candidate = self.history[j + 1];
            if self.is_exempt(candidate) {
                continue;
            }
            let max_len = self.max_match_length.min(j + 1);
            let mut match_len = 0usize;
            while match_len < max_len && self.history[n - 1 - match_len] == self.history[j - match_len] {
                match_len += 1;
            }
            if match_len >= self.allowed_length {
                max_match_for_next
                    .entry(candidate)
                    .and_modify(|prev| {
                        if match_len > *prev {
                            *prev = match_len;
                        }
                    })
                    .or_insert(match_len);
            }
        }

        if max_match_for_next.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let mut ids = Vec::with_capacity(max_match_for_next.len());
        let mut pens = Vec::with_capacity(max_match_for_next.len());
        for (tok, len) in max_match_for_next {
            let exp = (len - self.allowed_length) as i32;
            let penalty = (self.multiplier * self.base.powi(exp)).min(self.max_penalty);
            ids.push(tok);
            pens.push(penalty);
        }
        (ids, pens)
    }
}
