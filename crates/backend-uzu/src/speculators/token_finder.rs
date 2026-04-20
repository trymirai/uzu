pub struct TokenFinder;

impl TokenFinder {
    pub fn find_candidate_pred_token(
        sequence: &[u64],
        max_ngram_size: usize,
    ) -> Option<u64> {
        let input_length = sequence.len();

        for ngram_size in (1..=max_ngram_size).rev() {
            if ngram_size >= input_length {
                continue;
            }

            let ngram_start = input_length - ngram_size;
            let ngram = &sequence[ngram_start..input_length];

            let max_start_index = input_length - ngram_size;
            for window_start in 0..max_start_index {
                let window = &sequence[window_start..(window_start + ngram_size)];

                if window == ngram {
                    let start_idx = window_start + ngram_size;

                    if start_idx < (input_length - ngram_size) {
                        return Some(sequence[start_idx]);
                    }
                }
            }
        }

        None
    }
}
