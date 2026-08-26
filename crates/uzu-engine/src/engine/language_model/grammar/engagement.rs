/// Decides when the grammar starts constraining sampling. `Always` engages from
/// the first generated token. `Triggered` stays dormant until the generated
/// stream contains the trigger token sequence (a reasoning model's
/// end-of-thinking tag), so reasoning tokens flow unconstrained and only the
/// final answer is grammar-constrained.
pub enum GrammarEngagementState {
    Always,
    Triggered {
        trigger_sequence: Vec<u64>,
        /// Match-automaton state after each accepted token: the length of the
        /// longest prefix of `trigger_sequence` that is a suffix of the stream,
        /// held at `trigger_sequence.len()` once the trigger has completed. One
        /// entry per accepted token so `rollback` restores any earlier state
        /// exactly.
        match_history: Vec<usize>,
    },
}

/// The last `matched` tokens of the stream equal `trigger_sequence[..matched]`,
/// so the new match length is the longest prefix of `trigger_sequence` that is
/// a suffix of `trigger_sequence[..matched]` followed by `token_id`.
fn advance(
    trigger_sequence: &[u64],
    matched: usize,
    token_id: u64,
) -> usize {
    if matched == trigger_sequence.len() {
        return matched;
    }
    let mut length = matched + 1;
    while length > 0 {
        if trigger_sequence[length - 1] == token_id
            && trigger_sequence[matched + 1 - length..matched] == trigger_sequence[..length - 1]
        {
            return length;
        }
        length -= 1;
    }
    0
}

impl GrammarEngagementState {
    pub fn is_engaged(&self) -> bool {
        match self {
            Self::Always => true,
            Self::Triggered {
                trigger_sequence,
                match_history,
            } => match_history.last() == Some(&trigger_sequence.len()),
        }
    }

    pub fn accept_token(
        &mut self,
        token_id: u64,
    ) {
        match self {
            Self::Always => (),
            Self::Triggered {
                trigger_sequence,
                match_history,
            } => {
                let matched = match_history.last().copied().unwrap_or(0);
                match_history.push(advance(trigger_sequence, matched, token_id));
            },
        }
    }

    pub fn rollback(
        &mut self,
        num_tokens: usize,
    ) -> usize {
        match self {
            Self::Always => num_tokens,
            Self::Triggered {
                trigger_sequence,
                match_history,
            } => {
                let full_match = trigger_sequence.len();
                let mut num_grammar_tokens = 0;
                for _ in 0..num_tokens {
                    if match_history.pop().is_none() {
                        break;
                    }
                    // A token was fed to the matcher iff the trigger was already
                    // complete when it was accepted, i.e. the state before it
                    // was a full match.
                    if match_history.last().copied().unwrap_or(0) == full_match {
                        num_grammar_tokens += 1;
                    }
                }
                num_grammar_tokens
            },
        }
    }
}

#[cfg(test)]
#[path = "../../../../unit/engine/language_model/grammar/engagement_test.rs"]
mod tests;
