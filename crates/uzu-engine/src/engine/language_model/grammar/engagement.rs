pub enum GrammarEngagementState {
    Always,
    Triggered {
        trigger_token_id: u64,
        trigger_distance: Option<usize>,
    },
}

impl GrammarEngagementState {
    pub fn is_engaged(&self) -> bool {
        match self {
            Self::Always => true,
            Self::Triggered {
                trigger_token_id: _,
                trigger_distance,
            } => trigger_distance.is_some(),
        }
    }

    pub fn accept_token(
        &mut self,
        token_id: u64,
    ) {
        match self {
            Self::Always => (),
            Self::Triggered {
                trigger_token_id,
                trigger_distance,
            } => {
                if let Some(trigger_distance) = trigger_distance {
                    *trigger_distance += 1;
                } else if token_id == *trigger_token_id {
                    *trigger_distance = Some(0);
                }
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
                trigger_token_id: _,
                trigger_distance,
            } => {
                let num_grammar_tokens = usize::min(trigger_distance.unwrap_or(0), num_tokens);
                *trigger_distance = trigger_distance.and_then(|x| x.checked_sub(num_tokens));
                num_grammar_tokens
            },
        }
    }
}
