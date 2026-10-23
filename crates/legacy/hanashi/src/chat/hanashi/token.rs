use shoji::types::basic::Token;
use token_stream_parser::types::Token as ParserToken;

use super::{Error, HanashiEncodingImpl};

pub trait ToParserToken {
    type Output;

    fn to_parser_token(self) -> Self::Output;
}

impl ToParserToken for Token {
    type Output = ParserToken;

    fn to_parser_token(self) -> Self::Output {
        ParserToken {
            id: self.id,
            value: self.value,
            is_special: self.is_special,
        }
    }
}

impl HanashiEncodingImpl {
    pub fn flush(&mut self) -> Result<bool, Error> {
        if self.pending_framing_tokens.is_empty() {
            return Ok(false);
        }
        for token in std::mem::take(&mut self.pending_framing_tokens) {
            self.push_token_to_parser(&token, true)?;
        }
        self.parser.flush_extraction();
        self.update_messages_from_parser_state()?;
        Ok(true)
    }

    pub(super) fn push_decoded_token(
        &mut self,
        token: &Token,
    ) -> Result<(), Error> {
        if token.is_special || token.value.is_empty() || self.framing_tokens.contains(&token.value) {
            self.flush()?;
            return self.push_token_to_parser(token, true);
        }
        loop {
            let text: String = self
                .pending_framing_tokens
                .iter()
                .map(|token| token.value.as_str())
                .chain(std::iter::once(token.value.as_str()))
                .collect();
            let marker = self
                .framing_tokens
                .iter()
                .filter(|marker| {
                    !self.tokenizer.get_added_vocabulary().is_special_token(marker) && marker.starts_with(&text)
                })
                .filter_map(|marker| self.tokenizer.token_to_id(marker).map(|id| (marker.len() == text.len(), id)))
                .min_by_key(|(complete, _)| !complete);
            match marker {
                Some((true, id)) => {
                    self.pending_framing_tokens.clear();
                    // Normalize the parser view only; State.tokens retains the sampled IDs.
                    let marker = self.resolve_token(id, false)?;
                    return self.push_token_to_parser(&marker, true);
                },
                Some((false, _)) => {
                    self.pending_framing_tokens.push(token.clone());
                    return Ok(());
                },
                None if self.pending_framing_tokens.is_empty() => return self.push_token_to_parser(token, true),
                None => {
                    self.flush()?;
                },
            }
        }
    }
}
