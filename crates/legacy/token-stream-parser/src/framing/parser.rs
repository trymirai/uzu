use std::{collections::HashSet, convert::Infallible};

use crate::{
    Parser,
    framing::{FramingParserConfig, FramingParserOutput, FramingParserSection, FramingParserState},
    types::Token,
};

pub struct FramingParser {
    marker_tokens: HashSet<String>,
    state: FramingParserState,
}

impl Parser for FramingParser {
    type Config = FramingParserConfig;
    type Input = Token;
    type Output = FramingParserOutput;
    type State = FramingParserState;
    type Error = Infallible;

    fn new(config: Self::Config) -> Result<Self, Self::Error> {
        Ok(Self {
            marker_tokens: config.tokens.into_iter().collect(),
            state: FramingParserState::new(),
        })
    }

    #[tracing::instrument(skip_all, fields(token = %input))]
    fn push(
        &mut self,
        input: &Token,
    ) -> Result<FramingParserOutput, Infallible> {
        let token = input.clone();
        if self.marker_tokens.contains(&input.value) {
            let output = FramingParserOutput::Added(FramingParserSection::Marker(token.clone()));
            self.state.sections.push(FramingParserSection::Marker(token));
            Ok(output)
        } else {
            match self.state.sections.last_mut() {
                Some(FramingParserSection::Text(tokens)) => {
                    tokens.push(token.clone());
                    Ok(FramingParserOutput::Extended(token))
                },
                _ => {
                    let output = FramingParserOutput::Added(FramingParserSection::Text(vec![token.clone()]));
                    self.state.sections.push(FramingParserSection::Text(vec![token]));
                    Ok(output)
                },
            }
        }
    }

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn reset(&mut self) {
        self.state.sections.clear();
    }
}
