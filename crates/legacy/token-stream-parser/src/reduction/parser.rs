use crate::{
    Parser,
    framing::{FramingParserOutput, FramingParserSection},
    reduction::{ReductionParserConfig, ReductionParserError, ReductionParserGroupStack, ReductionParserState},
    types::Token,
};

pub struct ReductionParser {
    state: ReductionParserState,
    stack: ReductionParserGroupStack,
}

impl Parser for ReductionParser {
    type Config = ReductionParserConfig;
    type Input = FramingParserOutput;
    type Output = ();
    type State = ReductionParserState;
    type Error = ReductionParserError;

    fn new(config: Self::Config) -> Result<Self, Self::Error> {
        config.validate()?;
        Ok(Self {
            state: ReductionParserState::new(),
            stack: ReductionParserGroupStack::new(config.groups),
        })
    }

    #[tracing::instrument(skip_all, fields(input = %input))]
    fn push(
        &mut self,
        input: &FramingParserOutput,
    ) -> Result<(), ReductionParserError> {
        match input {
            FramingParserOutput::Added(section) => match section {
                FramingParserSection::Marker(token) => self.handle_frame_marker(token.clone()),
                FramingParserSection::Text(tokens) => self.handle_frame_text_new(tokens.clone()),
            },
            FramingParserOutput::Extended(token) => self.handle_frame_text_extend(token.clone()),
        }
    }

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn reset(&mut self) {
        self.state.sections.clear();
        self.stack.clear();
    }
}

impl ReductionParser {
    #[tracing::instrument(skip(self), fields(token = %token))]
    fn handle_frame_marker(
        &mut self,
        token: Token,
    ) -> Result<(), ReductionParserError> {
        // Close token for an ancestor group
        if self.stack.close_groups_matching_token(&token, &mut self.state)? {
            return Ok(());
        }

        // Open token for a child or root group
        if self.stack.open_group_matching_token(&token, &mut self.state)? {
            return Ok(());
        }

        // No match, add marker as content to current group
        self.stack.open_greedy_group_if_needed(&mut self.state)?;
        self.stack.append_frame_marker(token, &mut self.state)
    }

    #[tracing::instrument(skip_all, fields(tokens = tokens.len()))]
    fn handle_frame_text_new(
        &mut self,
        tokens: Vec<Token>,
    ) -> Result<(), ReductionParserError> {
        self.stack.open_greedy_group_if_needed(&mut self.state)?;
        self.stack.append_frame_text(tokens, &mut self.state)
    }

    #[tracing::instrument(skip(self), fields(token = %token))]
    fn handle_frame_text_extend(
        &mut self,
        token: Token,
    ) -> Result<(), ReductionParserError> {
        self.stack.open_greedy_group_if_needed(&mut self.state)?;
        self.stack.extend_frame_text(token, &mut self.state)
    }
}
