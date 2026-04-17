use std::{collections::HashSet, convert::Infallible};

use json_transform::TransformSchema;

use crate::{
    Parser,
    extraction::{ExtractionParserResolver, ExtractionParserState},
    reduction::ReductionParserState,
};

pub struct ExtractionParserConfig {
    pub schema: Option<TransformSchema>,
    pub sections_compose_groups: HashSet<String>,
}

pub struct ExtractionParser {
    state: ExtractionParserState,
    tree: ExtractionParserResolver,
}

impl Parser for ExtractionParser {
    type Config = ExtractionParserConfig;
    type Input = ReductionParserState;
    type Output = ();
    type State = ExtractionParserState;
    type Error = Infallible;

    fn new(config: ExtractionParserConfig) -> Result<Self, Self::Error> {
        Ok(Self {
            state: ExtractionParserState::new(),
            tree: ExtractionParserResolver::new(config.sections_compose_groups, config.schema),
        })
    }

    #[tracing::instrument(skip_all)]
    fn push(
        &mut self,
        input: &ReductionParserState,
    ) -> Result<(), Infallible> {
        self.state.value = self.tree.compute_output(input);
        Ok(())
    }

    fn state(&self) -> &ExtractionParserState {
        &self.state
    }

    fn reset(&mut self) {
        self.state = ExtractionParserState::new();
        self.tree.reset();
    }
}
