use crate::{
    Parser,
    extraction::{ExtractionParser, ExtractionParserConfig, ExtractionParserState},
    framing::FramingParser,
    reduction::ReductionParser,
    token_stream::{TokenStreamParserConfig, TokenStreamParserError},
    types::Token,
};

pub struct TokenStreamParser {
    framing: FramingParser,
    reduction: ReductionParser,
    extraction: ExtractionParser,
}

impl Parser for TokenStreamParser {
    type Config = TokenStreamParserConfig;
    type Input = Token;
    type Output = ();
    type State = ExtractionParserState;
    type Error = TokenStreamParserError;

    fn new(config: Self::Config) -> Result<Self, Self::Error> {
        let sections_compose_groups = config.reduction.collect_sections_compose_groups();
        let framing = match FramingParser::new(config.framing_config()) {
            Ok(parser) => parser,
            Err(infallible) => match infallible {},
        };
        let reduction = ReductionParser::new(config.reduction)?;
        let extraction = match ExtractionParser::new(ExtractionParserConfig {
            schema: Some(config.transformation),
            sections_compose_groups,
        }) {
            Ok(parser) => parser,
            Err(infallible) => match infallible {},
        };
        Ok(Self {
            framing,
            reduction,
            extraction,
        })
    }

    #[tracing::instrument(skip_all, fields(token = %input))]
    fn push(
        &mut self,
        input: &Token,
    ) -> Result<(), TokenStreamParserError> {
        let event = match self.framing.push(input) {
            Ok(event) => event,
            Err(infallible) => match infallible {},
        };
        self.reduction.push(&event)?;
        match self.extraction.push(self.reduction.state()) {
            Ok(()) => {},
            Err(infallible) => match infallible {},
        }
        Ok(())
    }

    fn state(&self) -> &ExtractionParserState {
        self.extraction.state()
    }

    fn reset(&mut self) {
        self.framing.reset();
        self.reduction.reset();
        self.extraction.reset();
    }
}

impl TokenStreamParser {
    pub fn framing(&self) -> &FramingParser {
        &self.framing
    }

    pub fn reduction(&self) -> &ReductionParser {
        &self.reduction
    }

    pub fn extraction(&self) -> &ExtractionParser {
        &self.extraction
    }
}
