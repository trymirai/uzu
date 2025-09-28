use regex::{Regex, RegexBuilder};

use crate::session::types::Error;

pub struct ParsingResult {
    pub chain_of_thought: Option<String>,
    pub response: Option<String>,
}

pub struct OutputParser {
    regex: Option<Regex>,
}

impl OutputParser {
    pub fn new(regex_str: Option<String>) -> Result<Self, Error> {
        let regex = match regex_str {
            Some(regex_str) => {
                let re = RegexBuilder::new(&regex_str)
                    .dot_matches_new_line(true)
                    .build()
                    .map_err(|_| Error::UnableToBuildOutputParserRegex)?;
                Some(re)
            },
            None => None,
        };
        Ok(Self {
            regex,
        })
    }

    pub fn parse(
        &self,
        text: String,
    ) -> ParsingResult {
        match &self.regex {
            Some(regex) => match regex.captures(&text) {
                Some(captures) => {
                    let chain_of_thought = captures
                        .name("chain_of_thought")
                        .map(|m| m.as_str().to_string());
                    let response = captures
                        .name("response")
                        .map(|m| m.as_str().to_string());
                    ParsingResult {
                        chain_of_thought,
                        response,
                    }
                },
                None => ParsingResult {
                    chain_of_thought: None,
                    response: Some(text),
                },
            },
            None => ParsingResult {
                chain_of_thought: None,
                response: Some(text),
            },
        }
    }
}
