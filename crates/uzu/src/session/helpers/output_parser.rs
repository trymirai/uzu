use regex::{Regex, RegexBuilder};

use crate::session::types::{Error, SplittedText, Text};

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
    ) -> Text {
        let splitted_text = match &self.regex {
            Some(regex) => match regex.captures(&text) {
                Some(captures) => {
                    let chain_of_thought = captures
                        .name("chain_of_thought")
                        .map(|m| m.as_str().to_string());
                    let response = captures
                        .name("response")
                        .map(|m| m.as_str().to_string());
                    SplittedText {
                        chain_of_thought,
                        response,
                    }
                },
                None => SplittedText {
                    chain_of_thought: None,
                    response: Some(text.clone()),
                },
            },
            None => SplittedText {
                chain_of_thought: None,
                response: Some(text.clone()),
            },
        };
        Text {
            original: text.clone(),
            splitted: splitted_text,
        }
    }
}
