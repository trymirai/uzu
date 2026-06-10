use regex::{Regex, RegexBuilder};

use crate::session::types::{Error, ParsedText, Text};

pub struct OutputParser {
    regex: Option<Regex>,
}

impl OutputParser {
    pub fn new(regex_str: Option<String>) -> Result<Self, Error> {
        let regex = match regex_str {
            Some(regex_str) => {
                let regex_str = Self::preprocess_regex_string(regex_str);
                let re = RegexBuilder::new(&regex_str).dot_matches_new_line(true).build()?;
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
        enable_thinking: bool,
    ) -> Text {
        let parsed_text = match &self.regex {
            Some(regex) => match regex.captures(&text) {
                Some(captures) => {
                    let chain_of_thought = captures.name("chain_of_thought").map(|m| m.as_str().to_string());
                    let response = captures.name("response").map(|m| m.as_str().to_string());
                    if !enable_thinking {
                        ParsedText {
                            chain_of_thought: None,
                            response: Some(response.unwrap_or_else(|| text.clone())),
                        }
                    } else {
                        ParsedText {
                            chain_of_thought,
                            response,
                        }
                    }
                },
                None => ParsedText {
                    chain_of_thought: None,
                    response: Some(text.clone()),
                },
            },
            None => ParsedText {
                chain_of_thought: None,
                response: Some(text.clone()),
            },
        };
        Text {
            original: text.clone(),
            parsed: parsed_text,
        }
    }
}

impl OutputParser {
    // Needed because of differences in regex syntax between Python and Rust
    fn preprocess_regex_string(regex_str: String) -> String {
        let mut result = regex_str.clone();

        // Replace trailing \Z with \z
        if let Some(prefix) = result.strip_suffix("\\Z") {
            result = format!("{}\\z", prefix);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use test_macros::uzu_test;

    use super::*;

    const QWEN_REGEX: &str = r"(?s)(?:<think>)?(?P<chain_of_thought>.*?)(?:</think>\s*(?P<response>.*))?\Z";

    #[uzu_test]
    fn thinking_enabled_splits_on_close_tag() {
        let parser = OutputParser::new(Some(QWEN_REGEX.to_string())).unwrap();
        let parsed = parser.parse("Let me compute.</think>2 + 2 equals 4.".to_string(), true).parsed;
        assert_eq!(parsed.chain_of_thought.as_deref(), Some("Let me compute."));
        assert_eq!(parsed.response.as_deref(), Some("2 + 2 equals 4."));
    }

    #[uzu_test]
    fn thinking_disabled_keeps_answer_as_response() {
        let parser = OutputParser::new(Some(QWEN_REGEX.to_string())).unwrap();
        let parsed = parser.parse("2 + 2 equals 4.".to_string(), false).parsed;
        assert_eq!(parsed.chain_of_thought, None);
        assert_eq!(parsed.response.as_deref(), Some("2 + 2 equals 4."));
    }

    #[uzu_test]
    fn thinking_disabled_strips_emitted_reasoning_tags() {
        let parser = OutputParser::new(Some(QWEN_REGEX.to_string())).unwrap();
        let parsed = parser.parse("<think>wait, reconsider</think>2 + 2 equals 4.".to_string(), false).parsed;
        assert_eq!(parsed.chain_of_thought, None);
        assert_eq!(parsed.response.as_deref(), Some("2 + 2 equals 4."));
    }

    #[uzu_test]
    fn thinking_enabled_without_close_tag_is_still_reasoning() {
        let parser = OutputParser::new(Some(QWEN_REGEX.to_string())).unwrap();
        let parsed = parser.parse("Let me compute".to_string(), true).parsed;
        assert_eq!(parsed.chain_of_thought.as_deref(), Some("Let me compute"));
        assert_eq!(parsed.response, None);
    }
}
