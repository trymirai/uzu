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

    pub fn parse_raw(
        &self,
        raw_text: String,
        visible_text: String,
    ) -> Text {
        if let Some(parsed_text) = Self::parse_gemma_4_channels(&raw_text) {
            return Text {
                original: parsed_text.response.clone().unwrap_or_default(),
                parsed: parsed_text,
            };
        }

        let parsed_text = match &self.regex {
            Some(regex) => match regex.captures(&visible_text) {
                Some(captures) => {
                    let chain_of_thought = captures.name("chain_of_thought").map(|m| m.as_str().to_string());
                    let response = captures.name("response").map(|m| m.as_str().to_string());
                    ParsedText {
                        chain_of_thought,
                        response,
                    }
                },
                None => ParsedText {
                    chain_of_thought: None,
                    response: Some(visible_text.clone()),
                },
            },
            None => ParsedText {
                chain_of_thought: None,
                response: Some(visible_text.clone()),
            },
        };
        Text {
            original: parsed_text.response.clone().unwrap_or_else(|| visible_text.clone()),
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

    fn parse_gemma_4_channels(text: &str) -> Option<ParsedText> {
        const THOUGHT_OPEN: &str = "<|channel>thought";
        const THOUGHT_CLOSE: &str = "<channel|>";

        let thought_start = text.find(THOUGHT_OPEN)?;
        let thought_content_start = thought_start + THOUGHT_OPEN.len();
        let after_thought_open = &text[thought_content_start..];

        let Some(thought_close_relative) = after_thought_open.find(THOUGHT_CLOSE) else {
            return Some(ParsedText {
                chain_of_thought: Some(after_thought_open.to_string()),
                response: None,
            });
        };

        let thought_close = thought_content_start + thought_close_relative;
        let response_start = thought_close + THOUGHT_CLOSE.len();
        let response = Self::strip_gemma_4_stop_tokens(&text[response_start..]).to_string();

        Some(ParsedText {
            chain_of_thought: Some(text[thought_content_start..thought_close].to_string()),
            response: Some(response),
        })
    }

    fn strip_gemma_4_stop_tokens(text: &str) -> &str {
        const STOP_TOKENS: [&str; 2] = ["<turn|>", "<eos>"];

        let stop_index = STOP_TOKENS.iter().filter_map(|stop_token| text.find(stop_token)).min().unwrap_or(text.len());
        &text[..stop_index]
    }
}

#[cfg(test)]
mod tests {
    use super::OutputParser;

    #[test]
    fn test_output_parser_gemma_4_channels() {
        let parser = OutputParser::new(None).unwrap();
        let text = parser.parse_raw(
            "<|channel>thought\nThinking Process<channel|>Final answer.<turn|>".to_string(),
            "thought\nThinking ProcessFinal answer.".to_string(),
        );

        assert_eq!(text.original, "Final answer.");
        assert_eq!(text.parsed.response.as_deref(), Some("Final answer."));
        assert_eq!(text.parsed.chain_of_thought.as_deref(), Some("\nThinking Process"));
    }

    #[test]
    fn test_output_parser_gemma_4_incomplete_thought() {
        let parser = OutputParser::new(None).unwrap();
        let text = parser
            .parse_raw("<|channel>thought\nThinking Process".to_string(), "thought\nThinking Process".to_string());

        assert_eq!(text.original, "");
        assert_eq!(text.parsed.response, None);
        assert_eq!(text.parsed.chain_of_thought.as_deref(), Some("\nThinking Process"));
    }

    #[test]
    fn test_output_parser_visible_text_fallback() {
        let parser = OutputParser::new(None).unwrap();
        let text = parser.parse_raw("Hello.<turn|>".to_string(), "Hello.".to_string());

        assert_eq!(text.original, "Hello.");
        assert_eq!(text.parsed.response.as_deref(), Some("Hello."));
        assert_eq!(text.parsed.chain_of_thought, None);
    }
}
