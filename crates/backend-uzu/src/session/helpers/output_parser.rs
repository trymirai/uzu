use regex::{Regex, RegexBuilder};

use crate::session::types::{Error, ParsedText, ParsedToolCall, Text};

pub struct OutputParser {
    regex: Option<Regex>,
    tool_call_regex: Regex,
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
        let tool_call_regex = RegexBuilder::new(r"<tool_call>(.*?)</tool_call>").dot_matches_new_line(true).build()?;
        Ok(Self {
            regex,
            tool_call_regex,
        })
    }

    pub fn parse(
        &self,
        text: String,
        enable_thinking: bool,
    ) -> Text {
        let tool_calls = self.extract_tool_calls(&text);
        let stripped = self.tool_call_regex.replace_all(&text, "");
        let body = stripped.trim();
        let mut parsed_text = match &self.regex {
            Some(regex) => match regex.captures(body) {
                Some(captures) => {
                    let chain_of_thought = captures.name("chain_of_thought").map(|m| m.as_str().to_string());
                    let response = captures.name("response").map(|m| m.as_str().to_string());
                    if !enable_thinking {
                        ParsedText {
                            chain_of_thought: None,
                            response: Some(response.unwrap_or_else(|| body.to_string())),
                            tool_calls,
                        }
                    } else {
                        ParsedText {
                            chain_of_thought,
                            response,
                            tool_calls,
                        }
                    }
                },
                None => ParsedText {
                    chain_of_thought: None,
                    response: Some(body.to_string()),
                    tool_calls,
                },
            },
            None => ParsedText {
                chain_of_thought: None,
                response: Some(body.to_string()),
                tool_calls,
            },
        };
        if parsed_text.response.as_deref().is_some_and(str::is_empty) {
            parsed_text.response = None;
        }
        Text {
            original: text,
            parsed: parsed_text,
        }
    }
}

impl OutputParser {
    fn extract_tool_calls(
        &self,
        text: &str,
    ) -> Vec<ParsedToolCall> {
        self.tool_call_regex
            .captures_iter(text)
            .filter_map(|captures| {
                let body = captures.get(1)?.as_str().trim();
                let value: serde_json::Value = serde_json::from_str(body).ok()?;
                let name = value.get("name")?.as_str()?.to_string();
                let arguments =
                    value.get("arguments").map(|arguments| arguments.to_string()).unwrap_or_else(|| "{}".to_string());
                Some(ParsedToolCall {
                    name,
                    arguments,
                })
            })
            .collect()
    }

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
    use proc_macros::uzu_test;

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

    #[uzu_test]
    fn extracts_tool_call_and_clears_response() {
        let parser = OutputParser::new(Some(QWEN_REGEX.to_string())).unwrap();
        let parsed = parser
            .parse(
                "<think>need weather</think>\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>".to_string(),
                true,
            )
            .parsed;
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "get_weather");
        assert_eq!(parsed.tool_calls[0].arguments, "{\"city\":\"Paris\"}");
        assert_eq!(parsed.chain_of_thought.as_deref(), Some("need weather"));
        assert_eq!(parsed.response, None);
    }

    #[uzu_test]
    fn plain_response_has_no_tool_calls() {
        let parser = OutputParser::new(Some(QWEN_REGEX.to_string())).unwrap();
        let parsed = parser.parse("just text".to_string(), false).parsed;
        assert!(parsed.tool_calls.is_empty());
        assert_eq!(parsed.response.as_deref(), Some("just text"));
    }
}
