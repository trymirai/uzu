use std::collections::HashMap;

use regex::{Regex, RegexBuilder};
use tokenizers::Tokenizer;

use crate::{
    config::ToolCallFormat,
    session::types::{Error, ParsedSection, ParsedText, Text},
    tool_calling::{ToolCall, ToolCallParser, create_parser},
};

struct Token {
    value: String,
    is_special: bool,
}

struct SpecialTokenRange {
    start: usize,
    end: usize,
}

pub struct OutputParser {
    regex: Option<Regex>,
    tokens: Vec<Token>,
    tool_call_parser: Option<Box<dyn ToolCallParser>>,
    tool_call_cache: HashMap<String, Option<Vec<ToolCall>>>,
}

impl OutputParser {
    pub fn new(
        regex_str: Option<String>,
        tool_call_format: Option<ToolCallFormat>,
    ) -> Result<Self, Error> {
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
        let tool_call_parser =
            tool_call_format.and_then(|format| create_parser(&format));
        Ok(Self {
            regex,
            tokens: Vec::new(),
            tool_call_parser,
            tool_call_cache: HashMap::new(),
        })
    }

    pub fn append_token_id(
        &mut self,
        tokenizer: &Tokenizer,
        token_id: u64,
    ) -> Result<(), Error> {
        let value = tokenizer
            .decode(&[token_id as u32], false)
            .map_err(|_| Error::UnableToDecodeText)?;
        let is_special =
            tokenizer.get_added_vocabulary().is_special_token(value.as_str());
        self.tokens.push(Token {
            value,
            is_special,
        });
        Ok(())
    }

    pub fn append_token_ids(
        &mut self,
        tokenizer: &Tokenizer,
        token_ids: &[u64],
    ) -> Result<(), Error> {
        for token_id in token_ids {
            self.append_token_id(tokenizer, *token_id)?;
        }
        Ok(())
    }

    fn build_text_with_special_ranges(
        &self
    ) -> (String, Vec<SpecialTokenRange>) {
        let mut text = String::new();
        let mut special_token_ranges = Vec::new();
        for token in &self.tokens {
            let start = text.len();
            text.push_str(&token.value);
            if token.is_special {
                special_token_ranges.push(SpecialTokenRange {
                    start,
                    end: text.len(),
                });
            }
        }
        (text, special_token_ranges)
    }

    pub fn parse(&mut self) -> Result<Text, Error> {
        let (text, special_token_ranges) =
            self.build_text_with_special_ranges();
        let strip = |start: usize, end: usize| -> String {
            Self::strip_special_tokens(
                text.clone(),
                &special_token_ranges,
                start,
                end,
            )
        };

        let sections = match &self.regex {
            Some(regex) => {
                let mut sections = Vec::new();
                let mut last_end = 0;
                for captures in regex.captures_iter(&text) {
                    let full_match =
                        captures.get(0).ok_or(Error::UnableToDecodeText)?;
                    if full_match.start() > last_end {
                        let content = strip(last_end, full_match.start());
                        if !content.is_empty() {
                            sections.push(ParsedSection::Response(content));
                        }
                    }
                    if let Some(current_match) =
                        captures.name("chain_of_thought")
                    {
                        let content =
                            strip(current_match.start(), current_match.end());
                        if !content.is_empty() {
                            sections
                                .push(ParsedSection::ChainOfThought(content));
                        }
                    } else if let Some(current_match) =
                        captures.name("tool_call")
                    {
                        let content =
                            strip(current_match.start(), current_match.end());
                        if let Some(cached) = self.tool_call_cache.get(&content)
                        {
                            match cached {
                                Some(tool_calls) => {
                                    for tool_call in tool_calls {
                                        sections.push(ParsedSection::ToolCall(
                                            tool_call.clone(),
                                        ));
                                    }
                                },
                                None => {
                                    if !content.is_empty() {
                                        sections.push(
                                            ParsedSection::ToolCallCandidate(
                                                content,
                                            ),
                                        );
                                    }
                                },
                            }
                        } else if let Some(parser) = &self.tool_call_parser {
                            match parser.parse(content.clone()) {
                                Some(tool_calls) => {
                                    self.tool_call_cache.insert(
                                        content,
                                        Some(tool_calls.clone()),
                                    );
                                    for tool_call in tool_calls {
                                        sections.push(ParsedSection::ToolCall(
                                            tool_call,
                                        ));
                                    }
                                },
                                None => {
                                    self.tool_call_cache
                                        .insert(content.clone(), None);
                                    if !content.is_empty() {
                                        sections.push(
                                            ParsedSection::ToolCallCandidate(
                                                content,
                                            ),
                                        );
                                    }
                                },
                            }
                        }
                    } else if let Some(current_match) =
                        captures.name("tool_call_candidate")
                    {
                        let content =
                            strip(current_match.start(), current_match.end());
                        if !content.is_empty() {
                            sections.push(ParsedSection::ToolCallCandidate(
                                content,
                            ));
                        }
                    }
                    last_end = full_match.end();
                }
                if last_end < text.len() {
                    let content = strip(last_end, text.len());
                    if !content.is_empty() {
                        sections.push(ParsedSection::Response(content));
                    }
                }
                sections
            },
            None => {
                let content = strip(0, text.len());
                if content.is_empty() {
                    Vec::new()
                } else {
                    vec![ParsedSection::Response(content)]
                }
            },
        };
        Ok(Text {
            original: text,
            parsed: ParsedText {
                sections,
            },
        })
    }

    pub fn reset(&mut self) {
        self.tokens.clear();
        self.tool_call_cache.clear();
    }
}

impl OutputParser {
    // Needed because of differences in regex syntax between Python and Rust
    fn preprocess_regex_string(regex_str: String) -> String {
        regex_str.replace("\\Z", "\\z")
    }

    fn strip_special_tokens(
        text: String,
        special_token_ranges: &[SpecialTokenRange],
        range_start: usize,
        range_end: usize,
    ) -> String {
        let mut result = String::new();
        let mut cursor = range_start;
        for special_range in special_token_ranges {
            if special_range.end <= range_start
                || special_range.start >= range_end
            {
                continue;
            }
            if cursor < special_range.start {
                result.push_str(&text[cursor..special_range.start]);
            }
            cursor = special_range.end;
        }
        if cursor < range_end {
            result.push_str(&text[cursor..range_end]);
        }
        result
    }
}
