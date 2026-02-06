mod json_parser;
mod pythonic_parser;

use json_parser::ToolCallJsonParser;
use pythonic_parser::ToolCallPythonicParser;

use crate::{config::ToolCallFormat, tool_calling::ToolCall};

pub trait ToolCallParser {
    fn parse(
        &self,
        content: String,
    ) -> Option<Vec<ToolCall>>;
}

pub fn create_parser(
    format: &ToolCallFormat
) -> Option<Box<dyn ToolCallParser>> {
    match format {
        ToolCallFormat::Json {
            name_key,
            arguments_key,
            separator,
        } => Some(Box::new(ToolCallJsonParser::new(
            name_key.clone(),
            arguments_key.clone(),
            separator.clone(),
        ))),
        ToolCallFormat::Pythonic {
            function_regex,
            argument_separator,
            string_token,
        } => Some(Box::new(ToolCallPythonicParser::new(
            function_regex.clone(),
            argument_separator.clone(),
            string_token.clone(),
        )?)),
    }
}
