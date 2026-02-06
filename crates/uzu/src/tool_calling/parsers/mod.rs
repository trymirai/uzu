mod json_parser;

use crate::{config::ToolCallFormat, tool_calling::ToolCall};

pub trait ToolCallParser {
    fn parse(
        &self,
        content: String,
    ) -> Option<Vec<ToolCall>>;
}

pub fn create_parser(format: &ToolCallFormat) -> Box<dyn ToolCallParser> {
    match format {
        ToolCallFormat::Json {
            name_key,
            arguments_key,
            separator,
        } => Box::new(json_parser::ToolCallJsonParser::new(
            name_key.clone(),
            arguments_key.clone(),
            separator.clone(),
        )),
    }
}
