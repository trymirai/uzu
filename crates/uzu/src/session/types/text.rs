use crate::tool_calling::ToolCall;

#[derive(Debug, Clone)]
pub enum ParsedSection {
    ChainOfThought(String),
    Response(String),
    ToolCallCandidate(String),
    ToolCall(ToolCall),
}

#[derive(Debug, Clone)]
pub struct ParsedText {
    pub sections: Vec<ParsedSection>,
}

impl ParsedText {
    pub fn chain_of_thought(&self) -> Option<String> {
        self.sections.iter().find_map(|section| match section {
            ParsedSection::ChainOfThought(text) => {
                Some(text.clone().trim().to_string())
            },
            _ => None,
        })
    }

    pub fn tool_calls(&self) -> Vec<&ToolCall> {
        self.sections
            .iter()
            .filter_map(|section| match section {
                ParsedSection::ToolCall(tool_call) => Some(tool_call),
                _ => None,
            })
            .collect()
    }

    pub fn response(&self) -> Option<String> {
        let parts: Vec<String> = self
            .sections
            .iter()
            .filter_map(|section| match section {
                ParsedSection::Response(text) => Some(text.clone()),
                _ => None,
            })
            .collect();
        let response = parts.join("").trim().to_string();
        if response.is_empty() {
            None
        } else {
            Some(response)
        }
    }
}

#[derive(Debug, Clone)]
pub struct Text {
    pub original: String,
    pub parsed: ParsedText,
}
