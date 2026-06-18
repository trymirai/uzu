#[derive(Debug, Clone)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone)]
pub struct ParsedText {
    pub chain_of_thought: Option<String>,
    pub response: Option<String>,
    pub tool_calls: Vec<ParsedToolCall>,
}

#[derive(Debug, Clone)]
pub struct Text {
    pub original: String,
    pub parsed: ParsedText,
}
