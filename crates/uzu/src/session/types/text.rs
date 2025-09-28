#[derive(Debug, Clone)]
pub struct ParsedText {
    pub chain_of_thought: Option<String>,
    pub response: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Text {
    pub original: String,
    pub parsed: ParsedText,
}
