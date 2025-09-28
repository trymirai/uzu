#[derive(Debug, Clone)]
pub struct SplittedText {
    pub chain_of_thought: Option<String>,
    pub response: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Text {
    pub original: String,
    pub splitted: SplittedText,
}
