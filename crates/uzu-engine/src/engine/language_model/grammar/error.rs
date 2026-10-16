use thiserror::Error;

#[derive(Debug, Error)]
pub enum GrammarError {
    #[error("Grammar rejected the token")]
    GrammarReject,
    #[error("XGrammar error: {0}")]
    XGrammar(String),
    #[error("XGrammar compilation error: {0}")]
    Compile(#[from] xgrammar::CompileError),
}
