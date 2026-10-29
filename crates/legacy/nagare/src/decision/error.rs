#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum DecisionSessionError {
    #[error("Backend error: {message}")]
    Backend {
        message: String,
    },
    #[error("Unable to build prompt: {message}")]
    Encoding {
        message: String,
    },
    #[error("Question {index} is invalid: {message}")]
    InvalidQuestion {
        index: i64,
        message: String,
    },
    #[error("No questions")]
    NoQuestions {},
    #[error("Unsupported model")]
    UnsupportedModel {},
    #[error("Unable to perform operation in current state")]
    UnableToPerformOperationInCurrentState {},
    #[error("No response")]
    NoResponse {},
}
