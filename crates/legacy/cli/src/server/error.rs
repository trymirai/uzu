use serde::Serialize;

#[derive(Serialize)]
pub struct OaiErrorResponse {
    pub error: OaiError,
}

#[derive(Serialize)]
pub struct OaiError {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

/// nagare re-wraps backend errors at each layer, so a message can arrive as
/// "Backend error: Backend error: ..."; the wire message keeps only the inner text.
pub fn strip_backend_prefixes(message: &str) -> &str {
    let mut message = message;
    while let Some(inner) = message.strip_prefix("Backend error: ") {
        message = inner;
    }
    message
}
