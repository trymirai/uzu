use tokenizers::Tokenizer;

/// Builds a copy of `tokenizer` with `added_tokens` cleared so untrusted user
/// text is always encoded via pure BPE subwords and can never produce control
/// or multimodal special token IDs.
pub fn build_plain_tokenizer(tokenizer: &Tokenizer) -> Result<Tokenizer, String> {
    let raw = tokenizer.to_string(false).map_err(|e| e.to_string())?;
    let mut json: serde_json::Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
    if let Some(obj) = json.as_object_mut() {
        obj.insert("added_tokens".to_string(), serde_json::json!([]));
    }
    let bytes = serde_json::to_vec(&json).map_err(|e| e.to_string())?;
    Tokenizer::from_bytes(&bytes).map_err(|e| e.to_string())
}
