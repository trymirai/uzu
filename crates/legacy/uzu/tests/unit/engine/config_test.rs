use super::EngineConfig;

#[test]
fn debug_redacts_secret_values() {
    let config = EngineConfig::default().with_huggingface_api_key("hf-secret".to_string());
    let debug = format!("{config:?}");

    assert!(!debug.contains("hf-secret"), "debug output leaked a token: {debug}");
}

#[test]
fn serialization_omits_hugging_face_token() {
    let config = EngineConfig::default().with_huggingface_api_key("hf-secret".to_string());
    let json = serde_json::to_string(&config).unwrap();

    assert!(!json.contains("hf-secret"), "serialization leaked a token: {json}");
    assert!(!json.contains("huggingface_api_key"));
}
