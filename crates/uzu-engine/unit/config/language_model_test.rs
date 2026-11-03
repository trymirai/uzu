use serde_json::json;
use uzu_engine_macros::uzu_test;

use super::{LanguageModelConfig, VocabularyOrder};

#[uzu_test]
fn vocabulary_order_accepts_legacy_and_frequency_exports() {
    let mut config = json!({
        "type": "LanguageModelConfig",
        "token_codec_config": {"type": "RawTextCodecConfig"},
        "decoder_config": {
            "embedding_config": {"type": "TiedEmbeddingConfig", "input_scale": null,
                "logit_soft_cap": null, "logit_scale": null},
            "transformer_config": {
                "layer_configs": [], "model_dim": 128, "hidden_dim": 256,
                "output_norm_config": {"epsilon": 0.00001, "scale_offset": null,
                    "upcast_mode": "only_normalization", "subtract_mean": false,
                    "has_scale": true, "has_biases": false}
            },
            "vocab_size": 64, "ple_model_config": null, "embedding_norm_config": null
        },
        "generation_config": {
            "stop_token_ids": [], "temperature": null, "top_k": null, "top_p": null,
            "min_p": null, "banned_tokens": null, "repetition_penalty": null,
            "presence_penalty": null, "frequency_penalty": null, "suffix_repetition_length": null
        }
    });
    let legacy: LanguageModelConfig = serde_json::from_value(config.clone()).unwrap();
    assert_eq!(legacy.vocabulary_order, VocabularyOrder::Original);
    for (value, expected) in [("original", VocabularyOrder::Original), ("frequency", VocabularyOrder::Frequency)] {
        config["vocabulary_order"] = json!(value);
        let parsed: LanguageModelConfig = serde_json::from_value(config.clone()).unwrap();
        assert_eq!(parsed.vocabulary_order, expected);
        assert_eq!(serde_json::to_value(parsed).unwrap()["vocabulary_order"], value);
    }
    for invalid in [json!("unknown"), json!(null), json!(3)] {
        config["vocabulary_order"] = invalid;
        assert!(serde_json::from_value::<LanguageModelConfig>(config.clone()).is_err());
    }
    config.as_object_mut().unwrap().remove("vocabulary_order");
    config["vocabulary_oder"] = json!("frequency");
    assert!(serde_json::from_value::<LanguageModelConfig>(config).is_err());
}
