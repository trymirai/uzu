use std::path::Path;

use shoji::types::{basic::SamplingMethod, model::Model};
use uzu_engine::engine::{language_model::stream::SamplingMethod as UzuSamplingMethod, resolve_model_sampling_method};

use super::{Engine, EngineError};

impl Engine {
    /// Read a local model's sampling defaults without loading its weights or creating a session.
    pub async fn model_sampling_defaults(
        &self,
        model: &Model,
    ) -> Result<Option<SamplingMethod>, EngineError> {
        let Some(path) = self.model_path(model).await else {
            return Ok(None);
        };
        tokio::task::spawn_blocking(move || read_sampling_defaults(Path::new(&path))).await.map_err(|error| {
            EngineError::TokioError {
                message: error.to_string(),
            }
        })?
    }
}

fn read_sampling_defaults(model_path: &Path) -> Result<Option<SamplingMethod>, EngineError> {
    resolve_model_sampling_method(model_path).map(|method| method.map(sampling_method)).map_err(|error| {
        EngineError::ModelConfig {
            message: error.to_string(),
        }
    })
}

fn sampling_method(method: UzuSamplingMethod) -> SamplingMethod {
    match method {
        UzuSamplingMethod::Greedy => SamplingMethod::Greedy {},
        UzuSamplingMethod::Stochastic {
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            suffix_repetition_length,
        } => SamplingMethod::Stochastic {
            temperature: temperature.map(sampling_float),
            top_k: top_k.map(i64::from),
            top_p: top_p.map(sampling_float),
            min_p: min_p.map(sampling_float),
            repetition_penalty: repetition_penalty.map(sampling_float),
            suffix_repetition_length: suffix_repetition_length.map(i64::from),
        },
    }
}

fn sampling_float(value: f32) -> f64 {
    // Keep the shortest decimal that round-trips to the engine's f32 value.
    // Fixed decimal rounding would turn sufficiently small positive temperatures into zero.
    value.to_string().parse().expect("an f32 decimal parses as f64")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_sampling_defaults_without_weights_and_preserves_disabled_filters() {
        let directory = tempfile::tempdir().unwrap();
        let config = serde_json::json!({
            "type": "LanguageModelConfig",
            "token_codec_config": { "type": "RawTextCodecConfig" },
            "decoder_config": {
                "embedding_config": {
                    "type": "TiedEmbeddingConfig", "input_scale": null,
                    "logit_soft_cap": null, "logit_scale": null,
                },
                "transformer_config": {
                    "layer_configs": [], "model_dim": 4, "hidden_dim": 4,
                    "output_norm_config": {
                        "epsilon": 0.00001, "scale_offset": null,
                        "upcast_mode": "only_normalization", "subtract_mean": false,
                        "has_scale": true, "has_biases": false,
                    },
                },
                "vocab_size": 4, "ple_model_config": null, "embedding_norm_config": null,
            },
            "generation_config": {
                "stop_token_ids": [], "temperature": null, "top_k": 20, "top_p": null, "min_p": null,
                "banned_tokens": null, "repetition_penalty": null, "presence_penalty": null,
                "frequency_penalty": null, "suffix_repetition_length": null,
            },
        });
        std::fs::write(directory.path().join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();
        assert_eq!(
            read_sampling_defaults(directory.path()).unwrap(),
            Some(SamplingMethod::Stochastic {
                temperature: None,
                top_k: Some(20),
                top_p: None,
                min_p: None,
                repetition_penalty: None,
                suffix_repetition_length: None,
            })
        );
    }

    #[test]
    fn sampling_numbers_are_readable_and_round_trip_even_when_tiny() {
        assert_eq!(sampling_float(0.6), 0.6);
        assert_eq!(sampling_float(0.000_001), 0.000_001);
        for value in [f32::MIN_POSITIVE, f32::from_bits(1), 0.000_001, 0.6, 0.999_999_94, f32::MAX] {
            assert_eq!(sampling_float(value) as f32, value);
        }
    }
}
