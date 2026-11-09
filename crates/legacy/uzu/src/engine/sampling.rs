use std::path::Path;

use shoji::types::{basic::SamplingMethod, model::Model};
use uzu_engine::engine::{language_model::stream::SamplingMethod as UzuSamplingMethod, resolve_model_sampling_method};

use crate::engine::{Engine, EngineError};

impl Engine {
    /// Read a local model's sampling defaults without loading its weights or creating a session.
    pub async fn model_sampling_defaults(
        &self,
        model: &Model,
    ) -> Result<Option<SamplingMethod>, EngineError> {
        let Some(path) = self.model_path(model).await else {
            return Ok(None);
        };
        read_sampling_defaults(Path::new(&path)).await
    }
}

async fn read_sampling_defaults(model_path: &Path) -> Result<Option<SamplingMethod>, EngineError> {
    resolve_model_sampling_method(model_path).await.map(|method| method.map(sampling_method)).map_err(|error| {
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
    fn sampling_method_maps_all_fields() {
        assert_eq!(sampling_method(UzuSamplingMethod::Greedy), SamplingMethod::Greedy {});
        assert_eq!(
            sampling_method(UzuSamplingMethod::Stochastic {
                temperature: Some(0.6),
                top_k: Some(20),
                top_p: Some(0.95),
                min_p: Some(0.05),
                repetition_penalty: Some(1.1),
                suffix_repetition_length: Some(64),
            }),
            SamplingMethod::Stochastic {
                temperature: Some(0.6),
                top_k: Some(20),
                top_p: Some(0.95),
                min_p: Some(0.05),
                repetition_penalty: Some(1.1),
                suffix_repetition_length: Some(64),
            }
        );
        assert_eq!(
            sampling_method(UzuSamplingMethod::Stochastic {
                temperature: None,
                top_k: None,
                top_p: None,
                min_p: None,
                repetition_penalty: None,
                suffix_repetition_length: None,
            }),
            SamplingMethod::Stochastic {
                temperature: None,
                top_k: None,
                top_p: None,
                min_p: None,
                repetition_penalty: None,
                suffix_repetition_length: None,
            }
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
