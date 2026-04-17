use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ClassificationOutput {
    pub logits: Box<[f32]>,
    pub probabilities: HashMap<String, f32>,
    pub stats: ClassificationStats,
}

#[derive(Debug, Clone)]
pub struct ClassificationStats {
    // Timing breakdown
    pub preprocessing_duration: f64,
    pub forward_pass_duration: f64,
    pub postprocessing_duration: f64,
    pub total_duration: f64,

    // Token metrics
    pub tokens_count: u64,
    pub tokens_per_second: f64,

    pub predicted_label: String,
    pub confidence: f32,
}

impl ClassificationStats {
    pub fn new(
        preprocessing_duration: f64,
        forward_pass_duration: f64,
        postprocessing_duration: f64,
        total_duration: f64,
        tokens_count: u64,
        predicted_label: String,
        confidence: f32,
    ) -> Self {
        let tokens_per_second = if forward_pass_duration > 0.0 {
            tokens_count as f64 / forward_pass_duration
        } else {
            0.0
        };

        Self {
            preprocessing_duration,
            forward_pass_duration,
            postprocessing_duration,
            total_duration,
            tokens_count,
            tokens_per_second,
            predicted_label,
            confidence,
        }
    }
}
