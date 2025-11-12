use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ClassificationOutput {
    pub logits: Vec<f32>,
    pub probabilities: HashMap<String, f32>,
    pub stats: ClassificationStats,
}

#[derive(Debug, Clone)]
pub struct ClassificationStats {
    pub forward_pass_duration: f64,
    pub total_duration: f64,
}

impl ClassificationStats {
    pub fn new(
        forward_pass_duration: f64,
        total_duration: f64,
    ) -> Self {
        Self {
            forward_pass_duration,
            total_duration,
        }
    }
}
