use std::collections::HashMap;

pub struct ClassifierOutput {
    pub logits: Vec<f32>,
    pub probabilities: HashMap<String, f32>,
}
