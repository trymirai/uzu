use shoji::types::session::classification::{
    Message as ShojiMessage, Output as ShojiOutput, Role as ShojiRole, Stats as ShojiStats,
};

use crate::{
    classifier::ClassificationOutput,
    session::types::{Input, Message, Role},
};

pub fn build_input(messages: &[ShojiMessage]) -> Input {
    let input_messages: Vec<Message> = messages
        .iter()
        .map(|message| {
            let role = match message.role {
                ShojiRole::User => Role::User,
                ShojiRole::Assistant => Role::Assistant,
            };
            Message {
                role,
                content: message.content.clone(),
                reasoning_content: None,
            }
        })
        .collect();
    Input::Messages(input_messages)
}

pub fn build_output(output: &ClassificationOutput) -> ShojiOutput {
    let logits = output.logits.iter().map(|value| *value as f64).collect();
    let probabilities = output.probabilities.iter().map(|(label, value)| (label.clone(), *value as f64)).collect();
    let stats = ShojiStats {
        preprocessing_duration: output.stats.preprocessing_duration,
        forward_pass_duration: output.stats.forward_pass_duration,
        postprocessing_duration: output.stats.postprocessing_duration,
        total_duration: output.stats.total_duration,
        tokens_count: output.stats.tokens_count as i64,
        tokens_per_second: output.stats.tokens_per_second,
        predicted_label: output.stats.predicted_label.clone(),
        confidence: output.stats.confidence as f64,
    };
    ShojiOutput {
        logits,
        probabilities,
        stats,
    }
}
