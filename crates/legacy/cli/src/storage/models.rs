use std::collections::HashMap;

use shoji::types::model::ModelIdentifier;

use super::{app::ModelWithState, sections::Section};

pub struct ModelOrganizer;

impl ModelOrganizer {
    /// Filter models for a specific section
    pub fn filter_for_section(
        models: &HashMap<ModelIdentifier, ModelWithState>,
        section: Section,
    ) -> Vec<(ModelIdentifier, ModelWithState)> {
        let mut filtered: Vec<(ModelIdentifier, ModelWithState)> = models
            .iter()
            .filter(|(_, model_with_state)| {
                let model_section = Section::for_model(model_with_state);
                model_section == section
            })
            .map(|(id, model_with_state)| (id.clone(), model_with_state.clone()))
            .collect();

        // Sort by identifier
        filtered.sort_by(|a, b| a.0.cmp(&b.0));
        filtered
    }
}
