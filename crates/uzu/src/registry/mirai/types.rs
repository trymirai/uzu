use serde::{Deserialize, Serialize};
use shoji::types::model::{Accessibility, Entity, Model, Quantization, Specialization};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct ResponseModel {
    pub identifier: String,
    pub entity_identifiers: Vec<String>,
    pub specializations: Vec<Specialization>,
    pub number_of_parameters: Option<i64>,
    pub quantization: Option<Quantization>,
    pub accessibility: Accessibility,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Response {
    models: Vec<ResponseModel>,
    entities: Vec<Entity>,
}

impl Response {
    pub fn models(&self) -> Option<Vec<Model>> {
        let models = self
            .models
            .iter()
            .flat_map(|response_model| {
                let entities = self
                    .entities
                    .iter()
                    .filter(|entity| response_model.entity_identifiers.contains(&entity.identifier))
                    .map(|entity| entity.clone())
                    .collect::<Vec<_>>();

                if entities.len() != response_model.entity_identifiers.len() {
                    return None;
                }

                let model = Model {
                    identifier: response_model.identifier.clone(),
                    entities: entities,
                    specializations: response_model.specializations.clone(),
                    number_of_parameters: response_model.number_of_parameters.clone(),
                    quantization: response_model.quantization.clone(),
                    accessibility: response_model.accessibility.clone(),
                };
                return Some(model);
            })
            .collect::<Vec<_>>();

        if models.len() != self.models.len() {
            return None;
        }

        return Some(models);
    }
}
