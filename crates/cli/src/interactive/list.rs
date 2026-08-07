use std::collections::HashSet;

use shoji::types::model::Model;

pub struct ModelCheckpoint {
    pub id: String,
    pub name: String,
}

pub fn get_checkpoints(
    models: &[Model],
    model_id: &str,
) -> Vec<ModelCheckpoint> {
    models
        .iter()
        .filter(|model| {
            let Some(family) = &model.family else {
                return false;
            };
            let Some(properties) = &model.properties else {
                return false;
            };

            let family_id = family.identifier.rsplit(':').next().unwrap_or(&family.identifier);
            format!("{family_id}:{}", properties.identifier) == model_id
        })
        .map(|model| ModelCheckpoint {
            id: model.identifier.clone(),
            name: model.name(),
        })
        .collect()
}

pub struct ModelFamily {
    pub id: String,
    pub name: String,
}

pub fn get_families(models: &[Model]) -> Vec<ModelFamily> {
    let mut families = Vec::<ModelFamily>::new();
    let mut ids_set = HashSet::<String>::new();

    for model in models.iter() {
        if let Some(ref family) = model.family {
            if let Some(ref properties) = model.properties {
                let model_id = format!("{}:{}", family.identifier.split(":").last().unwrap(), properties.identifier);
                if !ids_set.contains(&model_id) {
                    ids_set.insert(model_id.clone());
                    families.push(ModelFamily {
                        id: model_id,
                        name: format!("{} {}", family.metadata.name, properties.metadata.name),
                    })
                }
            }
        }
    }

    families
}
