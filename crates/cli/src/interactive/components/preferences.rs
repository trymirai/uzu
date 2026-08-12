use serde::{Deserialize, Serialize};

use crate::{
    common::model_capabilities::ThinkingPreference,
    interactive::{components::Theme, sampling::SamplingPreferences},
};

#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct Preferences {
    pub theme: Theme,
    pub thinking: ThinkingPreference,
    pub sampling: SamplingPreferences,
    pub selected_model_id: Option<String>,
}
