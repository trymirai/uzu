use uzu::types::model::Model;

/// Emitted when the user picks an installed model to chat with.
pub enum LocalModelsEvent {
    UseModel(Model),
}
