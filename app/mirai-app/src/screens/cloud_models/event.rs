use uzu::types::model::Model;

/// Emitted to the shell to start a chat with the chosen cloud model.
pub enum CloudEvent {
    UseModel(Model),
}
