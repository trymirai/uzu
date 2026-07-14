use uzu::types::model::Model;

pub(super) struct PendingGeneration {
    pub(super) text: String,
    pub(super) model: Model,
    pub(super) vendor: String,
}
