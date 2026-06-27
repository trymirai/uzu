//! Cloud Models screen: connect provider API keys (keychain-backed) and pick a
//! cloud model to start a chat. Split into one-type-per-file submodules:
//! [`event`] (the shell event), [`vm`] (the row view-model), and [`view`] (the
//! `CloudModelsView` itself).

mod event;
mod view;
mod vm;

pub use event::CloudEvent;
pub use view::CloudModelsView;
