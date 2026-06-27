//! Local Models screen: a family grid and per-family detail list with
//! download/pause/cancel/delete. `format_size` is re-exported for the other
//! model screens.

mod event;
mod format;
mod view;
mod vm;

pub use event::LocalModelsEvent;
pub(crate) use format::format_size;
pub use view::LocalModelsView;
