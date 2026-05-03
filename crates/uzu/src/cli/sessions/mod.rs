pub mod chat;

use std::any::Any;

use crate::cli::components::HistoryCellType;

pub trait SessionState: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn is_busy(&self) -> bool;
    fn cancel(&self) -> bool;
    fn status_text(&self) -> Option<String>;
    fn pending_history_cell(&self) -> Option<HistoryCellType>;
}
