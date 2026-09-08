mod active_task;
#[cfg(target_vendor = "apple")]
mod apple;
mod backend;
mod backend_event;
mod backend_event_sender;
mod backend_progress;
mod download_generation;
mod universal;

pub use active_task::ActiveTask;
#[cfg(target_vendor = "apple")]
pub use apple::AppleBackend;
pub use backend::Backend;
pub use backend_event::BackendEvent;
pub use backend_event_sender::BackendEventSender;
pub use backend_progress::BackendProgress;
pub use download_generation::DownloadGeneration;
pub use universal::UniversalBackend;
