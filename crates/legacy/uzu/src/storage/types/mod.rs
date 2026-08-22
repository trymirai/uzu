mod download_phase;
mod download_state;
mod item;

pub use download_phase::DownloadPhase;
pub use download_state::DownloadState;
pub use item::Item;
use tokio::sync::broadcast::Sender as TokioBroadcastSender;
use tokio_stream::wrappers::BroadcastStream as TokioBroadcastStream;

pub type StorageDownloadEvent = (String, DownloadState);
pub type StorageDownloadEventSender = TokioBroadcastSender<StorageDownloadEvent>;
pub type StorageDownloadEventStream = TokioBroadcastStream<StorageDownloadEvent>;
