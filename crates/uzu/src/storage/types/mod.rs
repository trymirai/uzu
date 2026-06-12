mod crc_snapshot;
mod download_phase;
mod download_state;
mod item;

pub use crc_snapshot::CrcSnapshot;
pub use download_phase::DownloadPhase;
pub use download_state::{DownloadState, reduce_file_download_states};
pub use item::Item;
use tokio::sync::broadcast::Sender as TokioBroadcastSender;
use tokio_stream::wrappers::BroadcastStream as TokioBroadcastStream;

pub type StorageDownloadEvent = (String, DownloadState);
pub type StorageDownloadEventSender = TokioBroadcastSender<StorageDownloadEvent>;
pub type StorageDownloadEventStream = TokioBroadcastStream<StorageDownloadEvent>;
