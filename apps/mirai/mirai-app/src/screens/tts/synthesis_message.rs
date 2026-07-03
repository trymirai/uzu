use uzu::types::basic::{CancelToken, PcmBatch};

pub(super) enum SynthesisMessage {
    Started(CancelToken),
    Batch(PcmBatch),
    Done,
    Error(String),
}
