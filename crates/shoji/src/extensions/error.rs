use crate::extensions::PcmBatchError;

#[bindings::export(Error)]
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ExtensionsError {
    #[error(transparent)]
    PcmBatch(#[from] PcmBatchError),
}
