use crate::DownloadError;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum PublicProjection {
    #[default]
    None,
    LockedByOther(String),
    StickyError(DownloadError),
}

impl PublicProjection {
    pub(crate) fn failure(&self) -> Option<DownloadError> {
        match self {
            Self::StickyError(error) => Some(error.clone()),
            Self::None | Self::LockedByOther(_) => None,
        }
    }
}
