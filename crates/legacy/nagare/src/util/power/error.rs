#[cfg(not(target_vendor = "apple"))]
pub type Error = std::convert::Infallible;

#[cfg(target_vendor = "apple")]
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Keisoku(#[from] keisoku::KeisokuError),
}
