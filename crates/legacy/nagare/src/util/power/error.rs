#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub type Error = std::convert::Infallible;

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Keisoku(#[from] keisoku::KeisokuError),
}
