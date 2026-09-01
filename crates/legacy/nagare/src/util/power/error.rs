#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Keisoku(#[from] keisoku::KeisokuError),
}
