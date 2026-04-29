use bytesize::ByteSize;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("mock registry IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("mock registry HTTP build error: {0}")]
    Http(#[from] http::Error),

    #[error("mock registry HTTP parse error: {0}")]
    HttpParse(#[from] httparse::Error),

    #[error("mock registry invalid header value: {0}")]
    InvalidHeaderValue(#[from] http::header::ToStrError),

    #[error("mock registry request is missing method")]
    MissingRequestMethod,

    #[error("mock registry request is missing path")]
    MissingRequestPath,

    #[error("mock registry request ended before headers")]
    RequestEndedBeforeHeaders,

    #[error("mock registry request headers exceed {limit} bytes")]
    RequestHeadersTooLarge {
        limit: usize,
    },

    #[error("mock registry file not found: {name}")]
    FileNotFound {
        name: String,
    },

    #[error("mock registry file {name} size {size} does not fit {target_type}")]
    FileSizeOutOfRange {
        name: String,
        size: ByteSize,
        target_type: &'static str,
    },

    #[error("mock registry model must have at least one backend")]
    MissingBackend,

    #[error("mock registry file {name} must have a CRC32C hash")]
    MissingCrc32c {
        name: String,
    },
}

pub type Result<T> = std::result::Result<T, Error>;
