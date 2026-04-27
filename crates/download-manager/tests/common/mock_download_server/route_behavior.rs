#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RouteBehavior {
    Normal,
    SlowChunks {
        chunk_size: usize,
        delay_ms: u64,
    },
    StallAt {
        byte_offset: u64,
    },
    DisconnectAt {
        byte_offset: u64,
    },
    RetryThenOk {
        failures: u64,
        status: u16,
    },
    RedirectTo {
        target: String,
    },
    CorruptBody,
    WrongContentLength,
    NoContentLength,
    NoRangeSupport,
    InvalidRangeResponse,
}
