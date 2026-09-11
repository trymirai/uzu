use reqwest::StatusCode;

pub trait IsTransient {
    fn is_transient(&self) -> bool;
}

impl IsTransient for StatusCode {
    fn is_transient(&self) -> bool {
        self.is_server_error() || matches!(*self, StatusCode::REQUEST_TIMEOUT | StatusCode::TOO_MANY_REQUESTS)
    }
}

impl IsTransient for reqwest::Error {
    fn is_transient(&self) -> bool {
        !(self.is_builder() || self.is_redirect() || self.is_decode() || self.is_status())
    }
}
