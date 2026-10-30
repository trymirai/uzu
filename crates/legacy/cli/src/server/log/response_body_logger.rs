use std::{io, pin::Pin, task::Poll};

use rocket::response::Body;
use tokio::io::{AsyncRead, ReadBuf};

pub struct ResponseBodyLogger<'r> {
    body: Body<'r>,
    prefix: String,
    bytes: Vec<u8>,
    is_json: bool,
    logged: bool,
    span: tracing::Span,
}

impl<'r> ResponseBodyLogger<'r> {
    pub fn new(
        body: Body<'r>,
        prefix: String,
        is_json: bool,
        span: tracing::Span,
    ) -> Self {
        Self {
            body,
            prefix,
            bytes: Vec::new(),
            is_json,
            logged: false,
            span,
        }
    }

    fn log(
        &mut self,
        suffix: &str,
    ) {
        if self.logged {
            return;
        }
        self.logged = true;

        let body = if self.is_json {
            serde_json::from_slice::<serde_json::Value>(&self.bytes)
                .map(|value| value.to_string())
                .unwrap_or_else(|_| format!("{:?}", String::from_utf8_lossy(&self.bytes)))
        } else {
            format!("{:?}", String::from_utf8_lossy(&self.bytes))
        };
        self.span.in_scope(|| tracing::debug!("{} body={}{}\n", self.prefix, body, suffix));
    }
}

impl AsyncRead for ResponseBodyLogger<'_> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let span = this.span.clone();
        let _entered = span.enter();
        let filled_before = buffer.filled().len();

        match Pin::new(&mut this.body).poll_read(cx, buffer) {
            Poll::Ready(Ok(())) => {
                let filled_after = buffer.filled().len();
                if filled_after == filled_before {
                    this.log("");
                } else {
                    this.bytes.extend_from_slice(&buffer.filled()[filled_before..filled_after]);
                }
                Poll::Ready(Ok(()))
            },
            Poll::Ready(Err(error)) => {
                this.log(" [read error]");
                Poll::Ready(Err(error))
            },
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for ResponseBodyLogger<'_> {
    fn drop(&mut self) {
        self.log(" [incomplete]");
    }
}
