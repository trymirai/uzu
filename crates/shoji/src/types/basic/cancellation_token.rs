use tokio_util::sync::CancellationToken as TokioCancellationToken;

#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    inner: TokioCancellationToken,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            inner: TokioCancellationToken::new(),
        }
    }

    pub fn cancel(&self) {
        self.inner.cancel();
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }

    pub fn child(&self) -> Self {
        Self {
            inner: self.inner.child_token(),
        }
    }
}

impl CancellationToken {
    pub fn inner(&self) -> &TokioCancellationToken {
        &self.inner
    }

    pub fn into_inner(self) -> TokioCancellationToken {
        self.inner
    }
}

impl From<TokioCancellationToken> for CancellationToken {
    fn from(inner: TokioCancellationToken) -> Self {
        Self {
            inner,
        }
    }
}

impl From<CancellationToken> for TokioCancellationToken {
    fn from(token: CancellationToken) -> Self {
        token.inner
    }
}
