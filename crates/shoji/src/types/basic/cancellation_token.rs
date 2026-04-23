use tokio_util::sync::CancellationToken as TokioCancellationToken;

#[bindings::export(Class)]
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    inner: TokioCancellationToken,
}

#[bindings::export(Implementation)]
impl CancellationToken {
    #[bindings::export(Factory)]
    pub fn new() -> Self {
        Self {
            inner: TokioCancellationToken::new(),
        }
    }

    #[bindings::export(Method)]
    pub fn cancel(&self) {
        self.inner.cancel();
    }

    #[bindings::export(Getter)]
    pub fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
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
