use tokio_util::sync::CancellationToken as TokioCancellationToken;

#[bindings::export(Class)]
#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    inner: TokioCancellationToken,
}

impl CancelToken {
    pub fn new() -> Self {
        Self {
            inner: TokioCancellationToken::new(),
        }
    }
}

#[bindings::export(Implementation)]
impl CancelToken {
    #[bindings::export(Method(Factory))]
    pub fn create() -> Self {
        Self::new()
    }

    #[bindings::export(Method)]
    pub fn cancel(&self) {
        self.inner.cancel();
    }

    #[bindings::export(Method(Getter))]
    pub fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }
}

impl CancelToken {
    pub fn inner(&self) -> &TokioCancellationToken {
        &self.inner
    }

    pub fn into_inner(self) -> TokioCancellationToken {
        self.inner
    }
}

impl From<TokioCancellationToken> for CancelToken {
    fn from(inner: TokioCancellationToken) -> Self {
        Self {
            inner,
        }
    }
}

impl From<CancelToken> for TokioCancellationToken {
    fn from(token: CancelToken) -> Self {
        token.inner
    }
}
