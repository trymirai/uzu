use std::sync::Arc;

use tokio::sync::{Mutex, MutexGuard};

pub struct SharedAccess<T: ?Sized>(Arc<Mutex<T>>);

impl<T> SharedAccess<T> {
    pub fn new(value: T) -> Self {
        Self(Arc::new(Mutex::new(value)))
    }
}

impl<T: ?Sized> SharedAccess<T> {
    pub async fn lock(&self) -> MutexGuard<'_, T> {
        self.0.lock().await
    }
}

impl<T: ?Sized> Clone for SharedAccess<T> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}
