use std::sync::{Arc, LockResult, Mutex, MutexGuard};

pub struct SyncShared<T> {
    value: Arc<Mutex<T>>,
}

impl<T> SyncShared<T> {
    pub fn new(value: T) -> Self {
        Self {
            value: Arc::new(Mutex::new(value)),
        }
    }

    pub fn lock(&self) -> LockResult<MutexGuard<'_, T>> {
        self.value.lock()
    }
}

impl<T> Clone for SyncShared<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
        }
    }
}

unsafe impl<T> Send for SyncShared<T> {}

unsafe impl<T> Sync for SyncShared<T> {}
