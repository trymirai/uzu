use std::sync::{Arc, Mutex};

pub struct Container<T> {
    pub value: Arc<Mutex<T>>,
}

unsafe impl<T> Send for Container<T> {}
unsafe impl<T> Sync for Container<T> {}

impl<T> Container<T> {
    pub fn new(value: T) -> Self {
        Self {
            value: Arc::new(Mutex::new(value)),
        }
    }
}

impl<T> Clone for Container<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
        }
    }
}
