use std::{
    collections::HashMap,
    sync::{
        Arc, RwLock,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::generator::generator::Generator;

static CONTEXT_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_context_id() -> u64 {
    CONTEXT_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

pub struct ContextHandle {
    pub id: u64,
    pub tokens: Vec<u64>,
    generator: Generator,
}

impl ContextHandle {
    pub fn new(
        tokens: Vec<u64>,
        generator: Generator,
    ) -> Self {
        let id = next_context_id();
        Self {
            id,
            tokens,
            generator,
        }
    }

    pub fn clone_generator(&self) -> Generator {
        self.generator.clone_with_prefix()
    }
}

pub struct ContextRegistry {
    inner: RwLock<HashMap<u64, Arc<RwLock<ContextHandle>>>>,
}

impl ContextRegistry {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
        }
    }

    pub fn insert(
        &self,
        handle: ContextHandle,
    ) -> u64 {
        let id = handle.id;
        self.inner.write().unwrap().insert(id, Arc::new(RwLock::new(handle)));
        id
    }

    pub fn get(
        &self,
        id: &u64,
    ) -> Option<Arc<RwLock<ContextHandle>>> {
        self.inner.read().unwrap().get(id).cloned()
    }

    pub fn remove(
        &self,
        id: &u64,
    ) {
        self.inner.write().unwrap().remove(id);
    }
}
