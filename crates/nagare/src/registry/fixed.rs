use std::{future::Future, pin::Pin};

use tokio::sync::Mutex;

use crate::registry::{Error, Registry, types::Model};

pub struct FixedRegistry {
    identifier: String,
    models: Mutex<Vec<Model>>,
}

impl FixedRegistry {
    pub fn new(
        identifier: String,
        models: Vec<Model>,
    ) -> Self {
        Self {
            identifier,
            models: Mutex::new(models),
        }
    }
}

impl Registry for FixedRegistry {
    fn indentifier(&self) -> String {
        self.identifier.clone()
    }

    fn models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<Model>, Error>> + Send + '_>> {
        Box::pin(async {
            return Ok(self.models.lock().await.clone());
        })
    }
}
